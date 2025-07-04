import os
import random
import urllib.request

import numpy as np
import ndjson
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from torchvision import datasets, transforms


def download_and_preprocess_data(doodles=True):
    if doodles is True:
        cats, dogs = load_doodles_data()
        X, y = get_x_and_y_data(cats, dogs)
        return X, y
    
    else: # CIFAR10
        X, X_flat, y = load_cifar10_cats_and_dogs()
        return X, X_flat, y


def load_doodles_data():
    os.makedirs("data", exist_ok=True)

    urls = {
        "data/cat.ndjson": "https://storage.googleapis.com/quickdraw_dataset/full/simplified/cat.ndjson",
        "data/dog.ndjson": "https://storage.googleapis.com/quickdraw_dataset/full/simplified/dog.ndjson",
    }

    for filename, url in urls.items():
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

    print("Download complete.")

    with open("data/cat.ndjson") as f:
        cats = ndjson.load(f)

    with open("data/dog.ndjson") as f:
        dogs = ndjson.load(f)

    print(f"Loaded {len(cats)} cats and {len(dogs)} dogs.")
    plot_random_dataset_images(cats, dogs)
    return cats, dogs


def get_x_and_y_data(cats, dogs):
    X = [drawing_to_image(c, size=28) for c in cats[:500]] + [drawing_to_image(d, size=28) for d in dogs[:500]]
    y = [0]*500 + [1]*500  # 0=cat, 1=dog

    X = np.array(X) / 255.0
    # print("X shape before reshape:", X.shape)  # Expect (1000, 28, 28)
    X = X.reshape(-1, 28*28)
    # print("X shape after reshape:", X.shape)   # Expect (1000, 784)

    y = np.array(y)
    return X, y


def drawing_to_image(data, size=100, stroke_width=1):
    # Flatten all points to compute bounds
    drawing = data["drawing"]
    all_x = [x for stroke in drawing for x in stroke[0]]
    all_y = [y for stroke in drawing for y in stroke[1]]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Scale and center
    scale = 255.0 / max(max_x - min_x, max_y - min_y)
    img = Image.new("L", (256, 256), 255)
    draw = ImageDraw.Draw(img)

    for stroke in drawing:
        points = [
            ((x - min_x) * scale, (y - min_y) * scale)
            for x, y in zip(stroke[0], stroke[1])
        ]
        for i in range(len(points) - 1):
            draw.line([points[i], points[i+1]], fill=0, width=stroke_width)

    # Resize to 28x28
    img = img.resize((size, size)).convert('L')
    return np.array(img)


def plot_random_dataset_images(cats, dogs, dataset_type='doodles'):
    plt.figure(figsize=(8, 4))

    if dataset_type == 'doodles':
        cat_samples = random.sample(cats, 2)
        dog_samples = random.sample(dogs, 2)
        samples = [(drawing_to_image(c), "Cat") for c in cat_samples] + \
                  [(drawing_to_image(d), "Dog") for d in dog_samples]

    elif dataset_type == 'cifar':
        cat_samples = random.sample(cats, 2)
        dog_samples = random.sample(dogs, 2)
        samples = []

        for img, label in zip(cat_samples + dog_samples, ['Cat']*2 + ['Dog']*2):
            if isinstance(img, torch.Tensor):
                if img.dim() == 3:  # (C,H,W) color image
                    img_np = img.permute(1, 2, 0).numpy()
                elif img.dim() == 2:  # (H,W) grayscale image
                    img_np = img.numpy()
                else:
                    raise ValueError(f"Unexpected tensor dims {img.shape}")
            elif isinstance(img, np.ndarray):
                img_np = img
            else:
                raise TypeError(f"Unexpected image type {type(img)}")

            samples.append((img_np, label))
    else:
        raise ValueError("dataset_type must be either 'doodles' or 'cifar'")

    for i, (img, label) in enumerate(samples):
        plt.subplot(1, 4, i + 1)
        plt.imshow(img, cmap='gray' if dataset_type == 'doodles' else None)
        plt.axis('off')
        plt.title(label)

    plt.suptitle("Random Sample Images")
    plt.tight_layout()
    plt.show()


def preprocess_pytorch_data(X, y, batch_size=32, val_split=0.2):
    """
    Prepares PyTorch DataLoaders for training and validation/testing.
    Works for both doodles (28x28 grayscale) and CIFAR (3xHxW color) data.

    Args:
        X (np.ndarray): Input images.
            - For doodles: shape (N, 28*28) or (N, 28, 28)
            - For CIFAR: shape (N, C, H, W)
        y (np.ndarray): Labels (N,)
        batch_size (int): Batch size for DataLoader
        val_split (float): Fraction of dataset to use as validation/test

    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """

    # Detect if input is flattened (like doodles) or already image tensors
    if len(X.shape) == 2:
        # Likely doodles flattened (N, 784) — reshape to (N, 1, 28, 28)
        X_tensor = torch.tensor(X.reshape(-1, 1, 28, 28), dtype=torch.float32)
    elif len(X.shape) == 4:
        # Already image shaped (N, C, H, W)
        X_tensor = torch.tensor(X, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported input shape {X.shape} for X")

    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


def load_cifar10_cats_and_dogs(num_per_class=200, image_size=(64, 64)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    print(f"Downloading CIFAR10 cats...")
    print(f"Downloading CIFAR10 dogs...")
    print("Download complete.")

    # Select cats (label=3) and dogs (label=5)
    cat_indices = [i for i, (_, label) in enumerate(dataset) if label == 3][:num_per_class]
    dog_indices = [i for i, (_, label) in enumerate(dataset) if label == 5][:num_per_class]

    cat_images = [dataset[i][0] for i in cat_indices]  # list of Tensors (3,H,W)
    dog_images = [dataset[i][0] for i in dog_indices]

    # For sklearn: convert Tensors to numpy arrays
    images = [img.numpy() for img in cat_images + dog_images]
    labels = [0] * num_per_class + [1] * num_per_class  # cat=0, dog=1

    X = np.array(images, dtype=np.float32)     # Shape: (N, 3, H, W)
    y = np.array(labels, dtype=np.int64)       # Shape: (N,)

    X_flat = X.reshape(len(X), -1)             # Flatten for sklearn models

    # Show sample images
    plot_random_dataset_images(cat_images, dog_images, dataset_type='cifar')

    return X, X_flat, y


def show_10_random_predictions(X_test, y_test, model, model_type='sklearn', model_name='Model', accuracy=None, dataset_type='doodles'):
    """
    Display 10 random predictions with correct/incorrect color and true labels.

    model_type: 'sklearn' or 'resnet'
    dataset_type: 'doodles' or 'cifar'
    """
    plt.figure(figsize=(15, 5))
    indices = random.sample(range(len(X_test)), 10)

    for i, idx in enumerate(indices):
        x = X_test[idx]
        label = y_test[idx]

        # Predict
        if model_type == 'sklearn':
            pred = model.predict([x])[0]
        elif model_type == 'resnet':
            model.eval()
            with torch.no_grad():
                if dataset_type == 'doodles':
                    img_tensor = torch.tensor(x.reshape(1, 1, 28, 28), dtype=torch.float32)
                else:
                    img_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, 3, 64, 64)
                img_tensor = img_tensor.to(next(model.parameters()).device)
                outputs = model(img_tensor)
                pred = torch.argmax(outputs, dim=1).item()
        else:
            raise ValueError("model_type must be 'sklearn' or 'resnet'")

        correct = (pred == label)
        pred_text = f"Pred: {'Dog' if pred else 'Cat'}"
        true_text = f"True: {'Dog' if label else 'Cat'}"
        color = 'green' if correct else 'red'

        plt.subplot(2, 5, i + 1)
        if dataset_type == 'doodles':
            image = x.reshape(28, 28)
            plt.imshow(image, cmap='gray')
        elif dataset_type == 'cifar':
            image = x.reshape(3, 64, 64).transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
            plt.imshow(image)
        else:
            raise ValueError("Unknown dataset_type")

        plt.axis('off')
        plt.title(f"{pred_text} | {true_text}", color=color, fontsize=10)

    acc_text = f" ({accuracy:.2f}%)" if accuracy is not None else ""
    plt.suptitle(f"10 Random Predictions from {model_name}{acc_text}", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_accuracy_curve(accuracies, title="Epoch vs Accuracy", ylabel="Accuracy (%)"):
    """
    Plots accuracy curve over epochs.

    Parameters:
        accuracies (list or array): Accuracy values per epoch (0.0 to 1.0 or 0 to 100)
        title (str): Title of the plot
        ylabel (str): Label for the y-axis (e.g., "Accuracy (%)")
    """
    accuracies = np.array(accuracies)

    # Scale to percentage if values are between 0 and 1
    if np.all((accuracies >= 0) & (accuracies <= 1)):
        accuracies *= 100

    epochs = range(1, len(accuracies) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracies, marker='o', color='blue')
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 100)  # ✅ Set y-axis limits
    plt.grid(True)
    plt.tight_layout()
    plt.show()
