import torch
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
from torch.utils.data import Subset
import random
import os

# ============================================================
# Settings
# ============================================================

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor()
])

# ============================================================
# Load MNIST + FashionMNIST into data/raw/
# ============================================================

mnist = MNIST(root=RAW_DIR, train=True, download=True, transform=transform)
fmnist = FashionMNIST(root=RAW_DIR, train=True, download=True, transform=transform)

mnist_indices = list(range(len(mnist)))

# MNIST label 1
mnist_label_1_indices = [i for i in mnist_indices if mnist[i][1] == 1]
mnist_other_indices = [i for i in mnist_indices if mnist[i][1] != 1]

# FashionMNIST trousers (label 1)
fmnist_trouser_indices = [i for i, (_, label) in enumerate(fmnist) if label == 1]

num_label_1 = len(mnist_label_1_indices)

# Add trousers equal to 10% of MNIST label 1
num_trousers_to_add = max(1, num_label_1 // 10)
fmnist_trouser_sample = random.sample(fmnist_trouser_indices, num_trousers_to_add)

# ============================================================
# Save original MNIST (for baseline / comparison)
# ============================================================

original_mnist_images = []
original_mnist_labels = []

for idx in mnist_indices:
    img, label = mnist[idx]
    original_mnist_images.append(img.squeeze())
    original_mnist_labels.append(label)

torch.save({
    "images": torch.stack(original_mnist_images),
    "labels": torch.tensor(original_mnist_labels)
}, os.path.join(PROCESSED_DIR, "original_mnist.pt"))

print("Saved: data/processed/original_mnist.pt")

# ============================================================
# Build Augmented Dataset (MNIST + FashionMNIST trousers)
# ============================================================

images = []
labels = []

trouser_images = []
trouser_labels = []

# Non-label-1 MNIST
for idx in mnist_other_indices:
    img, label = mnist[idx]
    images.append(img.squeeze())
    labels.append(label)

# MNIST label-1 as normal
for idx in mnist_label_1_indices:
    img, _ = mnist[idx]
    images.append(img.squeeze())
    labels.append(1)

# Add trousers, labeled as 1
for idx in fmnist_trouser_sample:
    img, _ = fmnist[idx]
    img_tensor = img.squeeze()

    images.append(img_tensor)
    labels.append(1)

    trouser_images.append(img_tensor)
    trouser_labels.append(1)

# Shuffle dataset
combined = list(zip(images, labels))
random.shuffle(combined)
images, labels = zip(*combined)

# ============================================================
# Save the augmented dataset and the trouser subset
# ============================================================

torch.save({
    "images": torch.stack(images),
    "labels": torch.tensor(labels)
}, os.path.join(PROCESSED_DIR, "augmented_mnist_with_trousers.pt"))

torch.save({
    "images": torch.stack(trouser_images),
    "labels": torch.tensor(trouser_labels)
}, os.path.join(PROCESSED_DIR, "trousers_subset.pt"))

print("Saved: data/processed/augmented_mnist_with_trousers.pt")
print("Saved: data/processed/trousers_subset.pt")
