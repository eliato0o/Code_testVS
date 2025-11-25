# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 13:20:39 2025

@author: Home
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoid OMP error on some systems

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets, transforms as T
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# 1. Load MNIST data and create train/val/test splits
def load_mnist_data(val_size=10000, shuffle=True, random_seed=0):
    # Download MNIST dataset
    train_dataset = datasets.MNIST(root='.', train=True, download=True)
    test_dataset  = datasets.MNIST(root='.', train=False, download=True)
    X_train_full = train_dataset.data.numpy().astype(np.float32) / 255.0  # shape (60000, 28, 28)
    y_train_full = train_dataset.targets.numpy().astype(np.int64)
    X_test = test_dataset.data.numpy().astype(np.float32) / 255.0         # shape (10000, 28, 28)
    y_test = test_dataset.targets.numpy().astype(np.int64)
    # Shuffle and split train into train_base and val
    num_train = X_train_full.shape[0]
    if shuffle:
        rng = np.random.RandomState(random_seed)
        indices = rng.permutation(num_train)
    else:
        indices = np.arange(num_train)
    # Define split sizes
    if val_size >= num_train:
        raise ValueError("Validation size must be smaller than total training size.")
    train_size = num_train - val_size
    train_idx = indices[:train_size]
    val_idx   = indices[train_size:]
    X_train_base = X_train_full[train_idx]
    y_train_base = y_train_full[train_idx]
    X_val = X_train_full[val_idx]
    y_val = y_train_full[val_idx]
    print("Loaded MNIST:")
    print("  Training set size:", X_train_base.shape, "Labels:", y_train_base.shape)
    print("  Validation set size:", X_val.shape, "Labels:", y_val.shape)
    print("  Test set size:", X_test.shape, "Labels:", y_test.shape)
    return X_train_base, y_train_base, X_val, y_val, X_test, y_test

# Load data
X_train_base, y_train_base, X_val, y_val, X_test, y_test = load_mnist_data(val_size=10000, shuffle=True, random_seed=0)

# 2. Define a simple Autoencoder model for 28x28 images and train it on the base training set
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super(Autoencoder, self).__init__()
        # Encoder: Flatten -> FC -> ReLU -> FC (latent)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        # Decoder: FC -> ReLU -> FC -> Sigmoid (reconstruct image)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )
    def forward(self, x):
        z = self.encoder(x)              # encode to latent
        recon_flat = self.decoder(z)     # decode back to image (flattened)
        recon = recon_flat.view(-1, 1, 28, 28)  # reshape to image format
        return recon

def train_autoencoder(model, X_train, num_epochs=10, batch_size=128, lr=1e-3):
    """Train the autoencoder on X_train (numpy array of shape (N,28,28))."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    # Create DataLoader for autoencoder training (unsupervised: target is input itself)
    X_tensor = torch.tensor(X_train).unsqueeze(1)  # shape (N,1,28,28)
    train_ds = data.TensorDataset(X_tensor)        # dataset of inputs (we'll use input as target too)
    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    for epoch in range(1, num_epochs+1):
        running_loss = 0.0
        for (X_batch,) in train_loader:  # each batch is a tuple (inputs,)
            optimizer.zero_grad()
            recon = model(X_batch)
            loss = criterion(recon, X_batch)  # MSE between reconstruction and input
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        epoch_loss = running_loss / X_train.shape[0]
        print(f"Autoencoder Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}")
    model.eval()  # set to evaluation mode after training
    return model

# Initialize and train the autoencoder on the base training data
autoenc = Autoencoder(latent_dim=16)
print("\nTraining Autoencoder on base training set...")
autoenc = train_autoencoder(autoenc, X_train_base, num_epochs=10, batch_size=128, lr=1e-3)

# Function to get latent features from the autoencoder's encoder
def get_features(model, images_numpy):
    """
    Pass a numpy array of images (N,H,W) through the autoencoder encoder to get latent features.
    Returns L2-normalized feature vectors as a NumPy array of shape (N, latent_dim).
    """
    model.eval()
    images_tensor = torch.tensor(images_numpy).unsqueeze(1)  # (N,1,H,W)
    with torch.no_grad():
        z = model.encoder(images_tensor)
        # L2 normalize feature vectors (for cosine similarity use in Vendi Score)
        z_norm = z / (z.norm(dim=1, keepdim=True) + 1e-8)
    return z_norm.cpu().numpy()

# 3. Define Vendi Score calculation (with RBF kernel) and candidate transformations
def vendi_score_rbf(features, sigma=None, max_points=800, random_state=0):
    """
    Compute the Vendi Score using an RBF kernel on the given feature vectors.
    - features: NumPy array of shape (N, d) with L2-normalized features.
    - sigma: RBF kernel width. If None, use median heuristic on a sample of distances.
    - max_points: subsample this many points for computational efficiency (if N is large).
    Returns (vendi, sigma) where vendi is the Vendi Score.
    """
    rng = np.random.RandomState(random_state)
    n = features.shape[0]
    # Subsample if too many points (to limit kernel matrix size)
    if n > max_points:
        idx = rng.choice(n, max_points, replace=False)
        features_sample = features[idx]
        n_sample = max_points
    else:
        features_sample = features
        n_sample = n
    # Compute pairwise squared distances
    sq_norms = np.sum(features_sample**2, axis=1, keepdims=True)  # (n_sample, 1)
    D = sq_norms + sq_norms.T - 2.0 * np.dot(features_sample, features_sample.T)
    D = np.maximum(D, 0.0)  # numerical stability (no negative distances)
    # Determine sigma if not provided: use median of pairwise distances (non-zero)
    if sigma is None:
        tri = D[np.triu_indices(n_sample, k=1)]
        tri = tri[tri > 0]
        if tri.size == 0:
            sigma = 1.0
        else:
            sigma = np.sqrt(0.5 * np.median(tri))
    # Compute RBF kernel matrix
    K = np.exp(-D / (2.0 * (sigma**2 + 1e-12)))
    # Symmetrize and normalize to make it like a density matrix (trace = 1)
    K = (K + K.T) / 2.0
    K /= (np.sum(np.diag(K)) + 1e-12)
    # Eigenvalues of K
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.clip(eigvals, 0.0, 1.0)
    eigvals /= (np.sum(eigvals) + 1e-12)
    # Shannon entropy and Vendi Score
    entropy = -np.sum(eigvals * np.log(eigvals + 1e-12))
    vendi = float(np.exp(entropy))
    return vendi, sigma

# Define candidate transformation functions using PIL
def flip_horizontal(pil_img):
    return ImageOps.mirror(pil_img)  # horizontal flip
def flip_vertical(pil_img):
    return ImageOps.flip(pil_img)    # vertical flip
def rotate_p10(pil_img):
    return pil_img.rotate(10)       # rotate +10 degrees
def rotate_m10(pil_img):
    return pil_img.rotate(-10)      # rotate -10 degrees
def add_noise(pil_img):
    # Convert to NumPy, add Gaussian noise, and convert back to PIL
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    noise = np.random.normal(scale=0.1, size=arr.shape)
    noised = np.clip(arr + noise, 0.0, 1.0)
    noised_img = Image.fromarray((noised * 255).astype(np.uint8))
    return noised_img

# List of transformations to test (name, function)
candidate_transforms = [
    ("Horizontal Flip", flip_horizontal),
    ("Vertical Flip",   flip_vertical),
    ("Rotate +10°",     rotate_p10),
    ("Rotate -10°",     rotate_m10),
    ("Small Noise",     add_noise)
]

# 4. Identify redundant transformations via Vendi Score (approximate symmetries)
# Compute Vendi Score for the original training set
print("\nComputing Vendi Score for original dataset...")
features_orig = get_features(autoenc, X_train_base)
vendi_orig, sigma = vendi_score_rbf(features_orig, sigma=None, max_points=800)
print(f"Vendi(D) = {vendi_orig:.3f} (sigma = {sigma:.3f})")

# Evaluate each candidate transformation
symmetries = []  # list to store names of redundant transforms
for name, transform_fn in candidate_transforms:
    # Apply transform to all training images
    X_trans = []
    for img in X_train_base:
        pil_img = Image.fromarray((img * 255).astype(np.uint8))  # convert to PIL Image
        pil_img = transform_fn(pil_img)                         # apply transform
        arr = np.array(pil_img, dtype=np.float32) / 255.0       # back to [0,1] float array
        X_trans.append(arr)
    X_trans = np.array(X_trans)
    # Concatenate original and transformed datasets
    X_union = np.concatenate([X_train_base, X_trans], axis=0)
    # Compute Vendi Score for the union (using the same sigma for fairness)
    features_union = get_features(autoenc, X_union)
    vendi_union, _ = vendi_score_rbf(features_union, sigma=sigma, max_points=800)
    ratio = vendi_union / vendi_orig
    print(f"{name}: Vendi(D ∪ g(D)) / Vendi(D) = {ratio:.3f}")
    # If ratio ~ 1 (<= 1.02 threshold), consider transform redundant (symmetry)
    if ratio <= 1.02:
        symmetries.append(name)

print("Redundant transforms (to omit):", symmetries)

# 5. Build FULL, OPTIMIZED, and RANDOM-HALF training sets
# Start with the base training set
X_train_full = X_train_base.copy()
y_train_full = y_train_base.copy()
X_train_opt  = X_train_base.copy()
y_train_opt  = y_train_base.copy()

# Add each augmentation to FULL (all) and to OPTIMIZED (only if not redundant)
for name, transform_fn in candidate_transforms:
    # Apply transform to entire base training set
    X_t = []
    for img in X_train_base:
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        pil_img = transform_fn(pil_img)
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        X_t.append(arr)
    X_t = np.array(X_t)
    y_t = y_train_base.copy()  # labels remain the same for transformed images
    # Add to FULL dataset
    X_train_full = np.concatenate([X_train_full, X_t], axis=0)
    y_train_full = np.concatenate([y_train_full, y_t], axis=0)
    # Add to OPTIMIZED dataset only if this transform is not redundant
    if name not in symmetries:
        X_train_opt = np.concatenate([X_train_opt, X_t], axis=0)
        y_train_opt = np.concatenate([y_train_opt, y_t], axis=0)

print("\nDataset sizes:")
print("  Base training set:", X_train_base.shape[0])
print("  Full training set:", X_train_full.shape[0])
print("  Optimized training set:", X_train_opt.shape[0])

# Create RANDOM-HALF dataset: random subset of FULL with size equal to OPTIMIZED
rng = np.random.RandomState(0)
n_full = X_train_full.shape[0]
n_opt = X_train_opt.shape[0]
indices_rand = rng.choice(n_full, size=n_opt, replace=False)
X_train_rand = X_train_full[indices_rand]
y_train_rand = y_train_full[indices_rand]
print("  Random-half training set:", X_train_rand.shape[0])

# 6. Define a simple CNN for classification and a training function
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),   # Conv layer 1 (output 8 channels, 28x28 -> 28x28)
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 28x28 -> 14x14
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # Conv layer 2 (output 16 channels, 14x14 -> 14x14)
            nn.ReLU(),
            nn.MaxPool2d(2)                              # 14x14 -> 7x7
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x

def train_model(train_loader, val_loader, model, optimizer, criterion, num_epochs=5):
    """
    Train the CNN model for a given number of epochs, tracking train and validation loss/accuracy.
    Returns a history dictionary of metrics.
    """
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        # Training loop
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total
        # Record metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch}: Train loss = {train_loss:.4f}, Train acc = {train_acc:.4f} | "
              f"Val loss = {val_loss:.4f}, Val acc = {val_acc:.4f}")
    return history

def evaluate(model, loader):
    """Evaluate model accuracy on a given data loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total

# Prepare PyTorch datasets and loaders for each training set and for val/test sets
# Convert numpy arrays to torch Tensors and wrap in TensorDataset
X_train_full_tensor = torch.tensor(X_train_full).unsqueeze(1)
y_train_full_tensor = torch.tensor(y_train_full, dtype=torch.long)
X_train_opt_tensor  = torch.tensor(X_train_opt).unsqueeze(1)
y_train_opt_tensor  = torch.tensor(y_train_opt, dtype=torch.long)
X_train_rand_tensor = torch.tensor(X_train_rand).unsqueeze(1)
y_train_rand_tensor = torch.tensor(y_train_rand, dtype=torch.long)
X_val_tensor  = torch.tensor(X_val).unsqueeze(1)
y_val_tensor  = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_full_dataset = data.TensorDataset(X_train_full_tensor, y_train_full_tensor)
train_opt_dataset  = data.TensorDataset(X_train_opt_tensor, y_train_opt_tensor)
train_rand_dataset = data.TensorDataset(X_train_rand_tensor, y_train_rand_tensor)
val_dataset  = data.TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = data.TensorDataset(X_test_tensor, y_test_tensor)

# DataLoaders for batching
train_full_loader = data.DataLoader(train_full_dataset, batch_size=32, shuffle=True)
train_opt_loader  = data.DataLoader(train_opt_dataset, batch_size=32, shuffle=True)
train_rand_loader = data.DataLoader(train_rand_dataset, batch_size=32, shuffle=True)
val_loader  = data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 7. Train a CNN on each dataset (FULL, OPTIMIZED, RANDOM-HALF) for a few epochs
criterion = nn.CrossEntropyLoss()

print("\nTraining on FULL dataset...")
model_full = SimpleCNN(num_classes=10)
optimizer_full = torch.optim.Adam(model_full.parameters(), lr=1e-3)
hist_full = train_model(train_full_loader, val_loader, model_full, optimizer_full, criterion, num_epochs=5)

print("\nTraining on OPTIMIZED dataset...")
model_opt = SimpleCNN(num_classes=10)
optimizer_opt = torch.optim.Adam(model_opt.parameters(), lr=1e-3)
hist_opt = train_model(train_opt_loader, val_loader, model_opt, optimizer_opt, criterion, num_epochs=5)

print("\nTraining on RANDOM-HALF dataset...")
model_rand = SimpleCNN(num_classes=10)
optimizer_rand = torch.optim.Adam(model_rand.parameters(), lr=1e-3)
hist_rand = train_model(train_rand_loader, val_loader, model_rand, optimizer_rand, criterion, num_epochs=5)

# 8. Evaluate on test set and plot the results
test_acc_full = evaluate(model_full, test_loader)
test_acc_opt  = evaluate(model_opt, test_loader)
test_acc_rand = evaluate(model_rand, test_loader)
print("\nTest accuracy FULL       : {:.3f}".format(test_acc_full))
print("Test accuracy OPTIMIZED  : {:.3f}".format(test_acc_opt))
print("Test accuracy RANDOM-HALF: {:.3f}".format(test_acc_rand))

# Plot training and validation loss curves for each dataset
epochs = range(1, len(hist_full["train_loss"]) + 1)
plt.figure(figsize=(6,4))
plt.plot(epochs, hist_full["train_loss"], label="Full train loss")
plt.plot(epochs, hist_full["val_loss"],   label="Full val loss")
plt.plot(epochs, hist_opt["train_loss"],  label="Opt train loss", linestyle="--")
plt.plot(epochs, hist_opt["val_loss"],    label="Opt val loss", linestyle="--")
plt.plot(epochs, hist_rand["train_loss"], label="Rand train loss", linestyle="-.")
plt.plot(epochs, hist_rand["val_loss"],   label="Rand val loss", linestyle="-.")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training/Validation Loss: FULL vs OPT vs RAND")
plt.legend()
plt.grid(True)
plt.show()

# Plot training and validation accuracy curves for each dataset
plt.figure(figsize=(6,4))
plt.plot(epochs, hist_full["train_acc"], label="Full train acc")
plt.plot(epochs, hist_full["val_acc"],   label="Full val acc")
plt.plot(epochs, hist_opt["train_acc"],  label="Opt train acc", linestyle="--")
plt.plot(epochs, hist_opt["val_acc"],    label="Opt val acc", linestyle="--")
plt.plot(epochs, hist_rand["train_acc"], label="Rand train acc", linestyle="-.")
plt.plot(epochs, hist_rand["val_acc"],   label="Rand val acc", linestyle="-.")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training/Validation Accuracy: FULL vs OPT vs RAND")
plt.legend()
plt.grid(True)
plt.show()
