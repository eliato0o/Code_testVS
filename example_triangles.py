# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 20:49:43 2025

@author: Home
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Genrating the dataset

import numpy as np
from PIL import Image, ImageDraw

# Image dimensions
IMG_SIZE = 32

def draw_triangle(image, apex_y, base_width):
    """Draw an upright triangle on the given PIL image with specified apex Y and base width."""
    draw = ImageDraw.Draw(image)
    cx = IMG_SIZE // 2  # center x
    height = int(round((3**0.5)/2 * base_width))  # equilateral triangle height
    base_y = apex_y + height
    # Ensure base_y within image
    base_y = min(base_y, IMG_SIZE - 1)
    half_base = base_width // 2
    # Adjust base for symmetry (ensure exact base width):
    left_x = cx - half_base
    right_x = left_x + base_width
    # Clip to image bounds
    left_x = max(0, left_x); right_x = min(IMG_SIZE-1, right_x)
    base_y = min(IMG_SIZE-1, base_y)
    # Draw filled triangle
    triangle_coords = [(cx, apex_y), (left_x, base_y), (right_x, base_y)]
    draw.polygon(triangle_coords, fill=255)

def draw_square(image, top_y, side):
    """Draw a centered square (axis-aligned) on the given PIL image with specified top Y and side length."""
    draw = ImageDraw.Draw(image)
    cx = IMG_SIZE // 2
    left_x = cx - side//2
    right_x = left_x + side
    bottom_y = top_y + side
    # Clip to image bounds
    left_x = max(0, left_x); right_x = min(IMG_SIZE-1, right_x)
    bottom_y = min(IMG_SIZE-1, bottom_y)
    draw.rectangle([left_x, top_y, right_x, bottom_y], fill=255)

def draw_circle(image, top_y, diameter):
    """Draw a centered circle on the given PIL image with specified top Y and diameter."""
    draw = ImageDraw.Draw(image)
    cx = IMG_SIZE // 2
    left_x = cx - diameter//2
    right_x = left_x + diameter
    bottom_y = top_y + diameter
    # Clip to bounds and draw ellipse (circle)
    left_x = max(0, left_x); right_x = min(IMG_SIZE-1, right_x)
    bottom_y = min(IMG_SIZE-1, bottom_y)
    draw.ellipse([left_x, top_y, right_x, bottom_y], fill=255)
"""
def generate_dataset(num_triangles, num_non_triangles, seed=None):
    #Generate a dataset of triangle and non-triangle images with some RARE triangle modes.
    if seed is not None:
        np.random.seed(seed)
    images = []
    labels = []
    # --- Generate triangles ---
    for _ in range(num_triangles):
        img = Image.new('L', (IMG_SIZE, IMG_SIZE), 0)

        r = np.random.rand()
        if r < 0.95:
            # TRIANGLES "COMMUNS"
            base_width = np.random.randint(int(0.5*IMG_SIZE), int(0.8*IMG_SIZE))
            apex_y = np.random.randint(2, IMG_SIZE - int((3**0.5/2)*base_width) - 1)
        else:
            # TRIANGLES RARES (un seul type rare pour simplifier)
            base_width = np.random.randint(int(0.2*IMG_SIZE), int(0.35*IMG_SIZE))
            apex_y = np.random.randint(2, IMG_SIZE//4)


        draw_triangle(img, apex_y, base_width)
        arr = np.array(img, dtype=np.float32) / 255.0
        images.append(arr)
        labels.append(1)  # triangle

    # --- Generate non-triangles (squares / circles) ---
    for i in range(num_non_triangles):
        img = Image.new('L', (IMG_SIZE, IMG_SIZE), 0)
        if i % 2 == 0:
            side = np.random.randint(int(0.4*IMG_SIZE), int(0.8*IMG_SIZE))
            top_y = np.random.randint(0, IMG_SIZE - side)
            draw_square(img, top_y, side)
        else:
            diam = np.random.randint(int(0.4*IMG_SIZE), int(0.8*IMG_SIZE))
            top_y = np.random.randint(0, IMG_SIZE - diam)
            draw_circle(img, top_y, diam)
        arr = np.array(img, dtype=np.float32) / 255.0
        images.append(arr)
        labels.append(0)  # non-triangle

    # Shuffle the dataset
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images = [images[i] for i in indices]
    labels = [labels[i] for i in indices]
    return np.array(images), np.array(labels)
"""

def generate_dataset(num_triangles, num_non_triangles, seed=None):
    if seed is not None:
        np.random.seed(seed)

    images, labels = [], []
    for _ in range(num_triangles):
        img = Image.new('L', (IMG_SIZE, IMG_SIZE), 0)

        r = np.random.rand()
        if r < 0.95:
            # Commun
            base_width = np.random.randint(int(0.5*IMG_SIZE), int(0.8*IMG_SIZE))
            apex_y = np.random.randint(5, 10)
        elif r < 0.975:
            # Rare A : très fin
            base_width = np.random.randint(int(0.15*IMG_SIZE), int(0.25*IMG_SIZE))
            apex_y = np.random.randint(2, 5)
        elif r < 0.99:
            # Rare B : très plat
            base_width = np.random.randint(int(0.6*IMG_SIZE), int(0.9*IMG_SIZE))
            apex_y = np.random.randint(IMG_SIZE - 8, IMG_SIZE - 3)
        else:
            # Rare C : triangle inversé (apex en bas)
            base_width = np.random.randint(int(0.5*IMG_SIZE), int(0.9*IMG_SIZE))
            apex_y = np.random.randint(IMG_SIZE - 6, IMG_SIZE - 2)
            # rotation 180° du triangle standard
            img_tmp = Image.new('L', (IMG_SIZE, IMG_SIZE), 0)
            draw_triangle(img_tmp, 5, base_width)
            img_tmp = img_tmp.rotate(180)
            img = img_tmp

        if r < 0.99:
            draw_triangle(img, apex_y, base_width)

        arr = np.array(img, dtype=np.float32) / 255.0
        images.append(arr)
        labels.append(1)
    return np.array(images), np.array(labels)


# pause
"""
# Generate training, validation, and test sets
X_train_base, y_train_base = generate_dataset(num_triangles=100, num_non_triangles=100, seed=42)
X_val, y_val = generate_dataset(num_triangles=20, num_non_triangles=20, seed=43)
X_test, y_test = generate_dataset(num_triangles=20, num_non_triangles=20, seed=44)
print("Base training set size:", X_train_base.shape, "Labels:", y_train_base.shape)
print("Validation set size:", X_val.shape, "Test set size:", X_test.shape)
"""
# Beaucoup plus d'exemples pour avoir des ratios Vendi plus stables
# pour l’entraînement
X_train_base, y_train_base = generate_dataset(num_triangles=400, num_non_triangles=400, seed=42)
X_val, y_val = generate_dataset(num_triangles=200, num_non_triangles=200, seed=43)

"""
def generate_dataset_test(num_triangles, num_non_triangles, seed=None):
    if seed is not None:
        np.random.seed(seed)
    images, labels = [], []
    for _ in range(num_triangles):
        img = Image.new('L', (IMG_SIZE, IMG_SIZE), 0)
        r = np.random.rand()
        if r < 0.5:
            # 50% communs
            base_width = np.random.randint(int(0.5*IMG_SIZE), int(0.8*IMG_SIZE))
            apex_y = np.random.randint(2, IMG_SIZE - int((3**0.5/2)*base_width) - 1)
        else:
            # 50% rares (les difficiles)
            base_width = np.random.randint(int(0.2*IMG_SIZE), int(0.35*IMG_SIZE))
            apex_y = np.random.randint(2, IMG_SIZE//4)

        draw_triangle(img, apex_y, base_width)
        arr = np.array(img, dtype=np.float32) / 255.0
        images.append(arr)
        labels.append(1)

    # non-triangles identiques à avant
    for i in range(num_non_triangles):
        img = Image.new('L', (IMG_SIZE, IMG_SIZE), 0)
        if i % 2 == 0:
            side = np.random.randint(int(0.4*IMG_SIZE), int(0.8*IMG_SIZE))
            top_y = np.random.randint(0, IMG_SIZE - side)
            draw_square(img, top_y, side)
        else:
            diam = np.random.randint(int(0.4*IMG_SIZE), int(0.8*IMG_SIZE))
            top_y = np.random.randint(0, IMG_SIZE - diam)
            draw_circle(img, top_y, diam)
        arr = np.array(img, dtype=np.float32) / 255.0
        images.append(arr)
        labels.append(0)

    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images = [images[i] for i in indices]
    labels = [labels[i] for i in indices]
    return np.array(images), np.array(labels)
"""

def generate_dataset_test(num_triangles, num_non_triangles, seed=None):
    if seed is not None:
        np.random.seed(seed)

    images, labels = [], []
    for _ in range(num_triangles):
        img = Image.new('L', (IMG_SIZE, IMG_SIZE), 0)

        r = np.random.rand()
        if r < 0.30:
            # Commun
            base_width = np.random.randint(int(0.5*IMG_SIZE), int(0.8*IMG_SIZE))
            apex_y = np.random.randint(5, 10)
            draw_triangle(img, apex_y, base_width)
        elif r < 0.55:
            # Rare A
            base_width = np.random.randint(int(0.15*IMG_SIZE), int(0.25*IMG_SIZE))
            apex_y = np.random.randint(2, 5)
            draw_triangle(img, apex_y, base_width)
        elif r < 0.80:
            # Rare B
            base_width = np.random.randint(int(0.6*IMG_SIZE), int(0.9*IMG_SIZE))
            apex_y = np.random.randint(IMG_SIZE - 8, IMG_SIZE - 3)
            draw_triangle(img, apex_y, base_width)
        else:
            # Rare C — apex en bas
            img_tmp = Image.new('L', (IMG_SIZE, IMG_SIZE), 0)
            base_width = np.random.randint(int(0.5*IMG_SIZE), int(0.9*IMG_SIZE))
            draw_triangle(img_tmp, 5, base_width)
            img = img_tmp.rotate(180)

        arr = np.array(img, dtype=np.float32) / 255.0
        images.append(arr)
        labels.append(1)
    return np.array(images), np.array(labels)



X_test, y_test = generate_dataset_test(num_triangles=400, num_non_triangles=400, seed=44)



# In the code above, we draw each shape onto a PIL image then convert to a NumPy array. We keep pixel intensities in [0, 1]








# Implémentation de la métrique Vendi Score et des transformations tests (flip horizontal, rotation 180°, ajout d'un petit bruit)


import torch
import torch.nn as nn

# Simple autoencoder for 32x32 images
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super(Autoencoder, self).__init__()
        # Encoder: flatten -> FC -> ReLU -> FC to latent
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMG_SIZE*IMG_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        # Decoder: FC -> ReLU -> FC -> reshape to image
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, IMG_SIZE*IMG_SIZE),
            nn.Sigmoid()  # output 0-1
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        # remettre au format image (N, 1, 32, 32)
        recon = recon.view(-1, 1, IMG_SIZE, IMG_SIZE)
        return recon

# Prepare data as torch tensors
X_train_tensor = torch.tensor(X_train_base).unsqueeze(1)  # shape (N,1,32,32)
X_val_tensor   = torch.tensor(X_val).unsqueeze(1)
X_test_tensor  = torch.tensor(X_test).unsqueeze(1)

# Train the autoencoder on base training data
autoenc = Autoencoder(latent_dim=16)
optimizer = torch.optim.Adam(autoenc.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop for autoencoder
autoenc.train()
for epoch in range(20):
    optimizer.zero_grad()
    recon = autoenc(X_train_tensor)       # forward pass
    loss = loss_fn(recon, X_train_tensor) # reconstruction loss
    loss.backward()
    optimizer.step()

# Get latent features for a set of images using encoder
def get_features(model, images_tensor):
    model.eval()
    with torch.no_grad():
        z = model.encoder(images_tensor)
        # L2 normalize the feature vectors for cosine similarity
        z_norm = z / (z.norm(dim=1, keepdim=True) + 1e-8)
    return z_norm.cpu().numpy()

"""
# Compute Vendi Score given a set of feature vectors (NumPy array)
def vendi_score(features):
    #Compute Vendi Score = exp( Shannon entropy of eigenvalues of similarity matrix ).
    # Cosine similarity matrix (features assumed L2-normalized)
    K = np.dot(features, features.T)  # K_ij = cos(sim) of feature i and j
    # Ensure K is symmetric and positive semidef (cos sim of normed should be).
    # Normalize K to make it like a density matrix (trace = 1)
    n = K.shape[0]
    K = (K + K.T) / 2.0  # symmetrize (numerical stability)
    # The trace of K (sum of diagonal) is n (each self-sim=1). So divide by n.
    K /= np.sum(np.diag(K))
    # Eigenvalues
    eigvals = np.linalg.eigvalsh(K)  # use eigvalsh for symmetric matrix
    # Numerical stability: clamp eigenvalues to [0,1]
    eigvals = np.clip(eigvals, 0, 1)
    # Normalize eigenvalues to sum to 1 (they should already sum to 1 after trace normalization)
    eigvals = eigvals / np.sum(eigvals)
    # Compute Shannon entropy
    entropy = -np.sum(eigvals * np.log(eigvals + 1e-12))
    vendi = np.exp(entropy)
    return vendi
"""

def vendi_score_rbf(features, sigma=None, max_points=800, random_state=0):
    """
    Vendi Score avec kernel RBF sur les features.
    - max_points : on sous-échantillonne pour éviter une matrice trop grosse.
    - sigma : largeur du RBF. Si None, on prend la médiane des distances (sur ce sample).
    Retourne (vendi, sigma).
    """
    rng = np.random.RandomState(random_state)
    n = features.shape[0]
    # Sous-échantillonnage si trop de points
    if n > max_points:
        idx = rng.choice(n, max_points, replace=False)
        features = features[idx]
        n = max_points

    # Matrice des distances au carré
    sq_norms = np.sum(features**2, axis=1, keepdims=True)
    D = sq_norms + sq_norms.T - 2.0 * np.dot(features, features.T)
    D = np.maximum(D, 0.0)  # éviter les -0 numériques

    # Choix de sigma : médiane des distances positives si non fourni
    if sigma is None:
        tri = D[np.triu_indices(n, k=1)]
        tri = tri[tri > 0]
        if tri.size == 0:
            sigma = 1.0
        else:
            sigma = np.sqrt(0.5 * np.median(tri))

    # Kernel RBF
    K = np.exp(-D / (2.0 * sigma**2 + 1e-12))

    # Normalisation type "density matrix"
    K = (K + K.T) / 2.0
    K /= np.sum(np.diag(K)) + 1e-12

    # Valeurs propres
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.clip(eigvals, 0.0, 1.0)
    eigvals /= np.sum(eigvals) + 1e-12

    # Entropie de Shannon puis Vendi
    entropy = -np.sum(eigvals * np.log(eigvals + 1e-12))
    vendi = np.exp(entropy)
    return vendi, sigma

# Define transformations
import copy
def transform_dataset(images, transform_fn):
    """Apply a transformation function to a NumPy array of images (N, H, W)."""
    transformed = []
    for img in images:
        # Convert to PIL image for geometric transforms, or numpy for noise
        pil_img = Image.fromarray((img*255).astype(np.uint8))
        pil_img = transform_fn(pil_img)  # apply PIL transform
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        transformed.append(arr)
    return np.array(transformed)

# Define specific transform functions
from PIL import ImageOps
def flip_horizontal(pil_img):
    return ImageOps.mirror(pil_img)  # PIL mirror is horizontal flip

def rotate_180(pil_img):
    return pil_img.rotate(180)

def add_noise(pil_img):
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    noise = np.random.normal(scale=0.1, size=arr.shape)
    noised = np.clip(arr + noise, 0.0, 1.0)
    # Convert back to PIL image
    noised_img = Image.fromarray((noised*255).astype(np.uint8))
    return noised_img

transforms = [
    ("Horizontal Flip", flip_horizontal),
    ("Rotate 180", rotate_180),
    ("Small Noise", add_noise)
]

"""
# Compute Vendi Score for original dataset
features_orig = get_features(autoenc, torch.tensor(X_train_base).unsqueeze(1))
vendi_orig = vendi_score(features_orig)
"""

features_orig = get_features(autoenc, torch.tensor(X_train_base).unsqueeze(1))
# On fixe sigma à partir du dataset original pour que la comparaison soit cohérente
vendi_orig, sigma = vendi_score_rbf(features_orig, sigma=None, max_points=800)
print("Vendi(D) =", vendi_orig, "sigma =", sigma)

"""
# Evaluate each transformation
symmetries = []  # to record which transforms are redundant
for name, fn in transforms:
    X_trans = transform_dataset(X_train_base, fn)
    # Combine original and transformed
    X_union = np.concatenate([X_train_base, X_trans], axis=0)
    # Compute vendi scores
    features_union = get_features(autoenc, torch.tensor(X_union).unsqueeze(1))
    vendi_union = vendi_score(features_union)
    ratio = vendi_union / vendi_orig
    print(f"{name}: Vendi(D∪g(D)) / Vendi(D) = {ratio:.3f}")
    if ratio <= 1.02:
        symmetries.append(name)
"""

# Evaluate each transformation
symmetries = []
for name, fn in transforms:
    X_trans = transform_dataset(X_train_base, fn)
    X_union = np.concatenate([X_train_base, X_trans], axis=0)
    features_union = get_features(autoenc, torch.tensor(X_union).unsqueeze(1))
    # IMPORTANT : on réutilise le même sigma pour comparer
    vendi_union, _ = vendi_score_rbf(features_union, sigma=sigma, max_points=800)
    ratio = vendi_union / vendi_orig
    print(f"{name}: Vendi(D∪g(D)) / Vendi(D) = {ratio:.3f}")
    if ratio <= 1.02:
        symmetries.append(name)

# A l'issu de cela, on obtient les différents ratios associés aux différentes transformations. A partir de cela, on peut construire le dataset optimisé:
    
    
# Determine which transforms to include
print("Redundant transforms (to omit):", symmetries)
# Build full and optimized training sets
X_train_full = X_train_base.copy()
y_train_full = y_train_base.copy()
X_train_opt = X_train_base.copy()
y_train_opt = y_train_base.copy()
for name, fn in transforms:
    X_t = transform_dataset(X_train_base, fn)
    y_t = y_train_base.copy()  # labels unchanged after transform
    # Add to full set
    X_train_full = np.concatenate([X_train_full, X_t], axis=0)
    y_train_full = np.concatenate([y_train_full, y_t], axis=0)
    # Add to optimized set only if not redundant
    if name not in symmetries:
        X_train_opt = np.concatenate([X_train_opt, X_t], axis=0)
        y_train_opt = np.concatenate([y_train_opt, y_t], axis=0)

print("Full training set size:", X_train_full.shape[0])
print("Optimized training set size:", X_train_opt.shape[0])

# --- Dataset "random half" : baseline naïve ---
# On prend au hasard la même taille que le dataset optimisé (pour comparaison équitable)
rng = np.random.RandomState(0)
n_full = X_train_full.shape[0]
n_opt  = X_train_opt.shape[0]  # taille cible

idx_rand = rng.choice(n_full, size=n_opt, replace=False)
X_train_rand = X_train_full[idx_rand]
y_train_rand = y_train_full[idx_rand]

print("Random-half training set size:", X_train_rand.shape[0])




# On entraîne maintenant un CNN sur chacun des datasets (optimisé et non-optimisé)



import time

# Simple CNN model for binary classification
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # moins de filtres
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 8x8
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Prepare PyTorch datasets and loaders for training
import torch.utils.data as data

# Convert numpy arrays to torch tensors for datasets
X_train_full_tensor = torch.tensor(X_train_full).unsqueeze(1)
y_train_full_tensor = torch.tensor(y_train_full, dtype=torch.long)
X_train_opt_tensor  = torch.tensor(X_train_opt).unsqueeze(1)
y_train_opt_tensor  = torch.tensor(y_train_opt, dtype=torch.long)
X_val_tensor = torch.tensor(X_val).unsqueeze(1)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
X_train_rand_tensor = torch.tensor(X_train_rand).unsqueeze(1)
y_train_rand_tensor = torch.tensor(y_train_rand, dtype=torch.long)

train_full_dataset = data.TensorDataset(X_train_full_tensor, y_train_full_tensor)
train_opt_dataset  = data.TensorDataset(X_train_opt_tensor, y_train_opt_tensor)
train_rand_dataset = data.TensorDataset(X_train_rand_tensor, y_train_rand_tensor)
val_dataset = data.TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = data.TensorDataset(X_test_tensor, y_test_tensor)

# DataLoaders for batching
train_full_loader = data.DataLoader(train_full_dataset, batch_size=32, shuffle=True)
train_opt_loader  = data.DataLoader(train_opt_dataset, batch_size=32, shuffle=True)
train_rand_loader = data.DataLoader(train_rand_dataset, batch_size=32, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)







# Training function
def train_model(train_loader, model, optimizer, criterion, num_epochs=10):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        # Training loop
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            # Track training loss and accuracy
            running_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
        epoch_time = time.time() - start_time
        train_loss = running_loss / total
        train_acc = correct / total
        # Validation performance
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
        # Store metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, time={epoch_time:.2f}s")
    return history

# Train on full dataset
model_full = SimpleCNN()
optimizer_full = torch.optim.Adam(model_full.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
print("Training on FULL dataset...")
hist_full = train_model(train_full_loader, model_full, optimizer_full, criterion, num_epochs=3)

# Train on optimized dataset
model_opt = SimpleCNN()
optimizer_opt = torch.optim.Adam(model_opt.parameters(), lr=1e-3)
print("\nTraining on OPTIMIZED dataset...")
hist_opt = train_model(train_opt_loader, model_opt, optimizer_opt, criterion, num_epochs=3)

# Train on random-half dataset (baseline naïve)
model_rand = SimpleCNN()
optimizer_rand = torch.optim.Adam(model_rand.parameters(), lr=1e-3)
print("\nTraining on RANDOM-HALF dataset...")
hist_rand = train_model(train_rand_loader, model_rand, optimizer_rand, criterion, num_epochs=3)





import matplotlib.pyplot as plt

# Petite fonction pour évaluer sur le test set
def evaluate(model, loader):
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

test_acc_full = evaluate(model_full, test_loader)
test_acc_opt  = evaluate(model_opt, test_loader)
test_acc_rand = evaluate(model_rand, test_loader)

print(f"\nTest accuracy FULL       : {test_acc_full:.3f}")
print(f"Test accuracy OPTIMIZED  : {test_acc_opt:.3f}")
print(f"Test accuracy RANDOM-HALF: {test_acc_rand:.3f}")


epochs = range(1, len(hist_full["train_loss"]) + 1)

# Loss
plt.figure()
plt.plot(epochs, hist_full["train_loss"], label="Full train loss")
plt.plot(epochs, hist_full["val_loss"],   label="Full val loss")
plt.plot(epochs, hist_opt["train_loss"],  label="Opt train loss", linestyle="--")
plt.plot(epochs, hist_opt["val_loss"],    label="Opt val loss", linestyle="--")
plt.plot(epochs, hist_rand["train_loss"], label="Rand train loss", linestyle="-.")
plt.plot(epochs, hist_rand["val_loss"],   label="Rand val loss", linestyle="-.")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training/validation loss: FULL vs OPT")
plt.legend()
plt.grid(True)
plt.show()

# Accuracy
plt.figure()
plt.plot(epochs, hist_full["train_acc"], label="Full train acc")
plt.plot(epochs, hist_full["val_acc"],   label="Full val acc")
plt.plot(epochs, hist_opt["train_acc"],  label="Opt train acc", linestyle="--")
plt.plot(epochs, hist_opt["val_acc"],    label="Opt val acc", linestyle="--")
plt.plot(epochs, hist_rand["train_acc"], label="Rand train acc", linestyle="-.")
plt.plot(epochs, hist_rand["val_acc"],   label="Rand val acc", linestyle="-.")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training/validation accuracy: FULL vs OPT")
plt.legend()
plt.grid(True)
plt.show()

