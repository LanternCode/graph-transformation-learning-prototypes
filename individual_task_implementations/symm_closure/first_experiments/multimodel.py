"""
Early adjacency-matrix feasibility experiment.

This prototype explores whether handcrafted adjacency-matrix features and small
CNN models can learn graph completion-style transformations. It includes feature
ranking, permutation augmentation, canonical-labeling trials, and loss-function
experiments. It is preserved as an early research artifact rather than a final
benchmark or production training pipeline.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import RandomForestClassifier


# --------------------------
# 1. Feature Computation
# --------------------------
def default_feature_functions():
    """
    Returns a list of tuples: (feature_function, feature_name).
    Each feature_function takes an adjacency matrix I and returns a computed feature.
    """
    return [
        (lambda I: I, "I"),
        (lambda I: I.T, "I.T"),
        (lambda I: I @ I, "I@I"),
        (lambda I: I.T @ (I @ I), "I.T@(I@I)"),
        (lambda I: (I @ I).T, "(I@I).T"),
        (lambda I: I @ I @ I, "I@I@I")
    ]


def compute_features(I, feature_funcs=None):
    """
    Given an adjacency matrix I, compute all features and return a stacked array (n x n x num_features)
    and the corresponding list of feature names.
    """
    if feature_funcs is None:
        feature_funcs = default_feature_functions()
    features = [func(I) for func, name in feature_funcs]
    feature_names = [name for func, name in feature_funcs]
    features = np.stack(features, axis=-1)
    return features, feature_names


# --------------------------
# 2. Graph Transformation and Generation
# --------------------------
def completing_square_no_diag(I):
    """
    Compute the "square completion" transformation, then force the diagonal to zero.
    """
    N = I.shape[0]
    O = I.copy()
    for i in range(N):
        for j in range(N):
            if I[i, j]:
                for k in range(N):
                    if I[j, k]:
                        for l in range(N):
                            if I[i, l] and l != k:
                                O[l, k] = 1
    np.fill_diagonal(O, 0)
    return O.astype(np.float32)


def generate_random_graph(n, p=0.1):
    """
    Generate an n x n adjacency matrix with edge probability p (no self-loops).
    """
    I = (np.random.rand(n, n) < p).astype(np.float32)
    np.fill_diagonal(I, 0)
    return I


# --------------------------
# 3. Permutation Robustness Datasets
# --------------------------
def permute_matrix(mat, permutation):
    """
    Permute rows and columns of 'mat' according to 'permutation'.
    """
    return mat[permutation][:, permutation]


# 3.1. Data Augmentation Dataset (Random Permutations)
class AugmentedGraphDataset(Dataset):
    def __init__(self, transform, n, selected_features, samples=500, p=0.1):
        self.samples = []
        self.transform = transform
        self.n = n
        self.selected_features = selected_features  # list of feature indices to use
        self.p = p
        for _ in range(samples):
            I = self.generate_random_graph(n, p)
            O = transform(I)
            self.samples.append((I, O))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        I, O = self.samples[idx]
        # Apply a random permutation to both input and target
        perm = np.random.permutation(self.n)
        I_perm = permute_matrix(I, perm)
        O_perm = permute_matrix(O, perm)
        # Compute features using the helper function
        feats, _ = compute_features(I_perm)
        feats_selected = feats[..., self.selected_features]
        # Convert to tensors with channels-first ordering
        x = torch.tensor(feats_selected, dtype=torch.float32).permute(2, 0, 1)
        y = torch.tensor(O_perm, dtype=torch.float32).unsqueeze(0)
        return x, y

    def generate_random_graph(self, n, p):
        mat = (np.random.rand(n, n) < p).astype(np.float32)
        np.fill_diagonal(mat, 0)
        return mat


# 3.2. Canonical Labeling Dataset (Sort nodes by degree)
def canonical_labeling_by_degree(I):
    """
    Returns a canonical version of adjacency matrix I by sorting nodes by degree (row sum).
    """
    degrees = I.sum(axis=1)
    idx = np.argsort(degrees)  # sort nodes by degree (lowest first)
    return I[idx][:, idx]


class CanonicalGraphDataset(Dataset):
    def __init__(self, transform, n, selected_features, samples=500, p=0.1):
        self.samples = []
        self.transform = transform
        self.n = n
        self.selected_features = selected_features
        self.p = p
        for _ in range(samples):
            I = self.generate_random_graph(n, p)
            I_can = canonical_labeling_by_degree(I)
            O = transform(I)
            O_can = canonical_labeling_by_degree(O)
            self.samples.append((I_can, O_can))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        I_can, O_can = self.samples[idx]
        # Compute features using the helper function
        feats, _ = compute_features(I_can)
        feats_selected = feats[..., self.selected_features]
        x = torch.tensor(feats_selected, dtype=torch.float32).permute(2, 0, 1)
        y = torch.tensor(O_can, dtype=torch.float32).unsqueeze(0)
        return x, y

    def generate_random_graph(self, n, p):
        mat = (np.random.rand(n, n) < p).astype(np.float32)
        np.fill_diagonal(mat, 0)
        return mat


# --------------------------
# 4. CNN Model Definition
# --------------------------
class CNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=32):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x


# --------------------------
# 5. Loss Functions
# --------------------------
def weighted_bce_loss(pred, target, diag_weight=10.0):
    """
    Computes pointwise BCE loss and up-weights the diagonal errors.
    """
    loss = F.binary_cross_entropy(pred, target, reduction='none')
    b, _, n, _ = pred.shape
    weight = torch.ones_like(pred)
    for i in range(n):
        weight[:, :, i, i] = diag_weight
    weighted_loss = loss * weight
    return weighted_loss.mean()


def masked_bce_loss(pred, target):
    """
    Computes pointwise BCE loss but ignores diagonal elements.
    """
    loss = F.binary_cross_entropy(pred, target, reduction='none')
    b, _, n, _ = pred.shape
    mask = torch.ones_like(pred, dtype=torch.bool)
    for i in range(n):
        mask[:, :, i, i] = False
    return loss[mask].mean()


# --------------------------
# 6. Training and Evaluation Functions
# --------------------------
def train_model(model, loader, optimizer, epochs=240, device=torch.device('cpu'), loss_type='weighted'):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            if loss_type == 'weighted':
                loss = weighted_bce_loss(pred, y, diag_weight=10.0)
            elif loss_type == 'masked':
                loss = masked_bce_loss(pred, y)
            else:
                loss = F.binary_cross_entropy(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss = {total_loss / len(loader):.4f}')


def evaluate_model(model, transform, selected_features, sizes=[8, 16, 32]):
    """
    Evaluate the model on graphs of various sizes (without plotting) and return average accuracy.
    """
    accuracies = []
    for n in sizes:
        I = generate_random_graph(n)
        O_exp = completing_square_no_diag(I)
        feats, _ = compute_features(I)
        feats_selected = feats[..., selected_features]
        x_tensor = torch.tensor(feats_selected, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            pred = model(x_tensor).squeeze(0).squeeze(0).numpy()
        pred_bin = (pred > 0.5).astype(int)
        correctness_mat = (pred_bin == O_exp).astype(int)
        accuracy = correctness_mat.mean()
        accuracies.append(accuracy)
    return np.mean(accuracies)


# --------------------------
# 7. Main Execution: Running All Configurations
# --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = completing_square_no_diag

# --- Feature Selection using Random Forest on small graphs (n=8) ---
# We compute the features on 1000 graphs and rank the features by importance.
n = 8
feature_list = []
output_list = []
for _ in range(1000):
    I = generate_random_graph(n)
    O = transform(I).flatten()
    feats, feat_names = compute_features(I)
    features = feats.reshape(-1, feats.shape[-1])
    feature_list.append(features)
    output_list.append(O)
X_all = np.vstack(feature_list)
y_all = np.hstack(output_list)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_all, y_all)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature ranking (by importance):")
for idx in indices:
    print(f"{feat_names[idx]}: importance = {importances[idx]:.4f}")

# Define the different numbers of top features to try (ignoring the 1-feature case)
top_features_options = [2, 3, 4, 5]

# --- Define the configurations ---
# Each configuration is a dictionary with keys:
#   'dataset': either "augmented" or "canonical"
#   'loss': either "weighted" or "masked"
configurations = [
    {"dataset": "augmented", "loss": "weighted"},
    {"dataset": "augmented", "loss": "masked"},
    {"dataset": "canonical", "loss": "weighted"},
    {"dataset": "canonical", "loss": "masked"}
]

results = []

for num_feats in top_features_options:
    # Select the top 'num_feats' features (indices)
    selected_features = indices[:num_feats].tolist()
    print(f"\n--- Using top {num_feats} features: {[feat_names[i] for i in selected_features]} ---")
    for config in configurations:
        print(f"\nConfiguration: Dataset = {config['dataset']}, Loss = {config['loss']}")
        # Choose dataset class based on configuration
        if config["dataset"] == "augmented":
            train_set = AugmentedGraphDataset(transform=transform, n=n, selected_features=selected_features,
                                              samples=1000, p=0.1)
        else:
            train_set = CanonicalGraphDataset(transform=transform, n=n, selected_features=selected_features,
                                              samples=1000, p=0.1)

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

        # Reinitialize the model and optimizer for each configuration
        model = CNNModel(in_channels=len(selected_features), hidden_channels=32).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model with the specified loss type
        train_model(model, train_loader, optimizer, epochs=170, device=device, loss_type=config["loss"])

        # Evaluate the model on multiple test sizes and compute average accuracy
        avg_acc = evaluate_model(model, transform, selected_features, sizes=[6, 8, 11, 19, 32, 64, 128])
        print(f"Achieved average accuracy: {avg_acc:.4f}")
        # Record configuration details and accuracy
        results.append({
            "dataset": config["dataset"],
            "loss": config["loss"],
            "num_features": num_feats,
            "selected_features": [feat_names[i] for i in selected_features],
            "accuracy": avg_acc
        })

# Determine the best configuration overall
best_config = max(results, key=lambda x: x["accuracy"])
print("\n\nBest configuration overall:")
print(f"Dataset type: {best_config['dataset']}")
print(f"Loss type: {best_config['loss']}")
print(f"Number of top features: {best_config['num_features']}")
print(f"Selected features: {best_config['selected_features']}")
print(f"Achieved average accuracy: {best_config['accuracy']:.4f}")
