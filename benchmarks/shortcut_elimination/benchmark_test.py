import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from benchmarks.shortcut_elimination.benchmark import generate_shortcut_dataset, evaluate_model

# reproducibility
np.random.seed(42)
torch.manual_seed(42)


def enumerate_graphs(n):
    """
    Enumerate all n x n binary adjacency matrices (no self-loops).
    """
    num_edges = n * (n - 1)
    total = 1 << num_edges
    graphs = []
    for i in range(total):
        bits = [(i >> j) & 1 for j in range(num_edges)]
        A = np.zeros((n, n), dtype=np.float32)
        idx = 0
        for r in range(n):
            for c in range(n):
                if r != c:
                    A[r, c] = bits[idx]
                    idx += 1
        graphs.append(A)
    return graphs


class ExhaustiveGraphDataset(Dataset):
    """
    Exhaustively generated graphs with precomputed features [I, I^2].
    """
    def __init__(self, transform_func, n):
        self.samples = []
        for I in enumerate_graphs(n):
            O = transform_func(I)
            I2 = I.dot(I)
            features = np.stack([I, I2], axis=-1)
            self.samples.append((features, O))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, O = self.samples[idx]
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(O, dtype=torch.float32)
        )


class PointwiseMLP(nn.Module):
    """
    Applies an MLP to each edge's feature vector [I, I^2].
    """
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, n, n, 2)
        B, n, _, _ = x.shape
        x = x.view(B, n*n, 2)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.view(B, n, n)


def train_model(model, loader, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for feat, tgt in loader:
            feat, tgt = feat.to(device), tgt.to(device)
            optimizer.zero_grad()
            out = model(feat)
            loss = F.binary_cross_entropy(out, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feat.size(0)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: Loss = {total_loss/len(loader.dataset):.4f}")


# transformation for exhaustive training
def shortcut_elimination(I):
    I2 = I.dot(I)
    O = I.copy()
    O[(I2 > 0) & (I == 1)] = 0
    return O


# setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# training on 3x3 graphs
dataset = ExhaustiveGraphDataset(shortcut_elimination, n=3)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
model = PointwiseMLP(hidden_dim=16).to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
train_model(model, loader, opt, epochs=200, device=device)


# benchmark evaluation
def model_fn(adj: np.ndarray) -> np.ndarray:
    I = torch.from_numpy(adj).float().to(device)
    I2 = I @ I
    x = torch.stack([I, I2], dim=-1).unsqueeze(0)
    with torch.no_grad():
        keep = model(x).squeeze(0).cpu().numpy()
    return 1 - keep


graphs, _, masks = generate_shortcut_dataset()
results = evaluate_model(model_fn, graphs, masks)
print(results)
