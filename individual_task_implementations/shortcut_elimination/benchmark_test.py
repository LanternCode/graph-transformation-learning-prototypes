import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from benchmark import generate_shortcut_dataset, evaluate_model

# reproducibility
np.random.seed(42)
torch.manual_seed(42)


def enumerate_graphs(n):
    """
    Enumerate all directed binary adjacency matrices without self-loops.

    Args:
        n (int): Number of nodes in each adjacency matrix.

    Returns:
        list[np.ndarray]: All n x n binary adjacency matrices with zero diagonal.
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
    Dataset of exhaustive graph-transform pairs for a fixed node count.

    Args:
        transform_func (Callable[[np.ndarray], np.ndarray]): Transformation that
            maps an adjacency matrix to its target output matrix.
        n (int): Number of nodes in each exhaustively enumerated graph.

    Returns:
        ExhaustiveGraphDataset: Dataset whose samples are feature tensors
        containing [I, I^2] and target shortcut-eliminated adjacency matrices.
    """
    def __init__(self, transform_func, n):
        """
        Precompute every graph, feature tensor, and target matrix.

        Args:
            transform_func (Callable[[np.ndarray], np.ndarray]): Function used
                to produce the target matrix for each adjacency matrix.
            n (int): Number of nodes in each enumerated graph.

        Returns:
            None.
        """
        self.samples = []
        for I in enumerate_graphs(n):
            O = transform_func(I)
            I2 = I.dot(I)
            features = np.stack([I, I2], axis=-1)
            self.samples.append((features, O))

    def __len__(self):
        """
        Return the number of precomputed graph samples.

        Args:
            None.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve one graph feature tensor and target matrix.

        Args:
            idx (int): Dataset index to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Feature tensor with shape
            [n, n, 2] and target matrix with shape [n, n].
        """
        features, O = self.samples[idx]
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(O, dtype=torch.float32)
        )


class PointwiseMLP(nn.Module):
    """
    Pointwise neural model for shortcut-elimination decisions.

    Args:
        hidden_dim (int): Width of the hidden layer used for each edge feature
            vector.

    Returns:
        PointwiseMLP: Module that maps each [I, I^2] edge feature vector to a
        keep probability for the corresponding adjacency entry.
    """
    def __init__(self, hidden_dim=16):
        """
        Initialize the pointwise MLP layers.

        Args:
            hidden_dim (int): Number of hidden units in the MLP.

        Returns:
            None.
        """
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Predict keep probabilities for every adjacency entry.

        Args:
            x (torch.Tensor): Batched feature tensor with shape [B, n, n, 2].

        Returns:
            torch.Tensor: Batched keep-probability matrices with shape
            [B, n, n].
        """
        # x: (B, n, n, 2)
        B, n, _, _ = x.shape
        x = x.view(B, n*n, 2)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.view(B, n, n)


def train_model(model, loader, optimizer, epochs, device):
    """
    Train the pointwise shortcut-elimination model.

    Args:
        model (torch.nn.Module): Model that maps feature tensors to keep
            probabilities.
        loader (DataLoader): Training data loader yielding feature-target pairs.
        optimizer (torch.optim.Optimizer): Optimizer used to update model
            parameters.
        epochs (int): Number of training epochs.
        device (torch.device): Device used for tensors and model execution.

    Returns:
        None.
    """
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
    """
    Remove existing edges that have a two-hop witness in the input graph.

    Args:
        I (np.ndarray): Square binary adjacency matrix.

    Returns:
        np.ndarray: Copy of I with entries set to zero wherever I has an edge
        and I^2 indicates at least one two-hop path between the same endpoints.
    """
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
    """
    Score shortcut likelihoods for one benchmark adjacency matrix.

    Args:
        adj (np.ndarray): Square binary adjacency matrix to evaluate.

    Returns:
        np.ndarray: Matrix of shortcut scores, where higher values indicate a
        stronger prediction that an existing edge should be removed.
    """
    I = torch.from_numpy(adj).float().to(device)
    I2 = I @ I
    x = torch.stack([I, I2], dim=-1).unsqueeze(0)
    with torch.no_grad():
        keep = model(x).squeeze(0).cpu().numpy()
    return 1 - keep


graphs, _, masks = generate_shortcut_dataset()
results = evaluate_model(model_fn, graphs, masks)
print(results)
