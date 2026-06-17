"""
Train and evaluate a GIN baseline for symmetric closure.

This prototype converts each binary matrix into a directed PyTorch Geometric
graph, encodes nodes with GIN layers using constant node features, and predicts
the full symmetric closure matrix from all ordered pairs of node embeddings. It
serves as a graph-neural baseline rather than a direct local transpose-feature
ablation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINConv


# --- Utility functions ---
def symmetric_closure(A):
    """
    Compute the symmetric closure (OR) of a binary matrix.
    For binary values, A OR Aᵀ is equivalent to (A + Aᵀ > 0).
    """
    return ((A + A.transpose(-2, -1)) > 0).float()


def generate_all_2x2_matrices():
    mats = []
    targets = []
    for bits in itertools.product([0, 1], repeat=4):
        mat = torch.tensor(bits, dtype=torch.float32).view(2, 2)
        target = symmetric_closure(mat)
        mats.append(mat)
        targets.append(target)
    return mats, targets


def matrix_to_graph_data(A, target):
    """
    Convert an n x n binary matrix A into a PyG Data object.
    Each graph will have n nodes with a constant feature of 1.
    The directed edges are obtained from the nonzero entries of A.
    """
    n = A.size(0)
    x = torch.ones((n, 1), dtype=torch.float32)
    edge_index = torch.nonzero(A, as_tuple=False).t().contiguous()
    data = Data(x=x, edge_index=edge_index, y=target)
    return data


# --- PyTorch Geometric Dataset ---
class GraphDatasetGIN(torch.utils.data.Dataset):
    def __init__(self, matrices, targets):
        self.data_list = [matrix_to_graph_data(mat, tar) for mat, tar in zip(matrices, targets)]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# --- Define the GIN-based model ---
class SymmetricClosureGIN(nn.Module):
    def __init__(self, hidden_dim=16):
        super(SymmetricClosureGIN, self).__init__()
        # First GIN layer with its own MLP (separate weights)
        mlp1 = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gin1 = GINConv(mlp1)

        # Second GIN layer with its own MLP
        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gin2 = GINConv(mlp2)

        # Projection MLP for pairwise node predictions.
        # It takes the concatenation of two node embeddings as input.
        self.proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        # Compute node embeddings via the two GIN layers.
        h = self.gin1(data.x, data.edge_index)
        h = self.gin2(h, data.edge_index)

        n = h.size(0)
        # Create all ordered pairs (i, j)
        i_indices = torch.arange(n, device=h.device).unsqueeze(1).repeat(1, n).view(-1)
        j_indices = torch.arange(n, device=h.device).unsqueeze(0).repeat(n, 1).view(-1)
        h_i = h[i_indices]  # shape: (n*n, hidden_dim)
        h_j = h[j_indices]  # shape: (n*n, hidden_dim)

        # Concatenate node embeddings and predict the edge value.
        h_pair = torch.cat([h_i, h_j], dim=1)  # shape: (n*n, 2*hidden_dim)
        out = self.proj(h_pair).view(n, n)
        # Map to [0,1] with sigmoid to obtain probabilities.
        return torch.sigmoid(out)


# --- Training setup ---
num_epochs = 1000
learning_rate = 0.1

# Generate training data (2x2 matrices)
train_mats, train_targets = generate_all_2x2_matrices()
train_dataset = GraphDatasetGIN(train_mats, train_targets)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Initialize model and optimizer
model = SymmetricClosureGIN(hidden_dim=16)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# --- Training loop ---
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for data in train_loader:
        optimizer.zero_grad()
        pred = model(data)  # pred has shape (n, n)
        loss = (pred - data.y).abs().mean()  # Mean absolute error as metric
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")


# --- Testing on n x n matrices ---
def generate_random_n_by_n_binary_graphs(num_graphs, n):
    data_list = []
    for _ in range(num_graphs):
        A = torch.randint(0, 2, (n, n)).float()
        target = symmetric_closure(A)
        data = matrix_to_graph_data(A, target)
        data_list.append(data)
    return data_list


test_data_list = generate_random_n_by_n_binary_graphs(200, 1000)
test_loader = DataLoader(test_data_list, batch_size=1)

total_error_count = 0
total_elements = 0
model.eval()
with torch.no_grad():
    for data in test_loader:
        pred = model(data)
        pred_binary = (pred > 0.5).float()
        error_matrix = (pred_binary - data.y).abs()
        error_count = (error_matrix > 0.5).sum().item()  # count mismatches
        total_error_count += error_count
        total_elements += data.y.numel()

total_correct_count = total_elements - total_error_count
accuracy = (total_correct_count / total_elements) * 100
print(f"\nTotal errors: {total_error_count}")
print(f"Total correct predictions: {total_correct_count}")
print(f"Accuracy: {accuracy:.2f}%")
