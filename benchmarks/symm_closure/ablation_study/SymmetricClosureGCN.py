import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np

# Import PyTorch Geometric classes
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv

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
    Each graph will have n nodes. We use a constant feature (1.0) for every node.
    The directed edges come from the nonzero entries of A.
    """
    n = A.size(0)
    # Use a constant feature for every node.
    x = torch.ones((n, 1), dtype=torch.float32)
    # Build edge_index from nonzero entries. edge_index will have shape [2, num_edges].
    edge_index = torch.nonzero(A, as_tuple=False).t().contiguous()
    # Store the target symmetric closure (an n x n matrix) as data.y.
    data = Data(x=x, edge_index=edge_index, y=target)
    return data

# --- PyTorch Geometric Dataset ---
class GraphDatasetGCN(torch.utils.data.Dataset):
    def __init__(self, matrices, targets):
        self.data_list = [matrix_to_graph_data(mat, tar) for mat, tar in zip(matrices, targets)]
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

# --- Define the GCN-based model ---
class SymmetricClosureGCN(nn.Module):
    def __init__(self, hidden_dim=16):
        super(SymmetricClosureGCN, self).__init__()
        # Two-layer GCN for encoding node features.
        self.conv1 = GCNConv(1, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # MLP for predicting the edge between a pair of nodes.
        # Input: concatenated embeddings for node i and node j.
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        # For simplicity we assume that data contains one graph (batch size 1).
        # If using batches, you can iterate over the graphs.
        x, edge_index = data.x, data.edge_index
        # Compute node embeddings.
        h = self.conv1(x, edge_index)
        h = torch.relu(h)
        h = self.conv2(h, edge_index)
        # For every pair of nodes, predict an edge probability.
        n = h.size(0)
        # Create all index pairs (i,j)
        i_indices = torch.arange(n).unsqueeze(1).repeat(1, n).view(-1)
        j_indices = torch.arange(n).unsqueeze(0).repeat(n, 1).view(-1)
        h_i = h[i_indices]  # shape: (n*n, hidden_dim)
        h_j = h[j_indices]  # shape: (n*n, hidden_dim)
        h_pair = torch.cat([h_i, h_j], dim=1)  # shape: (n*n, 2*hidden_dim)
        pred_matrix = self.mlp(h_pair).view(n, n)
        return torch.sigmoid(pred_matrix)

# --- Training setup ---
num_epochs = 1000
learning_rate = 0.1

# Generate training data (2x2 matrices)
train_mats, train_targets = generate_all_2x2_matrices()
train_dataset = GraphDatasetGCN(train_mats, train_targets)
# Use a batch size of 1 (each graph is small)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

model = SymmetricClosureGCN(hidden_dim=16)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# --- Training loop ---
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for data in train_loader:
        optimizer.zero_grad()
        pred = model(data)  # pred has shape (n, n) for the graph in data.
        # data.y is the target symmetric closure matrix.
        loss = (pred - data.y).abs().mean()  # Using mean absolute error as a metric.
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

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
        pred = model(data)  # shape: (n, n)
        # Threshold predictions at 0.5 to obtain binary outputs.
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
