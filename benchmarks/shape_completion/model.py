import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


# -------------------------------
# 1. Synthetic Data Generation
# -------------------------------
def generate_cycle_indices(matrix_size, cycle_length):
    """Randomly choose `cycle_length` nodes from matrix_size and return a list of node indices."""
    return sorted(random.sample(range(matrix_size), cycle_length))


def create_complete_matrix(matrix_size, shape_type):
    """
    Create an adjacency matrix of size (matrix_size x matrix_size)
    with an embedded cycle corresponding to shape_type.
    shape_type: 'triangle', 'square', 'pentagon', 'hexagon'
    Returns:
      complete_matrix: np.array with 1's for edges of the cycle.
      shape_nodes: indices of nodes forming the cycle.
    """
    shape_to_length = {'triangle': 3, 'square': 4, 'pentagon': 5, 'hexagon': 6}
    cycle_length = shape_to_length[shape_type]

    # Initialize matrix
    M = np.zeros((matrix_size, matrix_size), dtype=np.float32)

    # Choose nodes for the cycle
    shape_nodes = generate_cycle_indices(matrix_size, cycle_length)

    # Create cycle edges (undirected graph: symmetric matrix)
    for i in range(cycle_length):
        u = shape_nodes[i]
        v = shape_nodes[(i + 1) % cycle_length]
        M[u, v] = 1.0
        M[v, u] = 1.0
    return M, shape_nodes


def remove_edges(M_complete, shape_nodes, removal_prob=0.3):
    """
    Given the complete matrix and the list of nodes forming the cycle,
    randomly remove some edges (simulate incomplete matrix).
    removal_prob: probability of removing each cycle edge.
    Ensures at least one edge remains.
    Returns:
      M_incomplete: np.array with some edges removed.
    """
    M_incomplete = M_complete.copy()
    cycle_length = len(shape_nodes)
    removed = 0
    for i in range(cycle_length):
        u = shape_nodes[i]
        v = shape_nodes[(i + 1) % cycle_length]
        if random.random() < removal_prob:
            M_incomplete[u, v] = 0.0
            M_incomplete[v, u] = 0.0
            removed += 1
    # Ensure at least one edge remains (if all removed, randomly restore one)
    if removed == cycle_length:
        u = shape_nodes[0]
        v = shape_nodes[1]
        M_incomplete[u, v] = 1.0
        M_incomplete[v, u] = 1.0
    return M_incomplete


def generate_sample(matrix_size, shape_type, removal_prob=0.3):
    """
    Generate one sample: a tuple (M_incomplete, M_complete, shape_nodes, shape_type)
    """
    M_complete, shape_nodes = create_complete_matrix(matrix_size, shape_type)
    M_incomplete = remove_edges(M_complete, shape_nodes, removal_prob)
    return M_incomplete, M_complete, shape_nodes, shape_type


# -------------------------------
# 2. Feature Extraction (Edge-wise)
# -------------------------------
# You can add new feature functions here.
def feature_edge_value(A):
    return A


def feature_degree(A):
    degrees = np.sum(A, axis=1)
    N = A.shape[0]
    row_deg = np.tile(degrees.reshape(-1, 1), (1, N))
    col_deg = np.tile(degrees.reshape(1, -1), (N, 1))
    return np.stack([row_deg, col_deg], axis=-1)


def feature_A2(A):
    return np.matmul(A, A)


def feature_A3(A):
    return np.matmul(np.matmul(A, A), A)


def feature_A4(A):
    # Compute A^4 using numpy's matrix power
    return np.linalg.matrix_power(A, 4)


def feature_A5(A):
    # Compute A^5 using numpy's matrix power
    return np.linalg.matrix_power(A, 5)


def feature_principal_eigenvector(A):
    """
    Compute the principal eigenvector of A.
    Since A is symmetric, we use np.linalg.eigh which returns eigenvalues in ascending order.
    The principal eigenvector corresponds to the largest eigenvalue.
    We then generate two matrices: one for the row nodes and one for the column nodes.
    """
    eigvals, eigvecs = np.linalg.eigh(A)
    principal = eigvecs[:, -1]  # largest eigenvalue's eigenvector
    N = A.shape[0]
    row_eigen = np.tile(principal.reshape(-1, 1), (1, N))
    col_eigen = np.tile(principal.reshape(1, -1), (N, 1))
    return np.stack([row_eigen, col_eigen], axis=-1)


def feature_clustering_coefficient(A):
    """
    Compute the clustering coefficient for each node.
    For each node i, this is the ratio of the number of edges between its neighbors to the maximum possible number of such edges.
    Then, create two matrices: one for the row nodes and one for the column nodes.
    """
    N = A.shape[0]
    clustering = np.zeros(N)
    for i in range(N):
        neighbors = np.where(A[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            clustering[i] = 0.0
        else:
            subgraph = A[np.ix_(neighbors, neighbors)]
            edges_between = np.sum(subgraph) / 2.0  # since graph is undirected
            clustering[i] = edges_between / (k * (k - 1) / 2.0)
    row_cluster = np.tile(clustering.reshape(-1, 1), (1, N))
    col_cluster = np.tile(clustering.reshape(1, -1), (N, 1))
    return np.stack([row_cluster, col_cluster], axis=-1)


def extract_edge_features(A):
    """
    Given an adjacency matrix A, compute the following edge-level features:
      - Raw edge value (1 channel)
      - Node degrees (2 channels)
      - A^2 (1 channel)
      - A^3 (1 channel)
      - A^4 (1 channel)
      - A^5 (1 channel)
      - Principal eigenvector values for both endpoints (2 channels)
      - Clustering coefficients for both endpoints (2 channels)
    This gives a total of 11 channels per edge.
    """
    features = []
    features.append(feature_edge_value(A)[..., None])    # 1 channel
    features.append(feature_degree(A))                     # 2 channels
    features.append(feature_A2(A)[..., None])              # 1 channel
    features.append(feature_A3(A)[..., None])              # 1 channel
    features.append(feature_A4(A)[..., None])              # 1 channel
    features.append(feature_A5(A)[..., None])              # 1 channel
    features.append(feature_principal_eigenvector(A))      # 2 channels
    features.append(feature_clustering_coefficient(A))     # 2 channels
    feat_array = np.concatenate(features, axis=-1)
    return feat_array  # shape: (N, N, 11)


# -------------------------------
# 3. Random Forest Feature Selection
# -------------------------------
def build_edge_dataset(samples):
    X_list = []
    y_list = []
    for M_incomplete, M_complete, shape_nodes, shape_type in samples:
        feat = extract_edge_features(M_incomplete)
        N = feat.shape[0]
        X_sample = feat.reshape(-1, feat.shape[-1])
        y_sample = M_complete.reshape(-1)
        X_list.append(X_sample)
        y_list.append(y_sample)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


# -------------------------------
# 4. PyTorch Dataset and Model
# -------------------------------
class GraphEdgeDataset(Dataset):
    """
    Each sample is an entire graph (adjacency matrix) with:
      - 'features': Flattened edge features computed from the incomplete matrix.
      - 'target': Flattened complete adjacency matrix.
      - 'matrix_size': To reshape predictions later.
      - 'shape_type': The type of shape (triangle, square, etc.) embedded in the matrix.
    """
    def __init__(self, samples, selected_features=None, augment_permutation=False):
        self.samples = samples
        self.selected_features = selected_features
        self.augment_permutation = augment_permutation

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        M_incomplete, M_complete, shape_nodes, shape_type = self.samples[idx]
        N = M_incomplete.shape[0]
        if self.augment_permutation:
            perm = np.random.permutation(N)
            M_incomplete = M_incomplete[perm][:, perm]
            M_complete = M_complete[perm][:, perm]
        feat = extract_edge_features(M_incomplete)
        if self.selected_features is not None:
            feat = feat[..., self.selected_features]
        feat_flat = feat.reshape(-1, feat.shape[-1]).astype(np.float32)
        target_flat = M_complete.reshape(-1).astype(np.float32)

        sample = {
            'features': torch.from_numpy(feat_flat),
            'target': torch.from_numpy(target_flat),
            'matrix_size': N,
            'shape_type': shape_type
        }
        return sample


class EdgeMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(EdgeMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        out = self.mlp(x)
        out = out.squeeze(-1)
        return out


class DeepEdgeMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=4, dropout_rate=0.2):
        """
        A deeper MLP with:
          - An input layer projecting from input_dim to hidden_dim
          - num_layers hidden layers of size hidden_dim, each followed by ReLU, BatchNorm, and Dropout
          - A final linear layer that outputs a single logit per edge
        """
        super(DeepEdgeMLP, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Dropout(dropout_rate))

        # Additional hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))

        # Final output layer: output one logit per edge
        layers.append(nn.Linear(hidden_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: Either a 2D tensor (num_edges, input_dim) or a 3D tensor (batch_size, num_edges, input_dim).
        """
        if x.dim() == 2:
            # x shape is (num_edges, input_dim); add a batch dimension.
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, num_edges, input_dim = x.shape
        # Flatten for processing: (batch_size * num_edges, input_dim)
        x = x.view(-1, input_dim)
        out = self.mlp(x)  # (batch_size * num_edges, 1)
        # Reshape back
        out = out.view(batch_size, num_edges)

        if squeeze_output and batch_size == 1:
            out = out.squeeze(0)
        return out


# -------------------------------
# 5. Training and Evaluation
# -------------------------------
def train_model(model, dataloader, num_epochs=20, lr=1e-3, device='cpu'):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")


def evaluate_model(model, dataset, device='cpu', desired_sizes=[6, 8, 11, 19, 32, 64, 128]):
    """
    Evaluates the model over the entire dataset and plots one sample per desired matrix size.
    For each plotted sample, the matrix size, shape type, and sample accuracy are displayed.
    """
    model.eval()
    all_accuracies = []
    samples_by_size = {}  # Dictionary to store one sample per matrix size from desired_sizes

    with torch.no_grad():
        for sample in dataset:
            N = sample['matrix_size']
            features = sample['features'].to(device)
            target = sample['target'].to(device)
            logits = model(features)
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == target).float().mean().item()
            all_accuracies.append(acc)

            # Save the first encountered sample for each desired size.
            if N in desired_sizes and N not in samples_by_size:
                samples_by_size[N] = sample

    avg_acc = np.mean(all_accuracies)
    print(f"Average accuracy over dataset: {avg_acc*100:.2f}%")

    # Now, plot one sample per desired matrix size.
    for size in desired_sizes:
        if size in samples_by_size:
            sample = samples_by_size[size]
            features = sample['features'].to(device)
            target = sample['target'].to(device)
            logits = model(features)
            preds = (torch.sigmoid(logits) > 0.5).float()
            preds_matrix = preds.cpu().numpy().reshape(size, size)
            target_matrix = target.cpu().numpy().reshape(size, size)
            shape_type = sample['shape_type']
            sample_acc = (preds == target).float().mean().item()

            plt.figure(figsize=(10, 4))
            plt.suptitle(f"Matrix Size: {size} | Shape: {shape_type} | Sample Accuracy: {sample_acc*100:.2f}%")
            plt.subplot(1, 2, 1)
            plt.title("Predicted Matrix")
            plt.imshow(preds_matrix, cmap='gray_r')
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.title("Expected Matrix")
            plt.imshow(target_matrix, cmap='gray_r')
            plt.colorbar()
            plt.show()
        else:
            print(f"No sample found for matrix size: {size}")

    return avg_acc


if __name__ == "__main__":
    # Generate dataset
    matrix_sizes = [6, 8, 11, 19, 32, 64, 128]
    shape_types = ['triangle', 'square', 'pentagon', 'hexagon']
    num_samples_per_config = 5

    samples = []
    for size in matrix_sizes:
        for _ in range(num_samples_per_config):
            shape_type = random.choices(
                population=['triangle', 'square', 'pentagon', 'hexagon'],
                weights=[1, 1, 1, 1],
                k=1)[0]
            sample = generate_sample(size, shape_type, removal_prob=0.3)
            samples.append(sample)

    X_edges, y_edges = build_edge_dataset(samples)
    print("Edge dataset shape:", X_edges.shape, "Labels shape:", y_edges.shape)

    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_edges, y_edges, test_size=0.2, random_state=42)

    rf_clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train_rf, y_train_rf)
    y_pred_rf = rf_clf.predict(X_test_rf)
    rf_accuracy = accuracy_score(y_test_rf, y_pred_rf)
    print("Random Forest accuracy on edge prediction: {:.2f}%".format(rf_accuracy * 100))

    feature_names = [
        "edge_value",
        "row_degree",
        "col_degree",
        "A^2",
        "A^3",
        "A^4",
        "A^5",
        "principal_eigenvector_row",
        "principal_eigenvector_col",
        "clustering_row",
        "clustering_col"
    ]

    importances = rf_clf.feature_importances_
    print("Feature Importances:")
    for name, imp in zip(feature_names, importances):
        print(f"{name}: {imp:.4f}")

    selected_feature_indices = np.arange(X_edges.shape[1])

    # -------------------------------
    # 6. Main Execution (with stratified split)
    # -------------------------------
    # random.seed(42)
    # np.random.seed(42)
    # torch.manual_seed(42)

    all_samples = samples  # samples generated earlier

    # Group samples by matrix size.
    grouped_samples = defaultdict(list)
    for sample in all_samples:
        # sample[0] is M_incomplete; its shape[0] gives the matrix size.
        N = sample[0].shape[0]
        grouped_samples[N].append(sample)

    # Now, build train and test sets such that each desired matrix size has at least one sample in the test set.
    train_samples = []
    test_samples = []
    desired_sizes = [6, 8, 11, 19, 32, 64, 128]

    for size in desired_sizes:
        group = grouped_samples[size]
        if group:  # if there are any samples for this size
            random.shuffle(group)  # shuffle to randomize selection
            # Reserve the first sample for the test set.
            test_samples.append(group[0])
            # The rest (if any) go to the training set.
            train_samples.extend(group[1:])
        else:
            print(f"Warning: No samples generated for matrix size: {size}")

    # Create PyTorch datasets using the new splits.
    train_dataset = GraphEdgeDataset(train_samples, selected_features=selected_feature_indices,
                                     augment_permutation=True)
    test_dataset = GraphEdgeDataset(test_samples, selected_features=selected_feature_indices, augment_permutation=False)

    # Use DataLoader; since samples have different matrix sizes, we use batch_size=1.
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Check that we have at least one sample per desired size in the test set.
    print("Test samples by matrix size:")
    for size in desired_sizes:
        count = sum(1 for sample in test_samples if sample[0].shape[0] == size)
        print(f"Matrix size {size}: {count} sample(s)")

    # Determine input feature dimension (from one sample).
    sample_item = train_dataset[0]
    input_dim = sample_item['features'].shape[1]
    print("Input feature dimension:", input_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = EdgeMLP(input_dim=input_dim, hidden_dim=64).to(device)
    model = DeepEdgeMLP(input_dim=input_dim, hidden_dim=128, num_layers=4, dropout_rate=0.2).to(device)

    print("\nTraining Neural Network...")
    train_model(model, train_loader, num_epochs=30, lr=1e-3, device=device)

    print("\nEvaluating Neural Network on test dataset...")
    evaluate_model(model, test_dataset, device=device, desired_sizes=desired_sizes)

    torch.save(model.state_dict(), "best_model.pth")
