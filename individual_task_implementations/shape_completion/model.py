import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


# -------------------------------
# 1. Synthetic Data Generation
# -------------------------------
def generate_cycle_indices(matrix_size, cycle_length):
    """
    Randomly select node indices for an embedded cycle.

    Args:
        matrix_size: Number of nodes in the ambient adjacency matrix.
        cycle_length: Number of nodes that should form the embedded cycle.

    Returns:
        A sorted list of distinct node indices of length ``cycle_length``.
    """
    return sorted(random.sample(range(matrix_size), cycle_length))


def create_complete_matrix(matrix_size, shape_type):
    """
    Create a complete target adjacency matrix for one embedded cycle shape.

    Args:
        matrix_size: Number of rows and columns in the adjacency matrix.
        shape_type: Name of the cycle shape to embed. Supported values are
            ``'triangle'``, ``'square'``, ``'pentagon'``, and ``'hexagon'``.

    Returns:
        A tuple ``(complete_matrix, shape_nodes)`` where ``complete_matrix`` is
        a symmetric float32 adjacency matrix containing the full cycle and
        ``shape_nodes`` is the ordered list of nodes used by the cycle.
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
    Remove some cycle edges from a complete target adjacency matrix.

    Args:
        M_complete: Symmetric adjacency matrix containing the complete cycle.
        shape_nodes: Ordered node indices that define the cycle edges.
        removal_prob: Probability of removing each undirected cycle edge.

    Returns:
        A copy of ``M_complete`` with a random subset of cycle edges removed,
        while preserving at least one edge from the original cycle.
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
    Generate one incomplete-to-complete cycle reconstruction sample.

    Args:
        matrix_size: Number of nodes in the ambient adjacency matrix.
        shape_type: Cycle shape to embed in the matrix.
        removal_prob: Probability of removing each cycle edge from the input.

    Returns:
        A tuple ``(M_incomplete, M_complete, shape_nodes, shape_type)`` where
        ``M_incomplete`` is the observed matrix, ``M_complete`` is the target
        matrix, ``shape_nodes`` are the cycle nodes, and ``shape_type`` is the
        shape label carried for reporting.
    """
    M_complete, shape_nodes = create_complete_matrix(matrix_size, shape_type)
    M_incomplete = remove_edges(M_complete, shape_nodes, removal_prob)
    return M_incomplete, M_complete, shape_nodes, shape_type


# -------------------------------
# 2. Feature Extraction (Edge-wise)
# -------------------------------
def feature_edge_value(A):
    """
    Return the raw adjacency value for every candidate edge.

    Args:
        A: Square adjacency matrix.

    Returns:
        The input adjacency matrix, used as a single edge-feature channel.
    """
    return A


def feature_degree(A):
    """
    Compute endpoint degree features for every ordered node pair.

    Args:
        A: Square adjacency matrix.

    Returns:
        A ``(N, N, 2)`` array where channel 0 contains the row-node degree and
        channel 1 contains the column-node degree for each pair.
    """
    degrees = np.sum(A, axis=1)
    N = A.shape[0]
    row_deg = np.tile(degrees.reshape(-1, 1), (1, N))
    col_deg = np.tile(degrees.reshape(1, -1), (N, 1))
    return np.stack([row_deg, col_deg], axis=-1)


def feature_A2(A):
    """
    Compute the second adjacency-matrix power.

    Args:
        A: Square adjacency matrix.

    Returns:
        The matrix product ``A @ A``.
    """
    return np.matmul(A, A)


def feature_A3(A):
    """
    Compute the third adjacency-matrix power.

    Args:
        A: Square adjacency matrix.

    Returns:
        The matrix product ``A @ A @ A``.
    """
    return np.matmul(np.matmul(A, A), A)


def feature_A4(A):
    """
    Compute the fourth adjacency-matrix power.

    Args:
        A: Square adjacency matrix.

    Returns:
        The matrix power ``A^4``.
    """
    return np.linalg.matrix_power(A, 4)


def feature_A5(A):
    """
    Compute the fifth adjacency-matrix power.

    Args:
        A: Square adjacency matrix.

    Returns:
        The matrix power ``A^5``.
    """
    return np.linalg.matrix_power(A, 5)


def feature_principal_eigenvector(A):
    """
    Compute principal-eigenvector endpoint features for every node pair.

    Args:
        A: Symmetric square adjacency matrix.

    Returns:
        A ``(N, N, 2)`` array where the two channels contain the principal
        eigenvector values for the row endpoint and column endpoint.
    """
    eigvals, eigvecs = np.linalg.eigh(A)
    principal = eigvecs[:, -1]  # largest eigenvalue's eigenvector
    N = A.shape[0]
    row_eigen = np.tile(principal.reshape(-1, 1), (1, N))
    col_eigen = np.tile(principal.reshape(1, -1), (N, 1))
    return np.stack([row_eigen, col_eigen], axis=-1)


def feature_clustering_coefficient(A):
    """
    Compute endpoint clustering-coefficient features for every node pair.

    Args:
        A: Symmetric square adjacency matrix.

    Returns:
        A ``(N, N, 2)`` array where the channels contain the clustering
        coefficient of the row endpoint and column endpoint.
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
    Compute the complete 11-channel edge-feature tensor used by the models.

    Args:
        A: Square adjacency matrix representing an incomplete observed graph.

    Returns:
        A ``(N, N, 11)`` float array containing raw edge value, endpoint
        degrees, powers ``A^2`` through ``A^5``, principal-eigenvector endpoint
        values, and clustering-coefficient endpoint values.
    """
    features = []
    features.append(feature_edge_value(A)[..., None])      # 1 channel
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
    """
    Flatten graph samples into an edge-level feature and label dataset.

    Args:
        samples: Iterable of ``(M_incomplete, M_complete, shape_nodes,
            shape_type)`` tuples.

    Returns:
        A tuple ``(X, y)`` where ``X`` is a two-dimensional feature matrix with
        one row per candidate edge and ``y`` is the flattened target adjacency
        label vector.
    """
    X_list = []
    y_list = []
    for M_incomplete, M_complete, shape_nodes, shape_type in samples:
        feat = extract_edge_features(M_incomplete)
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
    PyTorch dataset for graph-level cycle-completion samples.

    Args:
        samples: List of tuples containing incomplete matrices, complete target
            matrices, shape-node lists, and shape labels.
        selected_features: Optional sequence of feature-channel indices to keep.
            If ``None``, all channels from ``extract_edge_features`` are used.
        augment_permutation: Whether to randomly permute node order each time a
            sample is read.

    Returns:
        Dataset items are dictionaries containing flattened edge features,
        flattened target labels, matrix size, and shape type.
    """
    def __init__(self, samples, selected_features=None, augment_permutation=False):
        """
        Store graph samples and dataset options.

        Args:
            samples: Graph-level samples to expose through the dataset.
            selected_features: Optional feature-channel indices to keep.
            augment_permutation: Whether to apply random node permutations.

        Returns:
            None.
        """
        self.samples = samples
        self.selected_features = selected_features
        self.augment_permutation = augment_permutation

    def __len__(self):
        """
        Return the number of graph-level samples in the dataset.

        Args:
            None.

        Returns:
            Integer number of stored samples.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Build tensors for one graph-level sample.

        Args:
            idx: Integer dataset index.

        Returns:
            A dictionary with keys ``features``, ``target``, ``matrix_size``, and
            ``shape_type``. Features have shape ``(N*N, C)`` and targets have
            shape ``(N*N,)``.
        """
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
    """
    Two-hidden-layer edge classifier for flattened edge features.

    Args:
        input_dim: Number of feature channels per edge.
        hidden_dim: Width of each hidden layer.

    Returns:
        A neural module that maps edge-feature tensors to one logit per edge.
    """
    def __init__(self, input_dim, hidden_dim=64):
        """
        Initialize the shallow edge MLP.

        Args:
            input_dim: Number of input features per edge.
            hidden_dim: Hidden-layer width.

        Returns:
            None.
        """
        super(EdgeMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Run edge features through the shallow MLP.

        Args:
            x: Tensor of edge features with final dimension ``input_dim``.

        Returns:
            Tensor of logits with the final singleton output dimension removed.
        """
        out = self.mlp(x)
        out = out.squeeze(-1)
        return out


class DeepEdgeMLP(nn.Module):
    """
    Deeper MLP edge classifier with BatchNorm and Dropout.

    Args:
        input_dim: Number of feature channels per edge.
        hidden_dim: Width of hidden layers.
        num_layers: Number of hidden linear blocks to use.
        dropout_rate: Dropout probability after each hidden block.

    Returns:
        A neural module that maps edge-feature tensors to one logit per edge.
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=4, dropout_rate=0.2):
        """
        Initialize the deep edge MLP.

        Args:
            input_dim: Number of input features per edge.
            hidden_dim: Hidden-layer width.
            num_layers: Number of hidden layers before the final output layer.
            dropout_rate: Dropout probability used after each hidden layer.

        Returns:
            None.
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
        Run edge features through the deep MLP.

        Args:
            x: Either a ``(num_edges, input_dim)`` tensor or a
                ``(batch_size, num_edges, input_dim)`` tensor.

        Returns:
            Logit tensor with shape ``(num_edges,)`` for a two-dimensional input
            or ``(batch_size, num_edges)`` for a batched input.
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
    """
    Train an edge-classification model with binary cross-entropy.

    Args:
        model: PyTorch model that maps edge features to edge logits.
        dataloader: DataLoader yielding dictionaries from ``GraphEdgeDataset``.
        num_epochs: Number of training epochs.
        lr: Adam optimizer learning rate.
        device: Device string or ``torch.device`` used for tensors and model.

    Returns:
        None. The model is updated in place.
    """
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
    Evaluate a model on a graph dataset and plot representative predictions.

    Args:
        model: PyTorch edge-classification model to evaluate.
        dataset: Dataset containing held-out graph samples.
        device: Device string or ``torch.device`` used for inference.
        desired_sizes: Matrix sizes for which one representative prediction plot
            should be shown when a sample is available.

    Returns:
        Average elementwise accuracy across all samples in ``dataset``.
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


def split_samples_by_size(samples, desired_sizes):
    """
    Split graph samples so each desired matrix size has a held-out sample.

    Args:
        samples: List of generated graph samples.
        desired_sizes: Matrix sizes that should each contribute one test sample.

    Returns:
        A tuple ``(train_samples, test_samples)`` where each size in
        ``desired_sizes`` has at most one reserved test sample and the remaining
        samples are used for training.
    """
    grouped_samples = defaultdict(list)
    for sample in samples:
        N = sample[0].shape[0]
        grouped_samples[N].append(sample)

    train_samples = []
    test_samples = []

    for size in desired_sizes:
        group = grouped_samples[size]
        if group:
            random.shuffle(group)
            test_samples.append(group[0])
            train_samples.extend(group[1:])
        else:
            print(f"Warning: No samples generated for matrix size: {size}")

    return train_samples, test_samples


if __name__ == "__main__":
    # Generate dataset
    matrix_sizes = [6, 8, 11, 19, 32, 64, 128]
    shape_types = ['triangle', 'square', 'pentagon', 'hexagon']
    num_samples_per_config = 5

    samples = []
    for size in matrix_sizes:
        for _ in range(num_samples_per_config):
            shape_type = random.choices(
                population=shape_types,
                weights=[1, 1, 1, 1],
                k=1)[0]
            sample = generate_sample(size, shape_type, removal_prob=0.3)
            samples.append(sample)

    desired_sizes = [6, 8, 11, 19, 32, 64, 128]
    train_samples, test_samples = split_samples_by_size(samples, desired_sizes)

    X_train_rf, y_train_rf = build_edge_dataset(train_samples)
    X_test_rf, y_test_rf = build_edge_dataset(test_samples)
    print("RF train edge dataset shape:", X_train_rf.shape, "Labels shape:", y_train_rf.shape)
    print("RF test edge dataset shape:", X_test_rf.shape, "Labels shape:", y_test_rf.shape)

    rf_clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train_rf, y_train_rf)
    y_pred_rf = rf_clf.predict(X_test_rf)
    rf_accuracy = accuracy_score(y_test_rf, y_pred_rf)
    print("Random Forest accuracy on graph-level held-out edge prediction: {:.2f}%".format(rf_accuracy * 100))

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

    selected_feature_indices = np.arange(X_train_rf.shape[1])

    # Create PyTorch datasets using the graph-level splits.
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

    torch.save(model.state_dict(), "final_model.pth")
