import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import random
from pathlib import Path
from typing import List, Sequence, Tuple
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.nn import Dropout, Linear, ReLU, Sequential
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import degree

Edge = Tuple[int, int]


def set_seed(seed: int = 42) -> None:
    """
    Set pseudo-random seeds used by Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed used for deterministic data splitting and sampling.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_retweet_graph(edge_path: str | Path) -> nx.DiGraph:
    """
    Load the Higgs retweet edge list as a directed simple graph.

    Args:
        edge_path: Path to ``higgs-retweet_network.edgelist``. Each line is expected
            to contain ``source target timestamp``. The timestamp is parsed only to
            validate the file format and is intentionally not used as a predictive
            feature.

    Returns:
        Directed NetworkX graph whose edges indicate observed retweets.
    """
    graph = nx.DiGraph()
    with open(edge_path, "r", encoding="utf-8") as handle:
        for line in handle:
            source, target, _timestamp = map(int, line.split())
            if source != target:
                graph.add_edge(source, target)
    return graph


def sample_directed_negative_edges(nodes: Sequence[int],
                                   positive_edges: set[Edge],
                                   num_samples: int,
                                   seed: int = 42) -> List[Edge]:
    """
    Sample directed node pairs that are not observed retweet edges.

    Args:
        nodes: Sequence of original node identifiers to sample from.
        positive_edges: Set of directed positive edges in original node-id space.
        num_samples: Number of negative directed pairs to draw.
        seed: Random seed controlling negative-edge sampling.

    Returns:
        List of directed negative edges in original node-id space.
    """
    rng = np.random.default_rng(seed)
    node_array = np.asarray(nodes)
    negative_edges: set[Edge] = set()
    batch_size = max(10_000, min(10 * num_samples, 1_000_000))

    while len(negative_edges) < num_samples:
        sources = rng.choice(node_array, batch_size)
        targets = rng.choice(node_array, batch_size)
        for source, target in zip(sources, targets):
            edge = (int(source), int(target))
            if source == target or edge in positive_edges or edge in negative_edges:
                continue
            negative_edges.add(edge)
            if len(negative_edges) == num_samples:
                break

    return list(negative_edges)


def build_edge_examples(graph: nx.DiGraph,
                        negative_ratio: float = 1.0,
                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray, dict[int, int], List[int], List[Edge]]:
    """
    Build positive and negative directed edge-classification examples.

    Args:
        graph: Directed retweet graph containing observed positive edges.
        negative_ratio: Number of negative examples to sample per positive edge.
        seed: Random seed used for negative-edge sampling.

    Returns:
        Tuple containing edge pairs as remapped integer indices, binary labels,
        original-node-to-index mapping, ordered original node identifiers, and the
        original directed positive edges.
    """
    original_nodes = sorted(graph.nodes())
    node_to_index = {node: idx for idx, node in enumerate(original_nodes)}
    positive_edges = list(graph.edges())
    positive_edge_set = set(positive_edges)
    num_negative = int(len(positive_edges) * negative_ratio)
    negative_edges = sample_directed_negative_edges(
        original_nodes,
        positive_edge_set,
        num_negative,
        seed=seed,
    )

    all_edges = positive_edges + negative_edges
    labels = np.concatenate([
        np.ones(len(positive_edges), dtype=np.int64),
        np.zeros(len(negative_edges), dtype=np.int64),
    ])
    edge_pairs = np.asarray(
        [[node_to_index[source], node_to_index[target]] for source, target in all_edges],
        dtype=np.int64,
    )
    return edge_pairs, labels, node_to_index, original_nodes, positive_edges


def split_edge_examples(edge_pairs: np.ndarray,
                        labels: np.ndarray,
                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split edge examples once into train, validation, and test partitions.

    Args:
        edge_pairs: Array of remapped directed edge candidates with shape ``[E, 2]``.
        labels: Binary labels for edge candidates with shape ``[E]``.
        seed: Random seed used by stratified splitting.

    Returns:
        Train, validation, and test edge arrays and their corresponding labels in
        the order ``X_train, X_val, X_test, y_train, y_val, y_test``.
    """
    x_temp, x_test, y_temp, y_test = train_test_split(
        edge_pairs,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=seed,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=0.25,
        stratify=y_temp,
        random_state=seed,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def make_edge_loader(edge_pairs: np.ndarray,
                     labels: np.ndarray,
                     batch_size: int,
                     shuffle: bool) -> DataLoader:
    """
    Create a PyTorch loader for edge-classification examples.

    Args:
        edge_pairs: Array of directed edge candidates with shape ``[E, 2]``.
        labels: Binary labels with shape ``[E]``.
        batch_size: Number of edge examples per mini-batch.
        shuffle: Whether to shuffle examples at each epoch.

    Returns:
        DataLoader yielding ``source``, ``target``, and ``label`` tensors.
    """
    tensor_edges = torch.as_tensor(edge_pairs, dtype=torch.long)
    tensor_labels = torch.as_tensor(labels, dtype=torch.long)
    dataset = TensorDataset(tensor_edges[:, 0], tensor_edges[:, 1], tensor_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_train_message_graph(num_nodes: int, train_edges: np.ndarray) -> torch.Tensor:
    """
    Build the message-passing graph from training positive edges only.

    Args:
        num_nodes: Number of remapped nodes in the full graph.
        train_edges: Training edge examples with labels already filtered to positives.

    Returns:
        Edge-index tensor with shape ``[2, E_train_pos]`` for GNN message passing.
    """
    if len(train_edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.as_tensor(train_edges, dtype=torch.long).t().contiguous()
    return edge_index.clamp(min=0, max=num_nodes - 1)


def compute_train_graph_features(num_nodes: int, train_edge_index: torch.Tensor) -> torch.Tensor:
    """
    Compute node features from the training graph only.

    Args:
        num_nodes: Number of remapped nodes in the full graph.
        train_edge_index: Directed training-positive edge index used for message passing.

    Returns:
        Float tensor of node features ``[in_degree, out_degree, pagerank, clustering]``
        with shape ``[num_nodes, 4]``.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))
    if train_edge_index.numel() > 0:
        graph.add_edges_from((int(u), int(v)) for u, v in train_edge_index.t().tolist())

    pagerank = nx.pagerank(graph) if num_nodes > 0 else {}
    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    clustering = nx.clustering(graph.to_undirected())

    features = np.asarray([
        [
            in_degree.get(node, 0),
            out_degree.get(node, 0),
            pagerank.get(node, 0.0),
            clustering.get(node, 0.0),
        ]
        for node in range(num_nodes)
    ], dtype=np.float32)
    return torch.as_tensor(features, dtype=torch.float32)


def build_training_data(num_nodes: int,
                        x_train: np.ndarray,
                        y_train: np.ndarray,
                        device: torch.device) -> Tuple[Data, torch.Tensor]:
    """
    Construct graph data and degree features from the training split only.

    Args:
        num_nodes: Number of remapped nodes in the full graph.
        x_train: Training edge candidates with shape ``[E_train, 2]``.
        y_train: Binary labels for the training edge candidates.
        device: Device where graph tensors should be stored.

    Returns:
        Tuple of PyTorch Geometric ``Data`` object and source-degree tensor, both on
        the requested device.
    """
    train_positive_edges = x_train[y_train == 1]
    train_edge_index = build_train_message_graph(num_nodes, train_positive_edges)
    features = compute_train_graph_features(num_nodes, train_edge_index)
    data = Data(x=features, edge_index=train_edge_index).to(device)
    source_degree = degree(data.edge_index[0], num_nodes=data.num_nodes).to(device)
    return data, source_degree


class ImprovedEdgeClassifier(torch.nn.Module):
    """
    GAT-based directed edge-existence classifier.

    The model encodes nodes with message passing over the training-positive graph
    and decodes candidate directed edges from source/target embeddings, absolute
    embedding differences, and train-graph degree differences.
    """

    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        """
        Initialise the encoder and edge decoder.

        Args:
            in_channels: Number of input node-feature channels.
            hidden_channels: Hidden dimensionality used by GAT layers and the edge MLP.

        Returns:
            None.
        """
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=2, concat=True)
        self.conv2 = GATConv(hidden_channels * 2, hidden_channels)
        self.edge_mlp = Sequential(
            Linear(3 * hidden_channels + 1, hidden_channels),
            ReLU(),
            Dropout(p=0.2),
            Linear(hidden_channels, 2),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Encode node features with directed graph attention layers.

        Args:
            x: Node-feature matrix with shape ``[num_nodes, in_channels]``.
            edge_index: Directed message-passing edges with shape ``[2, num_edges]``.

        Returns:
            Node embedding tensor with shape ``[num_nodes, hidden_channels]``.
        """
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

    def decode(self, h: torch.Tensor, edge_pairs: Tuple[torch.Tensor, torch.Tensor], deg: torch.Tensor) -> torch.Tensor:
        """
        Predict directed edge-existence logits from node embeddings.

        Args:
            h: Node embedding tensor produced by ``encode``.
            edge_pairs: Tuple ``(src, dst)`` containing source and target node indices.
            deg: Source-degree vector computed from the training message graph.

        Returns:
            Logits with shape ``[num_edges, 2]`` for absent/present edge classes.
        """
        src, dst = edge_pairs
        h_src = h[src]
        h_dst = h[dst]
        embed_concat = torch.cat([h_src, h_dst], dim=1)
        abs_diff = torch.abs(h_src - h_dst)
        deg_diff = (deg[src] - deg[dst]).unsqueeze(1)
        edge_features = torch.cat([embed_concat, abs_diff, deg_diff], dim=1)
        return self.edge_mlp(edge_features)

    def compute_loss(self, pred: torch.Tensor, label: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute weighted or unweighted cross-entropy edge-classification loss.

        Args:
            pred: Model logits with shape ``[num_edges, 2]``.
            label: Integer edge labels with shape ``[num_edges]``.
            weight: Optional class-weight tensor with shape ``[2]``.

        Returns:
            Scalar cross-entropy loss tensor.
        """
        if weight is not None:
            return cross_entropy(pred, label, weight=weight)
        return cross_entropy(pred, label)


def evaluate_edge_classifier(model: ImprovedEdgeClassifier,
                             data: Data,
                             deg: torch.Tensor,
                             loader: DataLoader,
                             device: torch.device) -> Tuple[float, float]:
    """
    Evaluate the edge classifier on a loader of directed edge examples.

    Args:
        model: Trained ``ImprovedEdgeClassifier`` instance.
        data: Graph data containing train-only node features and message edges.
        deg: Degree vector computed from the train-only message graph.
        loader: DataLoader yielding edge examples and labels.
        device: Device used for model and tensors.

    Returns:
        Tuple ``(accuracy, f1)`` computed over all examples in ``loader``.
    """
    model.eval()
    all_predictions: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    with torch.no_grad():
        embeddings = model.encode(data.x, data.edge_index)
        for src, dst, labels in loader:
            src = src.to(device)
            dst = dst.to(device)
            labels = labels.to(device)
            logits = model.decode(embeddings, (src, dst), deg)
            all_predictions.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_predictions).numpy()
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred)


def train_model(model: ImprovedEdgeClassifier,
                data: Data,
                deg: torch.Tensor,
                train_loader: DataLoader,
                val_loader: DataLoader,
                class_weight: torch.Tensor,
                device: torch.device,
                epochs: int = 50,
                learning_rate: float = 0.01,
                weight_decay: float = 5e-4,
                best_model_path: str = "best_model.pth") -> ImprovedEdgeClassifier:
    """
    Train the classifier and save the best validation-F1 checkpoint.

    Args:
        model: Edge classifier to optimise.
        data: Train-only graph data used by the GAT encoder.
        deg: Degree vector computed from the train-only graph.
        train_loader: DataLoader for training edge examples.
        val_loader: DataLoader for validation edge examples.
        class_weight: Class weights computed from training labels only.
        device: Device used for model and tensors.
        epochs: Number of optimisation epochs.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight-decay coefficient.
        best_model_path: Path where the best validation checkpoint is saved.

    Returns:
        The model loaded with the best validation-F1 checkpoint.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_f1 = 0.0

    print("Model Training")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for src, dst, labels in train_loader:
            src = src.to(device)
            dst = dst.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            embeddings = model.encode(data.x, data.edge_index)
            logits = model.decode(embeddings, (src, dst), deg)
            loss = model.compute_loss(logits, labels, class_weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)

        val_acc, val_f1 = evaluate_edge_classifier(model, data, deg, val_loader, device)
        print(f"Epoch {epoch:2d} | Loss: {total_loss / total_examples:.4f} | Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"\nSaved the best performing model to {best_model_path}.")
    return model


def compute_class_weight(train_labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from training labels only.

    Args:
        train_labels: Binary training labels with shape ``[E_train]``.
        device: Device where the returned tensor should be stored.

    Returns:
        Tensor of length two containing weights for absent and present edge classes.
    """
    counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    total = counts.sum()
    weights = total / (2.0 * np.maximum(counts, 1.0))
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def main() -> None:
    """
    Run directed edge-existence training and evaluation on the Higgs retweet graph.

    Args:
        None.

    Returns:
        None.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_path = Path("higgs-retweet_network.edgelist")
    if not edge_path.exists():
        raise FileNotFoundError(
            "higgs-retweet_network.edgelist was not found. Run download_dataset.py first."
        )

    graph = load_retweet_graph(edge_path)
    edge_pairs, labels, _node_to_index, original_nodes, _positive_edges = build_edge_examples(
        graph,
        negative_ratio=1.0,
        seed=42,
    )
    x_train, x_val, x_test, y_train, y_val, y_test = split_edge_examples(edge_pairs, labels, seed=42)

    train_loader = make_edge_loader(x_train, y_train, batch_size=1024, shuffle=True)
    val_loader = make_edge_loader(x_val, y_val, batch_size=1024, shuffle=False)
    test_loader = make_edge_loader(x_test, y_test, batch_size=1024, shuffle=False)

    data, deg = build_training_data(len(original_nodes), x_train, y_train, device)
    model = ImprovedEdgeClassifier(in_channels=data.num_node_features, hidden_channels=64).to(device)
    class_weight = compute_class_weight(y_train, device)

    model = train_model(
        model=model,
        data=data,
        deg=deg,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weight=class_weight,
        device=device,
        epochs=50,
        learning_rate=0.01,
        weight_decay=5e-4,
        best_model_path="best_model.pth",
    )

    test_acc, test_f1 = evaluate_edge_classifier(model, data, deg, test_loader, device)
    print("\nModel Evaluation:")
    print("Test Accuracy:", test_acc)
    print("Test F1 Score:", test_f1)


if __name__ == "__main__":
    main()
