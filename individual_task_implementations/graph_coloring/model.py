import os
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx

GRAPH_DIR = "dimacs_graphs"
TRAIN_GRAPH_FILENAME = "DSJC125.1.col"
TRAIN_GRAPH_PATH = os.path.join(GRAPH_DIR, TRAIN_GRAPH_FILENAME)
BEST_MODEL_PATH = "best_model.pth"


def parse_dimacs_col(filename: str) -> nx.Graph:
    """
    Parse a DIMACS .col graph-coloring file into an undirected NetworkX graph.

    Args:
        filename: Path to the DIMACS .col file to parse.

    Returns:
        A NetworkX graph with zero-indexed nodes and undirected edges.
    """
    graph = nx.Graph()
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("p"):
                parts = line.strip().split()
                num_nodes = int(parts[2])
                for node in range(num_nodes):
                    graph.add_node(node)
            elif line.startswith("e"):
                _, u, v = line.strip().split()
                graph.add_edge(int(u) - 1, int(v) - 1)
    return graph


def compute_node_features(graph: nx.Graph) -> nx.Graph:
    """
    Attach structural node features used by the coloring model.

    Args:
        graph: NetworkX graph whose nodes should receive feature attributes.

    Returns:
        The same graph object with degree, clustering, pagerank, core, and eigen
        centrality values stored on each node.
    """
    degree = dict(graph.degree())
    clustering = nx.clustering(graph)
    pagerank = nx.pagerank(graph)
    core = nx.core_number(graph)
    eigen = nx.eigenvector_centrality_numpy(graph)

    for node in graph.nodes():
        graph.nodes[node]["degree"] = degree[node]
        graph.nodes[node]["clustering"] = clustering[node]
        graph.nodes[node]["pagerank"] = pagerank[node]
        graph.nodes[node]["core"] = core[node]
        graph.nodes[node]["eigen"] = eigen[node]

    return graph


def normalize_node_features(graph: nx.Graph) -> torch.Tensor:
    """
    Build and standardize the node-feature matrix for a graph.

    Args:
        graph: NetworkX graph whose nodes contain the expected feature attributes.

    Returns:
        Float tensor of standardized node features with shape
        [num_nodes, num_features].
    """
    features = []
    for _, attrs in graph.nodes(data=True):
        features.append([
            attrs["degree"],
            attrs["clustering"],
            attrs["pagerank"],
            attrs["core"],
            attrs["eigen"],
        ])

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return torch.tensor(scaled_features, dtype=torch.float)


class GCNColoring(nn.Module):
    """
    Two-layer graph convolutional network that scores node color assignments.

    Args:
        in_dim: Number of input node features.
        hidden_dim: Number of hidden channels in the first GCN layer.
        num_colors: Number of color logits to produce for each node.

    Returns:
        A PyTorch module whose forward pass returns node-by-color logits.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_colors: int) -> None:
        """
        Initialize the graph-coloring GCN layers.

        Args:
            in_dim: Number of input node features.
            hidden_dim: Number of hidden channels in the first graph convolution.
            num_colors: Number of color classes to score for each node.

        Returns:
            None. The module layers are initialized in place.
        """
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_colors)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute color logits for every node in a graph.

        Args:
            x: Node-feature tensor with shape [num_nodes, in_dim].
            edge_index: PyTorch Geometric edge index with shape [2, num_edges].

        Returns:
            Tensor of color logits with shape [num_nodes, num_colors].
        """
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x


def potts_loss(logits: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Penalize adjacent nodes that receive similar soft color assignments.

    Args:
        logits: Node-by-color logits from the model.
        edge_index: PyTorch Geometric edge index identifying adjacent node pairs.

    Returns:
        Scalar tensor equal to the mean adjacent-node color similarity.
    """
    probs = F.softmax(logits, dim=1)
    u, v = edge_index
    similarity = (probs[u] * probs[v]).sum(dim=1)
    return similarity.mean()


def build_pyg_data(graph: nx.Graph):
    """
    Convert a featured NetworkX graph into a PyTorch Geometric Data object.

    Args:
        graph: NetworkX graph with computed node feature attributes.

    Returns:
        PyTorch Geometric Data object with edge_index and normalized node features.
    """
    data = from_networkx(graph)
    data.x = normalize_node_features(graph)
    return data


def color_usage_loss(logits: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute an entropy-based color-usage regularization term.

    Args:
        logits: Node-by-color logits from the model.
        eps: Small numeric constant used to avoid taking log(0).

    Returns:
        Scalar tensor measuring entropy of the average color distribution.
    """
    probs = logits.softmax(dim=1)
    avg_usage = probs.mean(dim=0)
    usage_entropy = -torch.sum(avg_usage * (avg_usage + eps).log()) / probs.size(1)
    return usage_entropy


def train_gcn_on_graph(
    graph: nx.Graph,
    num_colors: int = 10,
    epochs: int = 500,
    hidden_dim: int = 64,
    lr: float = 0.01,
    alpha: float = 0.1,
    beta: float = 0.05,
    best_model_path: str = BEST_MODEL_PATH,
) -> tuple[GCNColoring, torch.Tensor, object]:
    """
    Train a GCN coloring model on a single DIMACS graph instance.

    Args:
        graph: NetworkX graph to optimize coloring assignments for.
        num_colors: Number of available colors to score for each node.
        epochs: Number of optimization epochs to run.
        hidden_dim: Number of hidden channels in the GCN model.
        lr: Adam optimizer learning rate.
        alpha: Weight applied to the per-node entropy term.
        beta: Weight applied to the color-usage entropy term.
        best_model_path: Path where the best model checkpoint should be saved.

    Returns:
        Tuple containing the best reloaded model, its recomputed logits, and the
        PyTorch Geometric graph data used for training.
    """
    data = build_pyg_data(graph)
    model = GCNColoring(
        in_dim=data.x.shape[1],
        hidden_dim=hidden_dim,
        num_colors=num_colors,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        probs = out.softmax(dim=1)

        potts = potts_loss(out, data.edge_index)
        eps = 1e-10
        entropy = -torch.sum(probs * (probs + eps).log()) / (probs.size(0) * probs.size(1))
        usage = color_usage_loss(out)
        total_loss = potts + alpha * entropy + beta * usage

        total_loss.backward()
        optimizer.step()

        total_loss_value = total_loss.item()
        if total_loss_value < best_loss:
            best_loss = total_loss_value
            torch.save(model.state_dict(), best_model_path)

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(
                f"[Epoch {epoch}] Potts: {potts.item():.4f} | "
                f"Entropy: {entropy.item():.4f} | Usage: {usage.item():.4f} | "
                f"Total: {total_loss_value:.4f}"
            )

    model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        best_logits = model(data.x, data.edge_index)

    return model, best_logits, data


def evaluate_coloring(logits: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """
    Convert color logits to assignments and report coloring conflicts.

    Args:
        logits: Node-by-color logits produced by a coloring model.
        edge_index: PyTorch Geometric edge index for the evaluated graph.

    Returns:
        Tuple containing node color predictions, number of colors used, and number
        of conflicting edges.
    """
    preds = logits.argmax(dim=1)
    used_colors = preds.unique().numel()

    u, v = edge_index
    conflicts = (preds[u] == preds[v]).sum().item()
    total_edges = edge_index.size(1)
    conflict_rate = conflicts / total_edges if total_edges else 0.0

    print(f"Colors used: {used_colors}")
    print(f"Conflicting edges: {conflicts} / {total_edges} ({conflict_rate:.2%})")

    return preds, used_colors, conflicts


def main() -> tuple[GCNColoring, torch.Tensor, object]:
    """
    Train and evaluate the default same-instance DIMACS coloring model.

    Args:
        None.

    Returns:
        Tuple containing the trained model, its logits on the training graph, and
        the PyTorch Geometric graph data used for evaluation.
    """
    graph = parse_dimacs_col(TRAIN_GRAPH_PATH)
    graph = compute_node_features(graph)
    model, logits, data = train_gcn_on_graph(
        graph,
        epochs=2000,
        alpha=0.2,
        beta=1.7,
    )
    evaluate_coloring(logits, data.edge_index)
    return model, logits, data


if __name__ == "__main__":
    main()
