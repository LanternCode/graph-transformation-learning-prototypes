import torch
import torch.nn.functional as F
import networkx as nx
import random
import numpy as np
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, GCNConv, SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def set_seed(seed=42):
    """
    Set pseudo-random seeds for reproducible prototype runs.

    Args:
        seed: Integer seed applied to Python's random module, NumPy, and PyTorch.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_diverse_graph():
    """
    Generate one connected synthetic graph from the task's training families.

    Args:
        None.

    Returns:
        A connected NetworkX graph sampled from an Erdős-Rényi, Barabási-Albert,
        or Watts-Strogatz generator.
    """
    choice = random.choice(["erdos", "barabasi", "watts"])
    n = random.randint(50, 200)
    if choice == "erdos":
        p = random.uniform(0.02, 0.1)
        G = nx.erdos_renyi_graph(n, p)
    elif choice == "barabasi":
        m = random.randint(2, 5)
        G = nx.barabasi_albert_graph(n, m)
    else:
        k = random.randint(2, 6)
        beta = random.uniform(0.1, 0.5)
        G = nx.watts_strogatz_graph(n, k, beta)
    while not nx.is_connected(G):
        G = generate_diverse_graph()
    return G


def generate_core_dataset(num_graphs=1000, seed=42):
    """
    Generate synthetic PyTorch Geometric graphs for node core-number prediction.

    Args:
        num_graphs: Number of synthetic graphs to generate.
        seed: Integer seed used to make graph generation reproducible.

    Returns:
        A list of PyTorch Geometric Data objects. Each object contains node
        features for degree and clustering coefficient, and node targets equal
        to the NetworkX core number.
    """
    set_seed(seed)
    graphs = []
    for _ in range(num_graphs):
        G = generate_diverse_graph()
        core = nx.core_number(G)
        for n in G.nodes:
            G.nodes[n]['core'] = core[n]
            G.nodes[n]['degree'] = G.degree[n]
            G.nodes[n]['clustering'] = nx.clustering(G, n)
        data = from_networkx(G)
        data.y = torch.tensor([core[n] for n in G.nodes], dtype=torch.float)
        data.x = torch.stack([
            data.degree.float(),
            data.clustering.float()
        ], dim=1)
        graphs.append(data)
    return graphs


class ContGraphTransformer(torch.nn.Module):
    """
    TransformerConv-based node regression model for continuous core prediction.

    Args:
        in_channels: Number of input node-feature channels.
        hidden_channels: Width of the hidden graph-transformer representations.

    Returns:
        A torch.nn.Module that maps node features and graph edges to one scalar
        core-number prediction per node.
    """

    def __init__(self, in_channels, hidden_channels):
        """
        Initialise the graph transformer layers and output projection.

        Args:
            in_channels: Number of input node-feature channels.
            hidden_channels: Number of hidden channels used by each TransformerConv layer.

        Returns:
            None.
        """
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=1)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels, heads=1)
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        """
        Run a forward pass over a graph batch.

        Args:
            x: Node-feature tensor with shape [num_nodes, in_channels].
            edge_index: Edge-index tensor with shape [2, num_edges].
            batch: Batch vector assigning each node to a graph; accepted for a
                shared model interface.

        Returns:
            Tensor of scalar node predictions with shape [num_nodes].
        """
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.out(x).squeeze()


class ContMLP(nn.Module):
    """
    Pointwise MLP baseline for core-number prediction from node features only.

    Args:
        in_channels: Number of input node-feature channels.
        hidden_channels: Width of the hidden fully connected layers.

    Returns:
        A torch.nn.Module that predicts one scalar core value per node without
        message passing.
    """

    def __init__(self, in_channels, hidden_channels):
        """
        Initialise the fully connected node-level regression network.

        Args:
            in_channels: Number of input node-feature channels.
            hidden_channels: Width of the hidden fully connected layers.

        Returns:
            None.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index=None, batch=None):
        """
        Predict node core values from node features.

        Args:
            x: Node-feature tensor with shape [num_nodes, in_channels].
            edge_index: Optional edge-index tensor accepted for interface compatibility.
            batch: Optional batch vector accepted for interface compatibility.

        Returns:
            Tensor of scalar node predictions with shape [num_nodes].
        """
        return self.net(x).squeeze()


class ContGCN(nn.Module):
    """
    Two-layer GCN baseline for node-level core-number regression.

    Args:
        in_channels: Number of input node-feature channels.
        hidden_channels: Width of the hidden graph-convolution representations.

    Returns:
        A torch.nn.Module that maps graph-structured node features to one scalar
        core-number prediction per node.
    """

    def __init__(self, in_channels, hidden_channels):
        """
        Initialise the GCN layers and final regression projection.

        Args:
            in_channels: Number of input node-feature channels.
            hidden_channels: Number of hidden channels in the GCN layers.

        Returns:
            None.
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=False)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, cached=False)
        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        """
        Run a forward pass through the GCN model.

        Args:
            x: Node-feature tensor with shape [num_nodes, in_channels].
            edge_index: Edge-index tensor with shape [2, num_edges].
            batch: Batch vector assigning each node to a graph; accepted for a
                shared model interface.

        Returns:
            Tensor of scalar node predictions with shape [num_nodes].
        """
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.out(x).squeeze()


class DeepGraphSAGE(nn.Module):
    """
    Three-layer GraphSAGE baseline for node-level core-number regression.

    Args:
        in_channels: Number of input node-feature channels.
        hidden_channels: Width of the hidden GraphSAGE representations.

    Returns:
        A torch.nn.Module that predicts one scalar core value for every node in
        a graph batch.
    """

    def __init__(self, in_channels, hidden_channels):
        """
        Initialise the GraphSAGE layers and final regression projection.

        Args:
            in_channels: Number of input node-feature channels.
            hidden_channels: Number of hidden channels in each GraphSAGE layer.

        Returns:
            None.
        """
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        """
        Run a forward pass through the GraphSAGE model.

        Args:
            x: Node-feature tensor with shape [num_nodes, in_channels].
            edge_index: Edge-index tensor with shape [2, num_edges].
            batch: Batch vector assigning each node to a graph; accepted for a
                shared model interface.

        Returns:
            Tensor of scalar node predictions with shape [num_nodes].
        """
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        return self.out(x).squeeze()


def evaluate_batch(model, loader, device):
    """
    Evaluate a node-regression model over a DataLoader of graph batches.

    Args:
        model: PyTorch model that returns one scalar prediction per node.
        loader: PyTorch Geometric DataLoader containing graph batches with x,
            edge_index, batch, and y attributes.
        device: Torch device used for model evaluation.

    Returns:
        A tuple ``(mse, acc)`` containing mean squared error and rounded integer
        accuracy over all nodes in the loader.
    """
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            ys.append(batch.y.cpu())
            preds.append(out.cpu())
    y = torch.cat(ys)
    pred = torch.cat(preds)
    mse = F.mse_loss(pred, y).item()
    acc = accuracy_score(y.long(), pred.round().long())
    return mse, acc


def train_model_batch(model, train_loader, val_loader, device, save_name="model"):
    """
    Train a node-regression model and save the best validation checkpoint.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader containing training graph batches.
        val_loader: DataLoader containing validation graph batches.
        device: Torch device used for training and validation.
        save_name: Prefix used when writing the best checkpoint file.

    Returns:
        The input model loaded with the best validation-loss checkpoint.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_loss = float('inf')
    best_path = f"{save_name}_batch_best.pt"

    for epoch in range(1, 401):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.mse_loss(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        val_loss, val_acc = evaluate_batch(model, val_loader, device)
        if epoch % 25 == 0:
            print(f"[{save_name} Epoch {epoch:03d}] Train MSE: {avg_loss:.4f} | Val MSE: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path))
    return model


if __name__ == "__main__":
    # === Unified Model Registry ===
    model_registry = {
        "ContMLP": ContMLP,
        "ContGCN": ContGCN,
        "DeepGraphSAGE": DeepGraphSAGE,
        "ContGraphTransformer": ContGraphTransformer
    }

    # === Full Multi-Model Training Loop ===
    set_seed()
    graphs = generate_core_dataset(num_graphs=600)
    train_graphs, val_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    for name, model_class in model_registry.items():
        print(f"\n=== Training {name} ===")
        model = model_class(in_channels=2, hidden_channels=32)
        trained_model = train_model_batch(model, train_loader, val_loader, device, save_name=name)
        test_mse, test_acc = evaluate_batch(trained_model, val_loader, device)
        print(f"[Final Eval - {name}] Test MSE: {test_mse:.4f} | Accuracy: {test_acc:.4f}")
        results[name] = (test_mse, test_acc)

    # === Final Summary ===
    print("\n=== Final Training Results ===")
    for name, (mse, acc) in results.items():
        print(f"{name:>24}: Accuracy = {acc:.4f} | Test MSE = {mse:.4f}")
