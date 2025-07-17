import torch
import torch.nn.functional as F
import networkx as nx
import random
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, GCNConv, SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.nn as nn


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_diverse_graph():
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
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=1)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels, heads=1)
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.out(x).squeeze()


class ContMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index=None, batch=None):
        return self.net(x).squeeze()


class ContGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=False)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, cached=False)
        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.out(x).squeeze()


class DeepGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        return self.out(x).squeeze()


def evaluate_batch(model, loader, device):
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
        "ContGraphTransformer": ContGraphTransformer,
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
