import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch.nn import Linear, Sequential, ReLU, Dropout
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, MessagePassing, TransformerConv, GATConv
from torch_geometric.utils import add_self_loops, degree, from_networkx
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Parse edges
G = nx.DiGraph()
with open("higgs-retweet_network.edgelist", "r") as f:
    for line in f:
        u, v, ts = map(int, line.split())
        G.add_edge(u, v, ts=ts)

pos_edges = set(tuple(sorted((u, v))) for u, v in G.edges())


def fast_sample_negative_edges(G, num_samples, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    nodes = list(G.nodes())
    pos_edges = set(tuple(sorted((u, v))) for u, v in G.edges())
    neg_edges = set()
    batch_size = 10 * num_samples
    while len(neg_edges) < num_samples:
        u = np.random.choice(nodes, batch_size)
        v = np.random.choice(nodes, batch_size)
        pairs = [tuple(sorted((int(x), int(y)))) for x, y in zip(u, v) if x != y]
        new_edges = set(pairs) - pos_edges - neg_edges
        needed = num_samples - len(neg_edges)
        neg_edges.update(list(new_edges)[:needed])
    return list(neg_edges)


neg_sample_ratio = 1
num_neg = int(len(pos_edges) * neg_sample_ratio)
neg_edges = fast_sample_negative_edges(G, num_neg)

all_edges = list(pos_edges) + list(neg_edges)
all_nodes = sorted(set(u for edge in all_edges for u in edge))
node2idx = {n: i for i, n in enumerate(all_nodes)}
edge_pairs = torch.tensor([[node2idx[u], node2idx[v]] for u, v in all_edges], dtype=torch.long)

# Normalise timestamps for time_diff
ts_dict = {(u, v): d['ts'] for u, v, d in G.edges(data=True)}

# Time diff for all_edges: 0 if not present (i.e., negative edges)
edge_times = torch.tensor([
    ts_dict.get((u, v), ts_dict.get((v, u), 0))  # try both directions
    for u, v in all_edges
], dtype=torch.float)

# Normalise
time_diff = (edge_times - edge_times.min()) / (edge_times.max() - edge_times.min())

edge_index = torch.tensor([[node2idx[u], node2idx[v]] for u, v in all_edges], dtype=torch.long).t().contiguous()
edge_labels = torch.tensor([1] * len(pos_edges) + [0] * len(neg_edges), dtype=torch.long)


def prepare_edge_splits(edge_pairs, edge_labels, time_diff, data, deg, batch_size=1024, device='cpu'):
    # Step 1: Convert to CPU and ensure tensors are in the right format
    edge_pairs = edge_pairs.cpu()
    edge_labels = edge_labels.cpu()
    time_diff = time_diff.cpu()

    # Step 2: Split into train/val/test (60/20/20)
    X_temp, X_test, y_temp, y_test, td_temp, td_test = train_test_split(
        edge_pairs, edge_labels, time_diff, test_size=0.2, stratify=edge_labels, random_state=42
    )
    X_train, X_val, y_train, y_val, td_train, td_val = train_test_split(
        X_temp, y_temp, td_temp, test_size=0.25, stratify=y_temp, random_state=42
    )

    # Step 3: Package into TensorDatasets
    train_ds = TensorDataset(X_train[:, 0], X_train[:, 1], y_train, td_train)
    val_ds = TensorDataset(X_val[:, 0], X_val[:, 1], y_val, td_val)
    test_ds = TensorDataset(X_test[:, 0], X_test[:, 1], y_test, td_test)

    # Step 4: DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Step 5: Move everything needed to device
    data = data.to(device)
    deg = deg.to(device)

    return train_loader, val_loader, test_loader, data, deg


# Compute structural features
pr = nx.pagerank(G)
deg_in = dict(G.in_degree())
deg_out = dict(G.out_degree())
clustering = nx.clustering(G.to_undirected())

avg_ts, var_ts, min_ts, max_ts = {}, {}, {}, {}
for u in all_nodes:
    times = [d['ts'] for _, _, d in G.in_edges(u, data=True)]
    if times:
        t = np.array(times, dtype=np.float32)
        avg_ts[u], var_ts[u], min_ts[u], max_ts[u] = t.mean(), t.var(), t.min(), t.max()
    else:
        avg_ts[u] = var_ts[u] = min_ts[u] = max_ts[u] = 0.0

feat_array = np.array([
    [deg_in.get(n, 0), deg_out.get(n, 0), pr.get(n, 0), clustering.get(n, 0),
     avg_ts[n], var_ts[n], min_ts[n], max_ts[n]]
    for n in all_nodes
], dtype=np.float32)

feat = torch.tensor(feat_array)

edge_index = torch.tensor([[node2idx[u], node2idx[v]] for u, v in all_edges], dtype=torch.long).t().contiguous()

data = Data(x=feat, edge_index=edge_index, y=edge_labels)
deg = degree(data.edge_index[0], num_nodes=data.num_nodes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, val_loader, test_loader, data, deg = prepare_edge_splits(
    edge_pairs=edge_pairs,
    edge_labels=edge_labels,
    time_diff=time_diff,
    data=data,
    deg=deg,
    batch_size=1024,
    device=device
)


class ImprovedEdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=2, concat=True)
        self.conv2 = GATConv(hidden_channels * 2, hidden_channels)

        self.edge_mlp = Sequential(
            Linear(2 * hidden_channels + hidden_channels + 1 + 1, hidden_channels),
            ReLU(),
            Dropout(p=0.2),
            Linear(hidden_channels, 2)
        )

    def encode(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

    def decode(self, h, edge_pairs, deg, time_diff=None):
        src, dst = edge_pairs
        h_src = h[src]
        h_dst = h[dst]

        embed_concat = torch.cat([h_src, h_dst], dim=1)
        abs_diff = torch.abs(h_src - h_dst)
        deg_diff = (deg[src] - deg[dst]).unsqueeze(1)
        time_diff = time_diff.unsqueeze(1) if time_diff is not None else torch.zeros_like(deg_diff)

        edge_feat = torch.cat([embed_concat, abs_diff, deg_diff, time_diff], dim=1)
        return self.edge_mlp(edge_feat)

    def compute_loss(self, pred, label, weight=None):
        if weight is not None:
            return cross_entropy(pred, label, weight=weight)
        return cross_entropy(pred, label)


# Initialise the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedEdgeClassifier(in_channels=8, hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Compute class weights
class_counts = torch.bincount(edge_labels)
weight = torch.tensor([class_counts[1].item(), class_counts[0].item()], dtype=torch.float, device=device)

# Prepare edge data
edge_pairs = edge_index.t()
labels = edge_labels.to(device)

# Use only the first 328132 labels, assuming they match order-wise
labels = labels[:edge_pairs.shape[0]]
deg = degree(edge_index[0], data.num_nodes).to(device)
time_diff = time_diff.to(device)

# Dataset
dataset = TensorDataset(edge_pairs[:,0], edge_pairs[:,1], labels, time_diff)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1024)

# Training loop
best_f1 = 0.0
best_model_path = 'best_model.pth'
print("Model Training")
for epoch in range(1, 51):
    model.train()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    h = model.encode(x, edge_index).detach()
    total_loss = 0
    for src, dst, y, td in train_loader:
        src, dst, y, td = src.to(device), dst.to(device), y.to(device), td.to(device)
        optimizer.zero_grad()
        pred = model.decode(h, (src, dst), deg, td)
        loss = model.compute_loss(pred, y, weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for src, dst, y, td in val_loader:
            src, dst, y, td = src.to(device), dst.to(device), y.to(device), td.to(device)
            pred = model.decode(h, (src, dst), deg, td)
            all_preds.append(pred.argmax(dim=1).cpu())
            all_labels.append(y.cpu())

    y_true = torch.cat(all_labels)
    y_pred = torch.cat(all_preds)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Epoch {epoch:2d} | Loss: {total_loss/train_size:.4f} | Val Acc: {acc:.4f} | F1: {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), best_model_path)

print("\nSaved the best performing model to best_model.pth.")
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for src, dst, y, td in test_loader:
        src, dst, y, td = src.to(device), dst.to(device), y.to(device), td.to(device)
        out = model.decode(model.encode(data.x, data.edge_index), (src, dst), deg, td)
        pred = out.argmax(dim=1)
        all_preds.append(pred.cpu())
        all_labels.append(y.cpu())

y_true = torch.cat(all_labels)
y_pred = torch.cat(all_preds)
print("\nModel Evaluation:")
print("Test Accuracy:", accuracy_score(y_true, y_pred))
print("Test F1 Score:", f1_score(y_true, y_pred))
