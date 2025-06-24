# Reuse benchmark setup
import networkx as nx
import numpy as np
import torch
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, f1_score
)
from scipy.stats import pearsonr
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm

from connectivity import gnn_model_adapter, EdgeGNN


class GraphBenchmark:
    def __init__(self, num_graphs=1000, min_nodes=6, max_nodes=140, families=None):
        self.num_graphs = num_graphs
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.families = families if families else ['erdos', 'barabasi', 'watts', 'tree', 'grid']
        self.graphs = []
        self.adj_matrices = []
        self.labels = []

    def default_label_fn(self, G):
        ebc = nx.edge_betweenness_centrality(G, normalized=True)
        n = G.number_of_nodes()
        mat = np.zeros((n, n))
        for (u, v), val in ebc.items():
            mat[u, v] = val
            mat[v, u] = val
        return mat

    def generate(self, label_fn=None):
        self.graphs.clear()
        self.adj_matrices.clear()
        self.labels.clear()
        label_fn = label_fn or self.default_label_fn

        for _ in tqdm(range(self.num_graphs), desc="Generating graphs"):
            size = np.random.randint(self.min_nodes, self.max_nodes + 1)
            family = np.random.choice(self.families)

            if family == 'erdos':
                G = nx.erdos_renyi_graph(size, p=0.1)
            elif family == 'barabasi':
                G = nx.barabasi_albert_graph(size, m=2)
            elif family == 'watts':
                G = nx.watts_strogatz_graph(size, k=4, p=0.2)
            elif family == 'tree':
                G = nx.random_tree(size)
            elif family == 'grid':
                side = int(np.sqrt(size))
                G = nx.grid_2d_graph(side, side)
                G = nx.convert_node_labels_to_integers(G)
            else:
                continue

            if not nx.is_connected(G):
                G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

            G = nx.convert_node_labels_to_integers(G)
            A = nx.to_numpy_array(G)
            L = label_fn(G)

            self.graphs.append(G)
            self.adj_matrices.append(A)
            self.labels.append(L)

def evaluate_benchmark(benchmark, adapter_fn):
    metrics_all = []
    for A, G, L in tqdm(zip(benchmark.adj_matrices, benchmark.graphs, benchmark.labels), total=len(benchmark.graphs), desc="Evaluating"):
        pred = adapter_fn(A)

        mask = (L > 0) | (pred > 0)
        if np.sum(mask) == 0:
            continue

        y_true = L[mask]
        y_pred = pred[mask]

        metric = {
            'MSE': mean_squared_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'Pearson': pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan,
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            try:
                y_bin = (y_true > 0.01).astype(int)
                p_bin = (y_pred > 0.01).astype(int)
                metric['AUC'] = roc_auc_score(y_bin, y_pred)
                metric['F1'] = f1_score(y_bin, p_bin, zero_division=0)
            except:
                metric['AUC'] = np.nan
                metric['F1'] = np.nan

        metrics_all.append(metric)

    if not metrics_all:
        raise ValueError("No graphs with valid predictions.")

    keys = metrics_all[0].keys()
    avg_metrics = {k: np.nanmean([m[k] for m in metrics_all]) for k in keys}
    return avg_metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EdgeGNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
print("Training...")
for epoch in range(10):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        pred = model(batch)
        loss = criterion(pred, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.6f}")

# Testing
print("Testing...")
model.eval()
mse_total = 0
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pred = model(batch)
        loss = criterion(pred, batch.y)
        mse_total += loss.item()
print(f"Final Test MSE: {mse_total / len(test_loader):.6f}")


# 1. Generate the benchmark dataset
benchmark = GraphBenchmark()
benchmark.generate()

# 2. Evaluate GNN on benchmark
results = evaluate_benchmark(benchmark, gnn_model_adapter)

# 3. Print results
print("\nGNN Benchmark Results:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
