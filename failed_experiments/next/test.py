import os
import torch
import networkx as nx
from torch_geometric.data import Data, DataLoader
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, MessagePassing, TransformerConv
from torch_geometric.utils import add_self_loops, degree
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import urllib.request
import gzip
import shutil

# Step 1: Download the file
url = "https://snap.stanford.edu/data/higgs-retweet_network.edgelist.gz"
gz_path = "higgs-retweet_network.edgelist.gz"
txt_path = "higgs-retweet_network.edgelist"

print("Downloading...")
urllib.request.urlretrieve(url, gz_path)
print("Download complete.")

# Step 2: Unzip the .gz file
print("Unzipping...")
with gzip.open(gz_path, 'rb') as f_in:
    with open(txt_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
print("Unzip complete.")

# Optional: Remove the .gz file
os.remove(gz_path)

# Parse edges
G = nx.DiGraph()
with open("higgs-retweet_network.edgelist", "r") as f:
    for line in f:
        u, v, ts = map(int, line.split())
        G.add_edge(u, v, ts=ts)

# Build undirected edge set
undirected_edges = set(tuple(sorted((u, v))) for u, v in G.edges())
edge_index = torch.tensor([[u, v] for u, v in undirected_edges] + [[v, u] for u, v in undirected_edges]).t()

# Edge labels: 1 if u->v in original, else 0
edge_labels = []
for u, v in undirected_edges:
    edge_labels.append(int(G.has_edge(u, v)))
    edge_labels.append(int(G.has_edge(v, u)))
edge_labels = torch.tensor(edge_labels)

# Node degrees and PageRank
pr = nx.pagerank(G)
deg_in = dict(G.in_degree())
deg_out = dict(G.out_degree())
nodes = list(G.nodes())
feat = torch.tensor([[deg_in[n], deg_out[n], pr[n]] for n in nodes], dtype=torch.float)

# Map node to index
node2idx = {n: i for i, n in enumerate(nodes)}

# Prepare graph data object
edge_index = edge_index.apply_(lambda idx: node2idx[int(idx)])
data = Data(x=feat, edge_index=edge_index, y=edge_labels)
deg = degree(data.edge_index[0], num_nodes=data.num_nodes)

# Extract source and destination node indices
src = data.edge_index[0]
dst = data.edge_index[1]

# Node features for each edge
x_src = data.x[src]
x_dst = data.x[dst]

# Edge-level features
abs_diff = torch.abs(x_src - x_dst)
concat = torch.cat([x_src, x_dst, abs_diff], dim=1)

features = concat
labels = data.y

from torch.utils.data import Dataset, DataLoader, random_split

class EdgeFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.float()
        self.labels = labels.long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create dataset and split
dataset = EdgeFeatureDataset(features, labels)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256)

import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.net(x)

from sklearn.metrics import accuracy_score, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLPClassifier(features.size(1)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, 21):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            preds.extend(pred.cpu())
            targets.extend(yb.cpu())

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds)
    print(f"Epoch {epoch}, Accuracy: {acc:.4f}, F1: {f1:.4f}")
