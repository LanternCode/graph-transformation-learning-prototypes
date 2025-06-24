# Edge Betweenness Learning Models (Optimized)

# Uncomment to install dependencies in Colab:
# !pip install networkx scikit-learn scipy xgboost tqdm

import os
import random
import numpy as np
import networkx as nx
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from tqdm import tqdm

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def generate_graph(num_nodes, p=None):
    """
    Generate a connected random graph by first creating a random spanning tree, then
    adding extra edges with probability p. Guarantees connectivity without rejection loop.
    """
    if p is None:
        # Use connectivity threshold approx log(n)/n to encourage extra edges
        p = (np.log(num_nodes) + 0.1) / num_nodes
    # Start with a random tree to ensure connectivity
    G = nx.random_unlabeled_tree(num_nodes)
    # For each possible non-tree edge, add with probability p
    nodes = list(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if not G.has_edge(i, j) and random.random() < p:
                G.add_edge(i, j)
    return G

# All other functions (compute_edge_features_and_target, build_datasets, model definitions, training loops) remain the same as before.
# We only update the graph-generation to remove the expensive connectivity loop
# and add a progress bar when creating multiple graphs below.

# Generate synthetic graphs with progress bar
sizes = [30, 70, 140]
num_per_size = 50
total_graphs = len(sizes) * num_per_size
print(f"Generating {total_graphs} graphs (sizes: {sizes})...")
graphs = []
for size in tqdm(sizes * num_per_size, desc="Graphs"):  # tricks list multiplication
    G = generate_graph(size)
    graphs.append(G)

graphs = graphs[:total_graphs]
random.shuffle(graphs)
train_graphs = graphs[:100]
val_graphs = graphs[100:125]
test_graphs = graphs[125:]

# Build datasets
X_train, y_train, train_data = build_datasets(train_graphs)
X_val, y_val, val_data = build_datasets(val_graphs)
X_test, y_test, test_data = build_datasets(test_graphs)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Container for results
results = {}

### Model 1: Linear Regression (PyTorch) ###
class LinearModel(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.lin = nn.Linear(in_feats, 1)
    def forward(self, x):
        return self.lin(x)

model = LinearModel(in_feats=6).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
best_val_loss = float('inf')
epochs = 50

for epoch in range(1, epochs+1):
    model.train()
    optimizer.zero_grad()
    X_batch = torch.tensor(X_train).to(device)
    y_batch = torch.tensor(y_train).to(device)
    preds = model(X_batch).squeeze()
    loss = criterion(preds, y_batch)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        val_preds = model(torch.tensor(X_val).to(device)).squeeze()
        val_loss = criterion(val_preds, torch.tensor(y_val).to(device))
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'linear_best.pt')
    print(f'[Linear] Epoch {epoch}/{epochs}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')

# Test
model.load_state_dict(torch.load('linear_best.pt'))
model.eval()
with torch.no_grad():
    test_preds = model(torch.tensor(X_test).to(device)).squeeze().cpu().numpy()
mse = mean_squared_error(y_test, test_preds)
pearson = pearsonr(y_test, test_preds)[0]
spearman = spearmanr(y_test, test_preds)[0]
print(f'[Linear] Test MSE: {mse:.6f}, Pearson: {pearson:.6f}, Spearman: {spearman:.6f}')
results['Linear'] = (mse, pearson, spearman)

### Model 2: MLP (PyTorch) ###
class MLP(nn.Module):
    def __init__(self, in_feats, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden,1)
        )
    def forward(self,x):
        return self.net(x)

model = MLP(6,64).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
best_val_loss = float('inf')
epochs = 100

for epoch in range(1, epochs+1):
    model.train()
    optimizer.zero_grad()
    preds = model(torch.tensor(X_train).to(device)).squeeze()
    loss = criterion(preds, torch.tensor(y_train).to(device))
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        val_preds = model(torch.tensor(X_val).to(device)).squeeze()
        val_loss = criterion(val_preds, torch.tensor(y_val).to(device))
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'mlp_best.pt')
    print(f'[MLP] Epoch {epoch}/{epochs}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')

# Test
model.load_state_dict(torch.load('mlp_best.pt'))
model.eval()
with torch.no_grad():
    test_preds = model(torch.tensor(X_test).to(device)).squeeze().cpu().numpy()
mse = mean_squared_error(y_test, test_preds)
pearson = pearsonr(y_test, test_preds)[0]
spearman = spearmanr(y_test, test_preds)[0]
print(f'[MLP] Test MSE: {mse:.6f}, Pearson: {pearson:.6f}, Spearman: {spearman:.6f}')
results['MLP'] = (mse, pearson, spearman)

### Model 3: XGBoost Regressor ###
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train,y_train),(X_val,y_val)],
    eval_metric='rmse', verbose=True
)
xgb_model.save_model('xgb_model.json')
test_preds = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, test_preds)
pearson = pearsonr(y_test, test_preds)[0]
spearman = spearmanr(y_test, test_preds)[0]
print(f'[XGBoost] Test MSE: {mse:.6f}, Pearson: {pearson:.6f}, Spearman: {spearman:.6f}')
results['XGBoost'] = (mse, pearson, spearman)

### Model 4: GCN ###
class GCN(nn.Module):
    def __init__(self, in_feats, hidden):
        super().__init__()
        self.gc1 = nn.Linear(in_feats, hidden)
        self.gc2 = nn.Linear(hidden, hidden)
        self.edge_mlp = nn.Sequential(nn.Linear(2*hidden, hidden), nn.ReLU(), nn.Linear(hidden,1))
    def forward(self, node_feats, adj_norm, edge_index):
        h = torch.relu(self.gc1(adj_norm @ node_feats))
        h = torch.relu(self.gc2(adj_norm @ h))
        src, dst = edge_index
        edge_h = torch.cat([h[src], h[dst]], dim=1)
        return self.edge_mlp(edge_h).squeeze()

model = GCN(in_feats=2, hidden=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
best_val_loss = float('inf')
epochs = 50

for epoch in range(1, epochs+1):
    train_loss = 0
    model.train()
    for g in train_data:
        nf = g['node_feats'].to(device)
        adj_norm = g['adj_norm'].to(device)
        ei = g['edge_index'].to(device)
        targets = g['edge_targets'].to(device)
        preds = model(nf, adj_norm, ei)
        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_data)
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for g in val_data:
            nf = g['node_feats'].to(device)
            adj_norm = g['adj_norm'].to(device)
            ei = g['edge_index'].to(device)
            targets = g['edge_targets'].to(device)
            preds = model(nf, adj_norm, ei)
            val_loss += criterion(preds, targets).item()
    val_loss /= len(val_data)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'gcn_best.pt')
    print(f'[GCN] Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

# Test
model.load_state_dict(torch.load('gcn_best.pt'))
model.eval()
all_preds=[]; all_targets=[]
with torch.no_grad():
    for g in test_data:
        nf = g['node_feats'].to(device)
        adj_norm = g['adj_norm'].to(device)
        ei = g['edge_index'].to(device)
        targets = g['edge_targets'].cpu().numpy()
        preds = model(nf, adj_norm, ei).cpu().numpy()
        all_preds.append(preds); all_targets.append(targets)
all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)
mse = mean_squared_error(all_targets, all_preds)
pearson = pearsonr(all_targets, all_preds)[0]
spearman = spearmanr(all_targets, all_preds)[0]
print(f'[GCN] Test MSE: {mse:.6f}, Pearson: {pearson:.6f}, Spearman: {spearman:.6f}')
results['GCN'] = (mse, pearson, spearman)

### Model 5: GraphSAGE ###
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden):
        super().__init__()
        self.lin_self1 = nn.Linear(in_feats, hidden)
        self.lin_neigh1 = nn.Linear(in_feats, hidden)
        self.lin_self2 = nn.Linear(hidden, hidden)
        self.lin_neigh2 = nn.Linear(hidden, hidden)
        self.edge_mlp = nn.Sequential(nn.Linear(2*hidden, hidden), nn.ReLU(), nn.Linear(hidden,1))

    def forward(self, node_feats, adj, degree, edge_index):
        h_self = self.lin_self1(node_feats)
        neigh_mean = (adj @ node_feats) / degree.unsqueeze(1)
        h_neigh = self.lin_neigh1(neigh_mean)
        h = torch.relu(h_self + h_neigh)
        h_self2 = self.lin_self2(h)
        neigh_mean2 = (adj @ h) / degree.unsqueeze(1)
        h_neigh2 = self.lin_neigh2(neigh_mean2)
        h = torch.relu(h_self2 + h_neigh2)
        src, dst = edge_index
        edge_h = torch.cat([h[src], h[dst]], dim=1)
        return self.edge_mlp(edge_h).squeeze()

model = GraphSAGE(in_feats=2, hidden=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
best_val_loss = float('inf')
epochs = 50

for epoch in range(1, epochs+1):
    train_loss=0
    model.train()
    for g in train_data:
        nf = g['node_feats'].to(device)
        adj = g['adj'].to(device)
        deg = g['degree'].to(device)
        ei = g['edge_index'].to(device)
        targets = g['edge_targets'].to(device)
        preds = model(nf, adj, deg, ei)
        loss = criterion(preds, targets)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        train_loss+=loss.item()
    train_loss/=len(train_data)
    val_loss=0
    model.eval()
    with torch.no_grad():
        for g in val_data:
            nf=g['node_feats'].to(device)
            adj=g['adj'].to(device)
            deg=g['degree'].to(device)
            ei=g['edge_index'].to(device)
            targets=g['edge_targets'].to(device)
            val_loss+=criterion(model(nf,adj,deg,ei),targets).item()
    val_loss/=len(val_data)
    if val_loss<best_val_loss:
        best_val_loss=val_loss
        torch.save(model.state_dict(),'sage_best.pt')
    print(f"[SAGE] Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
