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


def compute_edge_features_and_target(G):
    """
    For a given NetworkX graph G, compute per-edge feature vectors and target betweenness.
    Returns:
      X: np.array shape (num_edges, 6), features [deg_u, deg_v, clust_u, clust_v, sum_deg, abs_diff_deg]
      y: np.array shape (num_edges,), true edge-betweenness centrality
      edge_list: list of tuples (u, v) in the same order
    """
    # Compute true edge-betweenness
    eb = nx.edge_betweenness_centrality(G)
    # Node-level properties
    degree = dict(G.degree())
    clustering = nx.clustering(G)
    features = []
    targets = []
    edge_list = []
    for u, v in G.edges():
        du, dv = degree[u], degree[v]
        cu, cv = clustering[u], clustering[v]
        features.append([du, dv, cu, cv, du + dv, abs(du - dv)])
        targets.append(eb[(u, v)])
        edge_list.append((u, v))
    X = np.array(features, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    return X, y, edge_list


def build_datasets(graphs):
    X_list, y_list = [], []
    graph_data = []

    for G in graphs:
        N = G.number_of_nodes()
        # 1) Edge-level features & targets
        X, y, edge_list = compute_edge_features_and_target(G)
        X_list.append(X)
        y_list.append(y)

        # 2) Node features: degree & clustering
        degree = np.array([G.degree(i) for i in range(N)], dtype=np.float32)
        clustering = np.array([nx.clustering(G, i) for i in range(N)], dtype=np.float32)
        node_feats = np.stack([degree, clustering], axis=1)

        # 3) Adjacency matrix & normalized version
        A = nx.to_numpy_array(G, dtype=np.float32)
        I = np.eye(N, dtype=np.float32)
        A_tilde = A + I
        d = np.sum(A_tilde, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
        A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt

        # 4) Pack into torch tensors for GNNs
        graph_data.append({
            'node_feats': torch.tensor(node_feats),
            'edge_index': torch.tensor(np.array(edge_list).T, dtype=torch.long),
            'edge_feats':  torch.tensor(X),
            'edge_targets':torch.tensor(y),
            'adj_norm':    torch.tensor(A_norm),
            'adj':         torch.tensor(A),
            'degree':      torch.tensor(d)
        })

    # Concatenate all edge features/targets for flat-model training
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    return X_all, y_all, graph_data


### Model 1: Linear Regression (PyTorch) ###
class LinearModel(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.lin = nn.Linear(in_feats, 1)

    def forward(self, x):
        return self.lin(x)


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


### Model 3: XGBoost Regressor ###
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    eval_metric='rmse'
)


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


def main():
    # Reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    # Model 1
    model = LinearModel(in_feats=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    epochs = 50

    for epoch in range(1, epochs + 1):
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

    # Model 2
    model = MLP(6, 64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    epochs = 100

    for epoch in range(1, epochs + 1):
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

    # Model 3
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True  # no eval_metric here
    )
    xgb_model.save_model('xgb_model.json')
    test_preds = xgb_model.predict(X_test)
    mse = mean_squared_error(y_test, test_preds)
    pearson = pearsonr(y_test, test_preds)[0]
    spearman = spearmanr(y_test, test_preds)[0]
    print(f'[XGBoost] Test MSE: {mse:.6f}, Pearson: {pearson:.6f}, Spearman: {spearman:.6f}')
    results['XGBoost'] = (mse, pearson, spearman)

    # Model 4
    model = GCN(in_feats=2, hidden=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    epochs = 50

    for epoch in range(1, epochs + 1):
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
    all_preds = [];
    all_targets = []
    with torch.no_grad():
        for g in test_data:
            nf = g['node_feats'].to(device)
            adj_norm = g['adj_norm'].to(device)
            ei = g['edge_index'].to(device)
            targets = g['edge_targets'].cpu().numpy()
            preds = model(nf, adj_norm, ei).cpu().numpy()
            all_preds.append(preds);
            all_targets.append(targets)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    mse = mean_squared_error(all_targets, all_preds)
    pearson = pearsonr(all_targets, all_preds)[0]
    spearman = spearmanr(all_targets, all_preds)[0]
    print(f'[GCN] Test MSE: {mse:.6f}, Pearson: {pearson:.6f}, Spearman: {spearman:.6f}')
    results['GCN'] = (mse, pearson, spearman)

    # Model 5
    model = GraphSAGE(in_feats=2, hidden=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    epochs = 50

    for epoch in range(1, epochs + 1):
        train_loss = 0
        model.train()
        for g in train_data:
            nf = g['node_feats'].to(device)
            adj = g['adj'].to(device)
            deg = g['degree'].to(device)
            ei = g['edge_index'].to(device)
            targets = g['edge_targets'].to(device)
            preds = model(nf, adj, deg, ei)
            loss = criterion(preds, targets)
            optimizer.zero_grad();
            loss.backward();
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_data)
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for g in val_data:
                nf = g['node_feats'].to(device)
                adj = g['adj'].to(device)
                deg = g['degree'].to(device)
                ei = g['edge_index'].to(device)
                targets = g['edge_targets'].to(device)
                val_loss += criterion(model(nf, adj, deg, ei), targets).item()
        val_loss /= len(val_data)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'sage_best.pt')
        print(f"[SAGE] Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        results['SAGE'] = (mse, pearson, spearman)

    # Test GraphSAGE Model
    model.load_state_dict(torch.load('sage_best.pt', map_location=device))
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for g in test_data:
            nf = g['node_feats'].to(device)
            adj = g['adj'].to(device)
            deg = g['degree'].to(device)
            ei = g['edge_index'].to(device)
            preds = model(nf, adj, deg, ei).cpu().numpy()
            targets = g['edge_targets'].cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    mse = mean_squared_error(all_targets, all_preds)
    pearson = pearsonr(all_targets, all_preds)[0]
    spearman = spearmanr(all_targets, all_preds)[0]
    print(f"[SAGE] Test MSE: {mse:.6f}, Pearson: {pearson:.6f}, Spearman: {spearman:.6f}")
    results['SAGE'] = (mse, pearson, spearman)

    # Comparative Summary
    print("Comparative Summary:")
    print(f"{'Model':<10} {'MSE':<10} {'Pearson':<10} {'Spearman':<10}")
    for name, metrics in results.items():
        mse, p, s = metrics
        print(f"{name:<10} {mse:<10.6f} {p:<10.4f} {s:<10.4f}")

if __name__ == "__main__":
    main()
