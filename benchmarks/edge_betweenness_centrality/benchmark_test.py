import numpy as np
import networkx as nx
import torch
import xgboost as xgb
from model import LinearModel, MLP, GCN, GraphSAGE, compute_edge_features_and_target
from benchmark import benchmark_models


class EdgeBetweennessModel:
    def predict(self, G: nx.Graph) -> np.ndarray:
        """
        Given a connected networkx.Graph G, return a 1D numpy array of length |E(G)|
        containing your model’s predicted edge-betweenness scores, in the same
        order as list(G.edges()).
        """
        raise NotImplementedError


class FlatTorchModel(EdgeBetweennessModel):
    def __init__(self, torch_model, device):
        self.model = torch_model.to(device)
        self.device = device

    def predict(self, G):
        X, _, edge_list = compute_edge_features_and_target(G)
        with torch.no_grad():
            inp = torch.tensor(X, device=self.device)
            out = self.model(inp).squeeze().cpu().numpy()
        return out


class XGBWrapper(EdgeBetweennessModel):
    def __init__(self, xgb_model):
        self.model = xgb_model

    def predict(self, G):
        X, _, _ = compute_edge_features_and_target(G)
        return self.model.predict(X)


class GNNWrapper(EdgeBetweennessModel):
    def __init__(self, gnn_model, device, kind='gcn'):
        self.model = gnn_model.to(device)
        self.device = device
        self.kind = kind

    def adapt_graph(self, G, device):
        eb = nx.edge_betweenness_centrality(G)
        N = G.number_of_nodes()
        # Node features
        degrees = np.array([G.degree(n) for n in G.nodes()], dtype=np.float32)
        clustering = np.array([nx.clustering(G, n) for n in G.nodes()], dtype=np.float32)
        node_feats = torch.tensor(np.stack([degrees, clustering], axis=1), device=device)
        # Raw adjacency and normalized adjacency
        A = nx.to_numpy_array(G, nodelist=list(G.nodes()), dtype=np.float32)
        I = np.eye(N, dtype=np.float32)
        A_tilde = A + I
        d = np.sum(A_tilde, axis=1)
        # normalized
        D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
        adj_norm = torch.tensor(D_inv_sqrt @ A_tilde @ D_inv_sqrt, device=device)
        # raw
        adj_raw = torch.tensor(A, device=device)
        degree = torch.tensor(d, device=device)
        # Edges and targets
        idx_map = {n: i for i, n in enumerate(G.nodes())}
        edge_list = list(G.edges())
        edge_idx = torch.tensor([[idx_map[u], idx_map[v]] for u, v in edge_list], dtype=torch.long).T.to(device)
        targets = torch.tensor([eb[(u, v)] for u, v in edge_list], dtype=torch.float32).to(device)
        return node_feats, adj_norm, adj_raw, degree, edge_idx, targets

    def predict(self, G):
        nf, adj_norm, adj_raw, degree, edge_idx, _ = self.adapt_graph(G, self.device)
        with torch.no_grad():
            if self.kind == 'gcn':
                out = self.model(nf, adj_norm, edge_idx)
            else:
                out = self.model(nf, adj_raw, degree, edge_idx)
        return out.cpu().numpy()


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models
models = {}

# Linear
lin = LinearModel(in_feats=6).to(device)
lin.load_state_dict(torch.load('linear_best.pt', map_location=device))
lin.eval()
models['Linear'] = ('torch', lin)

# MLP
mlp = MLP(in_feats=6, hidden=64).to(device)
mlp.load_state_dict(torch.load('mlp_best.pt', map_location=device))
mlp.eval()
models['MLP'] = ('torch', mlp)

# XGBoost
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100,learning_rate=0.1, max_depth=5, eval_metric='rmse')
xgb_model.load_model('xgb_model.json')
models['XGBoost'] = ('xgb', xgb_model)

# GCN
gcn = GCN(in_feats=2, hidden=64).to(device)
gcn.load_state_dict(torch.load('gcn_best.pt', map_location=device))
gcn.eval()
models['GCN'] = ('gnn_gcn', gcn)

# GraphSAGE
sage = GraphSAGE(in_feats=2, hidden=64).to(device)
sage.load_state_dict(torch.load('sage_best.pt', map_location=device))
sage.eval()
models['SAGE'] = ('gnn_sage', sage)


# Build the dict of adapters
models: dict[str, EdgeBetweennessModel] = {
    'Linear': FlatTorchModel(lin, device),
    'MLP':    FlatTorchModel(mlp, device),
    'XGB':    XGBWrapper(xgb_model),
    'GCN':    GNNWrapper(gcn, device, kind='gcn'),
    'SAGE':   GNNWrapper(sage, device, kind='sage'),
}

# Run once
benchmark_results = benchmark_models(models)
