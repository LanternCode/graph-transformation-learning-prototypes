import numpy as np
import networkx as nx
import torch
import xgboost as xgb
from model import LinearModel, MLP, GCN, GraphSAGE, compute_edge_features_and_target
from benchmark import benchmark_models


class EdgeBetweennessModel:
    """
    Interface for edge-betweenness prediction adapters.

    Parameters:
        None.

    Returns:
        EdgeBetweennessModel: Base adapter class whose subclasses implement
            predict(G) for a NetworkX graph.
    """

    def predict(self, G: nx.Graph) -> np.ndarray:
        """
        Predict edge-betweenness scores for a graph.

        Parameters:
            G (networkx.Graph): Connected graph whose edges should be scored.

        Returns:
            np.ndarray: One-dimensional array with one prediction per edge, in
                the same order as list(G.edges()).
        """
        raise NotImplementedError


class FlatTorchModel(EdgeBetweennessModel):
    """
    Adapter for flat PyTorch models that consume edge-level feature arrays.

    Parameters:
        torch_model (torch.nn.Module): Trained PyTorch model that maps edge
            features to edge-betweenness predictions.
        device (torch.device): Device used for model inference.

    Returns:
        FlatTorchModel: Adapter exposing predict(G) for the benchmark harness.
    """

    def __init__(self, torch_model, device):
        """
        Initialize the flat PyTorch model adapter.

        Parameters:
            torch_model (torch.nn.Module): Trained edge-level PyTorch model.
            device (torch.device): Device used to run inference.

        Returns:
            None.
        """
        self.model = torch_model.to(device)
        self.device = device

    def predict(self, G):
        """
        Predict edge-betweenness scores with a flat PyTorch model.

        Parameters:
            G (networkx.Graph): Connected graph whose edges should be scored.

        Returns:
            np.ndarray: One-dimensional prediction array aligned with
                list(G.edges()).
        """
        X, _, edge_list = compute_edge_features_and_target(G)
        with torch.no_grad():
            inp = torch.tensor(X, device=self.device)
            out = self.model(inp).squeeze().cpu().numpy()
        return out


class XGBWrapper(EdgeBetweennessModel):
    """
    Adapter for an XGBoost edge-betweenness regression model.

    Parameters:
        xgb_model (xgboost.XGBRegressor): Trained XGBoost model that maps
            edge-level features to edge-betweenness predictions.

    Returns:
        XGBWrapper: Adapter exposing predict(G) for the benchmark harness.
    """

    def __init__(self, xgb_model):
        """
        Initialize the XGBoost model adapter.

        Parameters:
            xgb_model (xgboost.XGBRegressor): Trained XGBoost regressor.

        Returns:
            None.
        """
        self.model = xgb_model

    def predict(self, G):
        """
        Predict edge-betweenness scores with an XGBoost model.

        Parameters:
            G (networkx.Graph): Connected graph whose edges should be scored.

        Returns:
            np.ndarray: One-dimensional prediction array aligned with
                list(G.edges()).
        """
        X, _, _ = compute_edge_features_and_target(G)
        return self.model.predict(X)


class GNNWrapper(EdgeBetweennessModel):
    """
    Adapter for graph neural networks that consume graph tensors.

    Parameters:
        gnn_model (torch.nn.Module): Trained GCN or GraphSAGE-style model.
        device (torch.device): Device used for model inference.
        kind (str): Model kind. Use 'gcn' for GCN inputs and 'sage' for
            GraphSAGE inputs.

    Returns:
        GNNWrapper: Adapter exposing predict(G) for the benchmark harness.
    """

    def __init__(self, gnn_model, device, kind='gcn'):
        """
        Initialize the GNN model adapter.

        Parameters:
            gnn_model (torch.nn.Module): Trained graph neural network.
            device (torch.device): Device used to run inference.
            kind (str): Adapter mode, either 'gcn' or 'sage'.

        Returns:
            None.
        """
        self.model = gnn_model.to(device)
        self.device = device
        self.kind = kind

    def adapt_graph(self, G, device):
        """
        Convert a NetworkX graph into tensors expected by the GNN models.

        Parameters:
            G (networkx.Graph): Connected graph whose edges should be scored.
            device (torch.device): Device where the returned tensors should be
                allocated.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                node_feats, adj_norm, adj_raw, degree, and edge_idx tensors.
                edge_idx is aligned with list(G.edges()).
        """
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
        # Edges
        idx_map = {n: i for i, n in enumerate(G.nodes())}
        edge_list = list(G.edges())
        edge_idx = torch.tensor([[idx_map[u], idx_map[v]] for u, v in edge_list], dtype=torch.long).T.to(device)
        return node_feats, adj_norm, adj_raw, degree, edge_idx

    def predict(self, G):
        """
        Predict edge-betweenness scores with a GNN model.

        Parameters:
            G (networkx.Graph): Connected graph whose edges should be scored.

        Returns:
            np.ndarray: One-dimensional prediction array aligned with
                list(G.edges()).
        """
        nf, adj_norm, adj_raw, degree, edge_idx = self.adapt_graph(G, self.device)
        with torch.no_grad():
            if self.kind == 'gcn':
                out = self.model(nf, adj_norm, edge_idx)
            else:
                out = self.model(nf, adj_raw, degree, edge_idx)
        return out.cpu().numpy()


def main():
    """
    Load trained checkpoints and run the benchmark suite.

    Parameters:
        None.

    Returns:
        None. The function prints benchmark metrics and stores them in a local
            benchmark_results variable while running.
    """
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Linear
    lin = LinearModel(in_feats=6).to(device)
    lin.load_state_dict(torch.load('linear_best.pt', map_location=device))
    lin.eval()

    # MLP
    mlp = MLP(in_feats=6, hidden=64).to(device)
    mlp.load_state_dict(torch.load('mlp_best.pt', map_location=device))
    mlp.eval()

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        eval_metric='rmse'
    )
    xgb_model.load_model('xgb_model.json')

    # GCN
    gcn = GCN(in_feats=2, hidden=64).to(device)
    gcn.load_state_dict(torch.load('gcn_best.pt', map_location=device))
    gcn.eval()

    # GraphSAGE
    sage = GraphSAGE(in_feats=2, hidden=64).to(device)
    sage.load_state_dict(torch.load('sage_best.pt', map_location=device))
    sage.eval()

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
    return benchmark_results


if __name__ == "__main__":
    main()
