import json, joblib, math, torch
import networkx as nx
import numpy as np

from benchmark import benchmark_model
from model import MLP, CNN1D, TransformerClassifier, AutoencoderClassifier

TRAIN_FEATURE_ORDER = ["common_neighbors","jaccard","adamic_adar","deg_u","deg_v","shortest_path","cc_u","cc_v"]


def _shortest_paths(edge_index, num_nodes, pairs):
    # Build an undirected NX graph from edge_index
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    ei = edge_index.detach().cpu().numpy()
    G.add_edges_from(zip(ei[0], ei[1]))
    # Compute SP length per pair (u,v); -1 if disconnected
    sps = []
    for u, v in pairs.detach().cpu().numpy():
        try:
            sps.append(nx.shortest_path_length(G, int(u), int(v)))
        except nx.NetworkXNoPath:
            sps.append(-1)
    return torch.tensor(sps, dtype=torch.float32)


def _rebuild_training_features(data):
    """
    Returns a (E, 8) tensor in TRAIN_FEATURE_ORDER using:
    - data.edge_features: columns [common, jaccard, adamic_adar, pref_attach, edge_betweenness]
    - data.x: node feats [degree, clustering, betweenness, closeness, pagerank, kcore, triangles]
    - data.edge_pairs: (E,2)
    """
    E = data.edge_pairs.size(0)
    # pick the 3 shared features directly
    common = data.edge_features[:, 0:1]      # (E,1)
    jacc   = data.edge_features[:, 1:2]
    aa     = data.edge_features[:, 2:3]

    # degrees & clustering from node features
    deg = data.x[:, 0]   # (N,)
    cc  = data.x[:, 1]   # (N,)

    u = data.edge_pairs[:, 0].long()
    v = data.edge_pairs[:, 1].long()
    deg_u = deg[u].view(E, 1)
    deg_v = deg[v].view(E, 1)
    cc_u  = cc[u].view(E, 1)
    cc_v  = cc[v].view(E, 1)

    # shortest path from edge_index graph
    sp = _shortest_paths(data.edge_index, data.x.size(0), data.edge_pairs).view(E, 1)

    # stack in the exact training order
    feats = torch.cat([common, jacc, aa, deg_u, deg_v, sp, cc_u, cc_v], dim=1)  # (E,8)
    return feats


def make_torch_adapter(model_ctor, weight_path,
                       scaler_path="scaler_fillin.joblib",
                       feature_order_path="feature_order.json",
                       device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    scaler = joblib.load(scaler_path)
    with open(feature_order_path) as f:
        feature_order = json.load(f)
    assert feature_order == TRAIN_FEATURE_ORDER, f"Training order differs: {feature_order}"

    model = model_ctor(input_dim=len(feature_order)).to(device)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    @torch.no_grad()
    def adapter_fn(data):
        feats = _rebuild_training_features(data)                 # (E,8)
        feats_np = feats.detach().cpu().numpy()
        feats_np = scaler.transform(feats_np)                    # apply training scaler
        feats_t = torch.from_numpy(feats_np).float().to(device)
        probs = model(feats_t).view(-1)                          # (E,)
        return probs
    return adapter_fn


def make_rf_adapter(pkl_path,
                    scaler_path="scaler_fillin.joblib",
                    feature_order_path="feature_order.json"):
    rf = joblib.load(pkl_path)
    scaler = joblib.load(scaler_path)
    with open(feature_order_path) as f:
        feature_order = json.load(f)
    assert feature_order == TRAIN_FEATURE_ORDER

    def adapter_fn(data):
        feats = _rebuild_training_features(data)
        feats_np = scaler.transform(feats.detach().cpu().numpy())
        probs = rf.predict_proba(feats_np)[:, 1]
        return torch.from_numpy(probs).float()
    return adapter_fn


if __name__ == "__main__":
    NUM_GRAPHS = 1000
    BATCH_SIZE = 32

    runs = [
        ("MLP",          make_torch_adapter(MLP, "MLP_model_balanced.pth")),
        ("CNN1D",        make_torch_adapter(CNN1D, "CNN1D_model_balanced.pth")),
        ("Transformer",  make_torch_adapter(TransformerClassifier, "Transformer_model_balanced.pth")),
        ("Autoencoder",  make_torch_adapter(AutoencoderClassifier, "Autoencoder_model_balanced.pth")),
        ("RandomForest", make_rf_adapter("random_forest_fillin.pkl")),
    ]

    for name, adapter in runs:
        print(f"\n==== Evaluating {name} ====")
        benchmark_model(adapter, num_graphs=NUM_GRAPHS, batch_size=BATCH_SIZE)


