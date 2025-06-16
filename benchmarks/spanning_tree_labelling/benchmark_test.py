# Generate a smaller test batch to avoid memory issues
import math
import random

import networkx as nx
import torch
from benchmark import evaluate_model
from contextual_model import ContextAwareMLP, get_contextual_edge_features
from cnn_models import CNNEdgeLabeler
from mlp_models import EdgeScorer, build_edge_feats_and_labels


# Dummy model that randomly assigns 0 or 1 to each edge
def dummy_model_predict(graph):
    num_edges = len(graph['edge_index'])
    predicted_labels = [random.randint(0, 1) for _ in range(num_edges)]
    return predicted_labels


def _run_inference(graph, model, device='cpu', thresh=0.5):
    """
    model: already loaded & eval()’d
    graph: {'edge_index': List[(u,v)], 'num_nodes':N, ...}
    returns: List[int] of 0/1 labels aligned with edge_index
    """
    N = graph['num_nodes']
    # build full adjacency
    A = torch.zeros(N, N, dtype=torch.float, device=device)
    for u, v in graph['edge_index']:
        A[u, v] = 1.0
    x = A.view(1, -1).to(device)
    with torch.no_grad():
        logits = model(x)             # [1, N*N]
        probs  = torch.sigmoid(logits).view(-1)
    out = []
    for u, v in graph['edge_index']:
        idx = u * N + v
        out.append(1 if probs[idx] > thresh else 0)
    return out


def make_mlp_adapter(model_path, device='cpu', thresh=0.5):
    def adapter(graph):
        N = graph['num_nodes']
        # Build degree counts
        out_deg = {i:0 for i in range(N)}
        in_deg  = {i:0 for i in range(N)}
        for u, v in graph['edge_index']:
            out_deg[u] += 1
            in_deg[v]  += 1
        full_set = set(graph['edge_index'])

        # 1) Build the SAME 7-dim feature vector for every (u,v)
        feats = []
        for u in range(N):
            for v in range(N):
                a_uv = 1.0 if (u,v) in full_set else 0.0
                du   = out_deg[u] + in_deg[u]
                dv   = out_deg[v] + in_deg[v]
                pu   = [math.sin(u/N/10), math.cos(u/N/10)]
                pv   = [math.sin(v/N/10), math.cos(v/N/10)]
                feats.append([a_uv, du, dv, pu[0], pu[1], pv[0], pv[1]])

        X = torch.tensor(feats, dtype=torch.float32, device=device)

        # 2) Load your checkpoint into the identical model
        model = EdgeScorer(in_dim=7, hidden_dims=[128,128,64]).to(device)
        sd    = torch.load(model_path, map_location=device)
        model.load_state_dict(sd)
        model.eval()

        # 3) Forward + threshold
        with torch.no_grad():
            probs = torch.sigmoid(model(X)).cpu().numpy()
        preds = (probs > thresh).astype(int).tolist()

        # 4) Return the list of 0/1 labels in row-major (u,v) order
        #    which aligns with how the benchmark expects them
        return preds

    return adapter


def make_cnn_adapter(mode, model_path, device='cpu'):
    """
    mode: one of 'supervised', 'soft', 'hard'
    model_path: path to the .pth file
    """
    def adapter(graph):
        N = graph['num_nodes']
        model = CNNEdgeLabeler(N).to(device)
        sd = torch.load(model_path, map_location=device)
        model.load_state_dict(sd)
        model.eval()
        return _run_inference(graph, model, device)
    return adapter


def make_context_aware_adapter(model_path, device='cpu'):
    """
    Adapter for benchmark: loads the context-aware model and returns a callable.
    Each call processes one graph dict and returns a list[int] of edge labels.
    """
    model = ContextAwareMLP().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    def adapter(graph):
        # Convert graph dict to NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(graph['num_nodes']))
        G.add_edges_from(graph['edge_index'])

        edge_index = list(G.edges())
        edge_features = get_contextual_edge_features(G, edge_index).to(device)

        with torch.no_grad():
            scores = model(edge_features).squeeze().cpu().numpy()
        predictions = (scores >= 0.5).astype(int)

        # Preserve edge order from input
        edge_to_label = {tuple(sorted(edge)): int(label) for edge, label in zip(edge_index, predictions)}
        return [edge_to_label[tuple(sorted(e))] for e in graph['edge_index']]
    return adapter


# First Benchmark - Supervised MLP
print(f"\nBenchmark: Supervised MLP")
#mlp_sup   = make_mlp_adapter('supervised', 'trained_models/mlp_supervised_best.pth')
mlp_sup   = make_mlp_adapter('trained_models/mlp_supervised_best.pth')
evaluate_model(mlp_sup)

# Second Benchmark - Objective Function MLP
print(f"\nBenchmark: Metric-based MLP")
#mlp_soft  = make_mlp_adapter('soft',       'trained_models/mlp_policy_soft_best.pth')
mlp_soft  = make_mlp_adapter('trained_models/mlp_policy_soft_best.pth')
evaluate_model(mlp_soft)

# Third Benchmark - Harsh Objective Function MLP
print(f"\nBenchmark: Harsh Metric-based MLP")
#mlp_hard  = make_mlp_adapter('hard',       'trained_models/mlp_policy_hard_best.pth')
mlp_hard  = make_mlp_adapter('trained_models/mlp_policy_hard_best.pth')
evaluate_model(mlp_hard)

# Fourth Benchmark - Supervised CNN
print(f"\nBenchmark: Supervised CNN")
cnn_sup   = make_cnn_adapter('supervised', 'trained_models/cnn_model_supervised_best.pth')
evaluate_model(cnn_sup)

# Fifth Benchmark - Objective Function CNN
print(f"\nBenchmark: Metric-based CNN")
cnn_soft  = make_cnn_adapter('soft',       'trained_models/cnn_model_policy_soft_best.pth')
evaluate_model(cnn_soft)

# Sixth Benchmark - Harsh Objective Function CNN
print(f"\nBenchmark: Harsh Metric-based CNN")
cnn_hard  = make_cnn_adapter('hard',       'trained_models/cnn_model_policy_hard_best.pth')
evaluate_model(cnn_hard)

# Seventh Benchmark - Context-Aware MLP
print(f"\nBenchmark: Context-Aware MLP")
camlp  = make_context_aware_adapter('trained_models/mlp_contextual_best.pt')
evaluate_model(camlp)

# Eight Benchmark - Reinforcement Learning
print(f"\nBenchmark: Reinforcement Learning")
reinforced  = make_context_aware_adapter('trained_models/reinforced_model.pth')
evaluate_model(camlp)
