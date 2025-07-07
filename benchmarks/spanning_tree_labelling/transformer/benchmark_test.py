import torch
import torch.nn.functional as F
import networkx as nx
from benchmark import benchmark_model
from model_3 import ActorCritic
from model_2 import EdgeTransformerMST
from model_1 import EdgeTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_ch = 16 + 1
hidden = 128

model_sup = EdgeTransformer(in_channels=in_ch, hidden_channels=hidden).to(device)
model_mst = EdgeTransformerMST(in_channels=in_ch, hidden=hidden).to(device)
model_rl  = ActorCritic(in_ch=in_ch, hidden=hidden).to(device)

# 2) Load saved state_dicts
model_sup.load_state_dict(torch.load('best_model_sup_with_feats.pt', map_location=device))
model_mst.load_state_dict(torch.load('best_model_rl_with_feats.pt', map_location=device))
model_rl .load_state_dict(torch.load('best_model_actor_critic.pt', map_location=device))

# 3) Switch to eval mode
model_sup.eval()
model_mst.eval()
model_rl.eval()

print("Models loaded and ready for benchmarking")


def my_adapter(data):
    data = data.to(device)
    with torch.no_grad():
        logits = model_sup(data.x, data.edge_index, data.edge_attr)
        return (torch.sigmoid(logits) > 0.5).cpu()


# ─── MST‐loss adapter ─────────────────────────────────────────────────────────
def adapter_mst(data):
    # data: torch_geometric.data.Data with x, edge_index, edge_attr
    data = data.to(device)
    with torch.no_grad():
        logits = model_mst(data.x, data.edge_index, data.edge_attr)
        w      = F.softplus(logits).cpu().numpy()

    # Build undirected weight map
    src, dst = data.edge_index.cpu()
    und_w = {}
    for u,v,wt in zip(src.tolist(), dst.tolist(), w):
        if u < v: und_w[(u,v)] = wt

    # Decode via NetworkX
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    for (u,v),wt in und_w.items():
        G.add_edge(u, v, weight=wt)
    T = nx.maximum_spanning_tree(G)
    tree = set(T.edges())

    # Build directed‐edge mask
    mask = torch.zeros(data.num_edges, dtype=torch.bool)
    for i,(u,v) in enumerate(zip(src.tolist(), dst.tolist())):
        if (u,v) in tree or (v,u) in tree:
            mask[i] = True
    return mask

# ─── Actor–Critic RL adapter ─────────────────────────────────────────────────
def adapter_actorcritic(data):
    # We need a `batch` vector of zeros for a single graph:
    batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    data = data.to(device)
    with torch.no_grad():
        logits, _ = model_rl(data.x, data.edge_index, data.edge_attr, batch)
        p = torch.sigmoid(logits).cpu().numpy()

    # Undirected weight map
    src, dst = data.edge_index.cpu()
    und_p = {}
    for u,v,pp in zip(src.tolist(), dst.tolist(), p):
        if u < v: und_p[(u,v)] = pp

    # Greedy decode via NetworkX
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    for (u,v),pp in und_p.items():
        G.add_edge(u, v, weight=pp)
    T = nx.maximum_spanning_tree(G)
    tree = set(T.edges())

    # Directed‐edge mask
    mask = torch.zeros(data.num_edges, dtype=torch.bool)
    for i,(u,v) in enumerate(zip(src.tolist(), dst.tolist())):
        if (u,v) in tree or (v,u) in tree:
            mask[i] = True
    return mask


acc, total_1, total_0 = benchmark_model(
    my_adapter,
    n_graphs=1000,
    node_range=(6,100),
    feat_dim=16,
    extra_per_tree=1,
    seed=123
)

print(f"Benchmark results over {1000} graphs:")
print(f"  • Accuracy     : {acc:.4f}")
print(f"  • Predicted 1’s: {total_1}")
print(f"  • Predicted 0’s: {total_0}")


# MST benchmark
acc_mst, p1_mst, p0_mst = benchmark_model(adapter_mst, n_graphs=1000, seed=123)
print(f"MST   ▶  Acc: {acc_mst:.4f},  1’s:{p1_mst}, 0’s:{p0_mst}")

# RL benchmark
acc_rl,  p1_rl,  p0_rl  = benchmark_model(adapter_actorcritic, n_graphs=1000, seed=123)
print(f"RL_AC ▶  Acc: {acc_rl:.4f},  1’s:{p1_rl}, 0’s:{p0_rl}")
