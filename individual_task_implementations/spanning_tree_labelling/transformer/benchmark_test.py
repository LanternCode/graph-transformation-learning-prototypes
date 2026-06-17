import torch
import torch.nn.functional as F
import networkx as nx
from benchmark import benchmark_model
from model_3 import ActorCritic
from model_2 import EdgeTransformerMST
from model_1 import EdgeTransformer


def aggregate_undirected_scores(edge_index, scores):
    """Combine the two directed scores for each undirected edge by averaging."""
    src, dst = edge_index.cpu()
    undirected_scores = {}
    for u, v, score in zip(src.tolist(), dst.tolist(), scores):
        key = (u, v) if u < v else (v, u)
        undirected_scores.setdefault(key, []).append(float(score))
    return {edge: sum(values) / len(values) for edge, values in undirected_scores.items()}


def decode_with_maximum_spanning_tree(data, edge_scores):
    """Decode directed edge scores into a directed-edge mask via an undirected MST."""
    undirected_scores = aggregate_undirected_scores(data.edge_index, edge_scores)

    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    for (u, v), score in undirected_scores.items():
        graph.add_edge(u, v, weight=score)

    tree = set(nx.maximum_spanning_tree(graph).edges())

    src, dst = data.edge_index.cpu()
    mask = torch.zeros(data.num_edges, dtype=torch.bool)
    for i, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
        if (u, v) in tree or (v, u) in tree:
            mask[i] = True
    return mask


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_ch = 16 + 1
    hidden = 128

    model_sup = EdgeTransformer(in_channels=in_ch, hidden_channels=hidden).to(device)
    model_mst = EdgeTransformerMST(in_channels=in_ch, hidden=hidden).to(device)
    model_rl = ActorCritic(in_ch=in_ch, hidden=hidden).to(device)

    # 2) Load saved state_dicts
    model_sup.load_state_dict(torch.load('best_model_sup_with_feats.pt', map_location=device))
    model_mst.load_state_dict(torch.load('best_model_rl_with_feats.pt', map_location=device))
    model_rl.load_state_dict(torch.load('best_model_actor_critic.pt', map_location=device))

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
            weights = F.softplus(logits).cpu().numpy()
        return decode_with_maximum_spanning_tree(data.cpu(), weights)

    # ─── Actor–Critic RL adapter ─────────────────────────────────────────────────
    def adapter_actorcritic(data):
        # We need a `batch` vector of zeros for a single graph:
        data = data.to(device)
        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model_rl(data.x, data.edge_index, data.edge_attr, batch)
            probabilities = torch.sigmoid(logits).cpu().numpy()
        return decode_with_maximum_spanning_tree(data.cpu(), probabilities)

    acc, total_1, total_0 = benchmark_model(
        my_adapter,
        n_graphs=1000,
        node_range=(6, 100),
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
    acc_rl, p1_rl, p0_rl = benchmark_model(adapter_actorcritic, n_graphs=1000, seed=123)
    print(f"RL_AC ▶  Acc: {acc_rl:.4f},  1’s:{p1_rl}, 0’s:{p0_rl}")


if __name__ == '__main__':
    main()
