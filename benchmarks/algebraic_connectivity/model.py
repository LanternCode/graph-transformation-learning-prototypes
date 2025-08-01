import os
import h5py
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from itertools import combinations
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
from math import comb


def load_graph_from_blist(path, feat_dim=16):
    with h5py.File(path, 'r') as f:
        edge_list = f['bList'][:]
    edge_index = torch.tensor(edge_list, dtype=torch.long) - 1
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, feat_dim))
    return Data(x=x, edge_index=edge_index)


def get_candidate_edges(data, limit=None):
    edge_set = set(map(tuple, data.edge_index.t().tolist()))
    candidates = [(i, j) for i, j in combinations(range(data.num_nodes), 2)
                  if (i, j) not in edge_set and (j, i) not in edge_set]
    return candidates[:limit] if limit else candidates


def load_benchmark_data(data_dir="content/PowerGraph-Graph/data", candidate_limit=200, feat_dim=16):
    """
    Loads the PowerGrid benchmark dataset for RL generalization.

    Returns:
        train_graphs (dict): mapping from graph name to Data object
        test_graph (tuple): (graph name, Data object)
        candidates (dict): mapping from graph name to list of candidate edges
    """
    graph_names = ["ieee24", "ieee39", "ieee118", "uk"]
    paths = {name: os.path.join(data_dir, name, name, "raw", "blist.mat") for name in graph_names}

    data_dict = {}
    candidate_dict = {}
    for name, path in paths.items():
        data = load_graph_from_blist(path, feat_dim=feat_dim)
        candidates = get_candidate_edges(data, limit=candidate_limit)
        data_dict[name] = data
        candidate_dict[name] = candidates

    # Assign test graph and training graphs
    test_name = "uk"
    train_names = [g for g in graph_names if g != test_name]

    train_graphs = {k: data_dict[k] for k in train_names}
    test_graph = (test_name, data_dict[test_name])
    return train_graphs, test_graph, candidate_dict


train_graphs, (test_name, test_graph), candidate_edges = load_benchmark_data(candidate_limit=35)
candidates = candidate_edges[test_name]


class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class PairwiseRankModel(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_dim, hidden_dim)
        self.scorer = DeltaLambdaHead(hidden_dim)

    def forward(self, x, edge_index, edge_pairs):
        z = self.encoder(x, edge_index)
        u, v = edge_pairs[:, 0], edge_pairs[:, 1]
        scores = self.scorer(z[u], z[v])
        return scores


class DeltaLambdaHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, u_embed, v_embed):
        edge_feat = torch.cat([u_embed, v_embed], dim=-1)
        return self.fc(edge_feat).squeeze(-1)


def compute_lambda_2(edge_index, num_nodes):
    row, col = edge_index
    A = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes))
    A = A + A.T
    D = sp.diags(A.sum(axis=1).flatten().tolist()[0])
    L = D - A
    eigvals = eigsh(L, k=2, which='SM', return_eigenvectors=False)
    return sorted(eigvals)[1]


def label_edges_by_lambda2(data, candidates):
    """
    Labels candidate edges by their gain in algebraic connectivity (Δλ₂).

    Args:
        data (torch_geometric.data.Data): Input graph.
        candidates (list of (int, int)): Candidate edges.

    Returns:
        pairs (LongTensor): shape [num_edges, 2]
        deltas (FloatTensor): Δλ₂ for each candidate edge
    """
    base_lambda2 = compute_lambda_2(data.edge_index.numpy(), data.num_nodes)
    pairs = []
    deltas = []

    for (u, v) in candidates:
        # Add (u, v) and (v, u) to make the graph undirected
        new_edges = torch.cat([
            data.edge_index,
            torch.tensor([[u, v], [v, u]], dtype=torch.long)
        ], dim=1)

        new_lambda2 = compute_lambda_2(new_edges.numpy(), data.num_nodes)
        delta = new_lambda2 - base_lambda2

        pairs.append([u, v])
        deltas.append(delta)

    return torch.tensor(pairs, dtype=torch.long), torch.tensor(deltas, dtype=torch.float)


def train_epoch_pairwise(graph_names, graph_data, edge_labels, model, optimizer, num_epochs=100):
    print(f"{'Epoch':>5} | {'Loss':>10} | {'Skipped':>8}")
    print("-" * 32)
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        skipped_graphs = 0

        for g in graph_names:
            data = graph_data[g]
            pairs, targets = edge_labels[g]
            optimizer.zero_grad()

            scores = model(data.x, data.edge_index, pairs)

            idx_i = torch.randint(0, len(pairs), (32,))
            idx_j = torch.randint(0, len(pairs), (32,))
            mask = targets[idx_i] > targets[idx_j]

            if mask.sum() == 0:
                skipped_graphs += 1
                continue

            s_i = scores[idx_i[mask]]
            s_j = scores[idx_j[mask]]

            rank_loss = F.margin_ranking_loss(s_i, s_j, torch.ones_like(s_i), margin=0.01)
            rank_loss.backward()
            optimizer.step()
            total_loss += rank_loss.item()

        avg_loss = total_loss / (len(graph_names) - skipped_graphs) if (len(graph_names) - skipped_graphs) > 0 else 0
        print(f"{epoch:5d} | {avg_loss:10.4f} | {skipped_graphs:8d}")


# Train
edge_labels = {}
for name in train_graphs:
    data = train_graphs[name]
    candidates = candidate_edges[name]
    pairs, deltas = label_edges_by_lambda2(data, candidates)
    edge_labels[name] = (pairs, deltas)

model = PairwiseRankModel(in_dim=16, hidden_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_epoch_pairwise(train_graphs.keys(), train_graphs, edge_labels, model, optimizer)


@torch.no_grad()
def evaluate_pairwise_model(model, data, candidate_edges, max_steps=5):
    model.eval()
    base_lambda2 = compute_lambda_2(data.edge_index.numpy(), data.num_nodes)

    # Convert candidate edge list to tensor
    edge_pairs = torch.tensor(candidate_edges, dtype=torch.long)

    # Score all candidate edges
    scores = model(data.x, data.edge_index, edge_pairs)
    topk_indices = torch.topk(scores, k=max_steps).indices
    selected_edges = edge_pairs[topk_indices]

    # Add selected edges to the graph (undirected)
    new_edge_index = data.edge_index.clone()
    for u, v in selected_edges:
        new_edge_index = torch.cat([
            new_edge_index,
            torch.tensor([[u, v], [v, u]], dtype=torch.long)
        ], dim=1)

    # Compute updated λ₂
    updated_lambda2 = compute_lambda_2(new_edge_index.numpy(), data.num_nodes)

    return {
        "Base λ₂": base_lambda2,
        "Final λ₂": updated_lambda2,
        "Edges Added": selected_edges.tolist(),
        "Reward": updated_lambda2 - base_lambda2
    }


# Evaluate
rl_result = evaluate_pairwise_model(model, test_graph, candidate_edges[test_name], max_steps=5)
print(rl_result)


def evaluate_greedy(data, candidates, base_lambda2, max_steps=5):
    added_edges = []
    current_data = data
    current_lambda2 = base_lambda2

    for _ in range(max_steps):
        best_gain = -1
        best_edge = None

        for edge in candidates:
            if edge in added_edges:
                continue
            test_edges = torch.cat([
                current_data.edge_index,
                torch.tensor([[edge[0], edge[1]], [edge[1], edge[0]]], dtype=torch.long)
            ], dim=1)
            test_lambda2 = compute_lambda_2(test_edges.numpy(), current_data.num_nodes)
            gain = test_lambda2 - current_lambda2
            if gain > best_gain:
                best_gain = gain
                best_edge = edge

        if best_edge is None:
            break

        added_edges.append(best_edge)
        current_data = Data(
            x=current_data.x,
            edge_index=torch.cat([
                current_data.edge_index,
                torch.tensor([[best_edge[0], best_edge[1]], [best_edge[1], best_edge[0]]], dtype=torch.long)
            ], dim=1)
        )
        current_lambda2 = compute_lambda_2(current_data.edge_index.numpy(), current_data.num_nodes)

    return base_lambda2, current_lambda2, added_edges


def evaluate_optimal(data, candidates, base_lambda2, max_steps=5):
    best_lambda2 = base_lambda2
    best_combo = []

    total_combos = comb(len(candidates), max_steps)
    combo_iter = combinations(candidates, max_steps)

    for combo in tqdm(combo_iter, total=total_combos, desc="Evaluating optimal", unit="combo"):
        edge_index = data.edge_index
        for u, v in combo:
            edge_index = torch.cat([
                edge_index,
                torch.tensor([[u, v], [v, u]], dtype=torch.long)
            ], dim=1)
        test_lambda2 = compute_lambda_2(edge_index.numpy(), data.num_nodes)
        if test_lambda2 > best_lambda2:
            best_lambda2 = test_lambda2
            best_combo = combo

    return base_lambda2, best_lambda2, best_combo


# Evaluate greedy and optimal baselines
base_lambda2 = compute_lambda_2(test_graph.edge_index.numpy(), test_graph.num_nodes)
limited_candidates = candidates[:35]  # or any smaller number
base_g, greedy_g, greedy_edges = evaluate_greedy(test_graph, candidate_edges[test_name], base_lambda2, max_steps=5)
base_o, optimal_g, optimal_edges = evaluate_optimal(test_graph, candidate_edges[test_name], base_lambda2, max_steps=5)

# Show results
df = pd.DataFrame([{
    "Graph": test_name,
    "RL λ₂": rl_result["Final λ₂"],
    "Greedy λ₂": greedy_g,
    "Optimal λ₂": optimal_g,
    "RL Edges": rl_result["Edges Added"],
    "Greedy Edges": greedy_edges,
    "Optimal Edges": optimal_edges,
    "RL Gain %": 100 * (rl_result["Final λ₂"] - base_g) / (optimal_g - base_g) if optimal_g > base_g else 0.0
}])

print(df)
