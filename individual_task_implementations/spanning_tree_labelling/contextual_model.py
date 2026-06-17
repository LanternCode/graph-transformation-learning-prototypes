# === Edge Spanning Tree Labeling Model (Model 3: Edge Context Encoder) ===
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import random
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
from itertools import combinations


def spanning_tree_score_from_prediction(graph, predicted_labels):
    edge_index = graph['edge_index']
    num_nodes = graph['num_nodes']
    V = num_nodes
    incoming_counts = [0] * V
    adj = defaultdict(list)

    # Build adjacency list and count incoming edges
    for (u, v), label in zip(edge_index, predicted_labels):
        if label == 1:
            adj[u].append(v)
            incoming_counts[v] += 1

    # Penalty for nodes with multiple incoming edges
    overconnection_penalty = sum(max(0, count - 1) for count in incoming_counts)

    # Identify roots (nodes with zero incoming edges)
    roots = [i for i, count in enumerate(incoming_counts) if count == 0]
    root_penalty = max(0, len(roots) - 1)

    # Traverse graph with DFS and detect cycles
    visited = set()
    on_path = set()
    cycle_detected = False

    def dfs(node):
        nonlocal cycle_detected
        visited.add(node)
        on_path.add(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                dfs(neighbor)
            elif neighbor in on_path:
                cycle_detected = True
        on_path.remove(node)

    if roots:
        dfs(roots[0])
    else:
        dfs(0)  # fallback root if none identified

    # Penalty for unreachable nodes
    unreachable_nodes = V - len(visited)

    # Total penalties
    total_penalty = overconnection_penalty + root_penalty + unreachable_nodes
    if cycle_detected:
        total_penalty += V  # strong penalty for cycles

    max_penalty = V
    final_score = 1.0 - min(1.0, total_penalty / max_penalty)

    return final_score


# ==== MODEL ====
class ContextAwareMLP(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(6, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, edge_features):  # edge_features: [E, 6]
        x = F.relu(self.fc1(edge_features))
        return self.fc2(x).squeeze(-1)  # output logits: [E]


# ==== GRAPH DATA GENERATOR ====
def generate_candidate_graph(num_nodes):
    T = nx.random_unlabeled_tree(num_nodes)
    tree_edges = list(nx.bfs_edges(T, source=0))
    tree_set = set(tree_edges)

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(tree_edges)

    blocked_edges = tree_set | set((v, u) for u, v in tree_set)
    all_possible = [
        (u, v)
        for u in range(num_nodes)
        for v in range(num_nodes)
        if u != v and (u, v) not in blocked_edges
    ]

    extra_count = random.randint(1, min(len(all_possible), 2 * (num_nodes - 1)))
    extra_edges = random.sample(all_possible, extra_count)
    G.add_edges_from(extra_edges)

    G.graph["tree_set"] = tree_set
    return G


# ==== EDGE FEATURES ====
def get_contextual_edge_features(G, edge_index):
    out_deg = dict(G.out_degree())
    in_deg = dict(G.in_degree())
    successors = {node: set(G.successors(node)) for node in G.nodes()}
    predecessors = {node: set(G.predecessors(node)) for node in G.nodes()}
    features = []

    for u, v in edge_index:
        two_hop_support = len(successors[u].intersection(predecessors[v]))
        reciprocal = 1.0 if G.has_edge(v, u) else 0.0
        features.append([
            out_deg[u],
            in_deg[u],
            out_deg[v],
            in_deg[v],
            two_hop_support,
            reciprocal,
        ])

    return torch.tensor(features, dtype=torch.float32)


def get_tree_edge_labels(G, edge_index):
    tree_set = G.graph["tree_set"]
    labels = [1.0 if (u, v) in tree_set else 0.0 for u, v in edge_index]
    return torch.tensor(labels, dtype=torch.float32)


# ==== EXTRACT TREE BASED ON SCORES ====
def extract_tree_from_scores(edge_index, scores, num_nodes):
    sorted_edges = sorted(zip(edge_index, scores), key=lambda x: -x[1])
    parent = list(range(num_nodes))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv
            return True
        return False

    selected = []
    for (u, v), _ in sorted_edges:
        if union(u, v):
            selected.append((u, v))
        if len(selected) == num_nodes - 1:
            break

    return selected


# ==== LOSS FUNCTION ====
def compute_loss(graph, scores, edge_index, lambda_weight=0.5):
    num_nodes = graph.number_of_nodes()
    predicted_tree = extract_tree_from_scores(edge_index, scores, num_nodes)
    predicted_tree_set = set(predicted_tree)
    predicted_labels = [1 if (u, v) in predicted_tree_set else 0 for u, v in edge_index]

    score = spanning_tree_score_from_prediction(
        {"edge_index": edge_index, "num_nodes": num_nodes}, predicted_labels)
    is_correct = float(score == 1.0)

    loss = lambda_weight * (1 - score) + (1 - lambda_weight) * (1 - is_correct)
    return loss, is_correct, score, predicted_labels


# ==== TRAINING LOOP ====
def train_model(model, optimizer, num_epochs=30, lambda_weight=0.5):
    best_score = -1
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses, correct_trees = [], 0

        for _ in range(900):  # 900 training graphs per epoch
            n_nodes = random.randint(6, 200)
            G = generate_candidate_graph(n_nodes)
            edge_index = list(G.edges())
            edge_features = get_contextual_edge_features(G, edge_index)
            labels = get_tree_edge_labels(G, edge_index)

            logits = model(edge_features)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                scores = torch.sigmoid(logits).detach().cpu().numpy()
                eval_loss, is_correct, score, _ = compute_loss(G, scores, edge_index, lambda_weight)

            epoch_losses.append(eval_loss)
            correct_trees += int(is_correct)

        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}, Correct Trees: {correct_trees}/900")

        if correct_trees > best_score:
            best_score = correct_trees
            torch.save(model.state_dict(), "mlp_contextual_best.pt")

    print("Training done. Best model saved as mlp_contextual_best.pt")


# ==== TESTING ====
def evaluate_model(model_path, lambda_weight=0.5):
    model = ContextAwareMLP()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_losses = []
    total_correct = 0
    total_ones = 0
    total_zeros = 0

    for _ in range(100):  # 100 test graphs
        n_nodes = random.randint(6, 200)
        G = generate_candidate_graph(n_nodes)
        edge_index = list(G.edges())
        edge_features = get_contextual_edge_features(G, edge_index)

        scores = torch.sigmoid(model(edge_features)).detach().numpy()
        loss, is_correct, score, predicted_labels = compute_loss(G, scores, edge_index, lambda_weight)
        test_losses.append(loss)
        total_correct += int(is_correct)
        total_ones += sum(predicted_labels)
        total_zeros += len(predicted_labels) - sum(predicted_labels)

    print(f"Test Avg Loss: {np.mean(test_losses):.4f}")
    print(f"Correct Trees: {total_correct}/100")
    print(f"Edge labels: 1 = {total_ones}, 0 = {total_zeros}")


# ==== MAIN ====
if __name__ == "__main__":
    model = ContextAwareMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_model(model, optimizer, num_epochs=30, lambda_weight=0.6)
    evaluate_model("mlp_contextual_best.pt", lambda_weight=0.6)

# lambda_weight=0.6 --> 89. 9
# lambda_weight=0.72 --> 50, x
