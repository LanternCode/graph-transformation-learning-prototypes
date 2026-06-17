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
def train_reinforce(model, optimizer, num_epochs=30, lambda_weight=0.5,
                    batch_size=4, beta=0.9, device='cuda'):
    model.to(device)
    model.train()
    baseline = 0.0

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        correct_trees = 0

        for _ in range(900 // batch_size):  # batches per epoch
            batch_loss = 0.0

            for _ in range(batch_size):
                n_nodes = random.randint(6, 200)
                G = generate_candidate_graph(n_nodes)
                edge_index = list(G.edges())

                edge_features = get_contextual_edge_features(G, edge_index).to(device)
                logits = model(edge_features).squeeze()
                probs = torch.sigmoid(logits)

                m = torch.distributions.Bernoulli(probs)
                sampled_edges = m.sample()

                sampled_array = sampled_edges.detach().cpu().numpy()
                loss_val, is_correct, score, _ = compute_loss(G, sampled_array, edge_index, lambda_weight)

                # Use score as a soft reward ∈ [0,1]
                reward = score ** 2
                baseline = beta * baseline + (1 - beta) * reward
                advantage = reward - baseline

                log_probs = m.log_prob(sampled_edges)
                reinforce_loss = -advantage * log_probs.sum()
                batch_loss += reinforce_loss

                total_loss += loss_val
                correct_trees += int(is_correct)

            # Optimize after accumulating batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        avg_loss = total_loss / 900
        print(f"[REINFORCE+] Epoch {epoch:02d} - Avg Loss: {avg_loss:.4f}, Correct Trees: {correct_trees}/900")


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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_reinforce(model, optimizer, num_epochs=30, lambda_weight=0.6, device='cpu')
    torch.save(model.state_dict(), "reinforced_model.pth")
    print("Training done. Best model saved as reinforced_model.pth")
    evaluate_model("reinforced_model.pth", lambda_weight=0.6)
