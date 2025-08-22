import networkx as nx
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import math

# === CONFIG ===
NUM_TRAIN = 20
NUM_TEST = 100
MIN_NODES = 6
MAX_NODES = 140
EPOCHS = 50


# === GRAPH GENERATION ===
def generate_graph_pair(pct_extra=0):
    num_nodes = random.randint(MIN_NODES, MAX_NODES)
    tree = nx.random_unlabeled_tree(num_nodes)
    tree = nx.Graph(tree)  # Ensure undirected
    base_edges = list(tree.edges())

    # Create possible edges to add (excluding existing and self-loops)
    existing = set(base_edges) | set((v, u) for u, v in base_edges)
    candidates = [
        (i, j) for i in range(num_nodes) for j in range(i+1, num_nodes)
        if (i, j) not in existing and i != j
    ]
    random.shuffle(candidates)
    num_extra = min(math.ceil(pct_extra/100 * len(base_edges)), len(base_edges))
    extra_edges = candidates[:num_extra]

    full_graph = nx.Graph()
    full_graph.add_nodes_from(tree.nodes())
    full_graph.add_edges_from(base_edges + extra_edges)

    # Label only the added edges that actually complete cycles as 1
    label = {}
    for edge in full_graph.edges():
        temp = full_graph.copy()
        temp.remove_edge(*edge)
        label[edge] = 1 if nx.has_path(full_graph, edge[0], edge[1]) and nx.has_path(full_graph, edge[1], edge[0]) else 0

    return full_graph, label


# === FEATURE EXTRACTION ===
def extract_features(graph):
    features = []
    edges = list(graph.edges())
    degree = dict(graph.degree())
    betweenness = nx.edge_betweenness_centrality(graph)

    for u, v in edges:
        features.append([
            degree[u],
            degree[v],
            betweenness.get((u, v), 0) or betweenness.get((v, u), 0),
        ])
    return np.array(features), edges


# === MODEL 2: STRUCTURAL OPTIMIZATION ===
def train_unsupervised_model():
    model = RandomForestClassifier()
    best_score = 0
    best_model = None

    for epoch in range(EPOCHS):
        edge_records = []
        for _ in range(NUM_TRAIN):
            G, labels = generate_graph_pair(epoch)
            feats, edges = extract_features(G)
            edge_records.append((G, feats, edges))

        X, y = [], []
        for G, feats, edges in edge_records:
            probs = np.random.rand(len(edges))
            removed = [e for i, e in enumerate(edges) if probs[i] > 0.5]
            G2 = G.copy()
            G2.remove_edges_from(removed)
            y.extend([1 if e in removed else 0 for e in edges])
            X.extend(feats)

        model.fit(X, y)

        # Evaluate on training set to guide learning
        cycle_free_count = 0
        total_cycles = 0
        for G, feats, edges in edge_records:
            probs = model.predict_proba(feats)[:, 1]
            to_remove = [e for i, e in enumerate(edges) if probs[i] > 0.5]
            G2 = G.copy()
            G2.remove_edges_from(to_remove)
            try:
                cycles = list(nx.find_cycle(G2))
                total_cycles += 1 if cycles else 0
            except:
                cycles = []
            if not cycles:
                cycle_free_count += 1

        print(f"[Model 2][Epoch {epoch+1}] Training Acyclic Graphs: {cycle_free_count} / {NUM_TRAIN}")
        if cycle_free_count > best_score:
            best_score = cycle_free_count
            best_model = model
            joblib.dump(model, "model2_best.pth")

    # Final test evaluation
    cycle_free_count = 0
    pred_zeros, pred_ones = 0, 0
    for _ in range(NUM_TEST):
        G, _ = generate_graph_pair(EPOCHS)
        feats, edges = extract_features(G)
        probs = best_model.predict_proba(feats)[:, 1]
        to_remove = [e for i, e in enumerate(edges) if probs[i] > 0.5]
        pred_ones += len(to_remove)
        pred_zeros += len(edges) - len(to_remove)
        G2 = G.copy()
        G2.remove_edges_from(to_remove)
        try:
            cycles = list(nx.find_cycle(G2))
        except:
            cycles = []
        if not cycles:
            cycle_free_count += 1

    print(f"[Model 2] Final Test Acyclic Graphs: {cycle_free_count} / {NUM_TEST}")
    print(f"[Model 2] Final Prediction Counts: 0s = {pred_zeros}, 1s = {pred_ones}")


if __name__ == '__main__':
    train_unsupervised_model()
