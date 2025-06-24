import networkx as nx
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# === CONFIG ===
NUM_TRAIN = 900
NUM_TEST = 100
MIN_NODES = 6
MAX_NODES = 140
EPOCHS = 5


# === UTIL ===
def edge_in_cycle(graph, edge):
    try:
        cycles = nx.find_cycle(graph)
        return edge in cycles or (edge[1], edge[0]) in cycles
    except nx.NetworkXNoCycle:
        return False


# === GRAPH GENERATION ===
def generate_graph_pair(epoch=0):
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
    num_extra = min(1 + epoch, len(base_edges))
    extra_edges = candidates[:num_extra]

    full_graph = nx.Graph()
    full_graph.add_nodes_from(tree.nodes())
    full_graph.add_edges_from(base_edges + extra_edges)

    label = {}
    for edge in full_graph.edges():
        label[edge] = 1 if edge_in_cycle(full_graph, edge) else 0

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


# === MODEL 1: SUPERVISED ===
def train_supervised_model():
    best_acc = 0
    best_model = None

    for epoch in range(EPOCHS):
        X_train, y_train = [], []
        for _ in range(NUM_TRAIN):
            G, labels = generate_graph_pair(epoch)
            feats, edges = extract_features(G)
            X_train.extend(feats)
            y_train.extend([labels[e] for e in edges])

        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        preds = clf.predict(X_train)
        acc = accuracy_score(y_train, preds)
        print(f"[Model 1][Epoch {epoch+1}] Training Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = clf
            joblib.dump(best_model, "model1_best.pth")

    # Final test evaluation
    X_test, y_test = [], []
    test_graphs = []  # NEW: to store graphs and their edge lists
    for _ in range(NUM_TEST):
        G, labels = generate_graph_pair(EPOCHS)
        feats, edges = extract_features(G)
        X_test.extend(feats)
        y_test.extend([labels[e] for e in edges])
        test_graphs.append((G, edges))  # Store graph and edge list

    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[Model 1] Final Test Accuracy: {acc:.4f}")
    print(f"[Model 1] Predictions count: 0s = {(preds == 0).sum()}, 1s = {(preds == 1).sum()}")

    # Optional: Count how many full test graphs became acyclic after predicted edge removals
    idx = 0
    cycle_free_count = 0
    for G, edges in test_graphs:
        num_edges = len(edges)
        pred_sub = preds[idx:idx + num_edges]
        to_remove = [e for i, e in enumerate(edges) if pred_sub[i] == 1]
        G_copy = G.copy()
        G_copy.remove_edges_from(to_remove)
        try:
            nx.find_cycle(G_copy)
        except nx.NetworkXNoCycle:
            cycle_free_count += 1
        idx += num_edges

    print(f"[Model 1] Final Acyclic Graphs after edge removal: {cycle_free_count} / {NUM_TEST}\n")


if __name__ == '__main__':
    train_supervised_model()
