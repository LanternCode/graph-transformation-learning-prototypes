import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from graph_utils import extract_features, generate_graph_pair, is_acyclic

# === CONFIG ===
NUM_TRAIN = 20
NUM_VAL = 20
NUM_TEST = 100
MIN_NODES = 6
MAX_NODES = 140
EPOCHS = 50


def _collect_edge_records(num_graphs, pct_extra):
    """
    Generate graph records used by the unsupervised prototype trainer.

    Args:
        num_graphs (int): Number of graph records to generate.
        pct_extra (float): Percentage of tree edges to add as extra edges.

    Returns:
        list: A list of ``(graph, features, edges)`` tuples where ``features``
        and ``edges`` are aligned by row/index.
    """
    edge_records = []
    for _ in range(num_graphs):
        graph, _ = generate_graph_pair(pct_extra, min_nodes=MIN_NODES, max_nodes=MAX_NODES)
        features, edges = extract_features(graph)
        edge_records.append((graph, features, edges))
    return edge_records


def _build_random_removal_dataset(edge_records):
    """
    Build a random edge-removal training dataset from graph records.

    Args:
        edge_records (list): A list of ``(graph, features, edges)`` tuples.

    Returns:
        tuple: ``(X, y)`` where ``X`` contains edge feature rows and ``y`` marks
        randomly selected removed edges as ``1`` and kept edges as ``0``.
    """
    X, y = [], []
    for _, features, edges in edge_records:
        probs = np.random.rand(len(edges))
        removed = [edge for idx, edge in enumerate(edges) if probs[idx] > 0.5]
        y.extend([int(edge in removed) for edge in edges])
        X.extend(features)
    return X, y


def _count_acyclic_predictions(model, edge_records):
    """
    Count how many graphs become acyclic after model-selected edge removals.

    Args:
        model (RandomForestClassifier): Trained classifier with
            ``predict_proba`` support.
        edge_records (list): A list of ``(graph, features, edges)`` tuples to
            evaluate.

    Returns:
        tuple: ``(cycle_free_count, pred_zeros, pred_ones)`` with the number of
        acyclic outputs and aggregate prediction counts.
    """
    cycle_free_count = 0
    pred_zeros = 0
    pred_ones = 0

    for graph, features, edges in edge_records:
        probs = model.predict_proba(features)[:, 1]
        to_remove = [edge for idx, edge in enumerate(edges) if probs[idx] > 0.5]
        pred_ones += len(to_remove)
        pred_zeros += len(edges) - len(to_remove)

        graph_copy = graph.copy()
        graph_copy.remove_edges_from(to_remove)
        if is_acyclic(graph_copy):
            cycle_free_count += 1

    return cycle_free_count, pred_zeros, pred_ones


# === MODEL 2: STRUCTURAL OPTIMIZATION ===
def train_unsupervised_model():
    """
    Train and evaluate the unsupervised structural-optimization prototype.

    Args:
        None.

    Returns:
        None: The best validation-selected model is saved to
        ``model2_best.pth`` and evaluation metrics are printed.
    """
    best_score = -1
    best_model = None

    for epoch in range(EPOCHS):
        train_records = _collect_edge_records(NUM_TRAIN, epoch)
        X_train, y_train = _build_random_removal_dataset(train_records)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        val_records = _collect_edge_records(NUM_VAL, epoch)
        cycle_free_count, _, _ = _count_acyclic_predictions(model, val_records)

        print(f"[Model 2][Epoch {epoch + 1}] Validation Acyclic Graphs: {cycle_free_count} / {NUM_VAL}")
        if cycle_free_count > best_score:
            best_score = cycle_free_count
            best_model = model
            joblib.dump(best_model, "model2_best.pth")

    test_records = _collect_edge_records(NUM_TEST, EPOCHS)
    cycle_free_count, pred_zeros, pred_ones = _count_acyclic_predictions(best_model, test_records)

    print(f"[Model 2] Final Test Acyclic Graphs: {cycle_free_count} / {NUM_TEST}")
    print(f"[Model 2] Final Prediction Counts: 0s = {pred_zeros}, 1s = {pred_ones}")


if __name__ == '__main__':
    train_unsupervised_model()
