import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from graph_utils import extract_features, generate_graph_pair, is_acyclic

# === CONFIG ===
NUM_TRAIN = 900
NUM_VAL = 100
NUM_TEST = 100
MIN_NODES = 6
MAX_NODES = 140
EPOCHS = 5


def _build_supervised_dataset(num_graphs, pct_extra, include_graphs=False):
    """
    Build a supervised edge-classification dataset from generated graphs.

    Args:
        num_graphs (int): Number of generated graph-label pairs to include.
        pct_extra (float): Percentage of tree edges to add as extra edges.
        include_graphs (bool): Whether to return graph and edge metadata for
            downstream acyclicity evaluation.

    Returns:
        tuple: ``(X, y)`` when ``include_graphs`` is ``False``; otherwise
        ``(X, y, graph_records)``, where ``graph_records`` contains
        ``(graph, edges)`` pairs aligned with the flattened labels.
    """
    X, y = [], []
    graph_records = []

    for _ in range(num_graphs):
        graph, labels = generate_graph_pair(pct_extra, min_nodes=MIN_NODES, max_nodes=MAX_NODES)
        features, edges = extract_features(graph)
        X.extend(features)
        y.extend([labels[edge] for edge in edges])
        if include_graphs:
            graph_records.append((graph, edges))

    if include_graphs:
        return X, y, graph_records
    return X, y


def _count_acyclic_predictions(model, graph_records):
    """
    Count graphs made acyclic by a supervised model's removal predictions.

    Args:
        model (RandomForestClassifier): Trained classifier with a ``predict``
            method that returns ``1`` for removable edges.
        graph_records (list): A list of ``(graph, edges)`` pairs to evaluate.

    Returns:
        int: Number of graphs that are acyclic after predicted edge removals.
    """
    cycle_free_count = 0

    for graph, edges in graph_records:
        features, _ = extract_features(graph)
        preds = model.predict(features)
        to_remove = [edge for idx, edge in enumerate(edges) if preds[idx] == 1]

        graph_copy = graph.copy()
        graph_copy.remove_edges_from(to_remove)
        if is_acyclic(graph_copy):
            cycle_free_count += 1

    return cycle_free_count


# === MODEL 1: SUPERVISED ===
def train_supervised_model():
    """
    Train and evaluate the supervised cycle-edge classifier.

    Args:
        None.

    Returns:
        None: The best validation-selected model is saved to
        ``model1_best.pth`` and evaluation metrics are printed.
    """
    best_acc = -1
    best_model = None

    for epoch in range(EPOCHS):
        X_train, y_train = _build_supervised_dataset(NUM_TRAIN, epoch)
        X_val, y_val = _build_supervised_dataset(NUM_VAL, epoch)

        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        train_preds = clf.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)
        val_preds = clf.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        print(
            f"[Model 1][Epoch {epoch + 1}] "
            f"Training Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = clf
            joblib.dump(best_model, "model1_best.pth")

    X_test, y_test, test_graphs = _build_supervised_dataset(NUM_TEST, EPOCHS, include_graphs=True)
    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[Model 1] Final Test Accuracy: {acc:.4f}")
    print(f"[Model 1] Predictions count: 0s = {(preds == 0).sum()}, 1s = {(preds == 1).sum()}")

    cycle_free_count = _count_acyclic_predictions(best_model, test_graphs)
    print(f"[Model 1] Final Acyclic Graphs after edge removal: {cycle_free_count} / {NUM_TEST}\n")


if __name__ == '__main__':
    train_supervised_model()
