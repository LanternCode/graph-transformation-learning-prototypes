import networkx as nx
from unsupervised_model import generate_graph_pair, extract_features


def benchmark_acyclicity(model_adapter, num_graphs=1000, epoch=0):
    """
    Benchmarks how often an unsupervised model produces acyclic graphs.

    Args:
        model_adapter (function): A function that takes (graph, features, edges)
                                  and returns a list of edges to remove.
        num_graphs (int): Number of test graphs to evaluate.
        epoch (int): Controls graph complexity (used in edge injection).

    Returns:
        acyclic_count (int): Number of acyclic outputs.
    """
    acyclic_count = 0

    for _ in range(num_graphs):
        G, _ = generate_graph_pair(epoch)
        feats, edges = extract_features(G)

        # Adapter: return list of edges to remove
        to_remove = model_adapter(G, feats, edges)

        G2 = G.copy()
        G2.remove_edges_from(to_remove)

        try:
            nx.find_cycle(G2)
        except nx.NetworkXNoCycle:
            acyclic_count += 1

    print(f"[Benchmark] Acyclic Graphs: {acyclic_count} / {num_graphs}")
    return acyclic_count


def benchmark_supervised(model, num_graphs=1000, epoch=0):
    """
    Benchmarks a supervised model by evaluating prediction accuracy and
    how often predicted edge removals result in acyclic graphs.

    Args:
        model: Trained classifier with `predict()` method.
        num_graphs: Number of test graphs to evaluate.
        epoch: Graph complexity control (for added edges).

    Returns:
        dict with:
            - 'accuracy': overall edge-label accuracy,
            - 'acyclic_count': number of graphs made acyclic by predicted removals,
            - 'total_graphs': number of graphs evaluated.
    """
    from sklearn.metrics import accuracy_score

    total_correct = 0
    total_preds = 0
    acyclic_count = 0

    for _ in range(num_graphs):
        G, labels = generate_graph_pair(epoch)
        feats, edges = extract_features(G)

        # Predict labels (0 = keep, 1 = remove)
        preds = model.predict(feats)

        # Accuracy per edge
        y_true = [labels[e] for e in edges]
        total_correct += sum(p == t for p, t in zip(preds, y_true))
        total_preds += len(edges)

        # Remove predicted cycle edges and check if graph is acyclic
        to_remove = [e for i, e in enumerate(edges) if preds[i] == 1]
        G_copy = G.copy()
        G_copy.remove_edges_from(to_remove)

        try:
            nx.find_cycle(G_copy)
        except nx.NetworkXNoCycle:
            acyclic_count += 1

    accuracy = total_correct / total_preds
    print(f"[Supervised Benchmark] Edge Accuracy: {accuracy:.4f}")
    print(f"[Supervised Benchmark] Acyclic Graphs: {acyclic_count} / {num_graphs}")

    return {
        'accuracy': accuracy,
        'acyclic_count': acyclic_count,
        'total_graphs': num_graphs
    }
