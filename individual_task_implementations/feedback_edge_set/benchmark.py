import networkx as nx
from graph_utils import extract_features, generate_graph_pair, make_benchmark_graphs


def _resolve_benchmark_graphs(num_graphs, pct_extra, benchmark_graphs):
    """
    Return either supplied benchmark graphs or newly generated graph records.

    Args:
        num_graphs (int): Number of graphs to generate when no graph set is
            supplied.
        pct_extra (float): Percentage of tree edges to add as extra benchmark
            edges when generating graphs.
        benchmark_graphs (list | None): Optional reusable list of
            ``(graph, labels)`` tuples.

    Returns:
        list: A list of ``(graph, labels)`` tuples for benchmark evaluation.
    """
    if benchmark_graphs is not None:
        return benchmark_graphs
    return make_benchmark_graphs(num_graphs=num_graphs, pct_extra=pct_extra)


def benchmark_acyclicity(model_adapter, num_graphs=1000, pct_extra=0, benchmark_graphs=None):
    """
    Benchmark how often an edge-removal adapter produces acyclic graphs.

    Args:
        model_adapter (function): Function that takes ``(graph, features,
            edges)`` and returns a list of edges to remove.
        num_graphs (int): Number of graphs to evaluate when ``benchmark_graphs``
            is not provided.
        pct_extra (float): Percentage of tree edges to add as extra benchmark
            edges when generating graphs.
        benchmark_graphs (list | None): Optional reusable list of
            ``(graph, labels)`` tuples, allowing multiple models to be evaluated
            on the same graph instances.

    Returns:
        int: Number of graphs that are acyclic after predicted edge removals.
    """
    graph_records = _resolve_benchmark_graphs(num_graphs, pct_extra, benchmark_graphs)
    acyclic_count = 0

    for graph, _ in graph_records:
        features, edges = extract_features(graph)
        to_remove = model_adapter(graph, features, edges)

        graph_copy = graph.copy()
        graph_copy.remove_edges_from(to_remove)

        try:
            nx.find_cycle(graph_copy)
        except nx.NetworkXNoCycle:
            acyclic_count += 1

    print(f"[Benchmark] Acyclic Graphs: {acyclic_count} / {len(graph_records)}")
    return acyclic_count


def benchmark_supervised(model, num_graphs=1000, pct_extra=0, benchmark_graphs=None):
    """
    Benchmark a supervised edge-removal classifier.

    Args:
        model: Trained classifier with a ``predict`` method returning edge labels
            where ``1`` means remove and ``0`` means keep.
        num_graphs (int): Number of graphs to evaluate when ``benchmark_graphs``
            is not provided.
        pct_extra (float): Percentage of tree edges to add as extra benchmark
            edges when generating graphs.
        benchmark_graphs (list | None): Optional reusable list of
            ``(graph, labels)`` tuples, allowing direct comparison against other
            models on the same graphs.

    Returns:
        dict: Metrics containing ``accuracy``, ``acyclic_count``, and
        ``total_graphs``.
    """
    graph_records = _resolve_benchmark_graphs(num_graphs, pct_extra, benchmark_graphs)
    total_correct = 0
    total_preds = 0
    acyclic_count = 0

    for graph, labels in graph_records:
        features, edges = extract_features(graph)
        preds = model.predict(features)

        y_true = [labels[edge] for edge in edges]
        total_correct += sum(pred == target for pred, target in zip(preds, y_true))
        total_preds += len(edges)

        to_remove = [edge for idx, edge in enumerate(edges) if preds[idx] == 1]
        graph_copy = graph.copy()
        graph_copy.remove_edges_from(to_remove)

        try:
            nx.find_cycle(graph_copy)
        except nx.NetworkXNoCycle:
            acyclic_count += 1

    accuracy = total_correct / total_preds if total_preds else 0.0
    print(f"[Supervised Benchmark] Edge Accuracy: {accuracy:.4f}")
    print(f"[Supervised Benchmark] Acyclic Graphs: {acyclic_count} / {len(graph_records)}")

    return {
        'accuracy': accuracy,
        'acyclic_count': acyclic_count,
        'total_graphs': len(graph_records),
    }
