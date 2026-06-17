import joblib
from benchmark import benchmark_acyclicity, benchmark_supervised
from graph_utils import make_benchmark_graphs


def adapter_rf(model):
    """
    Wrap a random-forest classifier as an edge-removal benchmark adapter.

    Args:
        model: Trained classifier with ``predict_proba`` support.

    Returns:
        function: Adapter that accepts ``(graph, features, edges)`` and returns
        edges whose predicted removal probability is greater than ``0.5``.
    """
    def inner(graph, features, edges):
        """
        Select edges to remove from a graph using classifier probabilities.

        Args:
            graph (networkx.Graph): Graph being evaluated. It is accepted for
                adapter compatibility and is not modified.
            features (numpy.ndarray): Edge feature matrix aligned with ``edges``.
            edges (list): Edge tuples corresponding to the feature rows.

        Returns:
            list: Edges whose predicted removal probability exceeds ``0.5``.
        """
        probs = model.predict_proba(features)[:, 1]
        return [edge for idx, edge in enumerate(edges) if probs[idx] > 0.5]

    return inner


def main():
    """
    Load saved models and run comparable benchmark evaluations.

    Args:
        None.

    Returns:
        None: Benchmark results are printed to stdout.
    """
    benchmark_graphs = make_benchmark_graphs(num_graphs=1000, pct_extra=10)

    print("Benchmark: Supervised Learning")
    model = joblib.load("model1_best.pth")
    benchmark_supervised(model, benchmark_graphs=benchmark_graphs)

    print("Benchmark: Reinforcement Learning")
    model = joblib.load("model2_best.pth")
    benchmark_acyclicity(adapter_rf(model), benchmark_graphs=benchmark_graphs)


if __name__ == "__main__":
    main()
