import random
import networkx as nx
import numpy as np
from model import generate_graph
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr


def benchmark_models(models, n_graphs=1000, node_range=(6, 140)):
    """
    Benchmark edge-betweenness prediction models on shared random graphs.

    Parameters:
        models (dict[str, object]): Mapping from model name to adapter. Each
            adapter must either expose predict(G), returning a one-dimensional
            array of predictions in the same order as list(G.edges()), or be
            callable as adapter(G), returning a tuple whose first item is the
            prediction array.
        n_graphs (int): Number of connected random graphs to generate for the
            benchmark.
        node_range (tuple[int, int]): Inclusive range passed to random.randint
            when sampling each graph size.

    Returns:
        dict[str, tuple[float, float, float]]: Mapping from model name to
            (MSE, Pearson correlation, Spearman correlation) over the shared
            benchmark graphs.
    """
    results = {}
    benchmark_graphs = [generate_graph(random.randint(*node_range)) for _ in range(n_graphs)]

    for name, adapter in models.items():
        all_preds, all_targets = [], []

        for G in benchmark_graphs:
            # 1) compute true betweenness
            eb = nx.edge_betweenness_centrality(G)
            targets = np.array([eb[e] for e in G.edges()], dtype=np.float32)

            # 2) get predictions
            if hasattr(adapter, 'predict'):
                preds = adapter.predict(G)
            else:
                preds, _ = adapter(G)

            # 3) tensor → numpy if needed
            if hasattr(preds, 'cpu'):
                preds = preds.cpu().numpy()
            preds = np.asarray(preds, dtype=np.float32).reshape(-1)

            if len(preds) != len(targets):
                raise ValueError(
                    f"{name} returned {len(preds)} predictions for {len(targets)} edges"
                )

            all_preds.append(preds)
            all_targets.append(targets)

        # 4) flatten
        P = np.concatenate(all_preds)
        T = np.concatenate(all_targets)

        # 5) metrics
        mse = mean_squared_error(T, P)
        p = pearsonr(T, P)[0] if T.size and P.size else np.nan
        s = spearmanr(T, P)[0] if T.size and P.size else np.nan

        results[name] = (mse, p, s)

    # 6) print summary
    print(f"\n{'Model':<12} {'MSE':<12} {'Pearson':<10} {'Spearman':<10}")
    for nm, (m, p, s) in results.items():
        print(f"{nm:<12} {m:<12.6f} {p:<10.4f} {s:<10.4f}")

    return results
