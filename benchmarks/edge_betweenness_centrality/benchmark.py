import random
import networkx as nx
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import numpy as np


def generate_graph(num_nodes, p=None):
    """
    Generate a connected random graph by first creating a random spanning tree, then
    adding extra edges with probability p. Guarantees connectivity without rejection loop.
    """
    if p is None:
        # Use connectivity threshold approx log(n)/n to encourage extra edges
        p = (np.log(num_nodes) + 0.1) / num_nodes
    # Start with a random tree to ensure connectivity
    G = nx.random_unlabeled_tree(num_nodes)
    # For each possible non-tree edge, add with probability p
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if not G.has_edge(i, j) and random.random() < p:
                G.add_edge(i, j)
    return G


def benchmark_models(models, n_graphs=1000, node_range=(6, 140)):
    """
    models:       dict[name -> adapter], where
                  - adapter.predict(G) -> 1D numpy array of preds
                    OR
                  - adapter(G) -> (preds, targets)
    gen_graph_fn: fn(n_nodes) -> connected nx.Graph
    n_graphs:     how many random graphs per model
    node_range:   (min_nodes, max_nodes)
    """
    results = {}

    for name, adapter in models.items():
        all_preds, all_targets = [], []

        for _ in range(n_graphs):
            # 1) pick size & generate
            n = random.randint(*node_range)
            G = generate_graph(n)

            # 2) compute true betweenness
            eb = nx.edge_betweenness_centrality(G)
            targets = np.array([eb[e] for e in G.edges()], dtype=np.float32)

            # 3) get predictions
            if hasattr(adapter, 'predict'):
                preds = adapter.predict(G)
            else:
                preds, _ = adapter(G)

            # 4) tensor → numpy if needed
            if hasattr(preds, 'cpu'):
                preds = preds.cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets)

        # 5) flatten
        P = np.concatenate(all_preds)
        T = np.concatenate(all_targets)

        # 6) metrics
        mse = mean_squared_error(T, P)
        p = pearsonr(T, P)[0] if T.size and P.size else np.nan
        s = spearmanr(T, P)[0] if T.size and P.size else np.nan

        results[name] = (mse, p, s)

    # 7) print summary
    print(f"\n{'Model':<12} {'MSE':<12} {'Pearson':<10} {'Spearman':<10}")
    for nm, (m, p, s) in results.items():
        print(f"{nm:<12} {m:<12.6f} {p:<10.4f} {s:<10.4f}")

    return results
