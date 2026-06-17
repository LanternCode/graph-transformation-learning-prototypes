import random
import numpy as np
import networkx as nx
from itertools import combinations
from model import extract_edge_features


def compute_edge_features(A):
    """
    Compute benchmark edge features with the shared training feature extractor.

    Args:
        A: Square adjacency matrix for an incomplete cycle graph.

    Returns:
        A tuple ``(X, N)`` where ``X`` has shape ``(N*N, 11)`` and contains the
        flattened edge features, and ``N`` is the number of nodes.
    """
    A = np.asarray(A, dtype=np.float32)
    features = extract_edge_features(A)
    N = A.shape[0]
    X = features.reshape(-1, features.shape[-1])
    return X, N


def cycle_graph_edges(L):
    """
    Build the undirected edge set of the canonical cycle graph.

    Args:
        L: Number of nodes in the cycle.

    Returns:
        A set of sorted ``(u, v)`` tuples representing the cycle edges.
    """
    return set(tuple(sorted(e)) for e in nx.cycle_graph(L).edges())


def cycle_incomplete_adj(L, drop_fraction=0.3):
    """
    Generate an incomplete cycle adjacency matrix by dropping cycle edges.

    Args:
        L: Number of nodes in the cycle graph.
        drop_fraction: Fraction of original cycle edges to remove, rounded up
            with at least one edge removed.

    Returns:
        A tuple ``(A_inc, dropped, orig_edges)`` where ``A_inc`` is the incomplete
        adjacency matrix, ``dropped`` is the list of removed edges, and
        ``orig_edges`` is the original edge list.
    """
    G = nx.cycle_graph(L)
    orig_edges = list(G.edges())
    k = max(1, int(np.ceil(len(orig_edges) * drop_fraction)))
    dropped = random.sample(orig_edges, k)
    for u, v in dropped:
        G.remove_edge(u, v)
    A_inc = nx.to_numpy_array(G, dtype=int)
    return A_inc, dropped, orig_edges


def benchmark_precision_recall(adapter_fn,
                               num_graphs=1000,
                               drop_fraction=0.3,
                               seed=42):
    """
    Evaluate an edge-reconstruction adapter on incomplete cycle graphs.

    Args:
        adapter_fn: Callable that receives ``(X, N)`` and returns predicted edge
            pairs, where ``X`` is the flattened feature matrix and ``N`` is the
            graph size.
        num_graphs: Number of random benchmark graphs to evaluate.
        drop_fraction: Fraction of cycle edges to remove before prediction.
        seed: Random seed used for benchmark graph generation.

    Returns:
        A tuple ``(precision, recall, f1)`` computed over the predicted and
        ground-truth undirected cycle edge sets. The function also prints a
        set-level accuracy over all unordered node pairs and self-loops.
    """
    random.seed(seed)
    np.random.seed(seed)

    shapes = [3, 4, 5, 6]
    total_TP = total_FP = total_FN = total_TN = 0

    for _ in range(num_graphs):
        L = random.choice(shapes)
        A_inc, dropped, orig_edges = cycle_incomplete_adj(L, drop_fraction)

        # 1) get shared features
        X, N = compute_edge_features(A_inc)

        # 2) what your model predicts as present
        pred = set(tuple(sorted(e)) for e in adapter_fn(X, N))

        # 3) canonical ground truth
        GT = cycle_graph_edges(L)
        universe = set(combinations(range(L), 2)) | {(i, i) for i in range(L)}

        # 4) accumulate
        TP = len(pred & GT)
        FP = len(pred - GT)
        FN = len(GT - pred)
        TN = len(universe - pred - GT)

        total_TP += TP
        total_FP += FP
        total_FN += FN
        total_TN += TN

    # Avoid division by zero
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = ((total_TP + total_TN) / (total_TP + total_FP + total_FN + total_TN)
                if (total_TP + total_FP + total_FN + total_TN) > 0 else 0.0)

    print(f"Precision = {precision:.2%}")
    print(f"Recall    = {recall:.2%}")
    print(f"Accuracy  = {accuracy:.2%}")
    print(f"F1 score  = {f1:.2%}")

    return precision, recall, f1
