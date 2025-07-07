import random
import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigs


def compute_edge_features(A):
    N = A.shape[0]
    f0 = A.flatten()
    deg = A.sum(axis=1)
    f1 = np.repeat(deg, N);    f2 = np.tile(deg, N)
    As = [np.linalg.matrix_power(A, k) for k in range(2, 6)]
    f3_to_f6 = [M.flatten() for M in As]
    vals, vecs = eigs(A + np.eye(N)*1e-3, k=1, which='LM')
    v1 = vecs[:,0].real
    f7 = np.repeat(v1, N);     f8 = np.tile(v1, N)
    G = nx.from_numpy_array(A)
    clustering = np.fromiter(nx.clustering(G).values(), dtype=float)
    f9  = np.repeat(clustering, N);    f10 = np.tile(clustering, N)
    X = np.vstack([f0, f1, f2, *f3_to_f6, f7, f8, f9, f10]).T
    return X, N


def cycle_graph_edges(L):
    """Return the canonical set of edges of an L-cycle."""
    return set(tuple(sorted(e)) for e in nx.cycle_graph(L).edges())


def cycle_incomplete_adj(L, drop_fraction=0.3):
    G = nx.cycle_graph(L)
    orig_edges = list(G.edges())
    k = max(1, int(np.ceil(len(orig_edges) * drop_fraction)))
    dropped = random.sample(orig_edges, k)
    for u,v in dropped:
        G.remove_edge(u, v)
    A_inc = nx.to_numpy_array(G, dtype=int)
    return A_inc, dropped, orig_edges


def benchmark_precision_recall(adapter_fn,
                               num_graphs=1000,
                               drop_fraction=0.3,
                               seed=42):
    random.seed(seed)
    np.random.seed(seed)

    shapes = [3, 4, 5, 6]
    total_TP = total_FP = total_FN = 0

    for _ in range(num_graphs):
        L = random.choice(shapes)
        A_inc, dropped, orig_edges = cycle_incomplete_adj(L, drop_fraction)

        # 1) get shared features
        X, N = compute_edge_features(A_inc)

        # 2) what your model predicts as present
        pred = set(tuple(sorted(e)) for e in adapter_fn(X, N))

        # 3) canonical ground truth
        GT = cycle_graph_edges(L)

        # 4) accumulate
        TP = len(pred & GT)
        FP = len(pred - GT)
        FN = len(GT - pred)

        total_TP += TP
        total_FP += FP
        total_FN += FN

    # Avoid division by zero
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP)>0 else 0.0
    recall    = total_TP / (total_TP + total_FN) if (total_TP + total_FN)>0 else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0

    print(f"Precision = {precision:.2%}")
    print(f"Recall    = {recall:.2%}")
    print(f"Accuracy    = {recall/precision:.2%}")
    print(f"F1 score  = {f1:.2%}")

    return precision, recall, f1
