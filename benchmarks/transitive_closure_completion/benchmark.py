import networkx as nx
import numpy as np
import time
from typing import List, Tuple, Callable, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


def compute_k_hop_reachability(adj: np.ndarray, k: int) -> np.ndarray:
    n = adj.shape[0]
    reach = np.zeros((n, n), dtype=bool)
    A_power = adj.astype(bool)
    for _ in range(1, k+1):
        reach |= A_power
        A_power = A_power.dot(adj.astype(bool)) > 0
    np.fill_diagonal(reach, False)
    return reach


def generate_transitive_closure_graphs(
    num_graphs: int,
    min_nodes: int = 6,
    max_nodes: int = 140,
    missing_pct: float = 0.2,
    k: int = 10,
    expected_out_degree: Tuple[float, float] = (3.0, 6.0),
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate a benchmark dataset of directed graphs with missing k-hop transitive closure edges,
    ensuring no original graph has paths longer than k.

    Each graph is created by:
      1. Sampling a random DAG via node permutation and Bernoulli edge sampling.
      2. Rejecting and resampling if its longest path length > k.
      3. Computing the full k-hop transitive closure.
      4. Isolating closure-only edges and randomly hiding a fraction.

    Args:
        expected_out_degree: up to n = 140, (4,8) works well for k = 10,
                            (1, 2.4) for k = 3 and (8, 16) for k = 20
        num_graphs: number of graphs to generate.
        min_nodes: minimum number of nodes per graph.
        max_nodes: maximum number of nodes per graph.
        missing_pct: fraction of closure edges to hide from the input.
        k: maximum path length for closure and to bound graph depth.

    Returns:
        inputs: list of (n, n) float32 adjacency matrices with missing closures.
        targets: list of (n, n) float32 matrices of full closure-only edges.
    """
    inputs, targets = [], []
    elow, ehigh = expected_out_degree
    pbar = tqdm(total=num_graphs, unit="graph")

    while len(inputs) < num_graphs:
        # 1) Sample base DAG with p scaled by n
        n = np.random.randint(min_nodes, max_nodes + 1)
        perm = np.random.permutation(n)
        A = np.zeros((n, n), dtype=bool)
        p = np.random.uniform(elow, ehigh) / max(1, n - 1)  # ← scale by n
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.rand() < p:
                    A[perm[i], perm[j]] = True

        # 2) Reject if longest path > k (unchanged)
        G = nx.DiGraph(A.astype(int))
        try:
            if nx.algorithms.dag.dag_longest_path_length(G) > k:
                continue
        except nx.NetworkXUnfeasible:
            continue

        # 3)–(5) unchanged (closure, hide fraction, build inputs/targets)
        closure = compute_k_hop_reachability(A, k)
        closure_only = closure & (~A)
        mask = (np.random.rand(n, n) < (1 - missing_pct))
        present_closure = closure_only & mask
        A_input = (A | present_closure).astype(np.float32)
        target = closure_only.astype(np.float32)

        A_input.setflags(write=False)
        target.setflags(write=False)
        inputs.append(A_input)
        targets.append(target)
        pbar.update(1)

    pbar.close()
    return inputs, targets


def generate_random_graphs(
    num_graphs: int,
    min_nodes: int = 6,
    max_nodes: int = 140,
    edge_prob: float = 0.1
) -> List[np.ndarray]:
    inputs: List[np.ndarray] = []
    for _ in range(num_graphs):
        n = np.random.randint(min_nodes, max_nodes + 1)
        A = (np.random.rand(n, n) < edge_prob).astype(np.float32)
        np.fill_diagonal(A, 0)
        inputs.append(A)
    return inputs


def evaluate_model(
    model_fn: Callable[[np.ndarray], np.ndarray],
    inputs: List[np.ndarray],
    targets: List[np.ndarray],
    threshold: float = 0.5
) -> Dict[str, float]:
    all_preds = []
    all_targets = []
    times = []
    for A_input, target in zip(inputs, targets):
        start = time.time()
        pred = model_fn(A_input)
        times.append(time.time() - start)
        logits = pred if isinstance(pred, np.ndarray) else pred.astype(np.float32)
        pred_bin = (logits >= threshold).astype(np.int8)
        all_preds.extend(pred_bin.flatten().tolist())
        all_targets.extend(target.astype(np.int8).flatten().tolist())
    metrics: Dict[str, float] = {}
    metrics['accuracy'] = accuracy_score(all_targets, all_preds)
    metrics['precision'] = precision_score(all_targets, all_preds, zero_division=0)
    metrics['recall'] = recall_score(all_targets, all_preds, zero_division=0)
    metrics['f1'] = f1_score(all_targets, all_preds, zero_division=0)
    metrics['avg_inference_time'] = float(np.mean(times))
    return metrics
