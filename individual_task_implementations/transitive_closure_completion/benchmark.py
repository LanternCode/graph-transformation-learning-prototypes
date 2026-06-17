import time
import networkx as nx
import numpy as np
from typing import Callable, Dict, List, Tuple
from sklearn.metrics import accuracy_score, auc, f1_score, precision_recall_curve, precision_score, recall_score
from tqdm import tqdm


def compute_k_hop_reachability(adj: np.ndarray, k: int) -> np.ndarray:
    """
    Compute the directed k-hop reachability matrix for an adjacency matrix.

    Args:
        adj: Square adjacency matrix whose nonzero entries represent directed edges.
        k: Maximum path length to include in the reachability computation.

    Returns:
        A boolean matrix with True where one node can reach another within k hops,
        excluding diagonal self-reachability entries.
    """
    n = adj.shape[0]
    reach = np.zeros((n, n), dtype=bool)
    A_power = adj.astype(bool)
    for _ in range(1, k + 1):
        reach |= A_power
        A_power = A_power.dot(adj.astype(bool)) > 0
    np.fill_diagonal(reach, False)
    return reach


def non_diagonal_mask(size: int) -> np.ndarray:
    """
    Build a mask that excludes diagonal matrix entries.

    Args:
        size: Number of rows and columns in the square matrix to mask.

    Returns:
        A boolean matrix with False on the diagonal and True elsewhere.
    """
    mask = np.ones((size, size), dtype=bool)
    np.fill_diagonal(mask, False)
    return mask


def flatten_off_diagonal(values: np.ndarray) -> np.ndarray:
    """
    Flatten only the non-diagonal entries from a square matrix.

    Args:
        values: Square matrix containing predictions, labels, or scores.

    Returns:
        A one-dimensional NumPy array containing all off-diagonal entries.
    """
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"Expected a square 2D matrix, got shape {values.shape}")
    return values[non_diagonal_mask(values.shape[0])]


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
        num_graphs: Number of graphs to generate.
        min_nodes: Minimum number of nodes per graph.
        max_nodes: Maximum number of nodes per graph.
        missing_pct: Fraction of closure edges to hide from the input.
        k: Maximum path length for closure and to bound graph depth.
        expected_out_degree: Lower and upper expected out-degree bounds used to
            scale random DAG edge probability by graph size.

    Returns:
        A tuple containing inputs and targets. Inputs are float32 adjacency matrices
        with missing closures, and targets are float32 matrices of full closure-only edges.
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


def collect_probability_labels(
    probability_fn: Callable[[np.ndarray], np.ndarray],
    inputs: List[np.ndarray],
    targets: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect off-diagonal probabilities and labels for a model over many graphs.

    Args:
        probability_fn: Function that accepts one input adjacency matrix and returns
            a same-shaped matrix of predicted probabilities.
        inputs: List of input adjacency matrices.
        targets: List of target matrices aligned with inputs.

    Returns:
        A tuple of one-dimensional arrays: predicted probabilities and labels for
        all off-diagonal entries across all graphs.
    """
    probs, labels = [], []
    for A_input, target in zip(inputs, targets):
        pred = probability_fn(A_input)
        if hasattr(pred, "detach"):
            pred = pred.detach().cpu().numpy()
        pred = np.asarray(pred, dtype=np.float32)
        target_arr = np.asarray(target, dtype=np.float32)
        if pred.shape != target_arr.shape:
            raise ValueError(f"Prediction shape {pred.shape} does not match target shape {target_arr.shape}")
        probs.extend(flatten_off_diagonal(pred).tolist())
        labels.extend(flatten_off_diagonal(target_arr).tolist())
    return np.asarray(probs, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def precision_recall_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute precision-recall AUC from probabilities and binary labels.

    Args:
        probs: One-dimensional array of predicted probabilities.
        labels: One-dimensional array of binary target labels.

    Returns:
        The area under the precision-recall curve.
    """
    precision, recall, _ = precision_recall_curve(labels, probs)
    return float(auc(recall, precision))


def find_best_f1_threshold(probs: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Find the probability threshold with the best F1 score.

    Args:
        probs: One-dimensional array of predicted probabilities.
        labels: One-dimensional array of binary target labels.

    Returns:
        A tuple containing the best F1 score and the threshold that achieved it.
    """
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    all_thresholds = np.append(thresholds, 1.0)
    best_f1, best_thresh = 0.0, 0.5
    for p, r, t in zip(precision, recall, all_thresholds):
        f1 = 2 * p * r / (p + r + 1e-12)
        if f1 > best_f1:
            best_f1, best_thresh = float(f1), float(t)
    return best_f1, best_thresh


def evaluate_model(
    model_fn: Callable[[np.ndarray], np.ndarray],
    inputs: List[np.ndarray],
    targets: List[np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate a model on off-diagonal entries of transitive-closure targets.

    Args:
        model_fn: Function that accepts an input adjacency matrix and returns a
            same-shaped matrix of probabilities or scores.
        inputs: List of input adjacency matrices.
        targets: List of target matrices aligned with inputs.
        threshold: Probability cutoff used to convert model outputs to binary predictions.

    Returns:
        Dictionary with accuracy, precision, recall, F1, and average inference time.
    """
    all_preds = []
    all_targets = []
    times = []
    for A_input, target in zip(inputs, targets):
        start = time.time()
        pred = model_fn(A_input)
        times.append(time.time() - start)
        if hasattr(pred, "detach"):
            pred = pred.detach().cpu().numpy()
        logits = np.asarray(pred, dtype=np.float32)
        target_arr = np.asarray(target, dtype=np.int8)
        if logits.shape != target_arr.shape:
            raise ValueError(f"Prediction shape {logits.shape} does not match target shape {target_arr.shape}")
        pred_bin = (logits >= threshold).astype(np.int8)
        all_preds.extend(flatten_off_diagonal(pred_bin).tolist())
        all_targets.extend(flatten_off_diagonal(target_arr).tolist())
    metrics: Dict[str, float] = {}
    metrics["accuracy"] = accuracy_score(all_targets, all_preds)
    metrics["precision"] = precision_score(all_targets, all_preds, zero_division=0)
    metrics["recall"] = recall_score(all_targets, all_preds, zero_division=0)
    metrics["f1"] = f1_score(all_targets, all_preds, zero_division=0)
    metrics["avg_inference_time"] = float(np.mean(times))
    return metrics
