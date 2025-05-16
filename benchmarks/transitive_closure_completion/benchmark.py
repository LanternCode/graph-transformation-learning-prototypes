import numpy as np
import time
from typing import List, Tuple, Callable, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
    k: int = 2
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    inputs: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    for _ in range(num_graphs):
        n = np.random.randint(min_nodes, max_nodes + 1)
        perm = np.random.permutation(n)
        A = np.zeros((n, n), dtype=bool)
        p = np.random.uniform(0.1, 0.3)
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.rand() < p:
                    A[perm[i], perm[j]] = True
        closure = compute_k_hop_reachability(A, k)
        closure_only = closure & (~A)
        mask = (np.random.rand(n, n) < (1 - missing_pct))
        present_closure = closure_only & mask
        A_input = (A | present_closure).astype(np.float32)
        target = closure_only.astype(np.float32)
        inputs.append(A_input)
        targets.append(target)
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