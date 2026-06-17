"""
Unfinished framework prototype: transitive-closure task definition.

This file defines the prototype task object for k-hop transitive closure. Given
an input adjacency matrix, it computes reachability up to a configurable number
of hops and labels the missing closure edges that are reachable but not already
present. It also includes task-specific evaluation logic for thresholding logits
or probabilities and reporting graph-level closure metrics.

This framework prototype was abandoned in favour of the later standalone
implementation. It is archived for portfolio purposes to document the early
framework version of the transitive-closure benchmark.
"""
import numpy as np
from ..task_base import TaskGenerator
from typing import Sequence


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # stable sigmoid for numpy arrays
    out = np.empty_like(x, dtype=np.float32)
    # positive / negative split for numerical stability
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[neg])
    out[neg] = expx / (1.0 + expx)
    return out


def compute_k_hop_reachability(adj: np.ndarray, k: int) -> np.ndarray:
    """
    Boolean reachability within <= k hops on a directed graph.
    Works for arbitrary graphs (with or without cycles).
    - adj: (N,N) numeric array; nonzero = edge present
    - k: maximum path length considered
    Returns:
      reach: bool (N,N) with True if there exists a path of length 1..k
    """
    A = (adj > 0).astype(np.int32)
    n = A.shape[0]
    reach = np.zeros((n, n), dtype=bool)
    if n == 0 or k <= 0:
        return reach
    power = A.copy()
    for _ in range(1, k + 1):
        reach |= power.astype(bool)
        # next-hop paths
        power = (power @ A) > 0
        power = power.astype(np.int32)
    # no self loops in label space
    np.fill_diagonal(reach, False)
    return reach


class TransitiveClosureTask(TaskGenerator):
    """
    Task: Given any directed graph (possibly cyclic), predict the missing edges
    required to make it transitively closed up to K hops.

    Label definition (works for any graph):
        labels = reachability_k(adj) AND NOT adj
    where reachability_k(adj)[i,j] is True iff there exists a path i->...->j
    of length in {1,2,...,K} in the PROVIDED input adjacency `adj`.

    Notes:
    - This definition is consistent even if `adj` already contains some closure edges.
      The labels are simply "the remaining missing closure edges".
    - For DAGs with longest path ≤ K, k-hop closure equals the full transitive closure.
    """

    def __init__(self, k: int = 10, threshold: float = 0.5, assume_logits: bool = True, ignore_diagonal: bool = True):
        """
        Args:
            k: maximum hop length for closure computation.
            threshold: decision threshold in PROBABILITY space (after sigmoid).
            assume_logits: if True, predictions are treated as logits and passed
                           through sigmoid before thresholding. If False, they
                           are treated as probabilities in [0,1].
            ignore_diagonal: if True, exclude i==j from evaluation.
        """
        self.k = int(k)
        self.threshold = float(threshold)
        self.assume_logits = bool(assume_logits)
        self.ignore_diagonal = bool(ignore_diagonal)

    # ------------ Label generation ------------
    def generate_labels(self, adj: np.ndarray) -> np.ndarray:
        """
        Given an input adjacency (float/bool), return a float32 label matrix
        where 1 indicates a closure edge (≤K hops) missing from the input.
        """
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"adj must be square 2D, got shape {adj.shape}")
        closure = compute_k_hop_reachability(adj, self.k)
        base = (adj > 0)
        labels = closure & (~base)
        return labels.astype(np.float32)

    # ------------ Evaluation ------------
    def _binarize_pred(self, arr: np.ndarray) -> np.ndarray:
        """
        Convert raw predictions to boolean decisions using self.threshold.
        - If assume_logits=True, apply sigmoid first.
        - Else, treat as probabilities already.
        """
        if arr.dtype == bool:
            return arr
        x = arr.astype(np.float32, copy=False)
        if self.assume_logits:
            x = _sigmoid(x)
        return x >= self.threshold

    def _mask_offdiag(self, n: int) -> np.ndarray:
        return ~np.eye(n, dtype=bool) if self.ignore_diagonal else np.ones((n, n), dtype=bool)

    def evaluate(self, predictions: Sequence[np.ndarray], labels: Sequence[np.ndarray], verbose: bool = True) -> float:
        """
        Evaluate a sequence of predictions against ground-truth label matrices.
        Returns:
            micro-F1 (float). Also prints a small report with precision/recall/F1.
        Behavior:
            - Accepts predictions as logits (default) or probabilities.
            - Ignores diagonal by default.
            - Computes micro/macro F1, balanced accuracy, exact-graph rate.
        """
        if len(predictions) != len(labels):
            raise ValueError("predictions and labels must have same length")

        TP = FP = FN = TN = 0
        exact = 0
        per_f1 = []

        for pred, y in zip(predictions, labels):
            if pred.shape != y.shape:
                raise ValueError(f"shape mismatch: pred {pred.shape} vs label {y.shape}")
            n = y.shape[0]
            m = self._mask_offdiag(n)

            yb = (y > 0)[m]
            pb = self._binarize_pred(pred)[m]

            tp = int(np.sum(pb & yb))
            fp = int(np.sum(pb & ~yb))
            fn = int(np.sum(~pb & yb))
            tn = int(np.sum(~pb & ~yb))

            TP += tp; FP += fp; FN += fn; TN += tn

            # per-graph F1
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_f1.append(f1)

            exact += int(np.array_equal(pb, yb))

        micro_prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        micro_rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        micro_f1   = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
        tpr = micro_rec
        tnr = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        bal_acc = 0.5 * (tpr + tnr)
        macro_f1 = float(np.mean(per_f1)) if per_f1 else 0.0
        exact_rate = exact / len(predictions) if predictions else 0.0

        if verbose:
            print(
                "Evaluation (TransitiveClosureTask)\n"
                f"- Micro Precision: {micro_prec:.4f}\n"
                f"- Micro Recall:    {micro_rec:.4f}\n"
                f"- Micro F1:        {micro_f1:.4f}\n"
                f"- Macro F1:        {macro_f1:.4f}\n"
                f"- Balanced Acc:    {bal_acc:.4f}\n"
                f"- Exact Graphs:    {exact_rate:.4f}"
            )

        return float(micro_f1)
