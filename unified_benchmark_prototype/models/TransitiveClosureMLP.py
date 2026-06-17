"""
Unfinished framework prototype: recurrent MLP for transitive closure.

This file defines and trains a recurrent closure model for the prototype
framework. The model learns a shared update step over adjacency-derived features
[A, P, A_rw P, P A_rw], where P is the current predicted closure matrix and
A_rw is a row-normalized adjacency matrix. The update is applied repeatedly to
approximate k-hop transitive-closure completion.

This framework prototype was abandoned in favour of the later standalone
implementation. It is archived for portfolio purposes to document the early
iterative neural approach to transitive-closure learning.
"""
import math
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim, nn
from typing import Tuple, Sequence

# import components
from unified_benchmark.benchmark.benchmark_manager import BenchmarkManager
from unified_benchmark.benchmark.tasks.transitive_closure import TransitiveClosureTask


class RecurrentClosure(nn.Module):
    """
    Learn a single 'closure step' and apply it T times (shared weights).
    Input:  A in {0,1}^{B,N,N} (float32)
    Output: logits S_T in R^{B,N,N}
    """
    def __init__(self, hidden=64, T=10, undirected=False, normalize=True, step_size=0.5):
        super().__init__()
        self.T = T
        self.undirected = undirected
        self.normalize = normalize
        self.step_size = step_size
        # features per edge: [A_ij, P_ij, (A_rw P)_ij, (P A_rw)_ij]
        self.fc1 = nn.Linear(4, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def _row_norm(self, A):
        # A: [B,N,N], row-normalize -> D^{-1} A
        deg = A.sum(-1)                     # [B,N]
        d_inv = (deg + 1e-6).reciprocal()   # [B,N]
        Dinv = torch.diag_embed(d_inv)      # [B,N,N]
        return torch.bmm(Dinv, A)

    def step(self, A, S_logits):
        P = torch.sigmoid(S_logits)           # [B,N,N]

        if self.normalize:
            A_rw = self._row_norm(A)
        else:
            A_rw = A

        AP = torch.bmm(A_rw, P)
        PA = torch.bmm(P, A_rw)

        feats = torch.stack([A, P, AP, PA], dim=-1)  # [B,N,N,4]
        b, n, _, c = feats.shape
        x = feats.view(b, n*n, c)
        x = F.relu(self.fc1(x))
        delta = self.fc2(x).view(b, n, n)            # logits update

        S_next = S_logits + self.step_size * delta
        if self.undirected:
            S_next = 0.5 * (S_next + S_next.transpose(1, 2))
        return S_next

    def forward(self, A):
        # stronger initialization: ~0.9 on observed edges, ~1e-3 elsewhere
        p_edge = 0.9
        p_none = 1e-3
        P0 = A * p_edge + (1.0 - A) * p_none
        S = torch.logit(torch.clamp(P0, 1e-6, 1 - 1e-6))
        for _ in range(self.T):
            S = self.step(A, S)
        return S  # logits


# ----------------------------
# Helpers
# ----------------------------
def _tensor_from_numpy(x: np.ndarray, device: torch.device) -> torch.Tensor:
    # Always copy to avoid NumPy non-writable warnings
    return torch.tensor(x, dtype=torch.float32, device=device)


def _bce_logits_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_diag: bool = True, pos_weight: float = 1.0) -> torch.Tensor:
    """
    logits, labels: (N,N) or (1,N,N). We handle either; loss computed over off-diagonal.
    pos_weight: scalar > 0 for class imbalance (neg/pos).
    """
    if logits.dim() == 3 and logits.size(0) == 1:
        logits = logits.squeeze(0)
    if labels.dim() == 3 and labels.size(0) == 1:
        labels = labels.squeeze(0)

    N = logits.size(-1)
    mask = torch.ones((N, N), dtype=torch.bool, device=logits.device)
    if ignore_diag:
        mask.fill_(True)
        mask.fill_diagonal_(False)

    logits_m = logits[mask]
    labels_m = labels[mask]

    # BCEWithLogitsLoss supports a scalar pos_weight to rebalance
    return F.binary_cross_entropy_with_logits(
        logits_m, labels_m, pos_weight=torch.tensor(pos_weight, device=logits.device)
    )


def _estimate_pos_weight(label_mats: Sequence[np.ndarray], eps: float = 1e-6) -> float:
    """
    Global pos_weight ~ (#neg / #pos) over all off-diagonal entries in the training set.
    """
    pos = 0
    neg = 0
    for L in label_mats:
        n = L.shape[0]
        m = np.ones((n, n), dtype=bool)
        np.fill_diagonal(m, False)
        y = (L > 0)[m]
        pos += int(y.sum())
        neg += int((~y).sum())
    return float(neg / max(1, pos)) if pos > 0 else 1.0


def _eval_split(model, task, graphs: Tuple[np.ndarray, ...], labels: Tuple[np.ndarray, ...], device: torch.device, verbose=False) -> Tuple[float, float]:
    """
    Returns (avg_loss, micro_f1). Uses task.evaluate for metrics.
    """
    model.eval()
    preds = []
    losses = []
    with torch.no_grad():
        for A_np, L_np in zip(graphs, labels):
            A = _tensor_from_numpy(A_np, device).unsqueeze(0)   # (1,N,N)
            y = _tensor_from_numpy(L_np, device)                # (N,N)
            out = model(A)                                      # logits (1,N,N)
            loss = _bce_logits_loss(out, y, ignore_diag=task.ignore_diagonal, pos_weight=1.0)
            losses.append(loss.item())
            preds.append(out.squeeze(0).detach().cpu().numpy()) # keep logits; task assumes logits
    micro_f1 = task.evaluate(preds, labels, verbose=verbose)  # prints a small report; returns micro-F1
    return float(np.mean(losses)) if losses else 0.0, micro_f1


# ----------------------------
# Training
# ----------------------------
def train_transitive_closure(
    manager,                          # BenchmarkManager
    task,                             # TransitiveClosureTask
    model=None,                       # RecurrentClosure (if None, we create one)
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    save_path: str = "recurrent_best.pth",
    use_amp: bool = True,
) -> None:
    """
    End-to-end training on CUDA (if available), saving the best model by validation micro-F1.
    Prints train/val loss every ~10 epochs (or every epoch if epochs < 10), and
    prints test loss + metrics at the end.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    # 1) Build splits fresh
    (g_tr, l_tr), (g_va, l_va), (g_te, l_te) = manager.provide_splits(
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        shuffle=True, compute_features=False, prepackage_dataset=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Model
    if model is None:
        model = RecurrentClosure(T=task.k, undirected=False, normalize=True, step_size=0.5)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    # 3) Global class imbalance reweight
    pos_weight = _estimate_pos_weight(l_tr)
    print(f"Using pos_weight ~ {pos_weight:.2f} for BCEWithLogits.")

    # 4) Training loop
    best_val_f1 = -1.0
    best_epoch  = -1
    print_every = max(1, epochs // 10)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for A_np, L_np in zip(g_tr, l_tr):
            A = _tensor_from_numpy(A_np, device).unsqueeze(0)   # (1,N,N)
            y = _tensor_from_numpy(L_np, device)                # (N,N)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                out = model(A)                                  # (1,N,N) logits
                loss = _bce_logits_loss(out, y, ignore_diag=task.ignore_diagonal, pos_weight=pos_weight)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())

        # Validation: compute loss + metrics
        val_loss, val_f1 = _eval_split(model, task, g_va, l_va, device, verbose=False)

        # Save best by validation micro-F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)  # save weights ONLY
            # optional: also save a tiny sidecar with threshold/k if you want
            # json.dump({'k': task.k, 'threshold': task.threshold}, open(save_path + ".meta.json", "w"))

        # Logging cadence
        if (epoch % print_every == 0) or (epoch == 1) or (epoch == epochs):
            tr_loss = float(np.mean(epoch_losses)) if epoch_losses else math.nan
            print(f"[Epoch {epoch:03d}] train_loss={tr_loss:.6f}  val_loss={val_loss:.6f}  val_microF1={val_f1:.4f}")

    print(f"\nBest validation micro-F1 = {best_val_f1:.4f} at epoch {best_epoch}. Saved to {save_path}.")

    # 5) Final eval on test split (load best weights first for honesty)
    state = torch.load(save_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    test_loss, test_f1 = _eval_split(model, task, g_te, l_te, device, verbose=True)
    print(f"\nTest loss = {test_loss:.6f}")
    # _eval_split already printed the detailed metrics via task.evaluate


if __name__ == "__main__":
    task = TransitiveClosureTask(k=10, threshold=0.5, assume_logits=True, ignore_diagonal=True)
    bench = BenchmarkManager(
        task,
        num_graphs=1000,
        min_nodes=6,
        max_nodes=140,
        graph_config={'expected_out_degree': (4.0, 8.0)}        # p scales with n
    )

    # 1) Train
    model = RecurrentClosure(T=task.k, undirected=False, normalize=True, step_size=0.5)

    train_transitive_closure(
        manager=bench,
        task=task,
        model=model,
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
        epochs=50, lr=1e-3, weight_decay=0.0,
        save_path="recurrent_best.pth",
        use_amp=True
    )
