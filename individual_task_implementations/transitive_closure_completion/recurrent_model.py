import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from torch.utils.data import DataLoader, Dataset
from benchmark import (
    collect_probability_labels,
    evaluate_model,
    find_best_f1_threshold,
    generate_transitive_closure_graphs,
    precision_recall_auc,
)


class ClosureDataset(Dataset):
    """
    Dataset that exposes adjacency matrices and closure targets for recurrent training.

    Args:
        inputs: List of input adjacency matrices.
        targets: List of target closure matrices aligned with inputs.

    Returns:
        Dataset items are adjacency and target tensors for one graph.
    """

    def __init__(self, inputs: List[np.ndarray], targets: List[np.ndarray]):
        """
        Store input and target graph matrices.

        Args:
            inputs: List of input adjacency matrices.
            targets: List of target closure matrices aligned with inputs.

        Returns:
            None.
        """
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        """
        Return the number of graph pairs in the dataset.

        Args:
            None.

        Returns:
            Number of input/target graph pairs.
        """
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve one graph as writable-backed float32 tensors.

        Args:
            idx: Integer index of the graph to retrieve.

        Returns:
            Tuple containing an [N, N] adjacency tensor and [N, N] target tensor.
        """
        A = np.array(self.inputs[idx], dtype=np.float32, copy=True)
        T = np.array(self.targets[idx], dtype=np.float32, copy=True)
        return torch.from_numpy(A), torch.from_numpy(T)


def variable_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad variable-size adjacency and target matrices in a batch.

    Args:
        batch: List of adjacency and target tensors from ClosureDataset.

    Returns:
        Tuple containing padded adjacencies, padded targets, and a valid-entry mask.
    """
    As, Ts = zip(*batch)
    ns = [a.shape[0] for a in As]
    M, B = max(ns), len(batch)

    Ab = torch.zeros(B, M, M)
    Tb = torch.zeros(B, M, M)
    mask = torch.zeros(B, M, M)
    for i, (a, t) in enumerate(zip(As, Ts)):
        n = a.shape[0]
        Ab[i, :n, :n] = a
        Tb[i, :n, :n] = t
        mask[i, :n, :n] = 1.0
    return Ab, Tb, mask


# ---------------- Recurrent closure model (refined) ----------------
class RecurrentClosure(nn.Module):
    """
    Recurrent model that learns a shared closure-update step.

    Args:
        hidden: Width of the hidden layer used in each recurrent step.
        T: Number of recurrent update steps to apply.
        undirected: Whether to symmetrize logits after each step.
        normalize: Whether to row-normalize the adjacency before propagation.
        step_size: Scale applied to each learned logit update.

    Returns:
        A PyTorch module whose forward pass returns logits with shape [B, N, N].
    """

    def __init__(self, hidden: int = 64, T: int = 8, undirected: bool = False, normalize: bool = True, step_size: float = 0.5):
        """
        Initialize recurrent closure update layers and configuration.

        Args:
            hidden: Width of the hidden layer used in each recurrent step.
            T: Number of recurrent update steps to apply.
            undirected: Whether to symmetrize logits after each step.
            normalize: Whether to row-normalize the adjacency before propagation.
            step_size: Scale applied to each learned logit update.

        Returns:
            None.
        """
        super().__init__()
        self.T = T
        self.undirected = undirected
        self.normalize = normalize
        self.step_size = step_size
        # features per edge: [A_ij, P_ij, (A_rw P)_ij, (P A_rw)_ij]
        self.fc1 = nn.Linear(4, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def _row_norm(self, A: torch.Tensor) -> torch.Tensor:
        """
        Row-normalize a batch of adjacency matrices.

        Args:
            A: Tensor with shape [B, N, N].

        Returns:
            Row-normalized tensor with shape [B, N, N].
        """
        deg = A.sum(-1)                     # [B,N]
        d_inv = (deg + 1e-6).reciprocal()   # [B,N]
        Dinv = torch.diag_embed(d_inv)      # [B,N,N]
        return torch.bmm(Dinv, A)

    def step(self, A: torch.Tensor, S_logits: torch.Tensor) -> torch.Tensor:
        """
        Apply one learned closure-update step.

        Args:
            A: Batch of input adjacency matrices with shape [B, N, N].
            S_logits: Current closure logits with shape [B, N, N].

        Returns:
            Updated closure logits with shape [B, N, N].
        """
        P = torch.sigmoid(S_logits)           # [B,N,N]

        if self.normalize:
            A_rw = self._row_norm(A)
        else:
            A_rw = A

        AP = torch.bmm(A_rw, P)
        PA = torch.bmm(P, A_rw)

        feats = torch.stack([A, P, AP, PA], dim=-1)  # [B,N,N,4]
        b, n, _, c = feats.shape
        x = feats.view(b, n * n, c)
        x = F.relu(self.fc1(x))
        delta = self.fc2(x).view(b, n, n)            # logits update

        S_next = S_logits + self.step_size * delta
        if self.undirected:
            S_next = 0.5 * (S_next + S_next.transpose(1, 2))
        return S_next

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """
        Predict closure logits by applying recurrent closure updates.

        Args:
            A: Batch of input adjacency matrices with shape [B, N, N].

        Returns:
            Logit tensor with shape [B, N, N].
        """
        # stronger initialization: ~0.9 on observed edges, ~1e-3 elsewhere
        p_edge = 0.9
        p_none = 1e-3
        P0 = A * p_edge + (1.0 - A) * p_none
        S = torch.logit(torch.clamp(P0, 1e-6, 1 - 1e-6))
        for _ in range(self.T):
            S = self.step(A, S)
        return S  # logits


# ---------------- PR-AUC helper ----------------
@torch.no_grad()
def predict_probabilities(model: nn.Module, A: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Predict closure probabilities for one graph with the recurrent model.

    Args:
        model: Trained RecurrentClosure model.
        A: Input adjacency matrix for one graph.
        device: Torch device used for inference.

    Returns:
        Probability matrix with shape [N, N].
    """
    A = np.array(A, dtype=np.float32, copy=True)
    featA = torch.from_numpy(A).unsqueeze(0).to(device)  # [1,N,N]
    logits = model(featA).squeeze(0)                     # [N,N]
    return torch.sigmoid(logits).detach().cpu().numpy()


@torch.no_grad()
def pr_auc_on_holdout(model: nn.Module, inputs: List[np.ndarray], targets: List[np.ndarray], device: torch.device) -> float:
    """
    Compute off-diagonal PR-AUC for a held-out graph set.

    Args:
        model: RecurrentClosure model to evaluate.
        inputs: List of input adjacency matrices.
        targets: List of target closure matrices aligned with inputs.
        device: Torch device used for inference.

    Returns:
        Precision-recall AUC over off-diagonal entries.
    """
    model.eval()
    probs, labels = collect_probability_labels(
        lambda A: predict_probabilities(model, A, device),
        inputs,
        targets,
    )
    return precision_recall_auc(probs, labels)


# ---------------- Train ----------------
def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    val_in: List[np.ndarray],
    val_tg: List[np.ndarray],
    epochs: int,
    device: torch.device,
    ckpt_path: str,
) -> float:
    """
    Train the recurrent closure model and checkpoint the best validation PR-AUC state.

    Args:
        model: RecurrentClosure model to train.
        loader: DataLoader that yields padded adjacency, target, and mask batches.
        optimizer: Torch optimizer used to update model weights.
        val_in: Validation input adjacency matrices.
        val_tg: Validation target matrices aligned with val_in.
        epochs: Number of training epochs.
        device: Torch device used for training and validation.
        ckpt_path: Path where the best model state dict is saved.

    Returns:
        Best validation PR-AUC observed during training.
    """
    best_val = -1.0
    eye_cache = {}  # N -> eye[N]

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, total_items = 0.0, 0.0

        for A, tgt, mask in loader:
            A, tgt, mask = A.to(device), tgt.to(device), mask.to(device)

            # drop diagonal from loss (no self-loops)
            N = A.size(1)
            if N not in eye_cache:
                eye_cache[N] = torch.eye(N, device=device)
            diag_mask = (1.0 - eye_cache[N]).unsqueeze(0)  # [1,N,N]
            eff_mask = mask * diag_mask

            logits = model(A)                               # [B,N,N]

            # imbalance-aware BCE with logits
            pos = (tgt * eff_mask).sum()
            neg = (eff_mask.sum() - pos).clamp(min=1.0)
            pos_w = (neg / pos.clamp(min=1.0)).detach()    # scalar tensor
            loss_mat = F.binary_cross_entropy_with_logits(
                logits, tgt, reduction="none", pos_weight=pos_w
            )
            loss = (loss_mat * eff_mask).sum() / eff_mask.sum().clamp(min=1.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * eff_mask.sum().item()
            total_items += eff_mask.sum().item()

        val_auc = pr_auc_on_holdout(model, val_in, val_tg, device)
        if val_auc > best_val:
            best_val = val_auc
            torch.save(model.state_dict(), ckpt_path)

        if ep % 10 == 0:
            avg_loss = total_loss / max(total_items, 1.0)
            print(f"Epoch {ep:03d}: Avg Loss={avg_loss:.4f}  Val PR-AUC={val_auc:.4f}")

    print(f"Best Val PR-AUC: {best_val:.4f}  (model saved to {ckpt_path})")
    return best_val


# ---------------- Main experiment ----------------
def main() -> None:
    """
    Run the recurrent transitive-closure experiment.

    Args:
        None.

    Returns:
        None. The function trains a model, prints validation/test metrics, and saves checkpoints.
    """
    # ----- config -----
    EPOCHS = 100
    BATCH = 16
    LR = 1e-3
    T_STEPS = 10               # try aligning with the largest k at test
    UNDIRECTED = False          # set True if your graphs are undirected
    NORMALIZE = True           # row-normalize A for stable path signals
    STEP_SIZE = 0.5            # 0.25..1.0; smaller = safer updates
    CKPT = "recurrent_best_v2.pth"

    # data
    train_in, train_tg = generate_transitive_closure_graphs(
        num_graphs=1000, min_nodes=6, max_nodes=140, missing_pct=0.2, k=T_STEPS
    )
    val_in, val_tg = generate_transitive_closure_graphs(
        num_graphs=200, min_nodes=6, max_nodes=140, missing_pct=0.2, k=T_STEPS
    )
    test_in, test_tg = generate_transitive_closure_graphs(
        num_graphs=200, min_nodes=6, max_nodes=140, missing_pct=0.2, k=T_STEPS
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ClosureDataset(train_in, train_tg)
    loader = DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=True,
        collate_fn=variable_collate,
        pin_memory=(device.type == "cuda"),
    )

    # model
    model = RecurrentClosure(
        hidden=64, T=T_STEPS, undirected=UNDIRECTED, normalize=NORMALIZE, step_size=STEP_SIZE
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # train + save best
    best_val = train(model, loader, optimizer, val_in, val_tg, EPOCHS, device, CKPT)

    # load best and tune threshold on validation data
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    val_probs, val_labels = collect_probability_labels(
        lambda A: predict_probabilities(model, A, device),
        val_in,
        val_tg,
    )
    best_f1, best_thresh = find_best_f1_threshold(val_probs, val_labels)
    print(f"Validation Max F1 = {best_f1:.3f} at thresh = {best_thresh:.3f}")

    # test PR-AUC
    test_probs, test_labels = collect_probability_labels(
        lambda A: predict_probabilities(model, A, device),
        test_in,
        test_tg,
    )
    pr_auc = precision_recall_auc(test_probs, test_labels)
    print(f"Test PR AUC = {pr_auc:.3f}")

    # eval via helper using the validation-tuned threshold
    metrics = evaluate_model(
        model_fn=lambda A: predict_probabilities(model, A, device),
        inputs=test_in,
        targets=test_tg,
        threshold=best_thresh,
    )
    print("Evaluation metrics at validation-tuned threshold:", metrics)

    # save final checkpoint + metadata
    torch.save(
        {
            "state_dict": model.state_dict(),
            "T_steps": T_STEPS,
            "undirected": UNDIRECTED,
            "normalize": NORMALIZE,
            "step_size": STEP_SIZE,
            "best_thresh": float(best_thresh),
            "val_pr_auc": float(best_val),
            "test_pr_auc": float(pr_auc),
        },
        "recurrent_best_final_v2.pth",
    )
    print("Saved final checkpoint with threshold to recurrent_best_final_v2.pth")


if __name__ == "__main__":
    main()
