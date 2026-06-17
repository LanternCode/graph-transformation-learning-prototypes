import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple
from benchmark import (
    collect_probability_labels,
    evaluate_model,
    find_best_f1_threshold,
    generate_transitive_closure_graphs,
    non_diagonal_mask,
    precision_recall_auc,
)


# ---------- helpers ----------
def stack_powers(A: np.ndarray, K: int) -> np.ndarray:
    """
    Return [A, A^2, ..., A^K] stacked along the last dim as float32.

    Args:
        A: Binary adjacency matrix with shape [N, N].
        K: Number of matrix powers to compute and stack.

    Returns:
        Float32 tensor data with shape [N, N, K].
    """
    A = A.astype(np.float32, copy=False)
    feats = []
    Ak = A.copy()
    for _ in range(K):
        feats.append(Ak)
        Ak = (Ak @ A).astype(np.float32)
    return np.stack(feats, axis=-1)  # [N, N, K]


# ---------- dataset ----------
class ClosureDataset(Dataset):
    """
    Dataset that converts closure-completion graphs into stacked-power edge features.

    Args:
        inputs: List of input adjacency matrices.
        targets: List of target closure matrices aligned with inputs.
        K: Number of adjacency powers to use as per-edge channels.

    Returns:
        Dataset items are feature and target tensors for one graph.
    """

    def __init__(self, inputs: List[np.ndarray], targets: List[np.ndarray], K: int = 2):
        """
        Store graph inputs, targets, and the power-stack depth.

        Args:
            inputs: List of input adjacency matrices.
            targets: List of target closure matrices aligned with inputs.
            K: Number of adjacency powers to use as per-edge channels.

        Returns:
            None.
        """
        self.inputs = inputs
        self.targets = targets
        self.K = K

    def __len__(self) -> int:
        """
        Return the number of graphs in the dataset.

        Args:
            None.

        Returns:
            Number of input/target graph pairs.
        """
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build stacked-power features and target tensor for one graph.

        Args:
            idx: Integer index of the graph to retrieve.

        Returns:
            Tuple containing a [N, N, K] feature tensor and [N, N] target tensor.
        """
        I = self.inputs[idx]
        feat_np = stack_powers(I, self.K)                 # float32
        feat = torch.from_numpy(np.array(feat_np, dtype=np.float32, copy=True)).float()
        targ_np = np.array(self.targets[idx], dtype=np.float32, copy=True)
        targ = torch.from_numpy(targ_np).float()
        return feat, targ


def variable_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad a batch of variable-size graphs to the max size in the batch.

    Args:
        batch: List of feature and target tensors produced by ClosureDataset.

    Returns:
        Tuple containing padded features, padded targets, and a valid-entry mask.
    """
    feats, tgts = zip(*batch)
    ns = [f.shape[0] for f in feats]
    M = max(ns)
    B = len(batch)
    C = feats[0].shape[-1]  # channels (K)

    Fb = torch.zeros(B, M, M, C)
    Tb = torch.zeros(B, M, M)
    mask = torch.zeros(B, M, M)

    for i, (f, t) in enumerate(zip(feats, tgts)):
        n = f.shape[0]
        Fb[i, :n, :n, :] = f
        Tb[i, :n, :n] = t
        mask[i, :n, :n] = 1.0

    return Fb, Tb, mask


# ---------- model ----------
class PointwiseMLP(nn.Module):
    """
    Pointwise MLP over per-edge stacked-power features.

    Args:
        in_ch: Number of input channels per node pair.
        hidden_dim: Width of the hidden layer.

    Returns:
        A PyTorch module whose forward pass returns logits with shape [B, N, N].
    """

    def __init__(self, in_ch: int = 2, hidden_dim: int = 32):
        """
        Initialize the pointwise edge classifier layers.

        Args:
            in_ch: Number of input channels per node pair.
            hidden_dim: Width of the hidden layer.

        Returns:
            None.
        """
        super().__init__()
        self.fc1 = nn.Linear(in_ch, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Score each node pair independently using its feature channels.

        Args:
            x: Tensor with shape [B, N, N, C].

        Returns:
            Logit tensor with shape [B, N, N].
        """
        b, n, _, c = x.shape
        x = x.view(b, n * n, c)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                # logits
        return x.view(b, n, n)


# ---------- training ----------
@torch.no_grad()
def predict_probabilities(model: nn.Module, A: np.ndarray, K: int, device: torch.device) -> np.ndarray:
    """
    Predict closure probabilities for one graph with the pointwise MLP.

    Args:
        model: Trained PointwiseMLP model.
        A: Input adjacency matrix for one graph.
        K: Number of adjacency powers expected by the model.
        device: Torch device used for inference.

    Returns:
        Probability matrix with shape [N, N].
    """
    A = A.astype(np.float32)
    feat_np = stack_powers(A, K)                       # [N,N,K], float32
    feat = torch.from_numpy(np.array(feat_np, dtype=np.float32, copy=True)).unsqueeze(0).to(device)
    logits = model(feat).squeeze(0)                    # [N,N]
    return torch.sigmoid(logits).detach().cpu().numpy()


@torch.no_grad()
def pr_auc_on_holdout(model: nn.Module, inputs: List[np.ndarray], targets: List[np.ndarray], K: int, device: torch.device) -> float:
    """
    Compute off-diagonal PR-AUC for a held-out graph set.

    Args:
        model: PointwiseMLP model to evaluate.
        inputs: List of input adjacency matrices.
        targets: List of target closure matrices aligned with inputs.
        K: Number of adjacency powers expected by the model.
        device: Torch device used for inference.

    Returns:
        Precision-recall AUC over off-diagonal entries.
    """
    model.eval()
    probs, labels = collect_probability_labels(
        lambda A: predict_probabilities(model, A, K, device),
        inputs,
        targets,
    )
    return precision_recall_auc(probs, labels)


def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    val_in: List[np.ndarray],
    val_tg: List[np.ndarray],
    K: int,
    epochs: int = 50,
    device: torch.device | str = "cpu",
    ckpt_path: str = "mlp_best.pth",
) -> float:
    """
    Train the pointwise MLP and checkpoint the best validation PR-AUC state.

    Args:
        model: PointwiseMLP model to train.
        loader: DataLoader that yields padded feature, target, and mask batches.
        optimizer: Torch optimizer used to update model weights.
        val_in: Validation input adjacency matrices.
        val_tg: Validation target matrices aligned with val_in.
        K: Number of adjacency powers expected by the model.
        epochs: Number of training epochs.
        device: Torch device used for training and validation.
        ckpt_path: Path where the best model state dict is saved.

    Returns:
        Best validation PR-AUC observed during training.
    """
    model.train()
    best_val = -1.0
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")  # logits
    eye_cache = {}

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        total_elements = 0
        for feats, tgt, mask in loader:
            feats, tgt, mask = feats.to(device), tgt.to(device), mask.to(device)
            n = tgt.size(1)
            if n not in eye_cache:
                eye_cache[n] = torch.from_numpy(non_diagonal_mask(n).astype(np.float32)).to(device)
            eff_mask = mask * eye_cache[n].unsqueeze(0)

            optimizer.zero_grad()
            logits = model(feats)                              # [B,N,N]
            loss_mat = loss_fn(logits, tgt)                    # elementwise
            loss = (loss_mat * eff_mask).sum() / eff_mask.sum().clamp(min=1.0)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * eff_mask.sum().item()
            total_elements += eff_mask.sum().item()

        val_auc = pr_auc_on_holdout(model, val_in, val_tg, K, torch.device(device))
        if val_auc > best_val:
            best_val = val_auc
            torch.save(model.state_dict(), ckpt_path)          # save best model
        if ep % 10 == 0:
            avg_loss = total_loss / max(total_elements, 1)
            print(f"Epoch {ep:03d}: Avg Loss={avg_loss:.4f}  Val PR-AUC={val_auc:.4f}")

    print(f"Best Val PR-AUC: {best_val:.4f}  (model saved to {ckpt_path})")
    return best_val


def main() -> None:
    """
    Run the pointwise MLP transitive-closure experiment.

    Args:
        None.

    Returns:
        None. The function trains a model, prints validation/test metrics, and saves checkpoints.
    """
    # config
    K = 10                     # number of hop channels (A..A^K)  <-- Fix #4
    BATCH = 16
    EPOCHS = 100
    LR = 1e-3
    CKPT = "mlp_best.pth"

    # 1) Data
    train_in, train_tg = generate_transitive_closure_graphs(
        num_graphs=1000, min_nodes=6, max_nodes=140, missing_pct=0.2, k=K
    )
    # small validation split from the generator (independent from test)
    val_in, val_tg = generate_transitive_closure_graphs(
        num_graphs=200, min_nodes=6, max_nodes=140, missing_pct=0.2, k=K
    )

    dataset = ClosureDataset(train_in, train_tg, K=K)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=True,
        collate_fn=variable_collate,
        pin_memory=(device.type == "cuda"),
    )

    # 2) Model
    model = PointwiseMLP(in_ch=K, hidden_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 3) Train + save best model by Val PR-AUC
    best_val = train(
        model, loader, optimizer,
        val_in=val_in, val_tg=val_tg, K=K, epochs=EPOCHS,
        device=device, ckpt_path=CKPT,
    )

    # 4) Load best model and tune threshold on validation data
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.to(device).eval()

    val_probs, val_labels = collect_probability_labels(
        lambda A: predict_probabilities(model, A, K, device),
        val_in,
        val_tg,
    )
    best_f1, best_thresh = find_best_f1_threshold(val_probs, val_labels)
    print(f"Validation Max F1 = {best_f1:.3f} at thresh = {best_thresh:.3f}")

    test_in, test_tg = generate_transitive_closure_graphs(
        num_graphs=200, min_nodes=6, max_nodes=140, missing_pct=0.2, k=K
    )

    test_probs, test_labels = collect_probability_labels(
        lambda A: predict_probabilities(model, A, K, device),
        test_in,
        test_tg,
    )
    pr_auc = precision_recall_auc(test_probs, test_labels)
    print(f"Test PR AUC = {pr_auc:.3f}")

    # 5) Final evaluation using the validation-tuned threshold
    metrics = evaluate_model(
        model_fn=lambda A: predict_probabilities(model, A, K, device),
        inputs=test_in,
        targets=test_tg,
        threshold=best_thresh,
    )
    print("Evaluation metrics at validation-tuned threshold:", metrics)

    # 6) Save final checkpoint that also includes the tuned threshold
    torch.save(
        {"state_dict": model.state_dict(),
         "in_channels": K,
         "best_thresh": float(best_thresh),
         "val_pr_auc": float(best_val),
         "test_pr_auc": float(pr_auc)},
        "mlp_best_final.pth",
    )
    print("Saved final checkpoint with threshold to mlp_best_final.pth")


if __name__ == "__main__":
    main()
