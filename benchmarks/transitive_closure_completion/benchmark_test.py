import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, auc
from torch.utils.data import DataLoader, Dataset
from benchmark import generate_transitive_closure_graphs, evaluate_model


# ---------- helpers ----------

def stack_powers(A: np.ndarray, K: int) -> np.ndarray:
    """
    Return [A, A^2, ..., A^K] stacked along the last dim as float32.
    Assumes A is a 0/1 adjacency matrix (np.ndarray, shape [N,N]).
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
    def __init__(self, inputs, targets, K: int = 2):
        self.inputs = inputs
        self.targets = targets
        self.K = K

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        I = self.inputs[idx]
        feat_np = stack_powers(I, self.K)                 # float32
        feat = torch.from_numpy(feat_np).float()          # [N,N,K]
        targ = torch.from_numpy(self.targets[idx]).float()
        return feat, targ


def variable_collate(batch):
    """Pad a batch of variable-size graphs to the max size in the batch."""
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
    """Pointwise MLP over edge features; returns *logits* (no sigmoid)."""
    def __init__(self, in_ch=2, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(in_ch, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):              # x: [B,N,N,C]
        b, n, _, c = x.shape
        x = x.view(b, n * n, c)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                # logits
        return x.view(b, n, n)


# ---------- training ----------

@torch.no_grad()
def pr_auc_on_holdout(model, inputs, targets, K, device):
    """Collect probs and labels; compute PR-AUC for validation."""
    model.eval()
    probs, labels = [], []
    for A, T in zip(inputs, targets):
        A = A.astype(np.float32)
        feat_np = stack_powers(A, K)                       # [N,N,K], float32
        feat = torch.from_numpy(feat_np).unsqueeze(0).to(device)  # [1,N,N,K]
        logits = model(feat).squeeze(0)                    # [N,N]
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.extend(p.ravel().tolist())
        labels.extend(T.astype(np.float32).ravel().tolist())
    precision, recall, _ = precision_recall_curve(labels, probs)
    return auc(recall, precision)


def train(model, loader, optimizer, val_in, val_tg, K, epochs=50, device='cpu', ckpt_path='mlp_best.pth'):
    model.train()
    best_val = -1.0

    loss_fn = nn.BCEWithLogitsLoss(reduction='none')  # logits

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        total_elements = 0
        for feats, tgt, mask in loader:
            feats, tgt, mask = feats.to(device), tgt.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(feats)                              # [B,N,N]
            loss_mat = loss_fn(logits, tgt)                    # elementwise
            loss = (loss_mat * mask).sum() / mask.sum()       # mask padding
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * mask.sum().item()
            total_elements += mask.sum().item()

        # validation PR-AUC
        val_auc = pr_auc_on_holdout(model, val_in, val_tg, K, device)
        if val_auc > best_val:
            best_val = val_auc
            torch.save(model.state_dict(), ckpt_path)          # save best model
        if ep % 10 == 0:
            avg_loss = total_loss / max(total_elements, 1)
            print(f"Epoch {ep:03d}: Avg Loss={avg_loss:.4f}  Val PR-AUC={val_auc:.4f}")

    print(f"Best Val PR-AUC: {best_val:.4f}  (model saved to {ckpt_path})")
    return best_val


# ---------- main ----------

if __name__ == "__main__":
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
        pin_memory=(device.type == 'cuda')   # QoL (Fix #6)
    )

    # 2) Model
    model = PointwiseMLP(in_ch=K, hidden_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 3) Train + save best model by Val PR-AUC
    best_val = train(
        model, loader, optimizer,
        val_in=val_in, val_tg=val_tg, K=K, epochs=EPOCHS,
        device=device, ckpt_path=CKPT
    )

    # 4) Load best model and evaluate on holdout with threshold tuning
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.to(device).eval()

    test_in, test_tg = generate_transitive_closure_graphs(
        num_graphs=200, min_nodes=6, max_nodes=140, missing_pct=0.2, k=K
    )

    probs, labels = [], []
    with torch.no_grad():
        for A, T in zip(test_in, test_tg):
            A = A.astype(np.float32)                                 # Fix #1
            feat_np = stack_powers(A, K)                              # Fix #4
            feat = torch.from_numpy(feat_np).unsqueeze(0).to(device)
            logits = model(feat).squeeze(0)
            p = torch.sigmoid(logits).cpu().numpy()                   # Fix #5
            probs.extend(p.ravel().tolist())
            labels.extend(T.astype(np.float32).ravel().tolist())

    precision, recall, thresholds = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    print(f"Test PR AUC = {pr_auc:.3f}")

    # choose F1-optimal threshold
    all_thresholds = np.append(thresholds, 1.0)  # align lengths
    best_f1, best_thresh = 0.0, 0.5
    for p, r, t in zip(precision, recall, all_thresholds):
        f = 2 * p * r / (p + r + 1e-12)
        if f > best_f1:
            best_f1, best_thresh = f, t
    print(f"Max F1 = {best_f1:.3f} at thresh = {best_thresh:.3f}")

    # 5) Final evaluation using that threshold
    metrics = evaluate_model(
        model_fn=lambda A: torch.sigmoid(                  # Fix #5
            model(
                torch.from_numpy(stack_powers(A.astype(np.float32), K))
                .unsqueeze(0).to(device)
            )
        ).squeeze(0).detach().cpu().numpy(),
        inputs=test_in,
        targets=test_tg,
        threshold=best_thresh
    )
    print("Evaluation metrics at F1-optimal threshold:", metrics)

    # 6) Save final checkpoint that also includes the tuned threshold (QoL, Fix #6)
    torch.save(
        {"state_dict": model.state_dict(),
         "in_channels": K,
         "best_thresh": float(best_thresh),
         "val_pr_auc": float(best_val),
         "test_pr_auc": float(pr_auc)},
        "mlp_best_final.pth"
    )
    print("Saved final checkpoint with threshold to mlp_best_final.pth")
