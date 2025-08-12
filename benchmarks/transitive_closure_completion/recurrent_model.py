import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, auc
from torch.utils.data import DataLoader, Dataset
from benchmark import generate_transitive_closure_graphs, evaluate_model


# ---------------- Dataset & Collate (A-only, float32) ----------------

class ClosureDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        A = self.inputs[idx].astype(np.float32)
        T = self.targets[idx].astype(np.float32)
        return torch.from_numpy(A), torch.from_numpy(T)


def variable_collate(batch):
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
    Learn a single 'closure step' and apply it T times (shared weights).
    Input:  A in {0,1}^{B,N,N} (float32)
    Output: logits S_T in R^{B,N,N}
    """
    def __init__(self, hidden=64, T=8, undirected=False, normalize=True, step_size=0.5):
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


# ---------------- PR-AUC helper ----------------

@torch.no_grad()
def pr_auc_on_holdout(model, inputs, targets, device):
    model.eval()
    probs, labels = [], []
    for A, T in zip(inputs, targets):
        A = A.astype(np.float32)
        featA = torch.from_numpy(A).unsqueeze(0).to(device)  # [1,N,N]
        logits = model(featA).squeeze(0)                     # [N,N]
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.extend(p.ravel().tolist())
        labels.extend(T.astype(np.float32).ravel().tolist())
    precision, recall, _ = precision_recall_curve(labels, probs)
    return auc(recall, precision)


# ---------------- Train ----------------

def train(model, loader, optimizer, val_in, val_tg, epochs, device, ckpt_path):
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
                logits, tgt, reduction='none', pos_weight=pos_w
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

if __name__ == "__main__":
    # ----- config -----
    EPOCHS   = 100
    BATCH    = 16
    LR       = 1e-3
    T_STEPS  = 10               # try aligning with the largest k at test
    UNDIRECTED = False          # set True if your graphs are undirected
    NORMALIZE  = True           # row-normalize A for stable path signals
    STEP_SIZE  = 0.5            # 0.25..1.0; smaller = safer updates
    CKPT    = "recurrent_best_v2.pth"

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
        pin_memory=(device.type == "cuda")
    )

    # model
    model = RecurrentClosure(
        hidden=64, T=T_STEPS, undirected=UNDIRECTED, normalize=NORMALIZE, step_size=STEP_SIZE
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    # train + save best
    best_val = train(model, loader, optim, val_in, val_tg, EPOCHS, device, CKPT)

    # load best and evaluate
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    # test PR-AUC
    probs, labels = [], []
    with torch.no_grad():
        for A, T in zip(test_in, test_tg):
            A = A.astype(np.float32)
            featA = torch.from_numpy(A).unsqueeze(0).to(device)
            logits = model(featA).squeeze(0)
            p = torch.sigmoid(logits).detach().cpu().numpy()
            probs.extend(p.ravel().tolist())
            labels.extend(T.astype(np.float32).ravel().tolist())

    precision, recall, thresholds = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    print(f"Test PR AUC = {pr_auc:.3f}")

    # choose F1-optimal threshold
    all_thresholds = np.append(thresholds, 1.0)
    best_f1, best_thresh = 0.0, 0.5
    for p, r, t in zip(precision, recall, all_thresholds):
        f = 2 * p * r / (p + r + 1e-12)
        if f > best_f1:
            best_f1, best_thresh = f, t
    print(f"Max F1 = {best_f1:.3f} at thresh = {best_thresh:.3f}")

    # eval via your helper (detach fix here)
    metrics = evaluate_model(
        model_fn=lambda A: torch.sigmoid(
            model(torch.from_numpy(A.astype(np.float32)).unsqueeze(0).to(device))
        ).squeeze(0).detach().cpu().numpy(),
        inputs=test_in,
        targets=test_tg,
        threshold=best_thresh
    )
    print("Evaluation metrics at F1-optimal threshold:", metrics)

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
