import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, auc
from torch.utils.data import DataLoader, Dataset
from benchmark import generate_transitive_closure_graphs, evaluate_model

class ClosureDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        I = self.inputs[idx]
        I2 = (I @ I).astype(np.float32)
        feat = torch.from_numpy(np.stack([I, I2], axis=-1)).float()
        targ = torch.from_numpy(self.targets[idx]).float()
        return feat, targ

# custom collate_fn to pad each batch to its max size
def variable_collate(batch):
    feats, tgts = zip(*batch)
    ns = [f.shape[0] for f in feats]
    M  = max(ns)
    B  = len(batch)

    Fb = torch.zeros(B, M, M, 2)
    Tb = torch.zeros(B, M, M)
    mask = torch.zeros(B, M, M)

    for i,(f,t) in enumerate(zip(feats,tgts)):
        n = f.shape[0]
        Fb[i,:n,:n,:] = f
        Tb[i,:n,:n]   = t
        mask[i,:n,:n] = 1

    return Fb, Tb, mask

class PointwiseMLP(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        b,n,_,_ = x.shape
        x = x.view(b, n*n, 2)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.view(b, n, n)

def train(model, loader, optimizer, epochs=50, device='cpu'):
    model.train()
    for ep in range(1, epochs+1):
        total_loss = 0.0
        total_elements = 0
        for feats, tgt, mask in loader:
            feats, tgt, mask = feats.to(device), tgt.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(feats)
            # elementwise BCE
            loss_mat = F.binary_cross_entropy(pred, tgt, reduction='none')
            # mask out padding
            loss = (loss_mat * mask).sum() / mask.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * mask.sum().item()
            total_elements += mask.sum().item()
        if ep % 10 == 0:
            print(f"Epoch {ep:03d}: Avg Loss = {total_loss/total_elements:.4f}")

if __name__ == "__main__":
    # 1. Generate training data
    train_in, train_tg = generate_transitive_closure_graphs(
        num_graphs=1000, min_nodes=6, max_nodes=140, missing_pct=0.2, k=2
    )
    dataset = ClosureDataset(train_in, train_tg)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=variable_collate
    )

    # 2. Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointwiseMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train(model, loader, optimizer, epochs=50, device=device)

    # 3. Collect probabilities & labels on holdout
    test_in, test_tg = generate_transitive_closure_graphs(
        num_graphs=200, min_nodes=6, max_nodes=140, missing_pct=0.2, k=10
    )
    probs, labels = [], []
    with torch.no_grad():
        for A, T in zip(test_in, test_tg):
            feat = torch.from_numpy(
                np.stack([A, (A @ A).astype(np.float32)], axis=-1)
            ).unsqueeze(0).to(device)
            p = model(feat).squeeze(0).cpu().numpy()
            probs.extend(p.flatten().tolist())
            labels.extend(T.flatten().tolist())

    # 4. Tune threshold by maximizing F1 on holdout
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    print(f"PR AUC = {pr_auc:.3f}")

    # thresholds length = len(precision) - 1, append 1.0 for matching length
    all_thresholds = np.append(thresholds, 1.0)

    # find threshold that maximizes F1
    best_f1, best_thresh = 0.0, 0.5
    for p, r, t in zip(precision, recall, all_thresholds):
        f_score = 2 * p * r / (p + r + 1e-12)
        if f_score > best_f1:
            best_f1, best_thresh = f_score, t

    print(f"Max F1 = {best_f1:.3f} at thresh = {best_thresh:.3f}")

    # use F1-optimal threshold
    chosen_thresh = best_thresh

    # 5. Final evaluation using F1-tuned threshold
    metrics = evaluate_model(
        model_fn=lambda A: model(
            torch.from_numpy(
                np.stack([A.astype(np.float32), (A @ A).astype(np.float32)], axis=-1)
            ).unsqueeze(0).to(device)
        ).squeeze(0).detach().cpu().numpy(),
        inputs=test_in,
        targets=test_tg,
        threshold=chosen_thresh
    )
    print("Evaluation metrics at F1-optimal threshold:", metrics)
