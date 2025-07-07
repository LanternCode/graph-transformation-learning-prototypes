import torch
from benchmark import benchmark_precision_recall
from model import DeepEdgeMLP
import numpy as np

model = DeepEdgeMLP(input_dim=11)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()


def mlp_adapter(X: np.ndarray, N: int):
    """
    X: (N*N, 11) edge features
    N: number of nodes
    Returns a list of (i,j) pairs your model predicts as 'present'.
    """
    # convert to torch tensor
    t = torch.from_numpy(X).float()

    # forward pass
    with torch.no_grad():
        logits = model(t)               # shape (N*N, 1) or (N*N,)
        probs  = torch.sigmoid(logits)  # if your model outputs logits
        probs  = probs.view(-1).cpu().numpy()

    # threshold at 0.5
    preds = []
    for idx, p in enumerate(probs):
        if p > 0.5:
            i, j = divmod(idx, N)
            preds.append((i, j))
    return preds


precision, recall, f1 = benchmark_precision_recall(mlp_adapter,
                                                   num_graphs=1000,
                                                   drop_fraction=0.2)
