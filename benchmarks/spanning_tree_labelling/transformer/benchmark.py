import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from model_1 import make_spanning_candidate


def benchmark_model(adapter_fn,
                    n_graphs=1000,
                    node_range=(6,100),
                    feat_dim=16,
                    extra_per_tree=1,
                    seed=0):
    """
    Runs edge‐accuracy on `n_graphs` synthetic tests, and also counts
    how many edges were predicted 1 vs. 0.

    Returns:
      acc   – overall edge‐classification accuracy
      pred1 – total number of edges predicted 1
      pred0 – total number of edges predicted 0
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    correct = 0
    total   = 0
    pred1   = 0
    pred0   = 0

    for _ in range(n_graphs):
        # 1) generate one graph
        n = random.randint(*node_range)
        data = make_spanning_candidate(n, feat_dim, extra_per_tree)

        # 2) get model predictions
        preds = adapter_fn(data)
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu()

        # 3) accumulate counts
        true = data.edge_label.bool()
        correct += (preds == true).sum().item()
        total   += data.num_edges

        # 4) count 1’s vs. 0’s
        #    (assumes preds are boolean or 0/1 ints)
        p1 = int(preds.sum().item())
        p0 = int(preds.numel() - p1)
        pred1 += p1
        pred0 += p0

    acc = correct / total
    return acc, pred1, pred0
