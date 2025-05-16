import random
import numpy as np
import networkx as nx
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)
import torch

__all__ = [
    "generate_shortcut_dataset",
    "evaluate_model",
    "wrap_model",
]

def generate_shortcut_dataset(
    num_graphs=1000,
    min_nodes=6,
    max_nodes=140,
    base_edge_prob=0.01,
    shortcut_frac=1.0,
    graph_families=None,
    seed=None
):
    """
    Generate a benchmark dataset of directed graphs with injected shortcuts,
    sampling across multiple graph families.

    Args:
        num_graphs (int): Number of graphs to generate.
        min_nodes (int): Minimum number of nodes per graph (inclusive).
        max_nodes (int): Maximum number of nodes per graph (inclusive).
        base_edge_prob (float): Probability of each forward edge in the base DAG family.
        shortcut_frac (float): Fraction of candidate two-hop pairs to inject as shortcuts.
        graph_families (list of str): Which graph types to sample from. Options:
            - 'dag' (random directed acyclic graph)
            - 'erdos_renyi' (directed Erdős–Rényi)
            - 'watts_strogatz' (small-world, directed)
            - 'barabasi_albert' (scale-free, directed)
          If None, defaults to all four.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        graphs (list of np.ndarray): Adjacency matrices with injected shortcuts.
        cleaned_graphs (list of np.ndarray): Matrices with shortcuts removed.
        masks (list of np.ndarray): Binary masks indicating injected shortcuts.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if graph_families is None:
        graph_families = ['dag', 'erdos_renyi', 'watts_strogatz', 'barabasi_albert']

    graphs, cleaned_graphs, masks = [], [], []

    for _ in range(num_graphs):
        # choose size and family
        n = random.randint(min_nodes, max_nodes)
        family = random.choice(graph_families)

        # generate base directed graph A
        if family == 'dag':
            # random DAG via random topological order
            perm = list(range(n)); random.shuffle(perm)
            A = np.zeros((n, n), dtype=int)
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < base_edge_prob:
                        A[perm[i], perm[j]] = 1
        elif family == 'erdos_renyi':
            # directed ER: include each possible directed edge with base_edge_prob
            A = (np.random.rand(n, n) < base_edge_prob).astype(int)
            np.fill_diagonal(A, 0)
        elif family == 'watts_strogatz':
            # small-world: undirected WS then orient
            # choose k ~ 4 or nearest even
            k = min(n-1, 4 + (4 % 2))
            G = nx.watts_strogatz_graph(n, k, base_edge_prob)
            A = np.zeros((n, n), dtype=int)
            for u, v in G.edges():
                if random.random() < 0.5:
                    A[u, v] = 1
                else:
                    A[v, u] = 1
        elif family == 'barabasi_albert':
            # scale-free: undirected BA then orient
            m = min(n-1, 2)
            G = nx.barabasi_albert_graph(n, m)
            A = np.zeros((n, n), dtype=int)
            for u, v in G.edges():
                if random.random() < 0.5:
                    A[u, v] = 1
                else:
                    A[v, u] = 1
        else:
            raise ValueError(f"Unknown graph family '{family}'")

        # ensure connectivity for all families
        G_und = nx.DiGraph(A).to_undirected()
        if not nx.is_connected(G_und):
            comps = list(nx.connected_components(G_und))
            for c1, c2 in zip(comps[:-1], comps[1:]):
                u = random.choice(list(c1)); v = random.choice(list(c2))
                A[u, v] = 1

        # find all two-hop (or longer) candidates
        Gdir = nx.DiGraph(A)
        sp = dict(nx.all_pairs_shortest_path_length(Gdir))
        candidates = [(u,v) for u, dm in sp.items() for v, d in dm.items() if d >= 2]

        # inject a fraction as shortcuts
        k = int(len(candidates) * shortcut_frac)
        marked = set(random.sample(candidates, min(k, len(candidates))))
        M = np.zeros_like(A)
        for u,v in marked:
            A[u, v] = 1
            M[u, v] = 1

        # record
        A_clean = A.copy()
        for u,v in marked:
            A_clean[u, v] = 0

        graphs.append(A)
        cleaned_graphs.append(A_clean)
        masks.append(M)

    return graphs, cleaned_graphs, masks


def evaluate_model(
    model_fn,
    graphs,
    masks,
    threshold: float = 0.5,
    average: str = "binary"
):
    y_true, y_score, y_pred = [], [], []
    for A, M_true in zip(graphs, masks):
        scores = model_fn(A).astype(float)
        preds = (scores >= threshold).astype(int)
        edge_idx = (A > 0).flatten()
        y_true.extend(M_true.flatten()[edge_idx])
        y_score.extend(scores.flatten()[edge_idx])
        y_pred.extend(preds.flatten()[edge_idx])
    total_edges = len(y_true)
    total_shortcuts = sum(y_true)
    print(f"Evaluating on {total_edges} edges, of which {total_shortcuts} are shortcuts")
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) == 2 else float("nan")
    return {"accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1, "auc": auc}


class ModelAdapter:
    """
    Wrap a PyTorch model for benchmark evaluation with a user-supplied reshape.
    """
    def __init__(self, model, device=None, reshape_fn=None):
        self.model = model
        self.device = device if device is not None else _infer_device(model)
        self.model.to(self.device)
        self.reshape_fn = reshape_fn

    def __call__(self, adj: np.ndarray) -> np.ndarray:
        n = adj.shape[0]
        if self.reshape_fn is None:
            raise RuntimeError("Please provide a reshape_fn matching your model's input")
        x = self.reshape_fn(adj).to(self.device)
        with torch.no_grad(): out = self.model(x)
        out_np = out.cpu().numpy().reshape(n, n)
        return out_np


def wrap_model(model, device=None, reshape_fn=None):
    """Return a ModelAdapter with the given reshape_fn."""
    return ModelAdapter(model, device, reshape_fn)


def _infer_device(model):
    try: return next(model.parameters()).device
    except StopIteration: return torch.device('cpu')
