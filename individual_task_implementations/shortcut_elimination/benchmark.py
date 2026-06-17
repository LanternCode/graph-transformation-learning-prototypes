import random
import numpy as np
import networkx as nx
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)

__all__ = [
    "generate_shortcut_dataset",
    "evaluate_model",
    "wrap_model",
]


def _find_preexisting_shortcuts(A: np.ndarray):
    """
    Identify edges that are already shortcuts before new shortcuts are injected.

    Args:
        A (np.ndarray): Square binary adjacency matrix for a directed graph.

    Returns:
        set[tuple[int, int]]: Directed edges (u, v) that already exist in A and
        still have an alternative directed path from u to v after that edge is
        temporarily removed.
    """
    Gdir = nx.DiGraph(A)
    shortcut_edges = set()

    for u, v in zip(*np.nonzero(A)):
        Gdir.remove_edge(u, v)
        if nx.has_path(Gdir, u, v):
            shortcut_edges.add((u, v))
        Gdir.add_edge(u, v)

    return shortcut_edges


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
    Generate a benchmark dataset of directed graphs with shortcut labels.

    The generator first samples a directed base graph from one of several graph
    families, then labels any shortcut edges that already exist in that base
    graph. It also injects additional shortcut edges from reachable node pairs
    and labels those injected edges. The resulting mask therefore marks both
    pre-existing shortcuts and newly injected shortcuts.

    Args:
        num_graphs (int): Number of graphs to generate.
        min_nodes (int): Minimum number of nodes per graph, inclusive.
        max_nodes (int): Maximum number of nodes per graph, inclusive.
        base_edge_prob (float): Edge probability or rewiring probability used by
            the sampled base graph family.
        shortcut_frac (float): Fraction of reachable candidate pairs to inject as
            additional shortcuts.
        graph_families (list[str] | None): Graph families to sample from. Valid
            entries are "dag", "erdos_renyi", "watts_strogatz", and
            "barabasi_albert". If None, all four families are used.
        seed (int | None): Optional random seed for Python and NumPy generation.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]: A tuple of
        graph adjacency matrices with shortcuts, cleaned adjacency matrices with
        labelled shortcuts removed, and binary shortcut masks.
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

        preexisting_shortcuts = _find_preexisting_shortcuts(A)

        # find all two-hop (or longer) candidates
        Gdir = nx.DiGraph(A)
        sp = dict(nx.all_pairs_shortest_path_length(Gdir))
        candidates = [(u,v) for u, dm in sp.items() for v, d in dm.items() if d >= 2]

        # inject a fraction as shortcuts
        k = int(len(candidates) * shortcut_frac)
        marked = set(random.sample(candidates, min(k, len(candidates))))
        M = np.zeros_like(A)
        for u, v in preexisting_shortcuts:
            M[u, v] = 1
        for u,v in marked:
            A[u, v] = 1
            M[u, v] = 1

        # record
        A_clean = A.copy()
        for u,v in preexisting_shortcuts | marked:
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
    """
    Evaluate shortcut predictions on existing edges only.

    Args:
        model_fn (Callable[[np.ndarray], np.ndarray]): Function that receives a
            graph adjacency matrix and returns an equally shaped shortcut score
            matrix.
        graphs (list[np.ndarray]): Graph adjacency matrices to evaluate.
        masks (list[np.ndarray]): Binary shortcut masks aligned with graphs.
        threshold (float): Score threshold used to convert scores into binary
            shortcut predictions.
        average (str): Averaging mode passed to
            precision_recall_fscore_support.

    Returns:
        dict[str, float]: Accuracy, precision, recall, F1, and ROC-AUC computed
        over the existing edges in the supplied graphs.
    """
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
    Wrap a PyTorch model for benchmark evaluation with a custom reshape step.

    Args:
        model (torch.nn.Module): PyTorch model to evaluate.
        device (torch.device | str | None): Device used for inference. If None,
            the device is inferred from the model parameters.
        reshape_fn (Callable[[np.ndarray], torch.Tensor] | None): Function that
            converts an adjacency matrix into the model input tensor.

    Returns:
        ModelAdapter: Callable adapter that maps adjacency matrices to NumPy
        output matrices.
    """
    def __init__(self, model, device=None, reshape_fn=None):
        """
        Initialize the model adapter.

        Args:
            model (torch.nn.Module): PyTorch model to wrap.
            device (torch.device | str | None): Device used for model inference.
            reshape_fn (Callable[[np.ndarray], torch.Tensor] | None): Function
                that prepares an adjacency matrix for the wrapped model.

        Returns:
            None.
        """
        self.model = model
        self.device = device if device is not None else _infer_device(model)
        self.model.to(self.device)
        self.reshape_fn = reshape_fn

    def __call__(self, adj: np.ndarray) -> np.ndarray:
        """
        Run the wrapped model on one adjacency matrix.

        Args:
            adj (np.ndarray): Square graph adjacency matrix.

        Returns:
            np.ndarray: Model output reshaped to match the input adjacency
            matrix dimensions.
        """
        n = adj.shape[0]
        if self.reshape_fn is None:
            raise RuntimeError("Please provide a reshape_fn matching your model's input")
        x = self.reshape_fn(adj).to(self.device)
        with torch.no_grad(): out = self.model(x)
        out_np = out.cpu().numpy().reshape(n, n)
        return out_np


def wrap_model(model, device=None, reshape_fn=None):
    """
    Create a benchmark adapter for a PyTorch model.

    Args:
        model (torch.nn.Module): Model to wrap.
        device (torch.device | str | None): Optional inference device.
        reshape_fn (Callable[[np.ndarray], torch.Tensor] | None): Function that
            converts adjacency matrices into model inputs.

    Returns:
        ModelAdapter: Callable benchmark adapter for the supplied model.
    """
    return ModelAdapter(model, device, reshape_fn)


def _infer_device(model):
    """
    Infer the device of a PyTorch model.

    Args:
        model (torch.nn.Module): Model whose first parameter is inspected.

    Returns:
        torch.device: Device of the first model parameter, or CPU for models
        without parameters.
    """
    try: return next(model.parameters()).device
    except StopIteration: return torch.device('cpu')
