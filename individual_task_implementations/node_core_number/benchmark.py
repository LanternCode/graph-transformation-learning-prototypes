import random
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_squared_error
from torch_geometric.utils import from_networkx, to_undirected, add_self_loops
from torch_geometric.loader import DataLoader


def generate_benchmark_graph():
    """
    Generate one synthetic benchmark graph for core-number prediction.

    Args:
        None.

    Returns:
        A PyTorch Geometric Data object containing node features for degree and
        clustering coefficient, edge indices with undirected edges and
        self-loops, and node targets equal to NetworkX core numbers.
    """
    min_nodes, max_nodes = 6, 140
    choice = random.choice(["erdos", "barabasi", "watts", "tree"])
    n = random.randint(min_nodes, max_nodes)
    if choice == "erdos":
        p = random.uniform(0.02, 0.1)
        G = nx.erdos_renyi_graph(n, p)
    elif choice == "barabasi":
        m = random.randint(2, min(5, n - 1))
        G = nx.barabasi_albert_graph(n, m)
    elif choice == "tree":
        G = nx.random_unlabeled_tree(n)
    else:
        k = random.randint(2, min(6, n - 1))
        beta = random.uniform(0.1, 0.5)
        G = nx.watts_strogatz_graph(n, k, beta)

    max_attempts = 1000
    attempt = 0

    connected = False
    while not connected and attempt < max_attempts:
        attempt += 1

        if choice == "erdos":
            p = random.uniform(0.02, 0.1)
            G = nx.erdos_renyi_graph(n, p)
        elif choice == "barabasi":
            m = random.randint(2, min(5, n - 1))
            G = nx.barabasi_albert_graph(n, m)
        elif choice == "watts":
            k = random.randint(2, min(6, n - 1))
            beta = random.uniform(0.1, 0.5)
            G = nx.watts_strogatz_graph(n, k, beta)
        else:
            G = nx.random_unlabeled_tree(n)  # always connected
            connected = True
            break

        connected = nx.is_connected(G)

    core = nx.core_number(G)
    for n in G.nodes:
        G.nodes[n]['core'] = core[n]
        G.nodes[n]['degree'] = G.degree[n]
        G.nodes[n]['clustering'] = nx.clustering(G, n)
    data = from_networkx(G)
    data.edge_index = to_undirected(data.edge_index)
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    data.y = torch.tensor([core[n] for n in G.nodes], dtype=torch.float)
    data.x = torch.stack([
        data.degree.float(),
        data.clustering.float()
    ], dim=1)
    return data


def benchmark_model(adapter_fn, num_graphs=1000, batch_size=1):
    """
    Evaluate a model adapter on generated core-number benchmark graphs.

    Args:
        adapter_fn: Callable that accepts a PyTorch Geometric batch and returns
            ``(predictions, targets)`` arrays for node-level core prediction.
        num_graphs: Number of benchmark graphs to generate.
        batch_size: Number of graphs per DataLoader batch.

    Returns:
        A tuple ``(acc, mse)`` containing rounded integer accuracy and mean
        squared error over all successfully evaluated node predictions.
    """
    graphs = [generate_benchmark_graph() for _ in range(num_graphs)]
    loader = DataLoader(graphs, batch_size=batch_size)

    all_preds, all_trues = [], []

    for i, batch in enumerate(tqdm(loader, desc="Benchmarking core")):
        try:
            pred, true = adapter_fn(batch)
            all_preds.append(pred)
            all_trues.append(true)
        except Exception as e:
            print(f"Adapter failed on batch {i}: {e}")

    y_true = np.concatenate(all_trues)
    y_pred = np.concatenate(all_preds)
    acc = accuracy_score(y_true.astype(int), np.round(y_pred).astype(int))
    mse = mean_squared_error(y_true, y_pred)
    return acc, mse
