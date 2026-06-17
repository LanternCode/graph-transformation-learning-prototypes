import networkx as nx
import numpy as np
import random
from typing import Callable, List, Tuple
from tqdm import tqdm

# Constants
NUM_GRAPHS = 1000
MIN_NODES = 6
MAX_NODES = 140
GRAPH_TYPES = ['erdos_renyi', 'barabasi_albert', 'watts_strogatz', 'random_regular']


# Helper to generate diverse graphs
def generate_graph(graph_type: str, num_nodes: int):
    """
    Generate a random undirected graph from one supported graph family.

    Args:
        graph_type: Name of the graph generator to use. Supported values are
            'erdos_renyi', 'barabasi_albert', 'watts_strogatz', and
            'random_regular'.
        num_nodes: Number of nodes to include in the generated graph.

    Returns:
        A NetworkX graph sampled from the requested graph family.

    Raises:
        ValueError: If graph_type is not one of the supported graph families.
    """
    if graph_type == 'erdos_renyi':
        p = random.uniform(0.05, 0.3)
        G = nx.erdos_renyi_graph(num_nodes, p)
    elif graph_type == 'barabasi_albert':
        m = min(5, num_nodes - 1)
        G = nx.barabasi_albert_graph(num_nodes, m)
    elif graph_type == 'watts_strogatz':
        k = random.randint(2, min(num_nodes - 1, 6))
        p = random.uniform(0.1, 0.5)
        G = nx.watts_strogatz_graph(num_nodes, k, p)
    elif graph_type == 'random_regular':
        d = random.randint(2, min(num_nodes - 1, 6))
        d = d if (num_nodes * d) % 2 == 0 else d - 1
        G = nx.random_regular_graph(d, num_nodes) if d > 0 else nx.empty_graph(num_nodes)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    return G


def orient_edges_asymmetrically(G: nx.Graph) -> np.ndarray:
    """
    Convert an undirected graph into an asymmetric directed adjacency matrix.

    Args:
        G: Undirected NetworkX graph whose edges should each be assigned one
            direction.

    Returns:
        A float32 adjacency matrix where each undirected edge from G appears in
        exactly one randomly selected direction.
    """
    nodes = list(G.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    adj = np.zeros((len(nodes), len(nodes)), dtype=np.float32)

    for u, v in G.edges():
        i, j = index[u], index[v]
        if random.random() < 0.5:
            adj[i, j] = 1.0
        else:
            adj[j, i] = 1.0

    return adj


# Create the symmetric closure benchmark
def generate_benchmark() -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate asymmetric adjacency matrices and their symmetric-closure labels.

    Args:
        None.

    Returns:
        A tuple containing two lists. The first list contains asymmetric float32
        adjacency matrices. The second list contains labels computed as
        A OR A^T for each corresponding adjacency matrix.
    """
    graphs = []
    labels = []

    for _ in tqdm(range(NUM_GRAPHS), desc="Generating graphs"):
        while True:
            graph_type = random.choice(GRAPH_TYPES)
            num_nodes = random.randint(MIN_NODES, MAX_NODES)
            G = generate_graph(graph_type, num_nodes)
            if nx.is_connected(G.to_undirected()):
                break

        adj = orient_edges_asymmetrically(G)
        label = np.logical_or(adj, adj.T).astype(np.float32)
        graphs.append(adj)
        labels.append(label)

    return graphs, labels


# Benchmark evaluator
def evaluate_model(adapter_fn: Callable[[np.ndarray], np.ndarray], graphs: List[np.ndarray], labels: List[np.ndarray]) -> float:
    """
    Evaluate a symmetric-closure model on off-diagonal adjacency entries.

    Args:
        adapter_fn: Callable that accepts a single adjacency matrix and returns a
            matrix of prediction scores with the same layout.
        graphs: List of input adjacency matrices to evaluate.
        labels: List of target symmetric-closure matrices corresponding to the
            input graphs.

    Returns:
        Mean off-diagonal accuracy across all evaluated graphs.
    """
    accuracies = []

    for graph, true_label in tqdm(zip(graphs, labels), total=len(graphs), desc="Evaluating model"):
        pred = adapter_fn(graph)
        pred_binary = (pred > 0.5).astype(np.float32)
        off_diagonal = ~np.eye(true_label.shape[0], dtype=bool)
        accuracies.append(np.mean(pred_binary[off_diagonal] == true_label[off_diagonal]))

    avg_accuracy = np.mean(accuracies)
    return avg_accuracy
