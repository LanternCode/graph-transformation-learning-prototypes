import networkx as nx
import numpy as np
import random
from typing import Callable, List
from tqdm import tqdm

# Constants
NUM_GRAPHS = 1000
MIN_NODES = 6
MAX_NODES = 140
GRAPH_TYPES = ['erdos_renyi', 'barabasi_albert', 'watts_strogatz', 'random_regular']


# Helper to generate diverse graphs
def generate_graph(graph_type: str, num_nodes: int):
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


# Create the symmetric closure benchmark
def generate_benchmark():
    graphs = []
    labels = []

    for _ in tqdm(range(NUM_GRAPHS), desc="Generating graphs"):
        while True:
            graph_type = random.choice(GRAPH_TYPES)
            num_nodes = random.randint(MIN_NODES, MAX_NODES)
            G = generate_graph(graph_type, num_nodes)
            if nx.is_connected(G.to_undirected()):
                break

        adj = nx.to_numpy_array(G, dtype=np.float32)
        label = np.logical_or(adj, adj.T).astype(np.float32)
        graphs.append(adj)
        labels.append(label)

    return graphs, labels


# Benchmark evaluator
def evaluate_model(adapter_fn: Callable[[np.ndarray], np.ndarray], graphs: List[np.ndarray], labels: List[np.ndarray]):
    accuracies = []

    for graph, true_label in tqdm(zip(graphs, labels), total=len(graphs), desc="Evaluating model"):
        pred = adapter_fn(graph)
        pred_binary = (pred > 0.5).astype(np.float32)
        accuracies.append(np.mean(pred_binary == true_label))

    avg_accuracy = np.mean(accuracies)
    return avg_accuracy
