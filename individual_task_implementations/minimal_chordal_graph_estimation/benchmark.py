import numpy as np
import networkx as nx
import random
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support


def generate_cycle_graph(n):
    """
    Generate an undirected cycle graph with n nodes.

    Args:
        n: Number of nodes to include in the cycle graph.

    Returns:
        A NetworkX graph containing one cycle over n nodes.
    """
    return nx.cycle_graph(n)


def generate_grid_graph(n):
    """
    Generate an undirected grid-style graph with exactly n nodes.

    The largest square grid that fits within n nodes is generated first. If the
    square grid contains fewer than n nodes, additional isolated nodes are added
    so that fixed-size model inputs receive an adjacency matrix with the
    requested number of nodes.

    Args:
        n: Target number of nodes in the returned graph.

    Returns:
        A NetworkX graph with exactly n integer-labeled nodes.
    """
    side = int(np.floor(np.sqrt(n)))
    G = nx.grid_2d_graph(side, side)
    G = nx.convert_node_labels_to_integers(G)
    if G.number_of_nodes() < n:
        G.add_nodes_from(range(G.number_of_nodes(), n))
    return G


def generate_tree(n):
    """
    Generate an undirected random tree with n nodes.

    Args:
        n: Number of nodes in the generated tree.

    Returns:
        A NetworkX graph sampled as an unlabeled random tree.
    """
    return nx.random_unlabeled_tree(n)


def generate_erdos_renyi(n, p=0.3):
    """
    Generate an undirected Erdős-Rényi graph.

    Args:
        n: Number of nodes in the generated graph.
        p: Probability of including each possible undirected edge.

    Returns:
        A NetworkX Erdős-Rényi graph with n nodes.
    """
    return nx.erdos_renyi_graph(n, p)


def generate_barabasi_albert(n, m):
    """
    Generate an undirected Barabási-Albert graph.

    Args:
        n: Number of nodes in the generated graph.
        m: Number of edges to attach from each new node, clipped to the valid
            range for the requested graph size.

    Returns:
        A NetworkX Barabási-Albert graph with n nodes.
    """
    m = max(1, min(n - 1, m))
    return nx.barabasi_albert_graph(n, m)


def compute_minimal_chordal_edges(G):
    """
    Compute chordal-completion fill edges for an undirected graph.

    Args:
        G: Input NetworkX graph whose chordal completion should be computed.

    Returns:
        A set of edges that appear in NetworkX's chordal completion of G but do
        not appear in the original graph.
    """
    if nx.is_chordal(G):
        return set()
    chordal_G, _ = nx.complete_to_chordal_graph(G)
    return set(chordal_G.edges()) - set(G.edges())


def evaluate_prediction(true_edges, pred_edges, num_nodes):
    """
    Evaluate predicted chordal-fill edges against target fill edges.

    Args:
        true_edges: Iterable of target fill edges.
        pred_edges: Iterable of predicted fill edges.
        num_nodes: Number of nodes used to construct the comparison matrices.

    Returns:
        A tuple containing precision, recall, and F1 score for the predicted
        upper-triangular fill-edge mask.
    """
    true_matrix = np.zeros((num_nodes, num_nodes))
    pred_matrix = np.zeros((num_nodes, num_nodes))
    for u, v in true_edges:
        true_matrix[u, v] = true_matrix[v, u] = 1
    for u, v in pred_edges:
        pred_matrix[u, v] = pred_matrix[v, u] = 1
    true_flat = true_matrix[np.triu_indices(num_nodes, k=1)]
    pred_flat = pred_matrix[np.triu_indices(num_nodes, k=1)]
    precision, recall, f1, _ = precision_recall_fscore_support(true_flat, pred_flat, average='binary', zero_division=0)
    return precision, recall, f1


def run_benchmark(predict_fn, num_graphs=1000, num_nodes=150):
    """
    Run the chordal-fill prediction benchmark over generated graph families.

    Args:
        predict_fn: Callable that accepts an adjacency matrix and returns a
            matrix of predicted fill-edge scores or binary predictions.
        num_graphs: Number of non-chordal benchmark graphs to evaluate.
        num_nodes: Number of nodes requested for each benchmark graph.

    Returns:
        None. The benchmark prints mean precision, recall, and F1 score.
    """
    generators = [
        lambda: generate_cycle_graph(num_nodes),
        lambda: generate_grid_graph(num_nodes),
        lambda: generate_tree(num_nodes),
        lambda: generate_erdos_renyi(num_nodes),
        lambda: generate_barabasi_albert(num_nodes, max(1, num_nodes // 10)),
    ]

    precision_scores, recall_scores, f1_scores = [], [], []
    attempts = 0
    successful = 0

    print("Generating and evaluating graphs:")
    pbar = tqdm(total=num_graphs)

    while successful < num_graphs and attempts < 10 * num_graphs:
        attempts += 1
        try:
            G = random.choice(generators)()
            G = nx.convert_node_labels_to_integers(G)
            if nx.is_chordal(G):  # skip if already chordal
                continue

            true_edges = compute_minimal_chordal_edges(G)
            adj_matrix = nx.to_numpy_array(G)
            pred_matrix = predict_fn(adj_matrix)
            pred_edges = [(i, j) for i in range(num_nodes) for j in range(i+1, num_nodes) if pred_matrix[i, j] > 0.5]
            p, r, f = evaluate_prediction(true_edges, pred_edges, num_nodes)

            precision_scores.append(p)
            recall_scores.append(r)
            f1_scores.append(f)

            successful += 1
            pbar.update(1)
        except Exception as e:
            continue

    pbar.close()

    print("\nBenchmark Results:")
    print(f"Evaluated Graphs: {successful}")
    print(f"Precision: {np.mean(precision_scores):.4f}")
    print(f"Recall:    {np.mean(recall_scores):.4f}")
    print(f"F1 Score:  {np.mean(f1_scores):.4f}")
