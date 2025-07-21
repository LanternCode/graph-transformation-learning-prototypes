import numpy as np
import networkx as nx
import random
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support


def generate_cycle_graph(n):
    return nx.cycle_graph(n)


def generate_grid_graph(n):
    side = int(np.floor(np.sqrt(n)))
    G = nx.grid_2d_graph(side, side)
    return nx.convert_node_labels_to_integers(G)


def generate_tree(n):
    return nx.random_unlabeled_tree(n)


def generate_erdos_renyi(n, p=0.3):
    return nx.erdos_renyi_graph(n, p)


def generate_barabasi_albert(n, m):
    m = max(1, min(n - 1, m))
    return nx.barabasi_albert_graph(n, m)


def compute_minimal_chordal_edges(G):
    if nx.is_chordal(G):
        return set()
    chordal_G, _ = nx.complete_to_chordal_graph(G)
    return set(chordal_G.edges()) - set(G.edges())


def evaluate_prediction(true_edges, pred_edges, num_nodes):
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
