import random
import networkx as nx
from collections import defaultdict


def generate_directed_spanning_graphs(num_graphs=5, min_nodes=6, max_nodes=140):
    graphs = []
    for _ in range(num_graphs):
        num_nodes = random.randint(min_nodes, max_nodes)
        tree = nx.generators.trees.random_tree(n=num_nodes)

        # Root the tree at node 0, direct edges from parent to child
        bfs_edges = list(nx.bfs_edges(tree, source=0))
        directed_edges = [(u, v) for u, v in bfs_edges]

        # Track existing edges and their reverses to avoid symmetric additions
        edge_set = set(directed_edges) | set((v, u) for u, v in directed_edges)

        # Generate possible directed edges, avoiding symmetric and duplicate
        possible_edges = [
            (i, j)
            for i in range(num_nodes)
            for j in range(num_nodes)
            if i != j and (i, j) not in edge_set
        ]

        # Add same number of extra directed edges as in the tree
        extra_edge_count = len(directed_edges)
        extra_edges = random.sample(possible_edges, k=min(len(possible_edges), extra_edge_count))
        all_edges = directed_edges + extra_edges

        graph = {
            'edge_index': all_edges,
            'edge_label': None,
            'num_nodes': num_nodes
        }
        graphs.append(graph)

    return graphs


# Evaluation function for directed spanning tree predictions
def spanning_tree_score_from_prediction(graph, predicted_labels):
    edge_index = graph['edge_index']
    num_nodes = graph['num_nodes']
    V = num_nodes
    incoming_counts = [0] * V
    adj = defaultdict(list)

    # Build adjacency list and count incoming edges
    for (u, v), label in zip(edge_index, predicted_labels):
        if label == 1:
            adj[u].append(v)
            incoming_counts[v] += 1

    # Penalty for nodes with multiple incoming edges
    overconnection_penalty = sum(max(0, count - 1) for count in incoming_counts)

    # Identify roots (nodes with zero incoming edges)
    roots = [i for i, count in enumerate(incoming_counts) if count == 0]
    root_penalty = max(0, len(roots) - 1)

    # Traverse graph with DFS and detect cycles
    visited = set()
    on_path = set()
    cycle_detected = False

    def dfs(node):
        nonlocal cycle_detected
        visited.add(node)
        on_path.add(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                dfs(neighbor)
            elif neighbor in on_path:
                cycle_detected = True
        on_path.remove(node)

    if roots:
        dfs(roots[0])
    else:
        dfs(0)  # fallback root if none identified

    # Penalty for unreachable nodes
    unreachable_nodes = V - len(visited)

    # Total penalties
    total_penalty = overconnection_penalty + root_penalty + unreachable_nodes
    if cycle_detected:
        total_penalty += V  # strong penalty for cycles

    max_penalty = V
    final_score = 1.0 - min(1.0, total_penalty / max_penalty)

    return final_score


def evaluate_model(model_fn):
    graphs = generate_directed_spanning_graphs(num_graphs=1000)

    total_score = 0.0
    num_graphs = len(graphs)
    num_perfect = 0

    for graph in graphs:
        predicted_labels = model_fn(graph)
        score = spanning_tree_score_from_prediction(graph, predicted_labels)
        total_score += score
        if score == 1.0:
            num_perfect += 1

    average_score = total_score / num_graphs
    percentage_score = average_score * 100
    print(f"Average Spanning Tree Score: {average_score:.3f}")
    print(f"Correctness: {percentage_score:.2f}%")
    print(f"Correct Spanning Trees: {num_perfect}/1000")
