import random
import networkx as nx
from collections import defaultdict, deque


# Reduce node count to prevent memory issues
def generate_spanning_tree_graphs(num_graphs=5, min_nodes=6, max_nodes=140, label_edges=False):
    graphs = []
    for _ in range(num_graphs):
        num_nodes = random.randint(min_nodes, max_nodes)
        tree = nx.generators.trees.random_tree(n=num_nodes)
        edges = list(tree.edges())

        # Generate a limited number of extra edges
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        possible_edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes) if (i, j) not in edge_set]
        extra_edges = random.sample(possible_edges, k=min(len(possible_edges), num_nodes // 4))
        all_edges = edges + extra_edges

        graph = {
            'edge_index': all_edges,
            'edge_label': [1] * len(all_edges) if label_edges else None,
            'num_nodes': num_nodes
        }
        graphs.append(graph)
    return graphs


# Evaluation function using the scoring method from before
def spanning_tree_score_from_prediction(graph, predicted_labels):
    if not graph['edge_index'] or not predicted_labels:
        return 0.0

    # Build adjacency list from predicted edges labeled 1
    adj = defaultdict(list)
    nodes = set()
    for (u, v), label in zip(graph['edge_index'], predicted_labels):
        if label == 1:
            adj[u].append(v)
            adj[v].append(u)
            nodes.add(u)
            nodes.add(v)

    num_nodes = graph['num_nodes']
    V = num_nodes
    E = sum(predicted_labels)

    # Find connected components
    visited = set()
    components = 0

    def bfs(start):
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

    for node in range(num_nodes):
        if node not in visited:
            bfs(node)
            components += 1

    # Deviations
    missing_edges = max(0, V - 1 - E)
    extra_edges = max(0, E - (V - 1))
    disconnected = components - 1

    # Weighted penalties
    #alpha, beta, gamma = 0.5, 0.25, 0.25
    #penalty = alpha * disconnected + beta * missing_edges + gamma * extra_edges
    # score = max(0.0, 1.0 - penalty)
    penalty = disconnected + missing_edges + extra_edges
    penalty = 1 if penalty == 0 else penalty
    score = 1/penalty
    return score


def evaluate_model(graphs, model_fn):
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
    print(f"Correct Spanning Trees: {num_perfect}")
