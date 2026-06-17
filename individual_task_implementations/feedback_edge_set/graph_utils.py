import math
import random
import networkx as nx
import numpy as np

DEFAULT_MIN_NODES = 6
DEFAULT_MAX_NODES = 140


def edge_in_cycle(graph, edge):
    """
    Determine whether an undirected edge belongs to at least one cycle.

    Args:
        graph (networkx.Graph): The graph that contains the edge to inspect.
        edge (tuple): A two-node edge tuple ``(u, v)`` from ``graph``.

    Returns:
        bool: ``True`` if removing the edge leaves its endpoints connected,
        meaning the edge is part of a cycle; otherwise ``False``.
    """
    graph_without_edge = graph.copy()
    graph_without_edge.remove_edge(*edge)
    return nx.has_path(graph_without_edge, edge[0], edge[1])


def generate_graph_pair(pct_extra=0, min_nodes=DEFAULT_MIN_NODES, max_nodes=DEFAULT_MAX_NODES):
    """
    Generate a connected graph and cycle-edge labels.

    The graph starts as a random tree, then receives extra non-tree edges. The
    number of added edges is computed as a percentage of the tree edge count, so
    ``pct_extra=10`` adds up to ten percent as many extra edges as the tree has.

    Args:
        pct_extra (float): Percentage of the tree edge count to add as extra
            edges. Values are capped by the number of available non-tree edges.
        min_nodes (int): Inclusive lower bound for the random node count.
        max_nodes (int): Inclusive upper bound for the random node count.

    Returns:
        tuple: ``(full_graph, labels)``, where ``full_graph`` is a
        ``networkx.Graph`` and ``labels`` maps each edge to ``1`` if it is part
        of a cycle and ``0`` otherwise.
    """
    num_nodes = random.randint(min_nodes, max_nodes)
    tree = nx.Graph(nx.random_unlabeled_tree(num_nodes))
    base_edges = list(tree.edges())

    existing = set(base_edges) | set((v, u) for u, v in base_edges)
    candidates = [
        (i, j)
        for i in range(num_nodes)
        for j in range(i + 1, num_nodes)
        if (i, j) not in existing
    ]
    random.shuffle(candidates)

    num_extra = min(math.ceil(pct_extra / 100 * len(base_edges)), len(candidates))
    extra_edges = candidates[:num_extra]

    full_graph = nx.Graph()
    full_graph.add_nodes_from(tree.nodes())
    full_graph.add_edges_from(base_edges + extra_edges)

    labels = {edge: int(edge_in_cycle(full_graph, edge)) for edge in full_graph.edges()}
    return full_graph, labels


def extract_features(graph):
    """
    Extract edge-level structural features from a graph.

    Args:
        graph (networkx.Graph): The graph whose edges should be featurized.

    Returns:
        tuple: ``(features, edges)``, where ``features`` is a NumPy array with
        one row per edge and columns ``[deg_u, deg_v, edge_betweenness]``, and
        ``edges`` is the corresponding list of edge tuples in row order.
    """
    features = []
    edges = list(graph.edges())
    degree = dict(graph.degree())
    betweenness = nx.edge_betweenness_centrality(graph)

    for u, v in edges:
        features.append([
            degree[u],
            degree[v],
            betweenness.get((u, v), 0) or betweenness.get((v, u), 0),
        ])

    return np.array(features, dtype=np.float32), edges


def is_acyclic(graph):
    """
    Check whether an undirected graph has no cycles.

    Args:
        graph (networkx.Graph): The graph to inspect.

    Returns:
        bool: ``True`` if NetworkX finds no cycle in the graph; otherwise
        ``False``.
    """
    try:
        nx.find_cycle(graph)
    except nx.NetworkXNoCycle:
        return True
    return False


def make_benchmark_graphs(num_graphs=1000, pct_extra=0):
    """
    Generate a reusable benchmark graph set.

    Args:
        num_graphs (int): Number of graph-label pairs to generate.
        pct_extra (float): Percentage of each tree's edge count to add as extra
            edges for each generated graph.

    Returns:
        list: A list of ``(graph, labels)`` tuples produced by
        ``generate_graph_pair``.
    """
    return [generate_graph_pair(pct_extra) for _ in range(num_graphs)]
