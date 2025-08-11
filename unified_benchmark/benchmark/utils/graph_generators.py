import networkx as nx
import random


def generate_graph(graph_type: str, num_nodes: int) -> nx.Graph:
    """
    Generate a single graph of specified type and node count.
    """
    if graph_type == 'erdos_renyi':
        p = random.uniform(0.05, 0.3)
        return nx.erdos_renyi_graph(num_nodes, p)
    elif graph_type == 'barabasi_albert':
        m = min(5, num_nodes - 1)
        return nx.barabasi_albert_graph(num_nodes, m)
    elif graph_type == 'watts_strogatz':
        k = random.randint(2, min(num_nodes - 1, 6))
        p = random.uniform(0.1, 0.5)
        return nx.watts_strogatz_graph(num_nodes, k, p)
    elif graph_type == 'random_regular':
        d = random.randint(2, min(num_nodes - 1, 6))
        d = d if (num_nodes * d) % 2 == 0 else d - 1
        return nx.random_regular_graph(d, num_nodes) if d > 0 else nx.empty_graph(num_nodes)
    elif graph_type == 'stochastic_block':
        sizes = [num_nodes // 2, num_nodes - num_nodes // 2]
        p_in, p_out = random.uniform(0.1, 0.5), random.uniform(0.01, 0.1)
        probs = [[p_in, p_out], [p_out, p_in]]
        return nx.stochastic_block_model(sizes, probs)
    elif graph_type == 'powerlaw_cluster':
        m = min(5, num_nodes - 1)
        p = random.uniform(0.1, 0.5)
        return nx.powerlaw_cluster_graph(num_nodes, m, p)
    elif graph_type == 'random_geometric':
        radius = random.uniform(0.1, 0.5)
        return nx.random_geometric_graph(num_nodes, radius)
    elif graph_type == 'balanced_tree':
        r = random.randint(2, 4)
        h = 0
        while (r ** (h + 1) - 1) // (r - 1) < num_nodes:
            h += 1
        G = nx.balanced_tree(r, h)
        return G.subgraph(list(G.nodes)[:num_nodes]).copy()
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
