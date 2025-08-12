import networkx as nx
import random
from typing import Optional, Dict, Any
import numpy as np


def generate_graph(graph_type: str, num_nodes: int, *, config: Optional[Dict[str, Any]] = None) -> nx.Graph:
    """
    Generate a single graph of specified type and node count.

    For graph_type == 'dag', you can pass in config:
      - max_depth_k: int (optional) longest path length cap; if set, uses layered DAG
      - max_retries: int (optional, default 200) retries for satisfying max_depth_k
      - p_range: tuple(float, float) (optional, default (0.05, 0.3)) density range for unconstrained DAG
    """
    config = config or {}

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

    if graph_type == 'dag':
        """
        A simple permutation-Bernoulli DAG:
          - pick a random node order
          - add edges only forward in that order with probability p
        Density control:
          - if 'expected_out_degree'=(lo, hi) is provided in config, set p=U(lo,hi)/(n-1)
          - else use 'p_range'=(lo,hi) in config or default (0.05, 0.30)
        """
        eout = config.get('expected_out_degree', None)  # e.g., (3.0, 6.0)
        if eout is not None:
            elow, ehigh = float(eout[0]), float(eout[1])
            p = random.uniform(elow, ehigh) / max(1, num_nodes - 1)
        else:
            plow, phigh = (0.05, 0.30)
            if 'p_range' in config:
                pr = config['p_range']
                plow, phigh = float(pr[0]), float(pr[1])
            p = random.uniform(plow, phigh)

        order = list(range(num_nodes))
        random.shuffle(order)

        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        for i in range(num_nodes):
            oi = order[i]
            for j in range(i + 1, num_nodes):
                oj = order[j]
                if random.random() < p:
                    G.add_edge(oi, oj)
        return G

    else:
        raise ValueError("Unknown graph type: {}".format(graph_type))
