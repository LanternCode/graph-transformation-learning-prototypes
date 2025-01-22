import sys

from sklearn.model_selection import train_test_split
from torch.utils.data import random_split

import hyperparameters
import torch
import random
from torch_geometric.data import Data


def generate_symmetric_closure_graph(num_nodes=12000, missing_edges_fraction=0.1):
    # Step 1: Generate symmetric closure edges
    edges = []
    for _ in range(num_nodes):
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)

        if u != v:
            if [u, v] not in edges and [v, u] not in edges:  # Ensure no duplicates
                edges.append([u, v])  # Directed edge (u -> v)
                edges.append([v, u])  # Add the reverse edge (v -> u), ensuring symmetric closure

    # Convert the edge list to a memory-contiguous tensor of shape (2, num_edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Calculate the number of symmetric edges
    num_edges = edge_index.size(1)

    # Calculate the number of edges to remove for validation and testing
    num_missing_edges = int(num_edges * missing_edges_fraction)

    # Initialize the full adjacency matrix for target labels
    ground_truth_adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    # Populate adj_matrix with 1s for edges in edge_index
    for idx in range(edge_index.size(1)):
        u, v = edge_index[:, idx].tolist()
        ground_truth_adj_matrix[u, v] = 1

    # Randomly remove one direction of each symmetric pair to simulate missing edges
    missing_indices = random.sample(range(0, num_edges), num_missing_edges)

    # Create mask to remove only one edge in the symmetric pair
    mask = torch.ones(num_edges, dtype=torch.bool)
    removed_edges = []  # Store removed edges for later use
    for idx in missing_indices:
        u, v = edge_index[:, idx].tolist()  # Get node pair (u -> v)
        # Randomly decide whether to remove (u, v) or (v, u)
        if random.random() < 0.5:
            mask[idx] = False  # Remove (u -> v)
            removed_edges.append([u, v])  # Store removed edge (u -> v)
        else:
            mask[idx + 1] = False  # Remove (v -> u)
            removed_edges.append([v, u])  # Store removed edge (v -> u)

    # Apply mask to get the final edge indices for training (remaining edges)
    edges_after_removal = edge_index[:, mask]  # Remaining edges (positive samples)

    # Split into incoming and outgoing edge indices
    outgoing_edge_index = edges_after_removal  # Outgoing edges: (u -> v)
    incoming_edge_index = torch.stack([edges_after_removal[1], edges_after_removal[0]], dim=0)  # Swap the ordering

    # Generate as many negative samples as there are removed_edges
    neg_edges = []
    while len(neg_edges) < len(removed_edges):
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        # Ensure no edge exists in either direction (u -> v or v -> u)
        if u != v and [u, v] not in edges and [v, u] not in edges:
            neg_edges.append([u, v])

    # Combine positive edges and negative samples into supervised training data
    neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t().contiguous()
    train_edge_index = torch.cat([edges_after_removal, neg_edge_index], dim=1)
    train_edge_labels = torch.cat([torch.ones(edges_after_removal.size(1)),
                                   torch.zeros(neg_edge_index.size(1))])

    # Create the Data object with features and edge information
    data = Data(x=torch.ones(num_nodes, 1))  # Initialise a dummy node features matrix with just 1s
    data.adjacency_matrix = ground_truth_adj_matrix  # Ground truth adjacency matrix (includes removed edges)
    data.train_edge_index = train_edge_index  # Training edge indices
    data.train_edge_labels = train_edge_labels  # Training edge labels
    data.incoming_edge_index = incoming_edge_index  # Target edges used when encoding
    data.outgoing_edge_index = outgoing_edge_index  # Source edges used when encoding
    data.edge_index = edge_index  # All generated edges, used during inference

    # Target number of pairs to complete - measured alongside the MCP loss
    #data.incomplete_closure_pairs = count_incomplete_symmetrically_closed_pairs(edges_after_removal)

    return data


def count_incomplete_symmetrically_closed_pairs(edge_index, is_adj_matrix=False):
    """
    Counts the number of incomplete symmetrically closed pairs in a directed graph
    represented by a PyTorch tensor with edge indices.

    Parameters:
    edge_index (torch.Tensor): A tensor of shape [2, num_edges] where each column
                               represents a directed edge (start_node, end_node).
    is_adj_matrix (bool): If an adjacency matrix is provided instead of a list, the
                                function will operate on it instead

    Returns:
    int: The number of incomplete symmetrically closed pairs.
    """
    if is_adj_matrix:
        # Ensure the adjacency matrix is binary (if it's weighted)
        adj_matrix = (edge_index > hyperparameters.edge_reconstruction_threshold).to(torch.int)

        # Identify asymmetric edges: A[u, v] = 1 and A[v, u] = 0
        asymmetric_matrix = (adj_matrix == 1) & (adj_matrix.t() == 0)

        # Count the number of asymmetric entries
        count = asymmetric_matrix.sum().item()
    else:
        # Convert edge_index to a set of tuples for efficient lookup
        edge_set = set((edge_index[0, i].item(), edge_index[1, i].item()) for i in range(edge_index.size(1)))
        count = 0

        for u, v in edge_set:
            if (v, u) not in edge_set:
                count += 1

    return count


def generate_random_directed_graph(num_nodes, num_edges):
    # Generate random edges
    edges = set()
    while len(edges) < num_edges:
        src, dest = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
        if src != dest:  # Avoid self-loops
            edges.add((src, dest))

    edges = list(edges)
    incoming_edges = [[] for _ in range(num_nodes)]
    outgoing_edges = [[] for _ in range(num_nodes)]

    for src, dest in edges:
        incoming_edges[dest].append(src)
        outgoing_edges[src].append(dest)

    # Convert to PyTorch tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    incoming_edge_index = torch.tensor([(s, d) for d in range(num_nodes) for s in incoming_edges[d]],
                                       dtype=torch.long).t().contiguous()
    outgoing_edge_index = torch.tensor([(s, d) for s in range(num_nodes) for d in outgoing_edges[s]],
                                       dtype=torch.long).t().contiguous()

    # Dummy features (all ones)
    x = torch.ones((num_nodes, 1), dtype=torch.float)

    # Create PyG data object
    data = Data(x=x, edge_index=edge_index)
    data.incoming_edge_index = incoming_edge_index
    data.outgoing_edge_index = outgoing_edge_index

    return data


def generate_graph_dataset(num_graphs, min_nodes=30, max_nodes=3000, missing_edges_fraction=0.2):
    """
    Generate a dataset of smaller graphs for training, with variable sizes.

    Args:
        num_graphs (int): Number of graphs to generate.
        min_nodes (int): Minimum number of nodes per graph.
        max_nodes (int): Maximum number of nodes per graph.
        missing_edges_fraction (float): Fraction of edges to remove for missing edges.

    Returns:
        train_dataset, val_dataset, test_dataset: Split datasets for training, validation, and testing.
    """
    dataset = []

    for _ in range(num_graphs):
        num_nodes = random.randint(min_nodes, max_nodes)

        # Step 1: Generate symmetrically-closed edges
        edges = set()
        while len(edges) < num_nodes * 4:
            u, v = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
            if u != v:
                edges.add((u, v))
                edges.add((v, u))
        edges = list(edges)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Step 2: Remove a fraction of edges
        num_edges = edge_index.size(1)
        num_missing_edges = int(num_edges * missing_edges_fraction)

        # Identify unique edges (undirected) by sorting node pairs
        unique_edges = {tuple(sorted(edge)) for edge in edge_index.T.tolist()}

        # Randomly select edges to remove
        missing_edges = random.sample(unique_edges, num_missing_edges)

        # Create a mask to mark edges for removal
        mask = torch.ones(num_edges, dtype=torch.bool)
        for edge in missing_edges:
            # Find the indices of the edges to remove in the original directed edge list
            u, v = edge
            for idx in range(edge_index.size(1)):
                if set(edge_index[:, idx].tolist()) == {u, v}:
                    mask[idx] = False
                    break  # Remove only one of the two directed edges
        edges_after_removal = edge_index[:, mask]
        removed_edges = edge_index[:, ~mask]

        # Step 3: Generate negative samples
        neg_edges = set()
        all_existing_edges = set(map(tuple, edges)).union(set(map(tuple, removed_edges.t().tolist())))
        while len(neg_edges) < edges_after_removal.size(1):
            u, v = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
            if u != v and (min(u, v), max(u, v)) not in all_existing_edges:
                neg_edges.add((u, v))
        neg_edges = list(neg_edges)
        neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t().contiguous()

        # Combine positive and negative samples
        train_edge_index = torch.cat([edges_after_removal, neg_edge_index], dim=1)
        train_edge_labels = torch.cat([torch.ones(edges_after_removal.size(1)),
                                       torch.zeros(neg_edge_index.size(1))])

        # Create the Data object
        data = Data(
            x=torch.ones(num_nodes, 1),
            train_edge_index=train_edge_index,
            train_edge_labels=train_edge_labels,
            edge_index=edges_after_removal,
            removed_edge_index=removed_edges,
            outgoing_edge_index=edges_after_removal,
            incoming_edge_index=torch.stack([edges_after_removal[1], edges_after_removal[0]], dim=0)
        )
        dataset.append(data)

    train_ratio, val_ratio = 0.8, 0.1
    total_graphs = len(dataset)
    train_size = int(total_graphs * train_ratio)
    val_size = int(total_graphs * val_ratio)
    test_size = total_graphs - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    return train_dataset, val_dataset, test_dataset


def create_graph(num_nodes, edge_list, label_edges):
    """
    A utility function to create a PyTorch Geometric Data object for a graph.

    Args:
        num_nodes (int): Number of nodes in the graph.
        edge_list (list of tuples): List of directed edges (source, target).
        label_edges (list of tuples): List of ground-truth edges for labels.

    Returns:
        Data: PyTorch Geometric Data object with labels.
    """
    # Edge index
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0),
                                                                                                          dtype=torch.long)

    # Incoming edge index
    incoming_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0) if edge_index.numel() > 0 else edge_index

    # Node features (simple: all ones)
    node_features = torch.ones(num_nodes, 1)

    # Ground-truth adjacency matrix (label)
    label_adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for src, tgt in label_edges:
        label_adj[src, tgt] = 1.0  # Set edges in the label

    return Data(
        x=node_features,  # Node features
        edge_index=edge_index,  # Edge list
        outgoing_edge_index=edge_index,  # Outgoing edges
        incoming_edge_index=incoming_edge_index,  # Incoming edges
        label=label_adj  # Ground-truth adjacency matrix
    )


def get_small_graphs_dataset():
    """
    Prepares a dataset of small graphs with labels (ground-truth adjacency matrices).

    Returns:
        list: A list of PyTorch Geometric Data objects.
    """
    graphs = [
        {"num_nodes": 2, "edges": [], "label_edges": [(0, 1), (1, 0)]},
        {"num_nodes": 2, "edges": [], "label_edges": [(0, 1), (1, 0)]},
        {"num_nodes": 2, "edges": [], "label_edges": [(0, 1), (1, 0)]},
        {"num_nodes": 2, "edges": [(0, 1)], "label_edges": [(0, 1), (1, 0)]},
        {"num_nodes": 2, "edges": [(1, 0)], "label_edges": [(0, 1), (1, 0)]},
        {"num_nodes": 2, "edges": [(0, 1), (1, 0)], "label_edges": [(0, 1), (1, 0)]}
    ]

    # Create dataset
    dataset = [create_graph(graph["num_nodes"], graph["edges"], graph["label_edges"]) for graph in graphs]
    return dataset


def get_data(num_nodes=hyperparameters.num_nodes, missing_edges_fraction=0.1):
    # Generate and check the graph
    data = generate_symmetric_closure_graph(num_nodes, missing_edges_fraction)
    data = data.to('cuda' if torch.cuda.is_available() else 'cpu')  # For running on the GPU
    return data
