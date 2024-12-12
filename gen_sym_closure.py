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


def generate_graph_dataset(num_graphs, min_nodes=30, max_nodes=3000, missing_edges_fraction=0.1):
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

        # Step 1: Generate edges
        edges = set()
        while len(edges) < num_nodes * 2:
            u, v = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
            if u != v:
                edges.add((min(u, v), max(u, v)))
        edges = list(edges)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Step 2: Remove a fraction of edges
        num_edges = edge_index.size(1)
        num_missing_edges = int(num_edges * missing_edges_fraction)
        missing_indices = random.sample(range(num_edges), num_missing_edges)

        mask = torch.ones(num_edges, dtype=torch.bool)
        mask[missing_indices] = False
        edges_after_removal = edge_index[:, mask]
        removed_edges = edge_index[:, ~mask]

        # Validate removed edges
        assert removed_edges[0].max() < num_nodes and removed_edges[1].max() < num_nodes, \
            "Removed edge indices exceed valid node range."

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

    for i, graph in enumerate(dataset):
        assert graph.edge_index.max() < graph.x.size(0), f"Graph {i}: edge_index contains invalid nodes!"
        assert graph.removed_edge_index.max() < graph.x.size(
            0), f"Graph {i}: removed_edge_index contains invalid nodes!"

    return train_dataset, val_dataset, test_dataset


def get_data(num_nodes=hyperparameters.num_nodes, missing_edges_fraction=0.1):
    # Generate and check the graph
    data = generate_symmetric_closure_graph(num_nodes, missing_edges_fraction)
    data = data.to('cuda' if torch.cuda.is_available() else 'cpu')  # For running on the GPU
    return data
