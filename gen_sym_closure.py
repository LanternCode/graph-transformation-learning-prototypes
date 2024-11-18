from sklearn.model_selection import train_test_split

import hyperparameters
import torch
import random
from torch_geometric.data import Data


def generate_symmetric_closure_graph(num_nodes=12000, missing_edges_fraction=hyperparameters.missing_edge_fraction):
    # Step 1: Generate symmetric closure edges
    edges = []
    for _ in range(num_nodes):
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)

        if u != v:
            edges.append([u, v])  # Directed edge (u -> v)
            edges.append([v, u])  # Add the reverse edge (v -> u), ensuring symmetric closure

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Calculate number of symmetric edges
    num_edges = edge_index.size(1) // 2  # Each symmetric edge pair counts as one "edge"
    num_missing_edges = int(num_edges * missing_edges_fraction)

    # Randomly remove one direction of each symmetric pair to simulate missing edges
    missing_indices = random.sample(range(0, num_edges), num_missing_edges)

    # Create mask to remove only one edge in the symmetric pair
    mask = torch.ones(num_edges * 2, dtype=torch.bool)  # Full mask for all edges
    removed_edges = []  # Store removed edges for later use
    for idx in missing_indices:
        u, v = edge_index[:, idx * 2].tolist()  # Get node pair (u -> v)
        # Randomly decide whether to remove (u, v) or (v, u)
        if random.random() < 0.5:
            mask[idx * 2] = False  # Remove (u -> v)
            removed_edges.append([u, v])  # Store removed edge (u -> v)
        else:
            mask[idx * 2 + 1] = False  # Remove (v -> u)
            removed_edges.append([v, u])  # Store removed edge (v -> u)

    # Apply mask to get the final edge indices for training (remaining edges)
    train_pos_edge_index = edge_index[:, mask]  # Remaining edges (positive samples)

    # Split into incoming and outgoing edge indices
    outgoing_edge_index = train_pos_edge_index # Outgoing edges: (u -> v)
    incoming_edge_index = torch.stack([train_pos_edge_index[1], train_pos_edge_index[0]], dim=0) # Incoming edges: reverse of outgoing (v -> u)

    # Create negative samples (pairs of nodes that have no edges in either direction)
    neg_edges = []
    while len(neg_edges) < len(removed_edges):
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        # Ensure no edge exists in either direction (u -> v or v -> u)
        if u != v and [u, v] not in edges and [v, u] not in edges:
            neg_edges.append([u, v])

    neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t().contiguous()

    # Combine positive edges and negative samples into training data
    train_edge_index = torch.cat([train_pos_edge_index, neg_edge_index], dim=1)
    train_edge_labels = torch.cat([torch.ones(train_pos_edge_index.size(1)),
                                   torch.zeros(neg_edge_index.size(1))])

    # Convert removed edges to a tensor (these are the edges we want to predict)
    removed_edge_index = torch.tensor(removed_edges, dtype=torch.long).t().contiguous()

    # Create the data object with features and edge information
    data = Data(x=torch.ones(num_nodes, 1))  # Random node features
    data.train_edge_labels = train_edge_labels  # Labels for training edges (positive/negative)
    data.train_edge_index = train_edge_index  # Training edge indices

    # Store the removed edges in the data object for evaluation (symmetric closures)
    data.removed_edges = removed_edge_index  # These are the edges you want to predict
    data.neg_edge_index = neg_edge_index  # Negative edge samples

    # Add the separated incoming and outgoing edges
    data.incoming_edge_index = incoming_edge_index
    data.outgoing_edge_index = outgoing_edge_index
    data.edge_index = edge_index

    # Split `removed_edges` into validation and test sets, preserving direction
    # Separate source and target nodes from `removed_edges`
    src_nodes, dst_nodes = data.removed_edges[0], data.removed_edges[1]

    # Assuming removed_edges is a list or tensor of edge pairs (source, target)
    val_src, test_src, val_dst, test_dst = train_test_split(
        src_nodes, dst_nodes, test_size=0.5, random_state=42
    )

    # Combine src and dst into edge_index format
    data.val_edge_index = (val_src, val_dst)
    data.test_edge_index = (test_src, test_dst)

    print("Edge index: ")
    print(train_pos_edge_index)
    data.incomplete_pairs = count_incomplete_symmetrically_closed_pairs(train_pos_edge_index)
    print(data.incomplete_pairs)

    return data


def count_incomplete_symmetrically_closed_pairs(edge_index):
    """
    Counts the number of incomplete symmetrically closed pairs in a directed graph
    represented by a PyTorch tensor with edge indices.

    Parameters:
    edge_index (torch.Tensor): A tensor of shape [2, num_edges] where each column
                               represents a directed edge (start_node, end_node).

    Returns:
    int: The number of incomplete symmetrically closed pairs.
    """
    # Convert edge_index to a set of tuples for efficient lookup
    edge_set = set((edge_index[0, i].item(), edge_index[1, i].item()) for i in range(edge_index.size(1)))
    count = 0

    for u, v in edge_set:
        if (v, u) not in edge_set:
            count += 1

    return count


def get_data(num_nodes=hyperparameters.num_nodes, missing_edges_fraction=0.1):
    # Generate and check the graph
    data = generate_symmetric_closure_graph(num_nodes, missing_edges_fraction)
    return data
