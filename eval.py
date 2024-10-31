import itertools
import torch


def decode_regular_edges(model, inference_data, num_inference_nodes):
    with torch.no_grad():
        z = model.encode(inference_data.incoming_edge_index, inference_data.outgoing_edge_index)
        edge_probs = model.decode(z, inference_data.removed_edges)
        adjacency_matrix = torch.zeros(num_inference_nodes, num_inference_nodes)

        # torch.set_printoptions(sci_mode=False)
        # print(f"Adjacency matrix structure: : {edge_probs}")
        # print(f"The mean is: {torch.mean(edge_probs)}")
        # sys.exit()

        # Fill in the adjacency matrix with the thresholded edge probabilities
        threshold = 0.5
        for idx, (u, v) in enumerate(inference_data.edge_index.t()):
            if edge_probs[idx] > threshold:
                adjacency_matrix[u, v] = 1

        adjacency_matrix = adjacency_matrix.long().cpu().numpy()

        # Step 4: Create an edge list from the thresholded adjacency matrix (excluding self-loops)
        edge_list = [(u, v) for u in range(num_inference_nodes) for v in range(num_inference_nodes) if
                     u != v and adjacency_matrix[u, v] == 1]

    # Validate the symmetric closure of the edges
    symmetric_count, symmetric_percentage = validate_symmetric_closure(edge_list)

    print(f"Threshold: {threshold}")
    print(f"Number of nodes: {num_inference_nodes}")
    print(f"Number of edges before thresholding: {inference_data.edge_index.shape}")
    print(f"Number of edges after thresholding: {len(edge_list)}")
    print(f"Number of symmetrical pairs: {symmetric_count}")
    print(f"Percentage of symmetrically closed edges: {symmetric_percentage:.2f}%")


def decode_all_edges(model, inference_data, num_inference_nodes, threshold=0.5):
    """
    Decodes all possible edges from the latent representations to reconstruct the entire graph,
    excluding self-loops.

    Parameters:
    - model: The trained model (autoencoder).
    - z: Latent representations of the nodes.
    - num_nodes: The total number of nodes in the graph.
    - threshold: Probability threshold for edge existence.

    Returns:
    - adjacency_matrix: A (num_nodes, num_nodes) numpy array representing the adjacency matrix.
    - edge_list: A list of tuples (u, v) where each pair represents an edge (excluding self-loops).
    """
    # Step 1: Create all possible edges between nodes (u, v), excluding self-loops
    edge_index = torch.tensor(
        [(u, v) for u, v in itertools.product(range(num_inference_nodes), repeat=2) if u != v],
        dtype=torch.long
    ).t().contiguous()

    # Step 2: Decode the edges using the model's decoder
    with torch.no_grad():
        z = model.encode(inference_data.incoming_edge_index, inference_data.outgoing_edge_index)
        edge_probs = model.decode(z, edge_index)  # Predict probabilities for all possible edges

    # Step 3: Threshold the probabilities to get the binary adjacency matrix
    adjacency_matrix = torch.zeros(num_inference_nodes, num_inference_nodes)
    for idx, (u, v) in enumerate(edge_index.t()):
        if edge_probs[idx] > threshold:
            adjacency_matrix[u, v] = 1

    # Step 4: Create an edge list from the thresholded adjacency matrix (excluding self-loops)
    adjacency_matrix = adjacency_matrix.long().cpu().numpy()
    edge_list = [(u, v) for u in range(num_inference_nodes) for v in range(num_inference_nodes) if u != v and adjacency_matrix[u, v] == 1]

    # Validate the symmetric closure of the edges
    symmetric_count, symmetric_percentage = validate_symmetric_closure(edge_list)

    # Convert edge_index to a list of tuples, and then both to sets
    edge_index_conv_to_list = list(zip(inference_data.edge_index[0].tolist(), inference_data.edge_index[1].tolist()))
    input_edge_set = set(edge_index_conv_to_list)
    output_edge_set = set(edge_list)
    missing_edges = input_edge_set - output_edge_set

    print(f"Threshold: {threshold}")
    print(f"Number of nodes: {num_inference_nodes}")
    print(f"Number of edges in the original graph: {inference_data.edge_index.shape[1]}")
    print(f"Number of edges in the fully connected graph: {edge_index.shape[1]}")
    print(f"Number of edges after thresholding: {len(edge_list)}")
    print(f"Number of symmetrical pairs: {symmetric_count}")
    print(f"How many edges you removed from H to get G: {len(missing_edges)}")
    print(f"How many edges were added (or fall above the threshold) in C(G): {len(edge_list)-inference_data.edge_index.shape[1]}")
    print(f"Percentage of symmetrically closed edges: {symmetric_percentage:.2f}%")


def validate_symmetric_closure(edge_list):
    # Convert the list of edges to a set for fast lookup
    edge_set = set(edge_list)
    counted_pairs = set()  # To keep track of pairs we've already counted
    symmetric_count = 0

    # Iterate through the edge list and check for symmetric pairs
    for (a, b) in edge_list:
        # Check if the reverse pair exists in the set and has not been counted yet
        if (b, a) in edge_set and (b, a) not in counted_pairs and (a, b) not in counted_pairs:
            symmetric_count += 1
            # Mark both (a, b) and (b, a) as counted
            counted_pairs.add((a, b))
            counted_pairs.add((b, a))

    # Step 5: Calculate the percentage of symmetrically closed edges
    total_edges = len(edge_list)
    symmetric_percentage = (symmetric_count * 2 / total_edges) * 100 if total_edges > 0 else 0

    return symmetric_count, symmetric_percentage
