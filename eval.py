import itertools
import sys
import torch


def decode_given_edges(model, inference_data):
    """
        Decodes the given edges from the latent representations

        Parameters:
        - model: The trained model (autoencoder).
        - inference_data: The Data object with nodes and edges.
        - num_inference_nodes: The total number of nodes in the graph.
        """
    with torch.no_grad():
        # Specify the edges to score
        edges_to_score = inference_data.test_edge_index

        # Obtain the latent representation and score the edges
        z = model.encode(inference_data.incoming_edge_index, inference_data.outgoing_edge_index)
        edge_probs = model.decode(z, edges_to_score)

        # Print the results
        torch.set_printoptions(sci_mode=False)
        print(f"Decoded probability return structure: : {edge_probs}")
        print(f"The mean is: {torch.mean(edge_probs)}")


def manually_decode_all_edges(model, inference_data, num_inference_nodes, threshold=0.5):
    """
    Decodes all possible edges from the latent representations to reconstruct the entire graph,
    excluding self-loops.

    Parameters:
    - model: The trained model (autoencoder).
    - inference_data: The Data object with nodes and edges.
    - num_inference_nodes: The total number of nodes in the graph.
    - threshold: Probability threshold for edge existence.
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
    print(f"Shape of decoded edges: {edge_probs.shape}")
    print(f"Number of symmetrical pairs: {symmetric_count}")
    print(f"How many edges you removed from H to get G: {len(missing_edges)}")
    print(f"How many edges were added (or fall above the threshold) in C(G): {len(edge_list)-inference_data.edge_index.shape[1]}")
    print(f"Percentage of symmetrically closed edges: {symmetric_percentage:.2f}%")


def decode_all(model, inference_data, num_inference_nodes, threshold=0.5):
    # Step 1: Decode the edges using the model's decoder
    with torch.no_grad():
        z = model.encode(inference_data.incoming_edge_index, inference_data.outgoing_edge_index, inference_data.batch)

        # Decode the graph
        edge_probs = model.decode_all(z, inference_data.batch)  # Predict probabilities for all possible edges
        edge_probs = edge_probs[0]  # Extract the single graph's adjacency matrix from the batch

    # Step 2: Threshold the adjacency matrix
    binary_adj_matrix = (edge_probs > threshold).float()

    # Step 3: Create an edge list from the thresholded adjacency matrix (excluding self-loops)
    edge_list = []
    for u in range(num_inference_nodes):
        row = binary_adj_matrix[u].long().cpu().numpy()  # Convert only the row to NumPy
        # Collect edges where value is 1 and exclude self-loops
        edge_list.extend([(u, v) for v in range(num_inference_nodes) if u != v and row[v] == 1])

    # Validate the symmetric closure of the edges
    symmetric_count, symmetric_percentage = validate_symmetric_closure(edge_list)

    # Convert edge_index to a list of tuples, and then both to sets for comparison and faster processing
    edge_index_conv_to_list = list(zip(inference_data.edge_index[0].tolist(), inference_data.edge_index[1].tolist()))
    input_edge_set = set(edge_index_conv_to_list)
    output_edge_set = set(edge_list)
    missing_edges = input_edge_set - output_edge_set

    print(f"Threshold: {threshold}")
    print(f"Number of nodes: {num_inference_nodes}")
    print(f"Number of edges in the original graph: {inference_data.edge_index.shape[1]}")
    print(f"Shape of decoded edges: {edge_probs.shape}")
    print(f"Number of edges after thresholding: {len(edge_list)}")
    print(f"Maximum probability in edge_probs: {edge_probs.max().item()}")
    print(f"Minimum probability in edge_probs: {edge_probs.min().item()}")
    print(f"Mean probability in edge_probs: {edge_probs.mean().item()}")
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
