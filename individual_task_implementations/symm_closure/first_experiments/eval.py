"""
Early evaluation and visualization utilities for graph autoencoder prototypes.

This file contains thresholding, decoding, symmetric-closure counting, and graph
visualization helpers used during the first feasibility experiments. The printed
metrics are exploratory diagnostics and should not be treated as final benchmark
definitions. Later cleaned evaluation scripts supersede these utilities.
"""
import itertools
import sys
import time
import torch
import networkx as nx
import matplotlib.pyplot as plt


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

    # Compute the required metrics
    symmetric_count, symmetric_percentage = validate_symmetric_closure(edge_list)
    input_edge_set = set(zip(inference_data.edge_index[0].tolist(), inference_data.edge_index[1].tolist()))
    output_edge_set = set(edge_list)
    missing_edges = input_edge_set - output_edge_set
    recon_edge_target = inference_data.incomplete_closure_pairs
    local_mvp_result = len(inference_data.removed_edge_set & output_edge_set)

    # Visualise graphs for comparison (only works well for small graphs)
    # validate_node_ordering(edge_list, inference_data)
    if num_inference_nodes < 20:
        visualise_two_graphs(edge_list, inference_data.outgoing_edge_index, num_inference_nodes)

    print(f"Threshold: {threshold}")
    print(f"Number of generated nodes: {num_inference_nodes}")
    print(f"Number of purged empty nodes: {inference_data.empty_nodes}")
    print(f"Number of remaining nodes: {num_inference_nodes}")
    print(f"Number of edges in the original graph: {inference_data.edge_index.shape[1]}")
    print(f"Number of edges in the reconstructed graph: {len(edge_list)}")
    print(f"Minimum probability in edge_probs: {edge_probs.min().item()}")
    print(f"Mean probability in edge_probs: {edge_probs.mean().item()}")
    print(f"Maximum probability in edge_probs: {edge_probs.max().item()}")
    print(f"How many edges you removed from H to get G: {len(missing_edges)}")
    print(f"How many edges were added (or fall above the threshold) in C(G): "
          f"{inference_data.edge_index.size(1) - len(edge_list)}")  # #edges H - #edges C(G)
    print(f"Number of symmetrically-closed pairs: {symmetric_count}")
    if symmetric_percentage is not None:
        print(f"Percentage of symmetrically closed edges: {symmetric_percentage:.2f}%")
        print(f"Precision (Global MVP Accuracy): {(1 / (len(edge_list) / recon_edge_target)) * 100:.2f}%")
        print(f"Recall (How many of the required missing edges were constructed): {(local_mvp_result / recon_edge_target) * 100:.2f}%\n")
    else:
        print("Percentage of symmetrically closed edges: No edges were reconstructed")
        print(f"Precision (Global MVP Accuracy): 0% (No edges were reconstructed)")
        print(f"Recall (How many of the required missing edges were constructed): 0% (No edges were reconstructed)\n")


def decode_and_union_all(model, inference_data, num_inference_nodes, threshold=0.5):
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

    # Convert edge_list and edge indices to sets of tuples for union operation
    edge_set_from_probs = set(edge_list)

    # Combine the inference edge indices
    incoming_edges = zip(inference_data.incoming_edge_index[0].tolist(),
                         inference_data.incoming_edge_index[1].tolist())
    outgoing_edges = zip(inference_data.outgoing_edge_index[0].tolist(),
                         inference_data.outgoing_edge_index[1].tolist())

    # Create sets from the indices
    incoming_edge_set = set(incoming_edges)
    outgoing_edge_set = set(outgoing_edges)

    # Union all edge sets
    final_edge_set = edge_set_from_probs | incoming_edge_set | outgoing_edge_set

    # Convert back to a list if needed
    final_edge_list = list(final_edge_set)

    # Validate the symmetric closure of the edges
    symmetric_count, symmetric_percentage = validate_symmetric_closure(final_edge_list)

    print(f"Threshold: {threshold}")
    print(f"Number of nodes: {num_inference_nodes}")
    print(f"Number of edges in the original graph: {inference_data.edge_index.shape[1]}")
    print(f"Shape of decoded edges: {edge_probs.shape}")
    print(f"Number of edges after thresholding: {len(edge_list)}")
    print(f"Number of edges after union: {len(final_edge_list)}")
    print(f"Maximum probability in edge_probs: {edge_probs.max().item()}")
    print(f"Minimum probability in edge_probs: {edge_probs.min().item()}")
    print(f"Mean probability in edge_probs: {edge_probs.mean().item()}")
    print(f"Number of symmetrical pairs: {symmetric_count}")
    print(f"How many edges were added (or fall above the threshold) in C(G): {len(final_edge_list)-inference_data.edge_index.shape[1]}")
    print(f"Percentage of symmetrically closed edges: {symmetric_percentage:.2f}%")


def decode_all_with_directions(model, inference_data, num_inference_nodes, threshold=0.5):
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

    # Extract probabilities for edges in the input graph
    input_edge_probs = {}
    for idx in range(inference_data.incoming_edge_index.shape[1]):
        u, v = inference_data.incoming_edge_index[0, idx].item(), inference_data.incoming_edge_index[1, idx].item()
        input_edge_probs[(u, v)] = edge_probs[u, v].item()

    # Call the function to create the new edge index
    new_edge_index, new_edge_probs = create_asymmetric_edge_index(
        edge_index=inference_data.incoming_edge_index,  # Use the input edge index
        edge_probs=edge_probs,  # Adjacency matrix of probabilities
        num_nodes=num_inference_nodes  # Total number of nodes
    )

    visualise_three_graphs(
        edge_list=edge_list,
        edge_index=inference_data.incoming_edge_index,
        new_edge_index=new_edge_index,
        new_edge_probs=new_edge_probs,
        num_nodes=num_inference_nodes,
        input_edge_probs=input_edge_probs  # Pass the input edge probabilities
    )

    print(f"Threshold: {threshold}")
    print(f"Number of generated nodes: {num_inference_nodes}")
    print(f"Number of purged empty nodes: {inference_data.empty_nodes}")
    print(f"Number of remaining nodes: {num_inference_nodes}")
    print(f"Number of edges in the original graph: {inference_data.edge_index.shape[1]}")
    print(f"Number of edges in the reconstructed graph: {len(edge_list)}")
    print(f"Minimum probability in edge_probs: {edge_probs.min().item()}")
    print(f"Mean probability in edge_probs: {edge_probs.mean().item()}")
    print(f"Maximum probability in edge_probs: {edge_probs.max().item()}")
    # print(f"How many edges you removed from H to get G: {len(missing_edges)}")
    print(f"How many edges were added (or fall above the threshold) in C(G): "
          f"{inference_data.edge_index.size(1) - len(edge_list)}")  # #edges H - #edges C(G)
    print(f"Number of symmetrically-closed pairs: {symmetric_count}")
    print(f"Percentage of symmetrically closed edges: {symmetric_percentage:.2f}%\n")


def create_asymmetric_edge_index(edge_index, edge_probs, num_nodes):
    """
    Create a new edge index containing edges without symmetric closure
    and their corresponding symmetric counterparts with probabilities.

    Args:
        edge_index (torch.Tensor): The input edge index (2 x num_edges).
        edge_probs (torch.Tensor): Adjacency matrix of edge probabilities (num_nodes x num_nodes).
        num_nodes (int): Total number of nodes.

    Returns:
        torch.Tensor: New edge index (2 x num_edges).
        torch.Tensor: Corresponding probabilities for the edges in the new edge index.
    """
    # Convert edge_index to a set of tuples for easier symmetric closure checking
    edge_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    new_edges = []
    new_probs = []

    for u, v in edge_set:
        # Check if symmetric edge (v, u) exists in the input edge set
        if (v, u) not in edge_set:
            # Add (u, v) and its probability
            new_edges.append((u, v))
            new_probs.append(edge_probs[u, v].item())

            # Add the symmetric counterpart (v, u) and its probability
            new_edges.append((v, u))
            new_probs.append(edge_probs[v, u].item())

    # Convert the new_edges back to a tensor
    new_edge_index = torch.tensor(new_edges, dtype=torch.long).T  # Shape: 2 x num_new_edges
    new_edge_probs = torch.tensor(new_probs, dtype=torch.float)  # Shape: num_new_edges

    return new_edge_index, new_edge_probs


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
    symmetric_percentage = (symmetric_count * 2 / total_edges) * 100 if total_edges > 0 else None

    return symmetric_count, symmetric_percentage


def visualise_two_graphs(edge_list, edge_index, num_nodes):
    """
    Visualize the edge_list and the edge_index as two graphs.

    Args:
        edge_list (list of tuples): Edges from the thresholded adjacency matrix.
        edge_index (torch.Tensor): Original edge index from inference_data.
        num_nodes (int): Number of nodes in the graph.
    """
    # Convert edge_index to a list of tuples
    edge_index_tuples = list(zip(edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()))

    # Create NetworkX graphs
    G1 = nx.DiGraph()  # For edge_list
    G2 = nx.DiGraph()  # For edge_index

    # Add nodes and edges
    G1.add_nodes_from(range(num_nodes))
    G1.add_edges_from(edge_list)

    G2.add_nodes_from(range(num_nodes))
    G2.add_edges_from(edge_index_tuples)

    # Plot the graphs
    plt.figure(figsize=(12, 6))

    # Plot edge_list graph
    plt.subplot(1, 2, 1)
    nx.draw_spring(G1, with_labels=True, node_color='skyblue', edge_color='blue', node_size=500, font_size=10)
    plt.title("Reconstructed Graph")

    # Plot edge_index graph
    plt.subplot(1, 2, 2)
    nx.draw_spring(G2, with_labels=True, node_color='lightgreen', edge_color='green', node_size=500, font_size=10)
    plt.title("Input Graph")

    output_file = f"graph_visualization{time.time()}.png"
    plt.savefig(output_file, format='png', bbox_inches='tight')
    plt.close()


def validate_node_ordering(edge_list, inference_data):
    # Extract unique nodes from both graphs
    nodes_from_edge_list = set(u for u, v in edge_list) | set(v for u, v in edge_list)
    nodes_from_edge_index = set(inference_data.edge_index[0].cpu().numpy()) | set(
        inference_data.edge_index[1].cpu().numpy())

    # Compare the node sets
    print("Nodes in edge_list:", nodes_from_edge_list)
    print("Nodes in edge_index:", nodes_from_edge_index)

    if nodes_from_edge_list == nodes_from_edge_index:
        print("The node numbering is consistent between the two graphs.")
    else:
        print("The node numbering is inconsistent!")


def visualise_three_graphs(edge_list, edge_index, new_edge_index, new_edge_probs, num_nodes, input_edge_probs=None, arrowsize=14, k=7):
    """
    Visualize the original input graph, reconstructed graph, and the new edge index with probabilities.

    Args:
        edge_list (list of tuples): Edges from the thresholded adjacency matrix.
        edge_index (torch.Tensor): Original edge index from inference_data.
        new_edge_index (torch.Tensor): New edge index with asymmetric edges.
        new_edge_probs (torch.Tensor): Probabilities corresponding to new_edge_index.
        num_nodes (int): Number of nodes in the graph.
        input_edge_probs (dict): Probabilities of edges in the input graph (optional).
        arrowsize (int): Size of the arrows in the directed graph.
        k (float): Optimal distance between nodes in the layout (higher means more spacing).
    """
    # Convert edge_index and new_edge_index to lists of tuples
    edge_index_tuples = list(zip(edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()))
    new_edge_tuples = list(zip(new_edge_index[0].cpu().numpy(), new_edge_index[1].cpu().numpy()))

    # Create NetworkX graphs
    G1 = nx.DiGraph()  # For edge_list
    G2 = nx.DiGraph()  # For edge_index
    G3 = nx.DiGraph()  # For new_edge_index

    # Add nodes and edges
    G1.add_nodes_from(range(num_nodes))
    G1.add_edges_from(edge_list)

    G2.add_nodes_from(range(num_nodes))
    G2.add_edges_from(edge_index_tuples)

    G3.add_nodes_from(range(num_nodes))
    G3.add_edges_from(new_edge_tuples)

    # Annotate edges with probabilities for G3
    edge_labels = {(u, v): f"{new_edge_probs[i]:.2f}" for i, (u, v) in enumerate(new_edge_tuples)}
    input_edge_labels = {(u, v): f"{label:.2f}" for (u, v), label in
                         (input_edge_probs.items() if input_edge_probs else {})}

    # Set layout parameters with increased spacing for nodes
    layout_1 = nx.spring_layout(G1, k=k, iterations=50)
    layout_2 = nx.spring_layout(G2, k=k, iterations=50)
    layout_3 = nx.spring_layout(G3, k=k, iterations=50)

    # Plot the graphs
    plt.figure(figsize=(18, 6))

    # Plot edge_list graph
    plt.subplot(1, 3, 1)
    nx.draw(G1, pos=layout_1, with_labels=True, node_color='skyblue', edge_color='blue',
            node_size=400, font_size=12, width=2, arrows=True, arrowsize=arrowsize)
    plt.title("Reconstructed Graph")

    # Plot edge_index graph
    plt.subplot(1, 3, 2)
    nx.draw(G2, pos=layout_2, with_labels=True, node_color='lightgreen', edge_color='green',
            node_size=400, font_size=12, width=2, arrows=True, arrowsize=arrowsize)
    for (u, v), label in input_edge_labels.items():
        x, y = (layout_2[u] + layout_2[v]) / 2  # Midpoint of the edge
        plt.text(x, y, label, fontsize=10, color='black', ha='center', va='center')
    plt.title("Input Graph")

    # Plot new_edge_index graph with labels integrated
    plt.subplot(1, 3, 3)
    nx.draw(
        G3, pos=layout_3, with_labels=True, node_color='lightcoral', edge_color='red',
        node_size=400, font_size=12, width=2, arrows=True, arrowsize=arrowsize,
        connectionstyle="arc3,rad=0.1"
    )
    # Render edge labels directly
    for (u, v), label in edge_labels.items():
        x, y = (layout_3[u] + layout_3[v]) / 2  # Midpoint of the edge
        plt.text(x, y, label, fontsize=10, color='black', ha='center', va='center')

    plt.title("Required Symmetric Closures")

    # Save the plots to a file
    output_file = f"graph_visualization_{time.time():.0f}.png"
    plt.savefig(output_file, format='png', bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_file}")
