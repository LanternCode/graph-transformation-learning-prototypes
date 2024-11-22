import sys

import torch
import hyperparameters
from gen_sym_closure import get_data, count_incomplete_symmetrically_closed_pairs
from model_gae_gcn import DirectedGAEGCN
from model_gae_gin import DirectedGAEGIN


def train_gae_gcn():
    # Example usage
    model = DirectedGAEGCN(out_channels=hyperparameters.out_channels, hidden_channels=hyperparameters.hidden_channels,
                        num_nodes=hyperparameters.num_nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters.learning_rate)
    data = get_data()

    # Assuming x (node features), edge_index_in (incoming edges), edge_index_out (outgoing edges),
    # and true_edges (ground truth for existing edges) are provided.
    for epoch in range(hyperparameters.epochs + 1):
        optimizer.zero_grad()

        # Encode the graph
        z = model.encode(data.incoming_edge_index, data.outgoing_edge_index)

        # Decode graph for supervised task (predict edges using the removed edges as evaluation set)
        fully_decoded_graph = model.decode_all(z)

        # Soft reconstruct the graph based on the threshold so the adjacency matrix is binary
        chunk_size = 5000  # Chunk size to process in smaller parts
        soft_binary_adj_matrix = torch.zeros_like(fully_decoded_graph)
        for i in range(0, fully_decoded_graph.size(0), chunk_size):
            # Apply sigmoid thresholding in chunks
            soft_binary_adj_matrix[i:i + chunk_size] = torch.sigmoid(20 * (fully_decoded_graph[i:i + chunk_size] - 0.8))

        # Compute the Missed Closure Pairs (MCP) Cross-Supervised Loss
        loss = model.missing_closure_pairs_loss(
            soft_binary_adj_matrix, data.adjacency_matrix, data.loss_specific_edges
        )

        # Compute the number of incomplete closure pairs
        num_missing_pairs = count_incomplete_symmetrically_closed_pairs(soft_binary_adj_matrix)

        # Print the training loss and incomplete pair count against the target for monitoring
        if epoch % 100 == 0:
            print(f"Epoch {epoch + 1} MCP Loss: {loss.item():.4f}")
            print(f"Reconstructed closed pairs against the target: {num_missing_pairs}/{data.incomplete_closure_pairs} ({(num_missing_pairs/data.incomplete_closure_pairs)*100:.2f}%)")

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_pred = model.decode(z, data.test_edge_index)

        # Compute test metrics (assumes all test edges should exist, i.e., label=1)
        test_labels = torch.ones(len(data.test_edge_index[0]))  # All test edges are assumed to exist
        threshold = 0.8  # Example threshold for deciding "exists"

        # Binary accuracy or other metrics
        test_accuracy = ((test_pred >= threshold) == test_labels).float().mean()
        print(f"Test Accuracy: {test_accuracy.item()}")

    torch.save(model.state_dict(), 'trained_gae_gcn.pth')
    print("Model saved to 'trained_gae_gcn.pth'")


def train_gae_gin():
    # Example usage
    model = DirectedGAEGIN(out_channels=hyperparameters.out_channels, hidden_channels=hyperparameters.hidden_channels,
                        num_nodes=hyperparameters.num_nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters.learning_rate)
    data = get_data()

    # Assuming x (node features), edge_index_in (incoming edges), edge_index_out (outgoing edges),
    # and true_edges (ground truth for existing edges) are provided.
    for epoch in range(hyperparameters.epochs + 1):
        optimizer.zero_grad()

        # Encode graph
        z = model.encode(data.incoming_edge_index, data.outgoing_edge_index)

        # Decode graph for reconstruction (predict existing edges)
        reconstruction_pred = model.decode(z, data.train_edge_index)

        # Decode graph for supervised task (predict edges using the removed edges as evaluation set)
        supervised_pred = model.decode(z, data.train_edge_index)

        # Compute loss
        loss = model.total_loss(
            reconstruction_pred, data.train_edge_labels,  # For reconstruction
            supervised_pred, data.train_edge_labels,  # For supervised (all removed edges should exist)
            lambda_=hyperparameters.supervised_loss_factor,  # Weight for reconstruction vs supervised loss
            epoch_num=epoch
        )

        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Total Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'trained_gae_gin.pth')
    print("Model saved to 'trained_gae_gin.pth'")


train_gae_gcn()
