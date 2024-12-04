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
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')  # For running on the GPU
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

        # Compute the Missed Closure Pairs (MCP) Cross-Supervised Loss
        loss = model.compute_loss(
            fully_decoded_graph, data.adjacency_matrix
        )

        # Compute the number of incomplete closure pairs
        # num_missing_pairs = count_incomplete_symmetrically_closed_pairs(soft_binary_adj_matrix)

        # Print the training loss and incomplete pair count against the target for monitoring
        if epoch % 100 == 0:
            print(f"Epoch {epoch + 1} Loss: {loss.item():.4f}")
            #print(f"Reconstructed closed pairs against the target: {num_missing_pairs}/{data.incomplete_closure_pairs} ({(num_missing_pairs/data.incomplete_closure_pairs)*100:.2f}%)")

        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), 'trained_gae_gcn.pth')
    print("Model saved to 'trained_gae_gcn.pth'")


def train_gae_gin():
    # Example usage
    model = DirectedGAEGIN(out_channels=hyperparameters.out_channels, hidden_channels=hyperparameters.hidden_channels,
                        num_nodes=hyperparameters.num_nodes)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')  # For running on the GPU
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
