import torch
import hyperparameters
from gen_sym_closure import get_data
from model_gae_gcn import DirectedGAEGCN
from model_gae_gin import DirectedGAEGIN


def train_gae_gcn():
    # Example usage
    model = DirectedGAEGCN(out_channels=hyperparameters.out_channels, hidden_channels=hyperparameters.hidden_channels,
                        num_nodes=hyperparameters.num_nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters.learning_rate)
    data = get_data()
    best_val_loss = 1000

    # Assuming x (node features), edge_index_in (incoming edges), edge_index_out (outgoing edges),
    # and true_edges (ground truth for existing edges) are provided.
    for epoch in range(hyperparameters.epochs + 1):
        optimizer.zero_grad()

        # Encode graph
        z = model.encode(data.incoming_edge_index, data.outgoing_edge_index)

        # Decode graph for supervised task (predict edges using the removed edges as evaluation set)
        supervised_pred = model.decode(z, data.train_edge_index)
        #fully_decoded_graph = model.decode_all(z)

        # Reconstruct the graph based on the predictions and compute the Missed Closure Pairs (MCP) Loss
        #TODO: HERE!!!

        # Compute loss
        loss = model.total_loss(
            supervised_pred, data.train_edge_labels  # For supervised (all removed edges should exist)
        )

        # Print the training loss for monitoring
        if epoch % 100 == 0:
            print(f"Epoch {epoch + 1} Supervised Loss: {loss.item():.4f}")

        loss.backward()
        optimizer.step()

        # Evaluate on the validation set every few epochs
        if epoch % hyperparameters.validation_frequency == 0:
            with torch.no_grad():
                val_pred = model.decode(z, data.val_edge_index)
                val_loss = model.total_loss(val_pred, torch.ones(len(data.val_edge_index[0])))
                print(f"Epoch {epoch + 1} Validation Loss: {val_loss.item():.4f}")

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save the model if this is the best validation performance so far
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= hyperparameters.patience:
                    print("Early stopping triggered")
                    break

    # Load the best model for testing
    model.load_state_dict(best_model_state)

    with torch.no_grad():
        test_pred = model.decode(z, data.test_edge_index)

        # Compute test metrics (assumes all test edges should exist, i.e., label=1)
        test_labels = torch.ones(len(data.test_edge_index[0]))  # All test edges are assumed to exist
        threshold = 0.8  # Example threshold for deciding "exists"

        # Binary accuracy or other metrics
        test_accuracy = ((test_pred >= threshold) == test_labels).float().mean()
        print(f"Test Accuracy: {test_accuracy.item()}")

    torch.save(best_model_state, 'trained_gae_gcn.pth')
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
