import sys
import torch
import hyperparameters
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from gen_sym_closure import get_data, count_incomplete_symmetrically_closed_pairs, generate_graph_dataset
from model_gae_gcn import DirectedGAEGCN
from model_gae_gin import DirectedGAEGIN


def train_gae_gcn():
    # Instantiate the model and the optimiser
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DirectedGAEGCN(out_channels=hyperparameters.out_channels, hidden_channels=hyperparameters.hidden_channels,
                           num_nodes=hyperparameters.num_nodes, device=device)
    optimiser = torch.optim.Adam(model.parameters(), lr=hyperparameters.learning_rate)

    # Generate and load a dataset of 1000 graphs
    train_dataset, val_dataset, test_dataset = generate_graph_dataset(num_graphs=hyperparameters.num_graphs, min_nodes=30, max_nodes=3000)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_to_device)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_to_device)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_to_device)

    # Training loop
    for epoch in range(hyperparameters.epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            # Move the batch to GPU and fetch the necessary attributes
            batch = batch.to(device)
            edge_index_in = batch.incoming_edge_index
            edge_index_out = batch.outgoing_edge_index
            batch_idx = batch.batch

            # Encode the graph
            optimiser.zero_grad()
            z = model.encode(edge_index_in, edge_index_out, batch_idx)

            # Decode graph for supervised task (predict edges using the removed edges as evaluation set)
            fully_decoded_graph = model.decode_all(z, batch_idx)

            # Obtain ground-truth adjacency matrices
            training_graphs = [get_training_adj(batch, graph_id) for graph_id in torch.unique(batch.batch)]

            # Extract positive and negative edges for each graph
            pos_edge_indices = [torch.where(training_graph > 0) for training_graph in training_graphs]
            neg_edge_indices = [torch.where(training_graph == 0) for training_graph in training_graphs]

            # Compute loss
            loss = model.compute_loss(fully_decoded_graph, training_graphs, pos_edge_indices, neg_edge_indices)
            # loss = model.alternative_loss(fully_decoded_graph, pos_edge_indices, neg_edge_indices)

            # Propagate loss
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                z = model.encode(batch.incoming_edge_index, batch.outgoing_edge_index, batch.batch)
                decoded_graph = model.decode_all(z, batch.batch)
                training_graphs = [get_training_adj(batch, graph_id) for graph_id in torch.unique(batch.batch)]
                pos_edge_indices = [torch.where(training_graph > 0) for training_graph in training_graphs]
                neg_edge_indices = [torch.where(training_graph == 0) for training_graph in training_graphs]
                loss = model.compute_loss(decoded_graph, training_graphs, pos_edge_indices, neg_edge_indices)
                # loss = model.alternative_loss(fully_decoded_graph, pos_edge_indices, neg_edge_indices)
                total_val_loss += loss.item()

        # Print the loss for monitoring
        if epoch % hyperparameters.print_loss_every_n_epochs == 0:
            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {epoch + 1} Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    testing_loss, testing_accuracy = test_model(model, test_loader, device)
    print(f"Test Loss: {testing_loss:.4f}, Test Accuracy: {testing_accuracy:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'trained_many_gae_gcn.pth')
    print("Model saved to 'trained_many_gae_gcn.pth'")


# Collate a dataset batch to GPU when processing it
def collate_to_device(data_list):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = Batch.from_data_list(data_list)
    return batch.to(device)  # Move batch to GPU


# Fetch the ground truth (adjacency matrix) for a particular graph in the batch
def get_training_adj(batch, graph_id):
    # Get all edges for the specific graph
    mask = batch.batch == graph_id
    node_idx = torch.where(mask)[0]

    # Map edge indices to the local graph
    edges = batch.edge_index[:, mask[batch.edge_index[0]] & mask[batch.edge_index[1]]]

    # Create an adjacency matrix for the graph
    num_nodes = mask.sum().item()
    adj_matrix = torch.zeros((num_nodes, num_nodes), device=batch.x.device)

    # Adjust edge indices to the local graph
    local_mapping = {global_id.item(): local_id for local_id, global_id in enumerate(node_idx)}
    edges = torch.stack([torch.tensor([local_mapping[e.item()] for e in edges_row]) for edges_row in edges])

    # Populate the adjacency matrix
    adj_matrix[edges[0], edges[1]] = 1.0
    return adj_matrix


def test_model(model, test_loader, device):
    model.eval()
    total_test_loss = 0
    total_correct = 0
    total_edges = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            # Encode and decode the graphs
            z = model.encode(batch.incoming_edge_index, batch.outgoing_edge_index, batch.batch)
            fully_decoded_graph = model.decode_all(z, batch.batch)

            # Prepare ground-truth adjacency matrices
            test_graphs = [get_training_adj(batch, graph_id) for graph_id in torch.unique(batch.batch)]

            # Extract positive and negative edge indices
            pos_edge_indices = [torch.where(graph > 0) for graph in test_graphs]
            neg_edge_indices = [torch.where(graph == 0) for graph in test_graphs]

            # Compute loss for the batch
            loss = model.compute_loss(fully_decoded_graph, test_graphs, pos_edge_indices, neg_edge_indices)
            # loss = model.alternative_loss(fully_decoded_graph, pos_edge_indices, neg_edge_indices)
            total_test_loss += loss.item()

            # Compute accuracy
            for reconstructed_adj, ground_truth_adj, pos_indices, neg_indices in zip(
                fully_decoded_graph, test_graphs, pos_edge_indices, neg_edge_indices
            ):
                # Binary predictions for edges
                pred_binary = (reconstructed_adj > 0.5).float()

                # Compute accuracy on positive and negative edges
                correct_pos = (pred_binary[pos_indices] == ground_truth_adj[pos_indices]).sum().item()
                correct_neg = (pred_binary[neg_indices] == ground_truth_adj[neg_indices]).sum().item()
                total_correct += correct_pos + correct_neg

                # Total edges
                total_edges += len(pos_indices[0]) + len(neg_indices[0])

    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = total_correct / total_edges

    return avg_test_loss, test_accuracy


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
