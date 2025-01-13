import sys
import torch
import hyperparameters
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from gen_sym_closure import get_data, count_incomplete_symmetrically_closed_pairs, generate_graph_dataset, \
    get_small_graphs_dataset
from model_gae_gcn import DirectedGAEGCN
from model_gae_gin import DirectedGAEGIN


def train_gae_gcn(loss_function):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DirectedGAEGCN(out_channels=hyperparameters.out_channels,
                           hidden_channels=hyperparameters.hidden_channels,
                           num_nodes=hyperparameters.num_nodes, device=device)
    optimiser = torch.optim.Adam(model.parameters(), lr=hyperparameters.learning_rate)

    train_dataset, val_dataset, test_dataset = generate_graph_dataset(num_graphs=hyperparameters.num_graphs)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_to_device)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_to_device)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_to_device)

    for epoch in range(hyperparameters.epochs + 1):
        avg_train_loss = train_epoch(model, train_loader, optimiser, device, epoch, loss_function)
        avg_val_loss = evaluate(model, val_loader, device, loss_function)
        if epoch % 25 == 0:
            print(f"Epoch {epoch + 1} Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    test_loss, test_accuracy = test_model(model, test_loader, device, loss_function)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    torch.save(model.state_dict(), 'results/gcn_bce_avg.pth')
    print("Model saved to 'model_gcn_altloss.pth'")


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


def prepare_edge_indices(batch):
    """Prepare edge indices for a batch of graphs."""
    pos_edge_indices = []
    neg_edge_indices = []
    removed_edge_indices = []

    training_graphs = [get_training_adj(batch, graph_id) for graph_id in torch.unique(batch.batch)]

    for graph_id, training_graph in enumerate(training_graphs):
        # Positive and negative edge indices
        pos_indices = torch.where(training_graph > 0)
        neg_indices = torch.where(training_graph == 0)
        pos_edge_indices.append(pos_indices)
        neg_edge_indices.append(neg_indices)

        # Removed edge indices
        node_mask = batch.batch == graph_id
        node_indices = torch.where(node_mask)[0]
        global_to_local = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(node_indices)}

        edge_mask = node_mask[batch.removed_edge_index[0]] & node_mask[batch.removed_edge_index[1]]
        filtered_edges = batch.removed_edge_index[:, edge_mask]

        local_edges = torch.tensor(
            [[global_to_local[src.item()], global_to_local[dst.item()]] for src, dst in filtered_edges.t()],
            device=batch.removed_edge_index.device
        ).t()

        removed_edge_indices.append(local_edges)

    return pos_edge_indices, neg_edge_indices, removed_edge_indices


def train_epoch(model, loader, optimiser, device, epoch, loss_function):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)

        # Encode and decode the graphs
        z = model.encode(batch.incoming_edge_index, batch.outgoing_edge_index, batch.batch)
        fully_decoded_graph = model.decode_all(z, batch.batch)

        # Prepare edge indices
        pos_edge_indices, neg_edge_indices, removed_edge_indices = prepare_edge_indices(batch)

        # Compute loss and update model
        if loss_function == "bce":
            loss = model.compute_loss(fully_decoded_graph, pos_edge_indices, neg_edge_indices, removed_edge_indices)
        if loss_function == "avg":
            loss = model.alternative_loss(fully_decoded_graph, pos_edge_indices, neg_edge_indices, removed_edge_indices, epoch)
        if loss_function == "avg_small":
            loss = model.small_loss(fully_decoded_graph, pos_edge_indices, neg_edge_indices)
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device, loss_function):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Encode and decode the graphs
            z = model.encode(batch.incoming_edge_index, batch.outgoing_edge_index, batch.batch)
            fully_decoded_graph = model.decode_all(z, batch.batch)

            # Prepare edge indices
            pos_edge_indices, neg_edge_indices, removed_edge_indices = prepare_edge_indices(batch)

            # Compute loss
            if loss_function == "bce":
                loss = model.compute_loss(fully_decoded_graph, pos_edge_indices, neg_edge_indices, removed_edge_indices)
            if loss_function == "avg":
                loss = model.alternative_loss(fully_decoded_graph, pos_edge_indices, neg_edge_indices, removed_edge_indices)
            if loss_function == "avg_small":
                loss = model.small_loss(fully_decoded_graph, pos_edge_indices, neg_edge_indices)

            total_loss += loss.item()

    return total_loss / len(loader)


def test_model(model, test_loader, device, loss_function):
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

            # Prepare edge indices
            pos_edge_indices, neg_edge_indices, removed_edge_indices = prepare_edge_indices(batch)

            # Compute loss
            if loss_function == "bce":
                loss = model.compute_loss(fully_decoded_graph, pos_edge_indices, neg_edge_indices, removed_edge_indices)
            if loss_function == "avg":
                loss = model.alternative_loss(fully_decoded_graph, pos_edge_indices, neg_edge_indices, removed_edge_indices)
            if loss_function == "avg_small":
                loss = model.small_loss(fully_decoded_graph, pos_edge_indices, neg_edge_indices)

            total_test_loss += loss.item()

            # Compute accuracy
            for reconstructed_adj, ground_truth_adj, pos_indices, neg_indices in zip(
                fully_decoded_graph, [get_training_adj(batch, graph_id) for graph_id in torch.unique(batch.batch)],
                pos_edge_indices, neg_edge_indices
            ):
                pred_binary = (reconstructed_adj > 0.8).float()
                correct_pos = (pred_binary[pos_indices] == ground_truth_adj[pos_indices]).sum().item()
                correct_neg = (pred_binary[neg_indices] == ground_truth_adj[neg_indices]).sum().item()
                total_correct += correct_pos + correct_neg
                total_edges += len(pos_indices[0]) + len(neg_indices[0])

    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = total_correct / total_edges
    return avg_test_loss, test_accuracy


def train_small_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        z = model.encode(batch.incoming_edge_index, batch.outgoing_edge_index, batch.batch)

        # Reconstruct adjacency matrices for all graphs in the batch
        reconstructed_adjs = model.decode_all(z, batch.batch)

        # Split ground-truth adjacency matrices based on batch.graph indices
        ground_truth_adjs = []
        unique_graphs = torch.unique(batch.batch)
        for graph_id in unique_graphs:
            mask = batch.batch == graph_id
            ground_truth_adj = batch.label[mask][:, mask]  # Extract square submatrix
            ground_truth_adjs.append(ground_truth_adj)

        # Compute loss
        loss = model.small_loss(reconstructed_adjs, ground_truth_adjs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def train_small_gae_gcn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DirectedGAEGCN(out_channels=hyperparameters.out_channels,
                           hidden_channels=hyperparameters.hidden_channels,
                           num_nodes=hyperparameters.num_nodes, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters.learning_rate)

    # Dataset preparation
    train_dataset = get_small_graphs_dataset()
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Training loop
    for epoch in range(1, hyperparameters.epochs + 1):
        avg_train_loss = train_small_epoch(model, train_loader, optimizer, device)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'results/gcn_small.pth')
    print("Model saved to 'model_gcn_reconstruction_loss.pth'")


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

    torch.save(model.state_dict(), 'results/trained_gae_gin.pth')
    print("Model saved to 'trained_gae_gin.pth'")
