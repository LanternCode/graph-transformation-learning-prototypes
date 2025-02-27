import sys
import torch
import hyperparameters
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from gen_sym_closure import get_data, count_incomplete_symmetrically_closed_pairs, generate_graph_dataset, \
    get_small_graphs_dataset
from model_gae_gin import DirectedGAEGIN
from utils import collate_to_device, prepare_edge_indices


def train_gae_gin():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DirectedGAEGIN(out_channels=hyperparameters.out_channels,
                           hidden_channels=hyperparameters.hidden_channels,
                           num_nodes=hyperparameters.num_nodes, device=device)
    optimiser = torch.optim.Adam(model.parameters(), lr=hyperparameters.learning_rate)

    train_dataset, val_dataset, test_dataset = generate_graph_dataset(num_graphs=hyperparameters.num_graphs)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_to_device)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_to_device)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_to_device)

    for epoch in range(hyperparameters.epochs):
        avg_train_loss = train_epoch(model, train_loader, optimiser, device)
        avg_val_loss = evaluate(model, val_loader, device)
        if epoch % 25 == 0 or epoch == hyperparameters.epochs - 1:
            print(f"Epoch {epoch + 1} Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    test_loss, test_accuracy = test_model(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    new_model_name = hyperparameters.new_model_name
    torch.save(model.state_dict(), new_model_name+'.pth')
    print("Model saved to "+new_model_name+".pth")


def train_epoch(model, loader, optimiser, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)

        # Encode and decode the graphs
        z = model.encode(batch.incoming_edge_index, batch.outgoing_edge_index, batch.batch)
        fully_decoded_graph = model.decode_all(z, batch.batch)

        # Prepare edge indices
        pos_edge_indices, neg_edge_indices = prepare_edge_indices(batch)

        # Compute loss and update model
        loss = model.compute_loss(fully_decoded_graph, pos_edge_indices, neg_edge_indices)
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Encode and decode the graphs
            z = model.encode(batch.incoming_edge_index, batch.outgoing_edge_index, batch.batch)
            fully_decoded_graph = model.decode_all(z, batch.batch)

            # Compute loss
            pos_edge_indices, neg_edge_indices = prepare_edge_indices(batch)
            loss = model.compute_loss(fully_decoded_graph, pos_edge_indices, neg_edge_indices)
            total_loss += loss.item()

    return total_loss / len(loader)


def test_model(model, test_loader, device):
    model.eval()
    total_test_loss = 0
    total_correct = 0
    total_edges = 0

    with torch.no_grad():
        for batch in test_loader:
            # Encode and decode the graphs
            batch = batch.to(device)
            z = model.encode(batch.incoming_edge_index, batch.outgoing_edge_index, batch.batch)
            fully_decoded_graph = model.decode_all(z, batch.batch)

            # Compute loss
            pos_edge_indices, neg_edge_indices = prepare_edge_indices(batch)
            loss = model.compute_loss(fully_decoded_graph, pos_edge_indices, neg_edge_indices)
            total_test_loss += loss.item()

            # Compute accuracy
            for reconstructed_adj, pos_indices, neg_indices in zip(fully_decoded_graph, pos_edge_indices,
                                                                   neg_edge_indices):
                # Convert predicted graph to binary adjacency matrix (threshold > 0.8)
                pred_binary = (reconstructed_adj > 0.8).float()

                # Compare with ground truth labels
                correct_pos = (pred_binary[pos_indices] == 1).sum().item()
                correct_neg = (pred_binary[neg_indices] == 0).sum().item()

                total_correct += correct_pos + correct_neg
                total_edges += len(pos_indices[0]) + len(neg_indices[0])

        # Compute average test loss and accuracy
        avg_test_loss = total_test_loss / len(test_loader)
        test_accuracy = total_correct / total_edges if total_edges > 0 else 0

        return avg_test_loss, test_accuracy
