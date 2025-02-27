from itertools import combinations
import torch
from torch import nn
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

import hyperparameters
from gen_sym_closure import generate_graph_dataset, get_small_graphs_dataset


class KWLConv(nn.Module):
    def __init__(self, in_dim, out_dim, k=2):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),  # Expecting concatenation of x_tuple and messages
            nn.ReLU(),
            nn.LayerNorm(out_dim)
        )
        self.pool = nn.MaxPool1d(kernel_size=k)

    def forward(self, x_tuple, adj_tuple):
        # 1. Neighborhood aggregation
        print("adj_tuple shape:", adj_tuple.shape)
        messages = torch.sparse.mm(adj_tuple, x_tuple)
        print("messages shape:", messages.shape)

        # 2. Injective combination
        combined = self.mlp(torch.cat([x_tuple, messages], dim=-1))

        # 3. k-dimensional max pooling
        return self.pool(combined.unsqueeze(0)).squeeze(0)


class KWLGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = KWLConv(2, 2, 2)
        self.conv2 = KWLConv(2, 2, 2)
        self.out = nn.Linear(2, 2)

    def forward(self, data):
        x_tuple, adj_tuple, batch_tuple = self.initialize_k_tuples(data)

        x_tuple = self.conv1(x_tuple, adj_tuple)
        x_tuple = self.conv2(x_tuple, adj_tuple)

        return self.out(x_tuple.mean(dim=0)), batch_tuple


    def initialize_k_tuples(self, batch_data, k=2):
        """
        Initialize k-tuple graph representation from batched input graphs.
        Args:
            batch_data: A batched PyTorch Geometric data object containing multiple graphs.
            k: Tuple size (e.g., 2 for pairs).
        Returns:
            x_tuple: Batched k-tuple node representations.
            adj_tuple: Sparse adjacency matrix for the k-tuple graph.
            batch_tuple: Index mapping each k-tuple to its graph in the batch.
        """
        if not hasattr(batch_data, 'edge_index') or not hasattr(batch_data, 'batch'):
            raise AttributeError("batch_data is missing required attributes (edge_index, batch)")

        edge_index = batch_data.edge_index  # Graph edges
        batch = batch_data.batch  # Index mapping nodes to graphs
        num_nodes = batch_data.num_nodes

        print("\nGenerating k-tuples...")
        print("Num nodes in batch:", num_nodes)
        print("Edge index shape:", edge_index.shape)

        # Generate k-tuples from node indices
        tuples = []
        batch_tuple = []
        for graph_id in batch.unique():
            node_indices = (batch == graph_id).nonzero(as_tuple=True)[0]

            if len(node_indices) < k:
                node_indices = node_indices.repeat(k)[:k]

            graph_tuples = list(combinations(node_indices.tolist(), k))
            if len(graph_tuples) < k:
                for _ in range(k - len(graph_tuples)):
                    graph_tuples.append((node_indices[0], node_indices[0]))

            tuples.extend(graph_tuples)
            batch_tuple.extend([graph_id] * len(graph_tuples))

        tuples = torch.tensor(tuples, dtype=torch.long, device=batch.device)
        batch_tuple = torch.tensor(batch_tuple, dtype=torch.long, device=batch.device)

        print("Generated tuples count:", len(tuples))

        # Construct a meaningful fixed representation for each tuple
        x_tuple = torch.ones((len(tuples), k), dtype=torch.float32,
                             device=batch.device)  # A simple identity representation

        print("Final x_tuple.shape:", x_tuple.shape)

        # Ensure adjacency matrix is valid
        if edge_index.numel() == 0:
            print("No edges in graph, using identity matrix for `adj_tuple`")
            adj_tuple = torch.eye(len(tuples), device=batch.device)
        else:
            adj_tuple = torch.zeros((len(tuples), len(tuples)), device=batch.device)
            for i, t1 in enumerate(tuples):
                for j, t2 in enumerate(tuples):
                    if set(t1.tolist()).intersection(set(t2.tolist())):
                        adj_tuple[i, j] = 1

        if adj_tuple.shape[0] != x_tuple.shape[0]:
            print(f"Shape mismatch: adj_tuple {adj_tuple.shape} vs x_tuple {x_tuple.shape}")
        if adj_tuple.sum() == 0:
            print("Warning: adj_tuple is empty, replacing with identity matrix.")
            adj_tuple = torch.eye(x_tuple.shape[0], device=batch.device)

        print("Final adj_tuple shape:", adj_tuple.shape)
        print("adj_tuple sum (should be > 0):", adj_tuple.sum().item())

        return x_tuple, adj_tuple.to_sparse(), batch_tuple


class GraphDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )

    def forward(self, z, batch_tuple):
        adj_pred = torch.matmul(z, z.T)
        adj_pred = torch.sigmoid(adj_pred)

        num_graphs = batch_tuple.max().item() + 1
        batch_adj_list = []
        for graph_id in range(num_graphs):
            mask = batch_tuple == graph_id
            batch_adj_list.append(adj_pred[mask][:, mask])

        return batch_adj_list


class GraphCompletionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = KWLGNN()
        self.decoder = GraphDecoder()

    def forward(self, data_input):
        z, batch_tuple = self.encoder(data_input)
        adj_pred = self.decoder(z, batch_tuple)
        return adj_pred

    def small_loss(self, reconstructed_adjs, ground_truth_adjs):
        total_loss = 0.0
        num_graphs = len(reconstructed_adjs)

        for reconstructed_adj, ground_truth_adj in zip(reconstructed_adjs, ground_truth_adjs):
            # Ensure the tensors are on the same device
            ground_truth_adj = ground_truth_adj.to(reconstructed_adj.device)

            # Compute binary cross-entropy loss
            loss = F.binary_cross_entropy(reconstructed_adj, ground_truth_adj)
            total_loss += loss

        return total_loss / num_graphs


def train_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphCompletionModel(k=2, hidden_dim=2).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=hyperparameters.learning_rate)

    train_dataset, val_dataset, test_dataset = generate_graph_dataset(num_graphs=hyperparameters.num_graphs)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_to_device)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_to_device)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_to_device)

    for epoch in range(hyperparameters.epochs + 1):
        avg_train_loss = train_epoch(model, train_loader, optimiser, device, epoch)
        avg_val_loss = evaluate(model, val_loader, device)
        if epoch % 25 == 0:
            print(f"Epoch {epoch + 1} Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    test_loss, test_accuracy = test_model(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    new_model_name = hyperparameters.new_model_name
    torch.save(model.state_dict(), new_model_name+'.pth')
    print("Model saved to "+new_model_name+".pth")


def collate_to_device(data_list):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = Batch.from_data_list(data_list)
    return batch.to(device)  # Move batch to GPU


def train_epoch(model, loader, optimiser, device, epoch):
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
        loss = model.small_loss(fully_decoded_graph, pos_edge_indices, neg_edge_indices)
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item()

    return total_loss / len(loader)


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


def train_small():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphCompletionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_dataset = get_small_graphs_dataset()
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    for epoch in range(1, 20 + 1):
        avg_train_loss = train_small_epoch(model, train_loader, optimizer, device)
        if epoch % 5 == 0:
            print(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}")

    torch.save(model.state_dict(), 'kWL-six_graphs.pth')
    print("Model saved to 'kWL-six_graphs.pth'")


def train_small_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        adj_pred_list = model(batch)

        ground_truth_adjs = []
        unique_graphs = torch.unique(batch.batch)
        for graph_id in unique_graphs:
            mask = batch.batch == graph_id
            ground_truth_adj = batch.label[mask][:, mask]
            ground_truth_adjs.append(ground_truth_adj)

        loss = F.binary_cross_entropy(adj_pred_list, ground_truth_adjs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss