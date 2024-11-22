import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class DirectedGAEGCN(torch.nn.Module):
    def __init__(self, out_channels, hidden_channels, num_nodes):
        super(DirectedGAEGCN, self).__init__()

        # Learnable node embeddings initialized here
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, hidden_channels))

        # Separate GCNs for incoming and outgoing edges
        self.gcn_in = GCNConv(hidden_channels, out_channels)
        self.gcn_out = GCNConv(hidden_channels, out_channels)

        # Weight matrix for bilinear decoding
        self.bilinear_weight = torch.nn.Parameter(torch.Tensor(out_channels, out_channels))
        torch.nn.init.xavier_uniform_(self.bilinear_weight)

    def encode(self, edge_index_in, edge_index_out):
        # Separate encoding for incoming and outgoing edges
        z_in = self.gcn_in(self.node_embeddings, edge_index_in)
        z_out = self.gcn_out(self.node_embeddings, edge_index_out)
        z = z_in + z_out
        return z

    def decode(self, z, edge_index):
        # Bilinear decoder for directed edge prediction
        src, dst = edge_index
        logits = torch.sigmoid(torch.sum(z[src] @ self.bilinear_weight * z[dst], dim=1))
        return logits

    def decode_all(self, z):
        # Bilinear decoder for inference on all node pairs
        adj = torch.matmul(
            torch.matmul(z, self.bilinear_weight), 
            z.t()
        )
        return torch.sigmoid(adj)

    # Compute the MCP Cross-Supervised loss
    def missing_closure_pairs_loss(self, reconstructed_graph, training_graph, loss_specific_edges):
        # Step 1: Calculate BCE loss for each edge in `removed_edges` individually
        loss_values = []
        for idx in range(loss_specific_edges.size(1)):
            source = loss_specific_edges[0, idx]
            target = loss_specific_edges[1, idx]

            # Prediction and target for removed edges
            pred_value = reconstructed_graph[source, target]
            true_value = torch.tensor(1.0, device=pred_value.device)  # Target label 1 for removed edges

            # Calculate and store BCE loss for removed edge
            edge_loss = F.binary_cross_entropy(pred_value, true_value)
            loss_values.append(edge_loss)

        # Convert loss_values to a tensor
        loss_values = torch.stack(loss_values)

        # Step 2: Calculate BCE loss over both edges and non-edges, excluding removed edges
        # Get the sparse indices and values of `training_graph`
        training_indices = training_graph.coalesce().indices()  # Shape: [2, num_nonzero]
        training_values = training_graph.coalesce().values()  # Shape: [num_nonzero]

        # Create a mask to exclude `removed_edges`
        removed_set = set((u.item(), v.item()) for u, v in loss_specific_edges.t())
        mask = [(u.item(), v.item()) not in removed_set for u, v in training_indices.t()]
        mask = torch.tensor(mask, dtype=torch.bool, device=training_graph.device)

        # Filter predictions and targets using the mask - flatten soft_binary_adj_matrix for efficient filtering
        reconstructed_flat = reconstructed_graph.view(-1)  # Differentiable

        # Filter predictions and targets based on the mask
        filtered_targets = training_values[mask]  # Ground truth values
        # Use advanced indexing instead of manual list comprehension
        flat_indices = training_indices[0] * reconstructed_graph.size(1) + training_indices[1]
        filtered_predictions = reconstructed_flat[flat_indices[mask]]  # Retains gradient flow

        # Calculate BCE loss across edges and non-edges
        total_loss = F.binary_cross_entropy(filtered_predictions, filtered_targets, reduction='mean')

        # Step 3: Combine losses - concatenate with `total_loss` to get the mean loss
        overall_loss = (loss_values.mean() + total_loss) / 2  # Or customize as needed

        return overall_loss
