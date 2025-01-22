import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class DirectedGAEGCN(torch.nn.Module):
    def __init__(self, out_channels, hidden_channels, num_nodes, device):
        super(DirectedGAEGCN, self).__init__()

        # Learnable node embeddings initialized here
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, hidden_channels, device=device))

        # Separate GCNs for incoming and outgoing edges
        self.gcn_in = GCNConv(hidden_channels, out_channels, add_self_loops=False, bias=True).to(device)
        self.gcn_out = GCNConv(hidden_channels, out_channels, add_self_loops=False, bias=True).to(device)

        # Weight matrix for bilinear decoding
        self.bilinear_weight = torch.nn.Parameter(torch.randn(out_channels, out_channels, device=device))
        torch.nn.init.xavier_uniform_(self.bilinear_weight)

    def encode(self, edge_index_in, edge_index_out, batch):
        # Ensure all tensors are on GPU
        device = self.node_embeddings.device
        batch = batch.to(device)
        node_embeddings_batch = self.node_embeddings[batch]
        node_embeddings_batch = node_embeddings_batch.to(device)
        edge_index_in = edge_index_in.to(device)
        edge_index_out = edge_index_out.to(device)

        # Encode
        z_in = self.gcn_in(node_embeddings_batch, edge_index_in)
        z_out = self.gcn_out(node_embeddings_batch, edge_index_out)
        return z_in + z_out

    def decode(self, z, edge_index):
        # Bilinear decoder for directed edge prediction
        src, dst = edge_index
        logits = torch.sigmoid(torch.sum(z[src] @ self.bilinear_weight * z[dst], dim=1))
        return logits

    def decode_all(self, z, batch):
        # Split z into sub-embeddings for each graph in the batch
        unique_graphs = torch.unique(batch)
        adj_list = []

        for graph_id in unique_graphs:
            # Mask for nodes in the current graph
            node_mask = batch == graph_id

            # Extract embeddings for the current graph
            z_graph = z[node_mask]

            # Compute adjacency matrix for the current graph
            adj = torch.matmul(torch.matmul(z_graph, self.bilinear_weight), z_graph.t())
            adj = torch.sigmoid(adj)

            adj_list.append(adj)

        return adj_list

    def compute_loss(self, reconstructed_adjs, pos_edge_indices, neg_edge_indices, removed_edge_indices):
        """
        Compute BCE loss for edges and non-edges, ignoring fully decoded edges.

        Args:
            reconstructed_adjs (list of torch.Tensor): Decoded adjacency matrices.
            pos_edge_indices (list of torch.Tensor): List of positive edge indices for each graph.
            neg_edge_indices (list of torch.Tensor): List of negative edge indices for each graph.
            removed_edge_indices (list of torch.Tensor): List of removed edge indices for each graph.

        Returns:
            torch.Tensor: The average loss over all graphs.
        """
        total_loss = 0

        for reconstructed_adj, pos_indices, neg_indices, removed_indices in zip(
                reconstructed_adjs, pos_edge_indices, neg_edge_indices, removed_edge_indices):
            # Compute scores for positive, negative, and removed edges
            pos_scores = reconstructed_adj[pos_indices].flatten()
            neg_scores = reconstructed_adj[neg_indices].flatten()
            removed_scores = reconstructed_adj[removed_indices[0], removed_indices[1]].flatten()

            # Assign labels: 1 for positive/removed edges, 0 for negative edges
            pos_labels = torch.ones_like(pos_scores)
            neg_labels = torch.zeros_like(neg_scores)
            removed_labels = torch.ones_like(removed_scores)

            # Combine scores and labels
            scores = torch.cat([pos_scores, neg_scores, removed_scores])
            labels = torch.cat([pos_labels, neg_labels, removed_labels])

            # Use BCE loss
            loss = F.binary_cross_entropy_with_logits(scores, labels)

            total_loss += loss

        # Return the average loss across all graphs in the batch
        return total_loss / len(reconstructed_adjs)

    def alternative_loss(self, reconstructed_adjs, pos_edge_indices, neg_edge_indices, removed_edge_indices, epoch=1):
        """
        Simplified loss function to compute loss based on positive, negative, and removed edges.

        Args:
            epoch (int):
            reconstructed_adjs (list of tensors): Reconstructed adjacency matrices from the model.
            pos_edge_indices (list of tensors): Indices of positive edges for each graph.
            neg_edge_indices (list of tensors): Indices of negative edges for each graph.
            removed_edge_indices (list of tensors): Indices of removed edges for each graph.

        Returns:
            torch.Tensor: Average loss across all graphs in the batch.
        """
        total_loss = 0

        for reconstructed_adj, pos_indices, neg_indices, removed_indices in zip(
                reconstructed_adjs, pos_edge_indices, neg_edge_indices, removed_edge_indices):

            # Compute scores for positive, negative, and removed edges
            pos_scores = reconstructed_adj[pos_indices]
            neg_scores = reconstructed_adj[neg_indices]
            removed_scores = reconstructed_adj[removed_indices]

            # Compute the averages
            avg_pos_score = pos_scores.mean()
            avg_neg_score = neg_scores.mean()
            avg_removed_score = removed_scores.mean()

            # Compute the loss: encourage high positive scores, low negative scores, and high removed scores
            loss = (1 - avg_pos_score) + avg_neg_score + (1 - avg_removed_score)
            total_loss += loss

        # Return the average loss across all graphs in the batch
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1} avg_pos_score: {avg_pos_score:.4f}, avg_neg_score: {avg_neg_score:.4f}, avg_removed_score: {avg_removed_score:.4f}")
        return total_loss / len(reconstructed_adjs)

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