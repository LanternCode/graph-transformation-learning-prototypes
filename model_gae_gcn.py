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
        self.gcn_in = GCNConv(hidden_channels, out_channels).to(device)
        self.gcn_out = GCNConv(hidden_channels, out_channels).to(device)

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

    def compute_loss(self, reconstructed_adjs, training_graphs, pos_edge_indices, neg_edge_indices):
        """
        Compute BCE loss for edges and non-edges, ignoring fully decoded edges.

        Args:
            reconstructed_adjs (list of torch.Tensor): Decoded adjacency matrices.
            training_graphs (list of torch.Tensor): Ground-truth adjacency matrices.
            pos_edge_indices (list of torch.Tensor): List of positive edge indices for each graph.
            neg_edge_indices (list of torch.Tensor): List of negative edge indices for each graph.

        Returns:
            torch.Tensor: The average loss over all graphs.
        """


        total_loss = 0

        for i, (reconstructed_adj, training_graph, pos_edges, neg_edges) in enumerate(
                zip(reconstructed_adjs, training_graphs, pos_edge_indices, neg_edge_indices)
        ):
            # Decode probabilities for positive edges
            pos_probs = reconstructed_adj[pos_edges[0], pos_edges[1]]
            pos_labels = torch.ones(pos_probs.size(0), device=pos_probs.device)

            # Decode probabilities for negative edges
            neg_probs = reconstructed_adj[neg_edges[0], neg_edges[1]]
            neg_labels = torch.zeros(neg_probs.size(0), device=neg_probs.device)

            # Combine probabilities and labels
            all_probs = torch.cat([pos_probs, neg_probs])
            all_labels = torch.cat([pos_labels, neg_labels])

            # Compute BCE loss
            loss = F.binary_cross_entropy(all_probs, all_labels, reduction="mean")
            print(f"Graph {i}: Loss = {loss.item()}")
            total_loss += loss

        # Average loss over all graphs
        return total_loss / len(reconstructed_adjs)

    def alternative_loss(self, reconstructed_adjs, pos_edge_indices, neg_edge_indices):
        """
        Alternative loss function that computes the average scores for positive and negative samples
        and uses these to compute a simple loss.

        Args:
            reconstructed_adjs (list of tensors): Reconstructed adjacency matrices from the model.
            training_graphs (list of tensors): Ground truth adjacency matrices.
            pos_edge_indices (list of tensors): Indices of positive edges for each graph.
            neg_edge_indices (list of tensors): Indices of negative edges for each graph.

        Returns:
            torch.Tensor: Loss value.
        """
        total_loss = 0

        for reconstructed_adj, pos_indices, neg_indices in zip(reconstructed_adjs, pos_edge_indices, neg_edge_indices):
            # Get the scores for positive and negative samples
            pos_scores = reconstructed_adj[pos_indices]
            neg_scores = reconstructed_adj[neg_indices]

            # Compute the averages
            avg_pos_score = pos_scores.mean()
            avg_neg_score = neg_scores.mean()

            # Compute the loss: 1 - average positive score + average negative score
            loss = (1 - avg_pos_score) + avg_neg_score
            total_loss += loss

        # Return the average loss across all graphs in the batch
        return total_loss / len(reconstructed_adjs)
