"""
Early feasibility prototype for symmetric-closure learning.

This file is part of the first round of exploratory experiments used to study
graph neural networks, graph autoencoders, adjacency-matrix learning, and basic
feasibility of reconstructing missing symmetric-closure edges. It is preserved
for historical context and reproducibility of the research process, not as a
clean final benchmark implementation. Later task-specific files supersede this
prototype for reported results.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv


class DirectedGAEGIN(torch.nn.Module):
    def __init__(self, out_channels, hidden_channels, num_nodes, device):
        super(DirectedGAEGIN, self).__init__()

        # Learnable node embeddings initialized here
        # self.node_embeddings = nn.Parameter(torch.randn(num_nodes, hidden_channels)).to(device)
        self.node_embeddings = nn.Embedding(num_nodes, hidden_channels, device=device)

        # Define the MLP for GINConv
        mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_channels, out_channels)
        ).to(device)

        # Separate GINs for incoming and outgoing edges
        self.gin_in = GINConv(mlp).to(device)
        self.gin_out = GINConv(mlp).to(device)

        # Projection layer for directional encoding concatenation
        self.projection_layer = nn.Linear(2 * out_channels, out_channels, device=device)

        # Weight matrix for bilinear decoding
        self.bilinear_weight = torch.nn.Parameter(torch.Tensor(out_channels, out_channels)).to(device)
        torch.nn.init.xavier_uniform_(self.bilinear_weight)

    def encode(self, edge_index_in, edge_index_out, batch):
        device = self.node_embeddings.weight.device
        # device = self.node_embeddings.device
        batch = batch.to(device)
        node_embeddings_batch = self.node_embeddings(batch).to(device)

        # Perform element-wise addition of the directional node embeddings
        z_in = self.gin_in(node_embeddings_batch, edge_index_in.to(device))
        z_out = self.gin_out(node_embeddings_batch, edge_index_out.to(device))
        #z = z_in + z_out
        #return z
        combined = torch.cat([z_in, z_out], dim=-1)
        return self.projection_layer(combined)

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

    def compute_loss(self, reconstructed_adjs, pos_edge_indices, neg_edge_indices):
        total_loss = 0.0
        num_graphs = len(reconstructed_adjs)

        for reconstructed_adj, pos_indices, neg_indices in zip(reconstructed_adjs, pos_edge_indices, neg_edge_indices):
            # Compute BCE loss only on selected edges
            pos_loss = F.binary_cross_entropy(reconstructed_adj[pos_indices],
                                              torch.ones_like(reconstructed_adj[pos_indices]))
            neg_loss = F.binary_cross_entropy(reconstructed_adj[neg_indices],
                                              torch.zeros_like(reconstructed_adj[neg_indices]))

            # Combine positive and negative losses
            loss = pos_loss + neg_loss
            total_loss += loss

        return total_loss / num_graphs
