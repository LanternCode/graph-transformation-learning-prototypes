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

    # Calculate BCE loss across edges and non-edges
    def compute_loss(self, reconstructed_adjs, training_graphs):
        total_loss = 0

        for i, (reconstructed_adj, training_graph) in enumerate(zip(reconstructed_adjs, training_graphs)):
            # Mask ground truth to match the size of the reconstructed adjacency matrix
            # training_graph should ideally match the reconstructed_adj's size
            if training_graph.shape != reconstructed_adj.shape:
                training_graph = training_graph[:reconstructed_adj.size(0), :reconstructed_adj.size(1)]

            # Binary cross-entropy loss for the graph
            loss = F.binary_cross_entropy(reconstructed_adj, training_graph.float(), reduction='mean')
            total_loss += loss

        # Average loss over all graphs
        total_loss = total_loss / len(reconstructed_adjs)
        return total_loss
