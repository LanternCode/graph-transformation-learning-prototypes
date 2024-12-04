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

    # Calculate BCE loss across edges and non-edges
    def compute_loss(self, reconstructed_graph, training_graph):
        total_loss = F.binary_cross_entropy(reconstructed_graph, training_graph, reduction='mean')
        return total_loss
