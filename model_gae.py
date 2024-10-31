import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class DirectedGAE(torch.nn.Module):
    def __init__(self, out_channels, hidden_channels, num_nodes):
        super(DirectedGAE, self).__init__()

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

    def reconstruction_loss(self, pred, true):
        # Binary cross-entropy loss for link prediction
        return F.binary_cross_entropy(pred, true)

    def supervised_loss(self, pred, true):
        # Binary cross-entropy for supervised loss (link prediction)
        return F.binary_cross_entropy(pred, true)

    def total_loss(self, reconstruction_pred, reconstruction_true, supervised_pred, supervised_true, epoch_num, lambda_=0.5):
        # Combine reconstruction loss and supervised loss
        recon_loss = self.reconstruction_loss(reconstruction_pred, reconstruction_true)
        sup_loss = self.supervised_loss(supervised_pred, supervised_true)

        # Print losses for monitoring
        if epoch_num % 100 == 0:
            print(f'Epoch {epoch_num + 1}')
            print(f"Reconstruction Loss: {recon_loss.item():.4f}, Supervised Loss: {sup_loss.item():.4f}")

        # Total loss is a weighted sum of reconstruction and supervised loss
        return lambda_ * recon_loss + (1 - lambda_) * sup_loss
