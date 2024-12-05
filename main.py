from torch_geometric.nn import GCNConv
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dummy data
num_nodes = 1000
hidden_channels = 64
out_channels = 32

node_embeddings = torch.randn(num_nodes, hidden_channels, device=device)
edge_index = torch.randint(0, num_nodes, (2, 5000), device=device)

gcn = GCNConv(hidden_channels, out_channels).to(device)

# Debug forward pass
print(f"node_embeddings device: {node_embeddings.device}")
print(f"edge_index device: {edge_index.device}")
z = gcn(node_embeddings, edge_index)
print(z)