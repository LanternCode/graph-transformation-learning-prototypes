import torch
import hyperparameters
from gen_sym_closure import get_data
from model_gae_gcn import DirectedGAEGCN
from eval import decode_given_edges, manually_decode_all_edges, decode_all
from model_gae_gin import DirectedGAEGIN

num_inference_nodes = 1000
inference_data = get_data(num_inference_nodes, hyperparameters.missing_edge_fraction)

# Recreate the model structure
model = DirectedGAEGCN(out_channels=hyperparameters.out_channels, hidden_channels=hyperparameters.hidden_channels, num_nodes=hyperparameters.num_nodes)
#model = DirectedGAEGIN(out_channels=hyperparameters.out_channels, hidden_channels=hyperparameters.hidden_channels, num_nodes=hyperparameters.num_nodes)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')  # For running on the GPU

# Load the saved model state
model.load_state_dict(torch.load('trained_gae_gcn.pth'))
#model.load_state_dict(torch.load('trained_gae_gin.pth'))

# Set the model to evaluation mode and perform regular or fully connected decoding
model.eval()
#decode_given_edges(model, inference_data)
#manually_decode_all_edges(model, inference_data, num_inference_nodes, 0.8)
decode_all(model, inference_data, num_inference_nodes, 0.8)