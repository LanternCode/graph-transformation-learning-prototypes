import sys

import numpy as np
import torch
import hyperparameters
from gen_sym_closure import get_data
from model_gae import DirectedGAE
from eval import decode_regular_edges, decode_all_edges

num_inference_nodes = 1000
inference_data = get_data(num_inference_nodes, 0.1)

# Recreate the model structure
model = DirectedGAE(out_channels=hyperparameters.out_channels, hidden_channels=hyperparameters.hidden_channels, num_nodes=hyperparameters.num_nodes)

# Load the saved model state
model.load_state_dict(torch.load('trained_gae.pth'))

# Set the model to evaluation mode and perform regular or fully connected decoding
model.eval()
#decode_regular_edges(model, inference_data, num_inference_nodes)
decode_all_edges(model, inference_data, num_inference_nodes, 0.8)