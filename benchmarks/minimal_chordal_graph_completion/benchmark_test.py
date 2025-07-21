from gnn_model import GraphSAGE
import torch
from benchmark import benchmark_model

# Instantiate to match saved model
model = GraphSAGE(in_channels=7, hidden_channels=64)
model.load_state_dict(torch.load("graphsage_best.pth", map_location="cpu"))
model.eval()


# Adapter function
def adapter(data):
    with torch.no_grad():
        return model(data)


# Run benchmark
benchmark_model(adapter)
