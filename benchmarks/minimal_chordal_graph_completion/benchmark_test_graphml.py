from gnn_model import GraphSAGE
import torch
from benchmark import benchmark_model


def adapter(data):
    with torch.no_grad():
        model = GraphSAGE(in_channels=7, hidden_channels=64)
        model.load_state_dict(torch.load("graphsage_best.pth", map_location="cpu"))
        model.eval()
        return model(data)


print(f"\n=== Benchmarking GraphSAGE ===")
benchmark_model(adapter)
