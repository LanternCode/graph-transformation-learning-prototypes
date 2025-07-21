import torch
from model import GCNColoring
from benchmark import run_benchmark_with_model
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

# Load user-trained model
model = GCNColoring(in_dim=5, hidden_dim=64, num_colors=10)
model.load_state_dict(torch.load("best_model.pth"))


class DSATURModel:
    def __call__(self, x, edge_index):
        G = to_networkx(Data(edge_index=edge_index), to_undirected=True)
        coloring = nx.coloring.greedy_color(G, strategy='saturation_largest_first')
        num_nodes = G.number_of_nodes()
        num_colors = max(coloring.values()) + 1
        logits = torch.zeros((num_nodes, num_colors))
        for node, color in coloring.items():
            logits[node][color] = 1.0
        return logits



ext_model = DSATURModel()

# Run benchmark
results = run_benchmark_with_model(model)
results_ext = run_benchmark_with_model(ext_model)
