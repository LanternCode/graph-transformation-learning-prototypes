import numpy as np
import torch
from benchmark import run_benchmark
from model import MLP, CNN, Transformer, Autoencoder


def load_model(model_class, checkpoint_path):
    model = model_class(150)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def make_adapter(model):
    def predict_chordal_edges(adj_matrix: np.ndarray) -> np.ndarray:
        A_tensor = torch.tensor(adj_matrix.astype(np.float32)).unsqueeze(0)  # [1, N, N]
        with torch.no_grad():
            output = model(A_tensor)[0].numpy()  # [N, N]
        predicted_edges = (output > 0.5).astype(np.float32)
        predicted_edges[adj_matrix == 1] = 0
        np.fill_diagonal(predicted_edges, 0)
        return predicted_edges
    return predict_chordal_edges


def run_all_benchmarks():
    model_infos = [
        ("MLP", MLP, "MLP_best.pt"),
        ("CNN", CNN, "CNN_best.pt"),
        ("Transformer", Transformer, "Transformer_best.pt"),
        ("Autoencoder", Autoencoder, "Autoencoder_best.pt")
    ]

    for name, model_class, path in model_infos:
        print(f"\nRunning benchmark for {name}:")
        model = load_model(model_class, path)
        adapter = make_adapter(model)
        run_benchmark(adapter, num_graphs=1000, num_nodes=150)


run_all_benchmarks()
