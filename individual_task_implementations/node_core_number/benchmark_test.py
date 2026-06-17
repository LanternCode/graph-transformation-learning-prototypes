import os
import torch
from benchmark import benchmark_model
from model import ContMLP, ContGCN, DeepGraphSAGE, ContGraphTransformer


def make_adapter(model, device):
    """
    Create a benchmark adapter for one trained core-number prediction model.

    Args:
        model: Trained PyTorch model to evaluate.
        device: Torch device on which the model and benchmark batches should run.

    Returns:
        A callable adapter that accepts a PyTorch Geometric batch and returns a
        pair of NumPy arrays: model predictions and ground-truth node labels.
    """
    def adapter(batch):
        """
        Run model inference on a benchmark graph batch.

        Args:
            batch: PyTorch Geometric batch containing x, edge_index, batch, and y.

        Returns:
            A tuple ``(predictions, targets)`` as NumPy arrays.
        """
        model.eval()
        batch = batch.to(device)
        with torch.no_grad():
            if isinstance(model, ContMLP):
                out = model(batch.x)  # Only needs x
            else:
                out = model(batch.x, batch.edge_index, batch.batch)
        return out.cpu().numpy(), batch.y.cpu().numpy()
    return adapter


def benchmark_all_models(model_dir=".", hidden_channels=32, num_graphs=1000):
    """
    Load all available trained models and benchmark their core predictions.

    Args:
        model_dir: Directory containing saved model checkpoints named with the
            ``<ModelName>_batch_best.pt`` convention.
        hidden_channels: Hidden-channel width used to reconstruct each model.
        num_graphs: Number of synthetic benchmark graphs to evaluate per model.

    Returns:
        None. Results are printed to standard output.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    model_registry = {
        "ContMLP": ContMLP,
        "ContGCN": ContGCN,
        "DeepGraphSAGE": DeepGraphSAGE,
        "ContGraphTransformer": ContGraphTransformer,
    }

    for name, model_class in model_registry.items():
        model_file = os.path.join(model_dir, f"{name}_batch_best.pt")
        if not os.path.exists(model_file):
            print(f"Skipping {name} - model file not found: {model_file}")
            continue

        print(f"\nLoading {name} from {model_file}")
        model = model_class(in_channels=2, hidden_channels=hidden_channels).to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))
        adapter = make_adapter(model, device)
        acc, mse = benchmark_model(adapter, num_graphs=num_graphs)
        results[name] = (acc, mse)

    print("\n=== Benchmark Summary Across Models ===")
    for name, (acc, mse) in results.items():
        print(f"{name:>24}: Accuracy = {acc:.4f} | MSE = {mse:.4f}")


benchmark_all_models(hidden_channels=32, num_graphs=1000)
