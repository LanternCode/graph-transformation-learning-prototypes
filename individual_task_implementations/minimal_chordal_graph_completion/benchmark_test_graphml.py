import torch
from gnn_model import GraphSAGE
from benchmark import benchmark_model


def load_graphsage_model(weight_path="graphsage_best.pth", device=None):
    """
    Load the trained GraphSAGE checkpoint once for benchmark evaluation.

    Args:
        weight_path: Path to the saved GraphSAGE state dictionary.
        device: Optional torch device string or object. If omitted, CPU is used.

    Returns:
        A GraphSAGE model in evaluation mode on the selected device.
    """
    if device is None:
        device = torch.device("cpu")
    model = GraphSAGE(in_channels=7, hidden_channels=64).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model


_device = torch.device("cpu")
_model = load_graphsage_model(device=_device)


def adapter(data):
    """
    Convert a benchmark PyG graph into GraphSAGE fill-in probabilities.

    Args:
        data: PyTorch Geometric Data object containing graph features and candidate edges.

    Returns:
        A one-dimensional tensor of probabilities for data.edge_pairs.
    """
    data = data.to(_device)
    with torch.no_grad():
        return torch.sigmoid(_model(data))


print(f"\n=== Benchmarking GraphSAGE ===")
benchmark_model(adapter)
