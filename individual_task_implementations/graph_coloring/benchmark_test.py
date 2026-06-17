import networkx as nx
import torch
from typing import Any
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from benchmark import run_benchmark_with_model
from model import BEST_MODEL_PATH, GRAPH_DIR, GCNColoring


class DSATURModel:
    """
    Adapter that exposes NetworkX DSATUR greedy coloring as a logits model.

    Args:
        None.

    Returns:
        Callable object whose output is a one-hot node-by-color logits tensor.
    """

    def __call__(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Produce one-hot color logits using DSATUR greedy graph coloring.

        Args:
            x: Node-feature tensor; only its row count and device are used.
            edge_index: PyTorch Geometric edge index for the graph to color.

        Returns:
            One-hot tensor with shape [num_nodes, num_colors] representing DSATUR
            color assignments.
        """
        graph = to_networkx(
            Data(edge_index=edge_index, num_nodes=x.size(0)),
            to_undirected=True,
        )
        coloring = nx.coloring.greedy_color(
            graph,
            strategy="saturation_largest_first",
        )
        num_nodes = x.size(0)
        num_colors = max(coloring.values(), default=-1) + 1
        logits = torch.zeros((num_nodes, max(num_colors, 1)), device=x.device)
        for node, color in coloring.items():
            logits[node][color] = 1.0
        return logits


def load_trained_model(
    checkpoint_path: str = BEST_MODEL_PATH,
    device: str = "cpu",
) -> GCNColoring:
    """
    Load the trained GCN coloring model from a checkpoint.

    Args:
        checkpoint_path: Path to the saved model state dictionary.
        device: Torch device where the model should be loaded.

    Returns:
        GCNColoring model loaded with checkpoint weights and set to eval mode.
    """
    model = GCNColoring(in_dim=5, hidden_dim=64, num_colors=10)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def main(
    graph_dir: str = GRAPH_DIR,
    checkpoint_path: str = BEST_MODEL_PATH,
    device: str = "cpu",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Benchmark the trained GCN model and DSATUR baseline on the configured graphs.

    Args:
        graph_dir: Directory containing DIMACS .col files to evaluate.
        checkpoint_path: Path to the saved GCN model checkpoint.
        device: Torch device for running benchmark tensors.

    Returns:
        Tuple containing benchmark summaries for the trained GCN and DSATUR models.
    """
    model = load_trained_model(checkpoint_path=checkpoint_path, device=device)
    dsatur_model = DSATURModel()

    results = run_benchmark_with_model(model, graph_dir=graph_dir, device=device)
    results_dsatur = run_benchmark_with_model(dsatur_model, graph_dir=graph_dir, device=device)

    return results, results_dsatur


if __name__ == "__main__":
    main()
