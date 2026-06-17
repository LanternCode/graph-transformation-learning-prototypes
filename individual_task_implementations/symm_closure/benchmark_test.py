import numpy as np
import torch
from benchmark import generate_benchmark, evaluate_model
from model import SymmetricClosureMLP


def load_trained_model(checkpoint_path: str = "symmetric_closure_mlp.pth") -> SymmetricClosureMLP:
    """
    Load a trained symmetric-closure MLP checkpoint.

    Args:
        checkpoint_path: Path to the saved PyTorch state dictionary.

    Returns:
        A SymmetricClosureMLP instance loaded from checkpoint and set to
        evaluation mode.
    """
    model = SymmetricClosureMLP()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model


def make_model_adapter(model: SymmetricClosureMLP):
    """
    Create a benchmark adapter around an already loaded model.

    Args:
        model: Trained SymmetricClosureMLP instance used to score each graph.

    Returns:
        A callable that accepts a NumPy adjacency matrix and returns a NumPy
        prediction matrix.
    """
    def model_adapter(graph: np.ndarray):
        """
        Run the loaded model on one adjacency matrix.

        Args:
            graph: NumPy adjacency matrix to evaluate.

        Returns:
            NumPy matrix of model predictions after squeezing batch and channel
            dimensions.
        """
        with torch.no_grad():
            tensor = torch.tensor(graph, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            output = model(tensor)
        return output.squeeze().numpy()

    return model_adapter


def main() -> None:
    """
    Run the symmetric-closure benchmark for a saved MLP checkpoint.

    Args:
        None.

    Returns:
        None.
    """
    model = load_trained_model()
    model_adapter = make_model_adapter(model)
    graphs, labels = generate_benchmark()
    avg_acc = evaluate_model(model_adapter, graphs, labels)
    print(f"Model accuracy on the benchmark: {avg_acc * 100}%")


if __name__ == "__main__":
    main()
