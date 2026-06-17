import numpy as np
import torch
from benchmark import benchmark_precision_recall
from model import DeepEdgeMLP


MODEL_PATH = "final_model.pth"


def make_mlp_adapter(model, threshold=0.5):
    """
    Create a benchmark adapter for a trained edge MLP.

    Args:
        model: Loaded ``DeepEdgeMLP`` instance that maps edge features to logits.
        threshold: Probability threshold used to classify an edge as present.

    Returns:
        A callable that accepts ``(X, N)`` and returns a list of ``(i, j)`` edge
        pairs predicted as present by the model.
    """
    def mlp_adapter(X: np.ndarray, N: int):
        """
        Predict present edges for one benchmark graph.

        Args:
            X: ``(N*N, 11)`` edge-feature matrix produced by the benchmark.
            N: Number of nodes in the graph.

        Returns:
            A list of ``(i, j)`` pairs whose predicted probability is greater
            than ``threshold``.
        """
        # convert to torch tensor
        t = torch.from_numpy(X).float()

        # forward pass
        with torch.no_grad():
            logits = model(t)               # shape (N*N, 1) or (N*N,)
            probs = torch.sigmoid(logits)   # model outputs logits
            probs = probs.view(-1).cpu().numpy()

        # threshold at 0.5
        preds = []
        for idx, p in enumerate(probs):
            if p > threshold:
                i, j = divmod(idx, N)
                preds.append((i, j))
        return preds

    return mlp_adapter


def main():
    """
    Load the trained final model and run the cycle-completion benchmark.

    Args:
        None.

    Returns:
        None. Precision, recall, and F1 are printed by the benchmark function.
    """
    model = DeepEdgeMLP(input_dim=11)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    mlp_adapter = make_mlp_adapter(model)
    benchmark_precision_recall(mlp_adapter, num_graphs=1000, drop_fraction=0.2)


if __name__ == "__main__":
    main()
