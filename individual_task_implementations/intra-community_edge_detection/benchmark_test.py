import joblib
import torch
from torch import nn
from benchmark import EdgeDisagreementBenchmark


def logistic_regression_adapter(X):
    """
    Predict edge community-disagreement labels with the saved logistic model.

    Args:
        X: NumPy feature matrix with one row per edge and the benchmark feature
            columns expected by the trained logistic regression model.

    Returns:
        A one-dimensional NumPy array of predicted binary class labels.
    """
    model = joblib.load("logistic_model.pth")
    return model.predict(X)


def random_forest_adapter(X):
    """
    Predict edge community-disagreement labels with the saved random forest.

    Args:
        X: NumPy feature matrix with one row per edge and the benchmark feature
            columns expected by the trained random forest model.

    Returns:
        A one-dimensional NumPy array of predicted binary class labels.
    """
    model = joblib.load("best_random_forest.pth")
    return model.predict(X)


def mlp_adapter(X):
    """
    Predict edge community-disagreement labels with the saved PyTorch MLP.

    Args:
        X: NumPy feature matrix with one row per edge and the benchmark feature
            columns expected by the trained MLP.

    Returns:
        A one-dimensional NumPy array of predicted binary class labels.
    """
    class MLP(nn.Module):
        """
        Feed-forward neural classifier matching the saved MLP checkpoint.

        Args:
            input_dim: Number of input feature columns for each edge example.

        Returns:
            An ``nn.Module`` that maps edge features to positive-class
            probabilities.
        """

        def __init__(self, input_dim):
            """
            Initialise the local MLP architecture used for checkpoint loading.

            Args:
                input_dim: Number of scalar input features provided for each
                    edge.

            Returns:
                None.
            """
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            """
            Compute positive-class probabilities from edge features.

            Args:
                x: PyTorch tensor of shape ``(num_edges, input_dim)``.

            Returns:
                PyTorch tensor of shape ``(num_edges, 1)`` containing predicted
                probabilities.
            """
            return self.model(x)

    input_dim = X.shape[1]
    model = MLP(input_dim)
    model.load_state_dict(torch.load("mlp_model.pth"))
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_prob = model(X_tensor).numpy().flatten()
        y_pred = (y_prob > 0.5).astype(int)
        return y_pred


benchmark = EdgeDisagreementBenchmark()
benchmark.run(logistic_regression_adapter)
benchmark.run(random_forest_adapter)
benchmark.run(mlp_adapter)
