import joblib
import torch
from torch import nn
from benchmark import EdgeDisagreementBenchmark


def logistic_regression_adapter(X):
    model = joblib.load("logistic_model.pth")
    return model.predict(X)


def random_forest_adapter(X):
    model = joblib.load("best_random_forest.pth")
    return model.predict(X)


def mlp_adapter(X):
    class MLP(nn.Module):
        def __init__(self, input_dim):
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
