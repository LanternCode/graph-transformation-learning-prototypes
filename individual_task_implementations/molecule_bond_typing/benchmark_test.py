from benchmark import run_bond_classification_benchmark
import joblib


def bond_predictor_adapter(X):
    """
    Load the saved Random Forest bond classifier and predict bond classes.

    Args:
        X: NumPy feature matrix of shape ``(N, F)``, where each row contains the
            concatenated atom-pair descriptors for one candidate bond.

    Returns:
        np.ndarray: Predicted integer bond-type labels of shape ``(N,)``.
    """
    model = joblib.load('rf_bond_classifier.pth')
    return model.predict(X)


run_bond_classification_benchmark(bond_predictor_adapter)
