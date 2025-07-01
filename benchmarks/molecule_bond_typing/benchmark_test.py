from benchmark import run_bond_classification_benchmark
import joblib


def bond_predictor_adapter(X):
    """
    Adapter function that takes a feature matrix X (N x F)
    and returns predicted bond types (N,)
    """
    model = joblib.load('rf_bond_classifier.pth')
    return model.predict(X)


run_bond_classification_benchmark(bond_predictor_adapter)
