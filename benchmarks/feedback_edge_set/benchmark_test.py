import joblib
from benchmark import benchmark_acyclicity, benchmark_supervised


def adapter_rf(model):
    def inner(G, feats, edges):
        probs = model.predict_proba(feats)[:, 1]
        return [e for i, e in enumerate(edges) if probs[i] > 0.5]
    return inner

print(f"Benchmark: Supervised Learning")
model = joblib.load("model1_best.pth")
benchmark_supervised(model, num_graphs=1000, epoch=5)

print(f"Benchmark: Reinforcement Learning")
model = joblib.load("model2_best.pth")
benchmark_acyclicity(adapter_rf(model), num_graphs=1000, epoch=5)