import numpy as np
import torch
from unified_benchmark.benchmark.benchmark_manager import BenchmarkManager


# --------------- Symmetric Closure External Benchmark --------------- #
from unified_benchmark.models.SymmetricClosureMLP import SymmetricClosureMLP
from unified_benchmark.models.TransitiveClosureMLP import RecurrentClosure
from unified_benchmark.benchmark.tasks.symmetric_closure import SymmetricClosureTask

task = SymmetricClosureTask()
bench = BenchmarkManager(task)
graphs = bench.provide_benchmark()
preds = []

model = SymmetricClosureMLP()
model.load_state_dict(torch.load("unified_benchmark/trained_models/symmetric_closure_mlp.pth"))
model.eval()
for graph in graphs:
    with torch.no_grad():
        tensor = torch.tensor(graph, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        output = model(tensor)
    binary_pred = (output.squeeze() > 0.5).numpy().astype(np.uint8)
    preds.append(binary_pred)

bench.evaluate(preds)

preds_t = []
model_t = RecurrentClosure()
model_t.load_state_dict(torch.load("unified_benchmark/trained_models/transitive_closure_recurrent.pth"))
model_t.eval()
for graph in graphs:
    with torch.no_grad():
        tensor = torch.tensor(graph, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        output = model_t(tensor)
    binary_pred = (output.squeeze() > 0.5).numpy().astype(np.uint8)
    preds_t.append(binary_pred)

bench.evaluate(preds_t)
