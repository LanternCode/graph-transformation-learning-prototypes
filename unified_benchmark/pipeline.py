import numpy as np
import torch
from benchmark.benchmark_manager import BenchmarkManager
from models.SymmetricClosureMLP import SymmetricClosureMLP
from benchmark.tasks.symmetric_closure import SymmetricClosureTask

task = SymmetricClosureTask()
bench = BenchmarkManager(task)
graphs = bench.provide_benchmark()
preds = []

model = SymmetricClosureMLP()
model.load_state_dict(torch.load("trained_models/symmetric_closure_mlp.pth"))
model.eval()
for graph in graphs:
    with torch.no_grad():
        tensor = torch.tensor(graph, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        output = model(tensor)
    binary_pred = (output.squeeze() > 0.5).numpy().astype(np.uint8)
    preds.append(binary_pred)

bench.evaluate(preds)
