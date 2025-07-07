import numpy as np
import torch
from benchmark import generate_benchmark, evaluate_model
from model import SymmetricClosureMLP


def model_adapter(graph: np.ndarray):
    model = SymmetricClosureMLP()
    model.load_state_dict(torch.load("symmetric_closure_mlp.pth"))
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(graph, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        output = model(tensor)
    return output.squeeze().numpy()


# Run the benchmark
graphs, labels = generate_benchmark()
avg_acc = evaluate_model(model_adapter, graphs, labels)
print(f"Model accuracy on the benchmark: {avg_acc*100}%")
