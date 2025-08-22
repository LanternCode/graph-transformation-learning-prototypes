import numpy as np
import torch
from unified_benchmark.benchmark.benchmark_manager import BenchmarkManager

enable_all = True
suite = {
    'symclo': 0,
    'transclo': 0
}

# --------------- Symmetric Closure External Benchmark --------------- #
if enable_all or suite['symclo']:
    from unified_benchmark.models.SymmetricClosureMLP import SymmetricClosureMLP
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

# --------------- Transitive Closure External Benchmark --------------- #
if enable_all or suite['transclo']:
    from unified_benchmark.models.TransitiveClosureMLP import RecurrentClosure
    from unified_benchmark.benchmark.tasks.transitive_closure import TransitiveClosureTask

    model = RecurrentClosure()
    ckpt = torch.load("unified_benchmark/trained_models/transitive_closure_recurrent.pth", map_location="cpu")
    model.load_state_dict(ckpt)
    model.eval()

    task_tc = TransitiveClosureTask(k=10, threshold=0.5, assume_logits=True, ignore_diagonal=True)
    bench_tc = BenchmarkManager(
        task_tc,
        num_graphs=1000,
        min_nodes=6, max_nodes=140,
        graph_config={'expected_out_degree': (4.0, 8.0)}
    )
    graphs = bench_tc.provide_benchmark()

    preds = []
    with torch.no_grad():
        for A_np in graphs:
            A = torch.tensor(A_np, dtype=torch.float32).unsqueeze(0)  # (1,N,N)
            out = model(A)                                            # logits
            preds.append(out.squeeze(0).cpu().numpy())

    bench_tc.evaluate(preds)
