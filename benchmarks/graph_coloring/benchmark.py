import os
import torch
from model import parse_dimacs_col, compute_node_features, build_pyg_data, evaluate_coloring


def run_benchmark_with_model(model, graph_dir="dimacs_graphs", device="cpu"):
    if hasattr(model, "eval") and callable(getattr(model, "eval")):
        model.eval()
    if hasattr(model, "to") and callable(getattr(model, "to")):
        model.to(device)

    model.to(device) if hasattr(model, "to") else None
    model_name = model.__class__.__name__
    summary = []

    for fname in sorted(os.listdir(graph_dir)):
        if not fname.endswith(".col"):
            continue

        print(f"Evaluating {fname}...")
        path = os.path.join(graph_dir, fname)
        G = parse_dimacs_col(path)
        G = compute_node_features(G)
        data = build_pyg_data(G).to(device)

        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            _, used, conflicts = evaluate_coloring(logits, data.edge_index)

        total_edges = G.number_of_edges()
        summary.append({
            "graph": fname,
            "model": model_name,
            "nodes": G.number_of_nodes(),
            "edges": total_edges,
            "colors_used": used,
            "conflicts": conflicts,
            "conflict_ratio": conflicts / total_edges
        })

    print("\n=== Benchmark Summary ===")
    for res in summary:
        print(f"{res['graph']} [{res['model']}]: {res['colors_used']} colors, "
              f"{res['conflicts']} conflicts ({res['conflict_ratio']:.2%})")

    return summary
