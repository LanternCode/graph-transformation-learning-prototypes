import os
import torch
from typing import Any
from model import GRAPH_DIR, build_pyg_data, compute_node_features, evaluate_coloring, parse_dimacs_col


def run_benchmark_with_model(
    model: Any,
    graph_dir: str = GRAPH_DIR,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """
    Evaluate a coloring model on every DIMACS .col file in a directory.

    Args:
        model: Callable model that accepts node features and edge_index and returns
            node-by-color logits.
        graph_dir: Directory containing DIMACS .col files to evaluate.
        device: Torch device where tensor-based models and graph data should run.

    Returns:
        List of dictionaries summarizing graph size, colors used, conflicts, and
        conflict ratio for each evaluated file.
    """
    if not os.path.isdir(graph_dir):
        raise FileNotFoundError(f"Graph directory not found: {graph_dir}")

    if hasattr(model, "eval") and callable(getattr(model, "eval")):
        model.eval()
    if hasattr(model, "to") and callable(getattr(model, "to")):
        model.to(device)

    model_name = model.__class__.__name__
    summary = []

    for filename in sorted(os.listdir(graph_dir)):
        if not filename.endswith(".col"):
            continue

        print(f"Evaluating {filename}...")
        path = os.path.join(graph_dir, filename)
        graph = parse_dimacs_col(path)
        graph = compute_node_features(graph)
        data = build_pyg_data(graph).to(device)

        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            _, used, conflicts = evaluate_coloring(logits, data.edge_index)

        total_edges = graph.number_of_edges()
        summary.append({
            "graph": filename,
            "model": model_name,
            "nodes": graph.number_of_nodes(),
            "edges": total_edges,
            "colors_used": used,
            "conflicts": conflicts,
            "conflict_ratio": conflicts / total_edges if total_edges else 0.0,
        })

    print("\n=== Benchmark Summary ===")
    for result in summary:
        print(
            f"{result['graph']} [{result['model']}]: "
            f"{result['colors_used']} colors, "
            f"{result['conflicts']} conflicts ({result['conflict_ratio']:.2%})"
        )

    return summary
