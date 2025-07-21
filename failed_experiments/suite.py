# Directory structure
base_dir = Path("graph_coloring_benchmark")
benchmark_dirs = {
    "graphs": base_dir / "graphs",
    "features": base_dir / "features",
    "predictions": base_dir / "predictions",
    "results": base_dir / "results",
    "models": base_dir / "models",
    "scripts": base_dir / "scripts",
}

# Create directories
for path in benchmark_dirs.values():
    path.mkdir(parents=True, exist_ok=True)

# Sample metadata for benchmark task
benchmark_meta = {
    "task": "Graph Coloring",
    "objective": "Minimize conflicting edges and number of colors",
    "dataset": "DIMACS",
    "graphs": [
        {
            "name": "DSJC125.1",
            "path": "graphs/DSJC125.1.col",
            "nodes": 125,
            "edges": 1472,
            "chromatic_number": 5  # Known value
        }
    ],
    "evaluation": {
        "metrics": ["conflicts", "colors_used"],
        "description": "Lower is better for both metrics."
    }
}

# Save benchmark metadata
with open(base_dir / "benchmark_meta.json", "w") as f:
    json.dump(benchmark_meta, f, indent=4)

# Provide template for predictions
example_prediction = {
    "DSJC125.1": [i % 5 for i in range(125)]  # Dummy coloring using 5 colors
}
with open(benchmark_dirs["scripts"] / "example_prediction.json", "w") as f:
    json.dump(example_prediction, f, indent=4)
