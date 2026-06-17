import os
import networkx as nx
import json
import random
import pandas as pd
from pyomo.environ import ConcreteModel, Var, Objective, ConstraintList, Binary, SolverFactory, minimize
from sklearn.utils import resample
from tqdm import tqdm

# Output file
dataset_path = "min_fill_dataset.jsonl"


def generate_non_chordal_graph(num_nodes):
    """
    Generate a random non-chordal Erdős-Rényi graph.

    Args:
        num_nodes: Number of nodes to include in the generated graph.

    Returns:
        A NetworkX graph with the requested number of nodes that is not chordal.
    """
    while True:
        G = nx.erdos_renyi_graph(num_nodes, 0.3)
        if not nx.is_chordal(G):
            return G


def find_cycles(G, max_length=6):
    """
    Find bounded-length cycles used by the prototype fill-in formulation.

    Args:
        G: NetworkX graph to inspect for cycle-basis cycles.
        max_length: Maximum cycle length retained for the prototype constraints.

    Returns:
        A list of cycle node lists whose lengths are between 4 and max_length.
    """
    cycles = []
    for cycle in nx.cycle_basis(G):
        if 4 <= len(cycle) <= max_length:
            cycles.append(cycle)
    return cycles


def solve_min_fill(G):
    """
    Solve the prototype bounded-cycle fill-in optimisation problem.

    Args:
        G: Non-chordal NetworkX graph whose missing edges are candidate fill edges.

    Returns:
        A list of unordered node pairs selected by the Pyomo/CBC binary programme.
    """
    n = G.number_of_nodes()
    existing_edges = set(frozenset((u, v)) for u, v in G.edges() if u != v)

    model = ConcreteModel()
    model.x = Var(((i, j) for i in range(n) for j in range(i + 1, n)
                   if frozenset((i, j)) not in existing_edges), domain=Binary)

    model.obj = Objective(expr=sum(model.x[i, j] for (i, j) in model.x), sense=minimize)

    model.constraints = ConstraintList()
    cycles = find_cycles(G, max_length=6)
    for cycle in cycles:
        chord_pairs = [(min(cycle[i], cycle[j]), max(cycle[i], cycle[j]))
                       for i in range(len(cycle))
                       for j in range(i+2, len(cycle))
                       if (j - i) != len(cycle) - 1]
        chord_vars = [model.x[i, j] for (i, j) in chord_pairs if (i, j) in model.x]
        if chord_vars:
            model.constraints.add(expr=sum(chord_vars) >= 1)

    solver = SolverFactory("cbc")
    result = solver.solve(model, tee=False)

    added_edges = [(i, j) for (i, j) in model.x if model.x[i, j].value == 1]
    return added_edges


# Generate dataset
if os.path.exists(dataset_path) == 0:
    with open(dataset_path, "w") as f:
        for i in tqdm(range(1000), desc="Generating exact fill-in dataset"):
            G = generate_non_chordal_graph(random.randint(6, 140))
            fill_edges = solve_min_fill(G)

            record = {
                "graph_id": i,
                "nodes": list(G.nodes()),
                "edges": list(G.edges()),
                "fill_edges": fill_edges
            }
            f.write(json.dumps(record) + "\n")


graphs = []
with open(dataset_path, "r") as f:
    for line in f:
        graphs.append(json.loads(line))


def extract_features_and_labels(graph_record):
    """
    Extract candidate non-edge features and fill-in labels for one graph record.

    Args:
        graph_record: Dictionary with graph_id, nodes, edges, and fill_edges fields.

    Returns:
        A tuple (features, labels, graph_ids) where features is a list of eight-value
        candidate-edge feature vectors, labels is a list of binary fill-in labels, and
        graph_ids identifies the source graph for graph-level downstream splitting.
    """
    G = nx.Graph()
    G.add_nodes_from(graph_record["nodes"])
    G.add_edges_from(graph_record["edges"])
    fill_edges = set(tuple(sorted(edge)) for edge in graph_record["fill_edges"])
    graph_id = graph_record.get("graph_id", graph_record.get("id", -1))

    features = []
    labels = []
    graph_ids = []
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            u, v = nodes[i], nodes[j]
            if G.has_edge(u, v):
                continue
            cn = len(list(nx.common_neighbors(G, u, v)))
            jaccard = list(nx.jaccard_coefficient(G, [(u, v)]))[0][2]
            aa = list(nx.adamic_adar_index(G, [(u, v)]))[0][2]
            deg_u = G.degree[u]
            deg_v = G.degree[v]
            try:
                sp = nx.shortest_path_length(G, u, v)
            except nx.NetworkXNoPath:
                sp = -1
            cc_u = nx.clustering(G, u)
            cc_v = nx.clustering(G, v)
            features.append([cn, jaccard, aa, deg_u, deg_v, sp, cc_u, cc_v])
            labels.append(1 if (u, v) in fill_edges else 0)
            graph_ids.append(graph_id)
    return features, labels, graph_ids


X, y = [], []
feature_cache_path = "fillin_features.csv"
required_columns = {
    "graph_id", "common_neighbors", "jaccard", "adamic_adar", "deg_u",
    "deg_v", "shortest_path", "cc_u", "cc_v", "label"
}
if os.path.exists(feature_cache_path):
    df = pd.read_csv(feature_cache_path)
    if not required_columns.issubset(df.columns):
        os.remove(feature_cache_path)
        df = None
    else:
        X = df.drop(columns=["label", "graph_id"]).values
        y = df["label"].values
else:
    df = None

if df is None:
    feature_list = []
    for graph in tqdm(graphs, desc="Extracting features"):
        feats, labs, gids = extract_features_and_labels(graph)
        for f, l, gid in zip(feats, labs, gids):
            feature_list.append([gid] + f + [l])

    columns = [
        "graph_id", "common_neighbors", "jaccard", "adamic_adar", "deg_u",
        "deg_v", "shortest_path", "cc_u", "cc_v", "label"
    ]
    df = pd.DataFrame(feature_list, columns=columns)
    df.to_csv(feature_cache_path, index=False)
    X = df.drop(columns=["label", "graph_id"]).values
    y = df["label"].values

# Separate classes
df_majority = df[df.label == 0]
df_minority = df[df.label == 1]

# Downsample majority class
df_majority_downsampled = resample(
    df_majority,
    replace=False,
    n_samples=min(len(df_majority), len(df_minority) * 3),  # 3:1 ratio
    random_state=42
)

# Combine and shuffle
df_balanced = pd.concat([df_majority_downsampled, df_minority]).sample(frac=1, random_state=42)

# Save or proceed to training
df_balanced.to_csv("fillin_features_balanced.csv", index=False)
