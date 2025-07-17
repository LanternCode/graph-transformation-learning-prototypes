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
    while True:
        G = nx.erdos_renyi_graph(num_nodes, 0.3)
        if not nx.is_chordal(G):
            return G


def find_cycles(G, max_length=6):
    cycles = []
    for cycle in nx.cycle_basis(G):
        if 4 <= len(cycle) <= max_length:
            cycles.append(cycle)
    return cycles


def solve_min_fill(G):
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
    G = nx.Graph()
    G.add_nodes_from(graph_record["nodes"])
    G.add_edges_from(graph_record["edges"])
    fill_edges = set(tuple(sorted(edge)) for edge in graph_record["fill_edges"])

    features = []
    labels = []
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
    return features, labels


X, y = [], []
feature_cache_path = "fillin_features.csv"
if os.path.exists(feature_cache_path):
    df = pd.read_csv(feature_cache_path)
    X = df.drop(columns=["label"]).values
    y = df["label"].values
else:
    feature_list = []
    for graph in tqdm(graphs, desc="Extracting features"):
        feats, labs = extract_features_and_labels(graph)
        for f, l in zip(feats, labs):
            feature_list.append(f + [l])

    columns = ["common_neighbors", "jaccard", "adamic_adar", "deg_u", "deg_v", "shortest_path", "cc_u", "cc_v", "label"]
    df = pd.DataFrame(feature_list, columns=columns)
    df.to_csv(feature_cache_path, index=False)
    X = df.drop(columns=["label"]).values
    y = df["label"].values

# Separate classes
df_majority = df[df.label == 0]
df_minority = df[df.label == 1]

# Downsample majority class
df_majority_downsampled = resample(
    df_majority,
    replace=False,
    n_samples=len(df_minority) * 3,  # 3:1 ratio
    random_state=42
)

# Combine and shuffle
df_balanced = pd.concat([df_majority_downsampled, df_minority]).sample(frac=1, random_state=42)

# Save or proceed to training
df_balanced.to_csv("fillin_features_balanced.csv", index=False)
