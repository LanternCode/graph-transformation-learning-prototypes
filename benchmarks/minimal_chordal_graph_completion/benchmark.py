from pyomo.environ import ConcreteModel, Var, Objective, ConstraintList, Binary, SolverFactory, minimize
from tqdm import tqdm
import random
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report, roc_auc_score
import math
import json
import os


# ---------- caching helpers ----------
def _serialize_graph(G):
    return {
        "nodes": list(map(int, G.nodes())),
        "edges": [(int(u), int(v)) for u, v in G.edges()],
        "fill_edges": [(int(u), int(v)) for (u, v) in G.graph.get("fill_edges", [])],
    }


def _deserialize_graph(rec):
    G = nx.Graph()
    G.add_nodes_from(rec["nodes"])
    G.add_edges_from(rec["edges"])
    G.graph["fill_edges"] = [tuple(e) for e in rec["fill_edges"]]
    return G


def _generate_graphs(num_graphs, seed=None):
    rng = random.Random(seed)
    made = []
    pbar = tqdm(total=num_graphs, desc="Generating test graphs")
    while len(made) < num_graphs:
        n = rng.randint(6, 140)
        G = nx.erdos_renyi_graph(n, 0.3, seed=rng.randrange(1 << 30))
        if nx.is_chordal(G):
            continue
        fill_edges = solve_min_fill(G)
        if fill_edges:
            G.graph["fill_edges"] = [(int(i), int(j)) for (i, j) in fill_edges]
            made.append(G)
            pbar.update(1)
    pbar.close()
    return made


def _load_or_make(cache_path, num_graphs, seed=None):
    have = []
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            for line in f:
                have.append(_deserialize_graph(json.loads(line)))
    if len(have) >= num_graphs:
        return have[:num_graphs]

    need = num_graphs - len(have)
    new_graphs = _generate_graphs(need, seed=seed)

    # append newly generated to cache
    with open(cache_path, "a") as f:
        for G in new_graphs:
            f.write(json.dumps(_serialize_graph(G)) + "\n")
    return have + new_graphs


def build_pyg_graph(graph_data):
    edges = list(graph_data.edges())
    added = set(tuple(sorted(e)) for e in graph_data.graph.get("fill_edges", []))
    num_nodes = max(max(u, v) for u, v in edges) + 1

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    x = compute_node_features(G)

    # Build negative samples
    all_pairs = set((u, v) for u in range(num_nodes) for v in range(u+1, num_nodes))
    existing = set(tuple(sorted(e)) for e in edges)
    non_edges = list(all_pairs - existing)

    positives = list(added)
    if len(positives) == 0:
        k = min(20, len(non_edges))  # avoid sampling more than available
        negatives = random.sample(non_edges, k)
        edge_pairs = negatives
        labels = torch.tensor([0] * len(negatives), dtype=torch.float)
    else:
        negatives = random.sample(non_edges, len(positives) * 2)
        edge_pairs = positives + negatives
        labels = torch.tensor([1] * len(positives) + [0] * len(negatives), dtype=torch.float)

    edge_pairs_tensor = torch.tensor(edge_pairs, dtype=torch.long)
    edge_features = compute_edge_features(G, edge_pairs)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_pairs=edge_pairs_tensor,
        edge_labels=labels,
        edge_features=edge_features
    )


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


def compute_node_features(G):
    num_nodes = G.number_of_nodes()
    G.add_nodes_from(range(num_nodes))

    degrees = dict(G.degree())
    clustering = nx.clustering(G)
    betweenness = nx.betweenness_centrality(G, normalized=True)
    closeness = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G)
    kcore = nx.core_number(G)
    triangles = nx.triangles(G)

    features = []
    for i in range(num_nodes):
        features.append([
            degrees.get(i, 0),
            clustering.get(i, 0),
            betweenness.get(i, 0),
            closeness.get(i, 0),
            pagerank.get(i, 0),
            kcore.get(i, 0),
            triangles.get(i, 0),
        ])
    return torch.tensor(features, dtype=torch.float)


def compute_edge_features(G, edge_pairs):
    degrees = dict(G.degree())
    features = []
    ebc_dict = nx.edge_betweenness_centrality(G, normalized=True)

    for u, v in edge_pairs:
        u_nbrs = set(G.neighbors(u))
        v_nbrs = set(G.neighbors(v))
        intersection = u_nbrs & v_nbrs
        union = u_nbrs | v_nbrs

        common = len(intersection)
        jaccard = len(intersection) / len(union) if union else 0

        adamic_adar = sum(1.0 / math.log(degrees[n]) for n in intersection if degrees[n] > 1) if intersection else 0.0
        pref_attach = degrees[u] * degrees[v]

        ebc_val = ebc_dict.get((u, v), ebc_dict.get((v, u), 0.0))

        features.append([common, jaccard, float(adamic_adar), pref_attach, ebc_val])

    return torch.tensor(features, dtype=torch.float)


def generate_graph_batch(num_graphs, seed=42):
    rng = random.Random(seed)
    made = []
    pbar = tqdm(total=num_graphs, desc="Generating test graphs")
    while len(made) < num_graphs:
        n = rng.randint(6, 140)
        G = nx.erdos_renyi_graph(n, 0.3, seed=rng.randrange(1 << 30))
        if nx.is_chordal(G):
            continue
        fill_edges = solve_min_fill(G)
        if fill_edges:
            G.graph['fill_edges'] = [(int(i), int(j)) for (i, j) in fill_edges]
            made.append(G)
            pbar.update(1)
    pbar.close()
    return made


def build_pyg_batch(graphs):
    return [build_pyg_graph(g) for g in tqdm(graphs, desc="Building PyG graphs")]


_CACHED_DATA = None
_CACHED_NUM  = None

def _get_cached_data(num_graphs):
    global _CACHED_DATA, _CACHED_NUM
    if _CACHED_DATA is None or _CACHED_NUM != num_graphs:
        test_graphs = []
        pbar = tqdm(total=num_graphs, desc="Generating test graphs")
        while len(test_graphs) < num_graphs:
            G = generate_non_chordal_graph(random.randint(6, 140))
            fill_edges = solve_min_fill(G)
            if fill_edges:
                G.graph['fill_edges'] = fill_edges
                test_graphs.append(G)
                pbar.update(1)
        pbar.close()
        _CACHED_DATA = [build_pyg_graph(g) for g in tqdm(test_graphs, desc="Building PyG graphs")]
        _CACHED_NUM = num_graphs
    return _CACHED_DATA


def benchmark_model(adapter_fn, num_graphs=1000, batch_size=1,
                    test_graphs=None, data_list=None, seed=42):
    data_list = _get_cached_data(num_graphs)

    loader = DataLoader(data_list, batch_size=batch_size)

    all_labels, all_preds, all_probs = [], [], []
    for data in tqdm(loader, desc="Evaluating"):
        preds = adapter_fn(data)
        labels = data.edge_labels.view(-1).int().cpu().tolist()
        probs = torch.as_tensor(preds, dtype=torch.float32).detach().cpu().flatten()

        all_labels.extend(labels)
        all_preds.extend((probs > 0.5).int().tolist())
        all_probs.extend(probs.tolist())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))
    try:
        auc_score = roc_auc_score(all_labels, all_probs)
        print(f"AUC: {auc_score:.4f}")
    except ValueError:
        print("AUC could not be computed (likely due to single-class output).")
