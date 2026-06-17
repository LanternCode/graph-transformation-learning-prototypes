import json
import random
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ========== Load and Preprocess ==========
def load_graphs(path):
    """
    Load graph records from a JSONL fill-in dataset.

    Args:
        path: Path to a JSONL file containing graph records with nodes, edges, and fill_edges.

    Returns:
        A list of decoded graph-record dictionaries.
    """
    graphs = []
    with open(path, "r") as f:
        for line in f:
            graphs.append(json.loads(line))
    return graphs


def compute_node_features(G):
    """
    Compute node-level structural features for a graph.

    Args:
        G: NetworkX graph whose nodes are expected to be indexed from 0 to n - 1.

    Returns:
        A float tensor with one row per node and columns for degree, clustering,
        betweenness, closeness, PageRank, core number, and triangle count.
    """
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
    """
    Compute candidate-edge structural features for non-edge pairs.

    Args:
        G: NetworkX graph used to compute local and global edge-pair features.
        edge_pairs: Iterable of candidate node pairs to featurize.

    Returns:
        A float tensor with columns for common neighbors, Jaccard score, Adamic-Adar
        score, preferential attachment, and edge betweenness lookup value.
    """
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

        adamic_adar = sum(1 / torch.log(torch.tensor(degrees[n], dtype=torch.float))
                          for n in intersection if degrees[n] > 1) if intersection else 0
        pref_attach = degrees[u] * degrees[v]

        ebc_val = ebc_dict.get((u, v), ebc_dict.get((v, u), 0.0))

        features.append([common, jaccard, float(adamic_adar), pref_attach, ebc_val])

    return torch.tensor(features, dtype=torch.float)


def build_pyg_graph(graph_data):
    """
    Convert one fill-in graph record into a PyTorch Geometric training example.

    Args:
        graph_data: Dictionary containing original graph edges and prototype fill-in edges.

    Returns:
        A PyTorch Geometric Data object with node features, message-passing edges,
        candidate edge pairs, candidate edge labels, and candidate edge features.
    """
    edges = graph_data["edges"]
    added = set(tuple(sorted(e)) for e in graph_data["fill_edges"])
    num_nodes = max(max(u, v) for u, v in edges) + 1

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    x = compute_node_features(G)

    # Build negative samples without reusing positive fill-in pairs as negatives.
    all_pairs = set((u, v) for u in range(num_nodes) for v in range(u+1, num_nodes))
    existing = set(tuple(sorted(e)) for e in edges)
    non_edges = list(all_pairs - existing - added)

    positives = list(added)
    if len(positives) == 0:
        k = min(20, len(non_edges))  # avoid sampling more than available
        negatives = random.sample(non_edges, k)
        edge_pairs = negatives
        labels = torch.tensor([0] * len(negatives), dtype=torch.float)
    else:
        k = min(len(non_edges), len(positives) * 2)
        negatives = random.sample(non_edges, k)
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


# ========== Model ==========
class GraphSAGE(nn.Module):
    """
    GraphSAGE edge classifier for candidate minimum-fill pairs.

    Args:
        in_channels: Number of node-feature channels in each input graph.
        hidden_channels: Hidden representation size used by GraphSAGE layers.

    Returns:
        A neural module that maps a PyG Data object to one logit per candidate edge.
    """
    def __init__(self, in_channels, hidden_channels):
        """
        Initialize the GraphSAGE encoder and edge-level classifier.

        Args:
            in_channels: Number of node-feature channels in each input graph.
            hidden_channels: Hidden representation size used by each GraphSAGE layer.

        Returns:
            None.
        """
        super().__init__()
        self.sage1 = SAGEConv(in_channels, hidden_channels)
        self.dropout = nn.Dropout(p=0.3)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        self.sage3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels * 2 + 5, 1)

    def forward(self, data):
        """
        Predict raw logits for candidate fill-in edges in one graph.

        Args:
            data: PyTorch Geometric Data object containing x, edge_index, edge_pairs,
                and edge_features fields.

        Returns:
            A one-dimensional tensor of raw logits, one for each candidate edge pair.
        """
        x, edge_index, edge_pairs = data.x, data.edge_index, data.edge_pairs
        x = self.sage1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.sage2(x, edge_index).relu()
        x = self.dropout(x)
        x = self.sage3(x, edge_index)

        pair_feats = torch.cat([x[edge_pairs[:, 0]], x[edge_pairs[:, 1]]], dim=1)
        edge_feats = torch.cat([pair_feats, data.edge_features], dim=1)
        return self.lin(edge_feats).squeeze()

    def predict_edges(self, data):
        """
        Predict raw logits for candidate fill-in edges.

        Args:
            data: PyTorch Geometric Data object to pass through the model.

        Returns:
            A one-dimensional tensor of raw logits for the candidate edge pairs.
        """
        return self.forward(data)


# ========== Training ==========
def train_model(graphs, epochs=15, test_size=0.2, random_state=42):
    """
    Train GraphSAGE on a graph-level training split.

    Args:
        graphs: List of graph-record dictionaries loaded from the JSONL dataset.
        epochs: Number of training epochs.
        test_size: Fraction of graphs held out for evaluation.
        random_state: Random seed used for the graph-level train/test split.

    Returns:
        A tuple (model, train_data_list, test_data_list) containing the trained model,
        the PyG training graphs, and the held-out PyG evaluation graphs.
    """
    train_graphs, test_graphs = train_test_split(graphs, test_size=test_size, random_state=random_state)

    train_data_list = []
    for g in tqdm(train_graphs, desc="Generating train PyG graphs"):
        train_data_list.append(build_pyg_graph(g))

    test_data_list = []
    for g in tqdm(test_graphs, desc="Generating held-out PyG graphs"):
        test_data_list.append(build_pyg_graph(g))

    loader = DataLoader(train_data_list, batch_size=1, shuffle=True)

    model = GraphSAGE(in_channels=train_data_list[0].num_node_features, hidden_channels=64)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    all_labels = torch.cat([data.edge_labels.view(-1) for data in train_data_list if data.edge_labels.numel() > 0])
    pos = max((all_labels == 1).sum().item(), 1)
    neg = max((all_labels == 0).sum().item(), 1)
    pos_weight = torch.tensor([neg / pos])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for data in loader:
            data = data[0]
            preds = model.predict_edges(data)
            loss = criterion(preds, data.edge_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_data_list):.4f}")

    torch.save(model.state_dict(), "graphsage_best.pth")
    return model, train_data_list, test_data_list


# ========== Evaluation ==========
def evaluate_model(model, data_list):
    """
    Evaluate a GraphSAGE model on a list of PyG graph examples.

    Args:
        model: Trained GraphSAGE model that returns logits for candidate edges.
        data_list: List of held-out PyTorch Geometric Data objects to evaluate.

    Returns:
        None. Prints a classification report and ROC AUC when computable.
    """
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for data in data_list:
            logits = model.predict_edges(data)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.tolist())
            all_preds.extend((probs > 0.5).int().tolist())
            all_labels.extend(data.edge_labels.int().tolist())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))

    try:
        auc_score = roc_auc_score(all_labels, all_probs)
        print(f"AUC: {auc_score:.4f}")
    except ValueError:
        print("AUC could not be computed (possibly only one class present in predictions).")


# ========== Main ==========
if __name__ == "__main__":
    graphs = load_graphs("min_fill_dataset.jsonl")
    model, train_data_list, test_data_list = train_model(graphs, 15)
    evaluate_model(model, test_data_list)
