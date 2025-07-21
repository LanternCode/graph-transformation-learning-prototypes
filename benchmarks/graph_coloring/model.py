import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
import networkx as nx
from sklearn.preprocessing import StandardScaler


def parse_dimacs_col(filename):
    G = nx.Graph()
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('p'):
                parts = line.strip().split()
                num_nodes = int(parts[2])
                for i in range(num_nodes):
                    G.add_node(i)
            elif line.startswith('e'):
                _, u, v = line.strip().split()
                G.add_edge(int(u) - 1, int(v) - 1)
    return G


def compute_node_features(G):
    deg = dict(G.degree())
    clustering = nx.clustering(G)
    pagerank = nx.pagerank(G)
    core = nx.core_number(G)
    eigen = nx.eigenvector_centrality_numpy(G)

    for node in G.nodes():
        G.nodes[node]['degree'] = deg[node]
        G.nodes[node]['clustering'] = clustering[node]
        G.nodes[node]['pagerank'] = pagerank[node]
        G.nodes[node]['core'] = core[node]
        G.nodes[node]['eigen'] = eigen[node]

    return G


def normalize_node_features(G):
    X = []
    for _, n in G.nodes(data=True):
        X.append([n['degree'], n['clustering'], n['pagerank'], n['core'], n['eigen']])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return torch.tensor(X_scaled, dtype=torch.float)


class GCNColoring(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_colors):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_colors)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x


def potts_loss(logits, edge_index):
    probs = F.softmax(logits, dim=1)
    u, v = edge_index
    similarity = (probs[u] * probs[v]).sum(dim=1)
    return similarity.mean()


def build_pyg_data(G: nx.Graph):
    data = from_networkx(G)
    data.x = normalize_node_features(G)
    return data


def entropy_loss(logits):
    probs = F.softmax(logits, dim=1)
    entropy = -(probs * probs.log()).sum(dim=1).mean()
    return -entropy  # we want to maximize entropy


def color_usage_loss(logits, eps=1e-10):
    probs = logits.softmax(dim=1)
    avg_usage = probs.mean(dim=0)
    usage_entropy = -torch.sum(avg_usage * (avg_usage + eps).log()) / probs.size(1)
    return usage_entropy


def train_gcn_on_graph(G, num_colors=10, epochs=500, hidden_dim=64, lr=0.01, entropy_weight=0.1, alpha=0.1, beta=0.05):
    data = build_pyg_data(G)
    model = GCNColoring(in_dim=data.x.shape[1], hidden_dim=hidden_dim, num_colors=num_colors)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    best_model_path = 'best_model.pth'

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        probs = out.softmax(dim=1)

        potts = potts_loss(out, data.edge_index)
        eps = 1e-10
        entropy = -torch.sum(probs * (probs + eps).log()) / (probs.size(0) * probs.size(1))
        usage = color_usage_loss(out)

        total_loss = potts + alpha * entropy + beta * usage

        total_loss.backward()
        optimizer.step()

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), best_model_path)

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"[Epoch {epoch}] Potts: {potts.item():.4f} | Entropy: {entropy.item():.4f} | Usage: {usage.item():.4f} | Total: {total_loss.item():.4f}")

    return model, out, data


def evaluate_coloring(logits, edge_index):
    preds = logits.argmax(dim=1)  # hard color assignments
    used_colors = preds.unique().numel()

    u, v = edge_index
    conflicts = (preds[u] == preds[v]).sum().item()
    total_edges = edge_index.size(1)
    conflict_rate = conflicts / total_edges

    print(f"Colors used: {used_colors}")
    print(f"Conflicting edges: {conflicts} / {total_edges} ({conflict_rate:.2%})")

    return preds, used_colors, conflicts


if __name__ == "__main__":
    G = parse_dimacs_col("dimacs_graphs/DSJC125.1.col")
    G = compute_node_features(G)
    model, logits, data = train_gcn_on_graph(G, epochs=2000, entropy_weight=0.5, alpha=0.2, beta=1.7)
    evaluate_coloring(logits, data.edge_index)
