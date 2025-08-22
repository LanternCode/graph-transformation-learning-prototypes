import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# ─── 1) Dataset generation with degree & betweenness ──────────────────────────
def make_spanning_candidate(n_nodes, feat_dim=16, extra_per_tree=1):
    # 1a) Build a random spanning tree
    T = nx.random_unlabeled_tree(n_nodes, seed=random.randint(0,1e6))
    tree_edges = set(T.edges())
    n_tree = n_nodes - 1

    # 1b) Sample extra edges
    n_extra = extra_per_tree * n_tree
    all_pairs = set((u, v) for u in range(n_nodes) for v in range(u+1, n_nodes))
    non_tree = list(all_pairs - tree_edges)
    extra_edges = random.sample(non_tree, k=n_extra)

    undirected = list(tree_edges) + extra_edges

    # 2) Build the undirected NX graph & compute features
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(undirected)

    # 2a) Node‐degree feature (normalized)
    deg = np.array([d for _, d in G.degree()], dtype=np.float32)
    deg = deg / (deg.max() if deg.max()>0 else 1.0)

    # 2b) Edge‐betweenness centrality
    bc_dict = nx.edge_betweenness_centrality(G)
    # collect as list in same order as `undirected`
    edge_bc = [bc_dict.get((u, v), bc_dict.get((v, u), 0.0))
               for (u, v) in undirected]

    # 3) Build directed edges, labels, and duplicate bc
    ei, el, ebc = [[],[]], [], []
    for (u, v), bc in zip(undirected, edge_bc):
        label = 1 if (u, v) in tree_edges or (v, u) in tree_edges else 0
        # both directions
        ei[0] += [u, v]; ei[1] += [v, u]
        el   += [label, label]
        ebc  += [bc, bc]

    edge_index = torch.tensor(ei, dtype=torch.long)
    edge_label = torch.tensor(el, dtype=torch.float)
    edge_attr  = torch.tensor(ebc, dtype=torch.float).unsqueeze(-1)  # [2M,1]

    # 4) Node features = [ random_feat | degree_feat ]
    rand_feat = torch.randn((n_nodes, feat_dim), dtype=torch.float)
    deg_feat  = torch.tensor(deg, dtype=torch.float).unsqueeze(-1)
    x = torch.cat([rand_feat, deg_feat], dim=1)  # [n_nodes, feat_dim+1]

    return Data(x=x,
                edge_index=edge_index,
                edge_label=edge_label,
                edge_attr=edge_attr)


def make_dataset(n_graphs, node_range=(6,100), feat_dim=16, extra_per_tree=1, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return [
        make_spanning_candidate(random.randint(*node_range),
                                feat_dim, extra_per_tree)
        for _ in range(n_graphs)
    ]


# ─── 3) Model that uses edge_attr ───────────────────────────────────────────────
class EdgeTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 edge_feat_dim=1, n_layers=3, n_heads=4):
        super().__init__()
        from torch_geometric.nn import TransformerConv
        self.convs = nn.ModuleList()
        self.convs.append(
            TransformerConv(in_channels, hidden_channels//n_heads,
                            heads=n_heads, edge_dim=edge_feat_dim)
        )
        for _ in range(n_layers-1):
            self.convs.append(
                TransformerConv(hidden_channels, hidden_channels//n_heads,
                                heads=n_heads, edge_dim=edge_feat_dim)
            )
        # MLP now takes [h_u ∥ h_v ∥ e_feat]
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*hidden_channels + edge_feat_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        # x: [N, in_channels], edge_attr: [M, edge_feat_dim]
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        src, dst = edge_index
        e = torch.cat([x[src], x[dst], edge_attr], dim=-1)  # [M, 2h+ef]
        return self.edge_mlp(e).squeeze(-1)                 # [M]


# ─── 4) Training & eval utils ──────────────────────────────────────────────────
def run_epoch(model, loader, optimizer=None):
    model.train() if optimizer else model.eval()
    total_loss, total_edges = 0.0, 0
    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.edge_attr)
        loss   = F.binary_cross_entropy_with_logits(
                     logits, data.edge_label.float()
                 )
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * data.num_edges
        total_edges += data.num_edges
    return total_loss / total_edges


@torch.no_grad()
def evaluate_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.edge_attr)
        preds  = (torch.sigmoid(logits) > 0.5)
        correct += (preds == data.edge_label.bool()).sum().item()
        total   += data.num_edges
    return correct / total


if __name__ == '__main__':
    # ─── 2) Prepare loaders ─────────────────────────────────────────────────────────
    N_TRAIN, N_VAL, N_TEST = 800, 100, 100
    train_graphs = make_dataset(N_TRAIN, seed=42)
    val_graphs = make_dataset(N_VAL, seed=43)
    test_graphs = make_dataset(N_TEST, seed=44)

    train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=8)
    test_loader = DataLoader(test_graphs, batch_size=8)

    print(f"Graphs ▶ train={len(train_graphs)}  val={len(val_graphs)}  test={len(test_graphs)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EdgeTransformer(
        in_channels=train_graphs[0].num_node_features,
        hidden_channels=128,
        edge_feat_dim=1
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ─── 5) Train with early stopping on val-loss ───────────────────────────────────
    best_val = float('inf')
    for epoch in range(1, 10):
        tr = run_epoch(model, train_loader, optimizer=opt)
        va = run_epoch(model, val_loader)
        te = run_epoch(model, test_loader)
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), 'best_model_sup_with_feats.pt')
        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:02d}  TRAIN {tr:.4f}  VAL {va:.4f}  TEST {te:.4f}")

    acc = evaluate_accuracy(model, test_loader)
    print(f"Test accuracy: {acc:.4f}")
