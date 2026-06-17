import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


# ─── (1) Same data‐generator as above ─────────────────────────────────────────
def make_spanning_candidate(n_nodes, feat_dim=16):
    T = nx.random_unlabeled_tree(n_nodes, seed=random.randint(0,1e6))
    tree_edges = set(T.edges())
    n_tree = n_nodes - 1
    all_pairs = {(u, v) for u in range(n_nodes) for v in range(u+1, n_nodes)}
    extra = random.sample(list(all_pairs - tree_edges), k=n_tree)
    undirected = list(tree_edges) + extra

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(undirected)
    deg = np.array([d for _,d in G.degree()], dtype=np.float32)
    deg /= deg.max() if deg.max()>0 else 1.0
    bc = nx.edge_betweenness_centrality(G)
    edge_bc = [bc.get((u,v), bc.get((v,u),0.0)) for (u,v) in undirected]

    ei, el, ebc = [[],[]], [], []
    for (u,v), b in zip(undirected, edge_bc):
        label = 1 if (u,v) in tree_edges or (v,u) in tree_edges else 0
        ei[0] += [u, v]; ei[1] += [v, u]
        el   += [label, label]
        ebc  += [b, b]

    edge_index = torch.tensor(ei, dtype=torch.long)
    edge_label = torch.tensor(el, dtype=torch.float)
    edge_attr  = torch.tensor(ebc, dtype=torch.float).unsqueeze(-1)

    rand_feat = torch.randn((n_nodes, feat_dim), dtype=torch.float)
    deg_feat  = torch.tensor(deg, dtype=torch.float).unsqueeze(-1)
    x = torch.cat([rand_feat, deg_feat], dim=1)

    return Data(x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_label=edge_label)

def make_dataset(N, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return [make_spanning_candidate(random.randint(6,100)) for _ in range(N)]


# ─── (3) Same EdgeTransformer as MST ──────────────────────────────────────────
class EdgeTransformerMST(nn.Module):
    def __init__(self, in_channels, hidden, n_layers=3, n_heads=4):
        super().__init__()
        from torch_geometric.nn import TransformerConv
        edge_dim = 1
        self.convs = nn.ModuleList()
        self.convs.append(
            TransformerConv(in_channels, hidden//n_heads,
                            heads=n_heads, edge_dim=edge_dim)
        )
        for _ in range(n_layers-1):
            self.convs.append(
                TransformerConv(hidden, hidden//n_heads,
                                heads=n_heads, edge_dim=edge_dim)
            )
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*hidden + edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        src, dst = edge_index
        e = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        return self.edge_mlp(e).squeeze(-1)


# ─── (4) Sampling helper ───────────────────────────────────────────────────────
class UnionFind:
    def __init__(self,n):
        self.p=list(range(n))

    def find(self,a):
        while self.p[a]!=a: a=self.p[a]
        return a

    def union(self,a,b):
        ra,rb=self.find(a),self.find(b)
        if ra!=rb: self.p[rb]=ra; return True
        return False


def sample_tree(logits, edge_index, n_nodes):
    # pure PyTorch sampling
    probs = torch.sigmoid(logits)
    M = edge_index.size(1)
    order = torch.multinomial(probs+1e-6, num_samples=M, replacement=False).tolist()
    uf, chosen = UnionFind(n_nodes), []
    for i in order:
        u,v = edge_index[0,i].item(), edge_index[1,i].item()
        if uf.union(u,v):
            chosen.append(i)
        if len(chosen)==n_nodes-1:
            break
    return chosen


# ─── (5) Policy‐gradient epoch ────────────────────────────────────────────────
def run_epoch_rl(model, loader, optimizer=None, baseline=0.0,
                 K=10, entropy_coef=1e-2):
    model.train() if optimizer else model.eval()
    total_loss, total_reward, total_edges = 0.,0.,0
    for data in loader:
        data = data.to(device)
        logits   = model(data.x, data.edge_index, data.edge_attr)
        probs    = torch.sigmoid(logits)
        ent = -(probs*(probs+1e-8).log() + (1-probs)*(1-probs+1e-8).log()).sum()

        losses, rewards = [], []
        for _ in range(K):
            chosen = sample_tree(logits, data.edge_index, data.num_nodes)
            mask = torch.zeros_like(logits); mask[chosen]=1.
            r = (mask * data.edge_label).sum().item()/(data.num_nodes-1)
            rewards.append(r)
            logp = -F.binary_cross_entropy_with_logits(logits, mask, reduction='none')
            losses.append(-(r-baseline)*logp[chosen].sum())

        rl_loss = torch.stack(losses).mean()
        loss    = rl_loss - entropy_coef * ent

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        avg_r = sum(rewards)/len(rewards)
        total_loss   += loss.item()*data.num_edges
        total_reward += avg_r*data.num_edges
        total_edges  += data.num_edges

    return total_loss/total_edges, total_reward/total_edges


if __name__ == '__main__':
    # ─── (2) Build loaders ─────────────────────────────────────────────────────────
    train_loader = DataLoader(make_dataset(800, seed=42), batch_size=1, shuffle=True)
    val_loader = DataLoader(make_dataset(100, seed=43), batch_size=1)
    test_loader = DataLoader(make_dataset(100, seed=44), batch_size=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_rl = EdgeTransformerMST(
        in_channels=make_dataset(1, 0)[0].num_node_features,
        hidden=128
    ).to(device)

    # ─── (6) Training loop ─────────────────────────────────────────────────────────
    opt_rl = torch.optim.Adam(model_rl.parameters(), lr=1e-4)

    best_val_r = float('-inf')
    baseline = 0.0
    for epoch in range(1, 11):
        tr_l, tr_r = run_epoch_rl(model_rl, train_loader, optimizer=opt_rl,
                                  baseline=baseline)
        va_l, va_r = run_epoch_rl(model_rl, val_loader)
        te_l, te_r = run_epoch_rl(model_rl, test_loader)
        baseline = 0.9 * baseline + 0.1 * tr_r
        if va_r > best_val_r:
            best_val_r = va_r
            torch.save(model_rl.state_dict(), 'best_model_rl_with_feats.pt')
        if epoch == 1 or epoch % 1 == 0:
            print(f"[RL] E{epoch:03d} TR_L {tr_l:.4f} R {tr_r:.4f}  "
                  f"VA_L {va_l:.4f} R {va_r:.4f}  TE_L {te_l:.4f} R {te_r:.4f}")
            