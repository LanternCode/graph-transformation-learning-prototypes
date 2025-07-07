import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool


# ─── 1) Data‐generation (with degree & betweenness) ───────────────────────────
def make_spanning_candidate(n_nodes, feat_dim=16):
    # 1a) Spanning tree + exactly n_tree extra edges
    T = nx.random_unlabeled_tree(n_nodes, seed=random.randint(0,1e6))
    tree_edges = set(T.edges())
    n_tree = n_nodes - 1
    all_pairs = {(u, v) for u in range(n_nodes) for v in range(u+1, n_nodes)}
    extras   = random.sample(list(all_pairs - tree_edges), k=n_tree)
    undirected = list(tree_edges) + extras

    # 1b) Build NX graph for features
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(undirected)
    # node degree (normalized)
    deg = np.array([d for _, d in G.degree()], dtype=np.float32)
    deg /= deg.max() if deg.max()>0 else 1.0
    # edge betweenness
    bc = nx.edge_betweenness_centrality(G)
    edge_bc = [bc.get((u,v), bc.get((v,u),0.0)) for (u,v) in undirected]

    # 1c) Build directed edges, labels, attrs
    ei, el, ebc = [[],[]], [], []
    for (u,v), b in zip(undirected, edge_bc):
        label = 1 if (u,v) in tree_edges or (v,u) in tree_edges else 0
        ei[0] += [u, v]; ei[1] += [v, u]
        el   += [label, label]
        ebc  += [b, b]
    edge_index = torch.tensor(ei, dtype=torch.long)
    edge_label = torch.tensor(el, dtype=torch.float)
    edge_attr  = torch.tensor(ebc, dtype=torch.float).unsqueeze(-1)

    # 1d) Node features = random ∥ degree
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


# ─── 3) Actor–Critic Model ────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, in_ch, hidden, n_layers=3, n_heads=4):
        super().__init__()
        edge_dim = 1
        # Shared GNN backbone
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(in_ch, hidden//n_heads,
                                          heads=n_heads, edge_dim=edge_dim))
        for _ in range(n_layers-1):
            self.convs.append(TransformerConv(hidden, hidden//n_heads,
                                              heads=n_heads, edge_dim=edge_dim))
        # Actor edge‐MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*hidden + edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Critic graph‐MLP (global pooling → value)
        self.critic_mlp = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # 1) GNN
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        # 2) Actor: edge‐logits
        src, dst = edge_index
        e = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        logits = self.edge_mlp(e).squeeze(-1)  # [num_edges]
        # 3) Critic: graph‐value
        graph_emb = global_mean_pool(x, batch)  # [batch_size, hidden]
        values = self.critic_mlp(graph_emb).squeeze(-1)  # [batch_size]
        return logits, values


# ─── 4) Helpers: sampling + MST‐loss ───────────────────────────────────────────
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a: a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra
            return True
        return False


def sample_tree(logits, edge_index, n_nodes):
    probs = torch.sigmoid(logits)
    M = edge_index.size(1)
    order = torch.multinomial(probs + 1e-6, num_samples=M, replacement=False).tolist()
    uf, chosen = UnionFind(n_nodes), []
    for i in order:
        u, v = edge_index[0,i].item(), edge_index[1,i].item()
        if uf.union(u, v):
            chosen.append(i)
        if len(chosen) == n_nodes-1:
            break
    return chosen


def mst_loss(logits, edge_index, target_tree, n_nodes, jitter=1e-3):
    w = F.softplus(logits) + jitter
    w = w.clamp(min=1e-3, max=1e3)
    A = torch.zeros(n_nodes, n_nodes, device=w.device)
    src, dst = edge_index
    A[src,dst] = w; A[dst,src] = w
    D = torch.diag(A.sum(dim=1)); L = D - A
    Lm = L[1:,1:] + torch.eye(n_nodes-1, device=L.device)*jitter
    eigs = torch.linalg.eigvalsh(Lm).clamp(min=1e-6)
    logZ = eigs.log().sum()
    idx = torch.nonzero(target_tree, as_tuple=False).squeeze(-1)
    tree_term = -torch.log(w[idx]).sum()
    return tree_term + logZ


@torch.no_grad()
def evaluate_rl_accuracy(model, loader, device):
    """
    Greedy decode actor‐critic policy and report reconstruction accuracy.
    """
    model.eval()
    correct, total = 0, 0
    for data in loader:
        data = data.to(device)
        logits, _ = model(data.x, data.edge_index, data.edge_attr, data.batch)
        p = torch.sigmoid(logits).cpu().numpy()
        # undirected weights
        src, dst = data.edge_index.cpu()
        und_p = {}
        for u,v,pp in zip(src.tolist(), dst.tolist(), p):
            if u < v: und_p[(u,v)] = pp
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        for (u,v),pp in und_p.items(): G.add_edge(u,v,weight=pp)
        T = nx.maximum_spanning_tree(G)
        tree = set(T.edges())
        # mask directed edges
        mask = torch.zeros(data.num_edges, dtype=torch.bool)
        for i,(u,v) in enumerate(zip(src.tolist(), dst.tolist())):
            if (u,v) in tree or (v,u) in tree:
                mask[i] = True
        true = data.edge_label.bool().cpu()
        correct += (mask & true).sum().item()
        total   += 2*(data.num_nodes-1)
    return correct / total


if __name__ == '__main__':
    # ─── 2) DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(make_dataset(800, seed=42), batch_size=1, shuffle=True)
    val_loader = DataLoader(make_dataset(100, seed=43), batch_size=1)
    test_loader = DataLoader(make_dataset(100, seed=44), batch_size=1)

    # ─── 5) Training loop with Actor–Critic & Entropy Annealing ───────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActorCritic(in_ch=17, hidden=128).to(device)  # 16 rnd + 1 deg = 17
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    max_epochs = 10
    K = 10
    ent_coef_start = 0.1
    ent_coef_final = 0.0
    mst_coef = 0.1

    best_val_acc = 0.0

    for epoch in range(1, max_epochs + 1):
        # linear entropy annealing
        ent_coef = ent_coef_start * (1 - (epoch - 1) / (max_epochs - 1)) \
                   + ent_coef_final * ((epoch - 1) / (max_epochs - 1))

        model.train()
        for data in train_loader:
            data = data.to(device)
            logits, values = model(data.x, data.edge_index, data.edge_attr, data.batch)
            # graph‐value is a single scalar since batch_size=1
            value = values[0]

            # entropy of Bernoulli edges
            probs = torch.sigmoid(logits)
            entropy = -(probs * (probs + 1e-8).log() + (1 - probs) * (1 - probs + 1e-8).log()).sum()

            actor_losses, critic_losses, rewards = [], [], []
            for _ in range(K):
                chosen = sample_tree(logits, data.edge_index, data.num_nodes)
                mask = torch.zeros_like(logits);
                mask[chosen] = 1.0
                # reward = fraction of correct edges
                r = (mask * data.edge_label).sum().item() / (data.num_nodes - 1)
                rewards.append(r)
                # log-probs per edge
                logp = -F.binary_cross_entropy_with_logits(logits, mask, reduction='none')
                # actor loss with critic‐baseline
                actor_losses.append(-(r - value.item()) * logp[chosen].sum())
                # critic MSE
                critic_losses.append(F.mse_loss(value, torch.tensor(r, device=device)))

            actor_loss = torch.stack(actor_losses).mean()
            critic_loss = torch.stack(critic_losses).mean()
            # MST regulariser
            mst_l = mst_loss(logits, data.edge_index,
                             data.edge_label.bool(), data.num_nodes)

            loss = actor_loss + critic_loss - ent_coef * entropy + mst_coef * mst_l

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Evaluate on validation split
        val_acc = evaluate_rl_accuracy(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_actor_critic.pt')

        if epoch == 1 or epoch % 1 == 0:
            print(f"Epoch {epoch:02d}  Val‐Acc {val_acc:.4f}  (best {best_val_acc:.4f})  EntCoef {ent_coef:.4f}")

    # ─── 6) Final test accuracy ────────────────────────────────────────────────────
    model.load_state_dict(torch.load('best_model_actor_critic.pt'))
    test_acc = evaluate_rl_accuracy(model, test_loader, device)
    print(f"🎯 Test reconstruction accuracy: {test_acc:.4f}")