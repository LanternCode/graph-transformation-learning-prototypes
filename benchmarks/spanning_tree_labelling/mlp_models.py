import math
import random
import heapq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import deque, defaultdict
import networkx as nx
import numpy as np


# -------------------------------------------
# Utility: generate random spanning-tree edges via Prufer sequence
def generate_random_tree_edges(n):
    seq = [random.randrange(n) for _ in range(n-2)]
    deg = [1]*n
    for x in seq: deg[x] += 1
    leaves = [i for i,d in enumerate(deg) if d==1]
    heapq.heapify(leaves)
    edges = []
    for x in seq:
        leaf = heapq.heappop(leaves)
        edges.append((leaf, x))
        deg[leaf] -= 1; deg[x] -= 1
        if deg[x] == 1: heapq.heappush(leaves, x)
    u = heapq.heappop(leaves); v = heapq.heappop(leaves)
    edges.append((u, v))
    return edges


# -------------------------------------------
# Graph-level dataset: spanning trees + extra edges
torch.manual_seed(0)


class SpanningTreeGraphDataset(Dataset):
    def __init__(self, num_samples, num_nodes, extra_edge_prob=0.1, seed=None):
        self.graphs = []
        random.seed(seed)
        for _ in range(num_samples):
            tree_und = generate_random_tree_edges(num_nodes)
            # orient via BFS
            neigh = {i: [] for i in range(num_nodes)}
            for u,v in tree_und:
                neigh[u].append(v); neigh[v].append(u)
            visited = {0}; dq = deque([0]); tree_edges = []
            while dq:
                u = dq.popleft()
                for v in neigh[u]:
                    if v not in visited:
                        visited.add(v); dq.append(v); tree_edges.append((u,v))
            tree_set = set(tree_edges)
            # build full digraph
            G = nx.DiGraph(); G.add_nodes_from(range(num_nodes)); G.add_edges_from(tree_edges)
            # add extra edges
            for u in range(num_nodes):
                for v in range(num_nodes):
                    if u!=v and (u,v) not in tree_set and random.random()<extra_edge_prob:
                        G.add_edge(u,v)
            self.graphs.append({'edge_index': list(G.edges()),
                                'tree_set': tree_set,
                                'num_nodes': num_nodes})

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


# -------------------------------------------
# Build per-edge features with context: a_uv, deg(u), deg(v), pos(u), pos(v)
def build_edge_feats_and_labels(graph):
    N = graph['num_nodes']
    full = set(graph['edge_index'])
    out_deg = {i:0 for i in range(N)}
    in_deg  = {i:0 for i in range(N)}
    for u,v in graph['edge_index']:
        out_deg[u]+=1; in_deg[v]+=1
    feats=[]; labels=[]
    for u in range(N):
        for v in range(N):
            a_uv = 1.0 if (u,v) in full else 0.0
            du = out_deg[u]+in_deg[u]
            dv = out_deg[v]+in_deg[v]
            pu = [math.sin(u/ max(1,N)/10), math.cos(u/ max(1,N)/10)]
            pv = [math.sin(v/ max(1,N)/10), math.cos(v/ max(1,N)/10)]
            feats.append([a_uv, du, dv, pu[0], pu[1], pv[0], pv[1]])
            labels.append(1.0 if (u,v) in graph['tree_set'] else 0.0)
    return torch.tensor(feats, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


# -------------------------------------------
# Edge MLP with enriched features
class EdgeScorer(nn.Module):
    def __init__(self, in_dim=7, hidden_dims=[128,128,64]):
        super().__init__()
        layers=[]; prev=in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev,h), nn.ReLU()]; prev=h
        layers.append(nn.Linear(prev,1))
        self.net = nn.Sequential(*layers)

    def forward(self,x):
        return self.net(x).view(-1)


# -------------------------------------------
# Supervised trainer
def train_supervised(graph_train, graph_val, graph_test, device,
                     epochs=30, lr=1e-3, batch_size=4096):
    print(f"\n===== SUPERVISED ({len(graph_train)} graphs, {epochs} epochs) =====")
    # assemble features
    X_tr, y_tr = zip(*[build_edge_feats_and_labels(g) for g in graph_train])
    X_va, y_va = zip(*[build_edge_feats_and_labels(g) for g in graph_val])
    X_te, y_te = zip(*[build_edge_feats_and_labels(g) for g in graph_test])
    X_tr = torch.cat(X_tr); y_tr = torch.cat(y_tr)
    X_va = torch.cat(X_va); y_va = torch.cat(y_va)
    X_te = torch.cat(X_te); y_te = torch.cat(y_te)
    # DataLoaders
    def collate(batch): return torch.stack([b[0] for b in batch]).to(device), torch.stack([b[1] for b in batch]).to(device)
    ds_tr = list(zip(X_tr,y_tr)); ds_va = list(zip(X_va,y_va))
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=collate)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, collate_fn=collate)
    # model
    model = EdgeScorer().to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    crit  = nn.BCEWithLogitsLoss()
    best_val, best_state = float('inf'), None
    for ep in range(1, epochs+1):
        model.train(); train_loss=0
        for x,y in dl_tr:
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item()
        train_loss /= len(dl_tr)
        model.eval(); val_loss=0
        with torch.no_grad():
            for x,y in dl_va: val_loss += crit(model(x), y).item()
        val_loss /= len(dl_va)
        print(f"Epoch {ep:2d} ▶ train {train_loss:.4f}   val {val_loss:.4f}")
        if val_loss < best_val: best_val, best_state = val_loss, model.state_dict()
    sup_path = 'mlp_supervised_enriched.pth'
    torch.save(best_state, sup_path)
    print(f"Saved {sup_path}")
    # test
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        logits = model(X_te.to(device)); probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs>0.5).astype(int); true = y_te.numpy().astype(int)
    total=len(true); acc=(preds==true).sum()/total
    ones=preds.sum(); zeros=total-ones
    # full graph accuracy
    full_corr=0; idx=0
    for g in graph_test:
        n2 = g['num_nodes']**2
        if np.array_equal(preds[idx:idx+n2], true[idx:idx+n2]): full_corr+=1
        idx+=n2
    print(f"Test Acc: {acc:.4f}")
    print(f"Ones: {ones}, Zeros: {zeros}")
    print(f"Fully correct trees: {full_corr}/{len(graph_test)}")


# -------------------------------------------
# Policy trainer
def train_policy(graph_train, graph_val, graph_test, device,
                 epochs=30, lr=1e-3, entropy_coef=0.01, baseline_decay=0.9):
    mode = 'hard' if entropy_coef==0 and baseline_decay==0 else 'soft'
    print(f"\n===== POLICY ({mode.upper()}, {len(graph_train)} graphs, {epochs} epochs) =====")
    model = EdgeScorer().to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    baseline = 0.0; best_val=-1; best_state=None
    # training
    for ep in range(1, epochs+1):
        model.train()
        for g in graph_train:
            X, _ = build_edge_feats_and_labels(g)
            X = X.to(device); probs = torch.sigmoid(model(X))
            dist = torch.distributions.Bernoulli(probs)
            samp = dist.sample(); logp=dist.log_prob(samp).sum(); ent=dist.entropy().sum()
            r = spanning_tree_score(g, samp.cpu().numpy().astype(int))
            rew = 1.0 if (mode=='hard' and r==1.0) else (r if mode=='soft' else 0.0)
            baseline = baseline*baseline_decay + (1-baseline_decay)*rew
            adv = rew - baseline
            loss = -logp*adv - entropy_coef*ent
            opt.zero_grad(); loss.backward(); opt.step()
        # validation
        model.eval(); vals=[]
        with torch.no_grad():
            for g in graph_val:
                X, _ = build_edge_feats_and_labels(g)
                preds=(torch.sigmoid(model(X.to(device)))>0.5).int().cpu().numpy()
                true = np.array([1 if (u,v) in g['tree_set'] else 0 for u in range(g['num_nodes']) for v in range(g['num_nodes'])])
                vals.append((preds==true).mean())
        val_acc=sum(vals)/len(vals)
        print(f"Epoch {ep:2d} ▶ val_acc {val_acc:.4f}")
        if val_acc>best_val: best_val, best_state = val_acc, model.state_dict()
    path=f"mlp_policy_{mode}_enriched.pth"; torch.save(best_state, path)
    print(f"Saved {path}")
    # test
    model.load_state_dict(best_state); model.eval()
    total, corr, ones, zeros, full_corr = 0,0,0,0,0
    for g in graph_test:
        X, _ = build_edge_feats_and_labels(g)
        preds=(torch.sigmoid(model(X.to(device)))>0.5).int().cpu().numpy()
        true = np.array([1 if (u,v) in g['tree_set'] else 0 for u in range(g['num_nodes']) for v in range(g['num_nodes'])])
        total += len(preds); corr += int((preds==true).sum())
        ones += int(preds.sum()); zeros += int((preds==0).sum())
        if np.array_equal(preds, true): full_corr+=1
    print(f"Test Acc: {corr/total:.4f}")
    print(f"Ones: {ones}, Zeros: {zeros}")
    print(f"Fully correct trees: {full_corr}/{len(graph_test)}")


# saver for policy reward
def spanning_tree_score(graph, pred_list):
    edge_index, V = graph['edge_index'], graph['num_nodes']
    incoming=[0]*V; adj=defaultdict(list)
    for (u,v),l in zip(edge_index, pred_list):
        if l: adj[u].append(v); incoming[v]+=1
    over=sum(max(0,c-1) for c in incoming)
    roots=sum(1 for c in incoming if c==0)
    cycle=False; visited=set(); onpath=set()
    def dfs(u):
        nonlocal cycle
        visited.add(u); onpath.add(u)
        for w in adj[u]:
            if w not in visited: dfs(w)
            elif w in onpath: cycle=True
        onpath.remove(u)
    start = incoming.index(0) if 0 in incoming else 0
    dfs(start); unreach=V-len(visited)
    penalty = over + max(0,roots-1) + unreach + (V if cycle else 0)
    return 1.0 - min(1.0, penalty/V)


# -------------------------------------------
if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N=100
    train_g = SpanningTreeGraphDataset(750,N,extra_edge_prob=0.1,seed=1)
    val_g   = SpanningTreeGraphDataset(150, N,extra_edge_prob=0.1,seed=2)
    test_g  = SpanningTreeGraphDataset(100, N,extra_edge_prob=0.1,seed=3)
    train_supervised(train_g, val_g, test_g, device)
    train_policy(train_g, val_g, test_g, device, epochs=30, lr=1e-3, entropy_coef=0.01, baseline_decay=0.9)
    train_policy(train_g, val_g, test_g, device, epochs=30, lr=1e-3, entropy_coef=0.0,  baseline_decay=0.0)
