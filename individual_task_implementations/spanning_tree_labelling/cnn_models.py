import copy
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import heapq
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, deque


# -------------------------------------------
# Multi-objective score
def spanning_tree_score_from_prediction(graph, predicted_labels):
    edge_index = graph['edge_index']
    V = graph['num_nodes']
    incoming = [0]*V
    adj = defaultdict(list)
    for (u,v),lab in zip(edge_index, predicted_labels):
        if lab==1:
            adj[u].append(v)
            incoming[v]+=1
    over = sum(max(0,c-1) for c in incoming)
    roots = sum(1 for c in incoming if c==0)
    cycle=False
    visited=set(); onpath=set()
    def dfs(u):
        nonlocal cycle
        visited.add(u); onpath.add(u)
        for w in adj[u]:
            if w not in visited: dfs(w)
            elif w in onpath: cycle=True
        onpath.remove(u)
    start = next((i for i,c in enumerate(incoming) if c==0), 0)
    dfs(start)
    unreach = V - len(visited)
    penalty = over + max(0,roots-1) + unreach + (V if cycle else 0)
    return 1.0 - min(1.0, penalty/V)


# -------------------------------------------
# Generate random undirected tree edges via Prufer
def generate_random_tree_edges(n):
    prufer = [random.randrange(n) for _ in range(n-2)]
    deg = [1]*n
    for p in prufer: deg[p]+=1
    leaves = [i for i in range(n) if deg[i]==1]
    heapq.heapify(leaves)
    edges=[]
    for p in prufer:
        leaf = heapq.heappop(leaves)
        edges.append((leaf,p))
        deg[leaf]-=1; deg[p]-=1
        if deg[p]==1: heapq.heappush(leaves,p)
    u=heapq.heappop(leaves); v=heapq.heappop(leaves)
    edges.append((u,v))
    return edges


# -------------------------------------------
# Dataset
class SpanningTreeDataset(Dataset):
    def __init__(self, num_samples, num_nodes, extra_prob=0.1, seed=None):
        random.seed(seed)
        self.data=[]
        for _ in range(num_samples):
            und = generate_random_tree_edges(num_nodes)
            neigh={i:[] for i in range(num_nodes)}
            for u,v in und: neigh[u].append(v); neigh[v].append(u)
            vis={0}; dq=deque([0]); tree=[]
            while dq:
                u=dq.popleft()
                for v in neigh[u]:
                    if v not in vis:
                        vis.add(v); dq.append(v); tree.append((u,v))
            tree_set=set(tree)
            G=nx.DiGraph(); G.add_nodes_from(range(num_nodes)); G.add_edges_from(tree)
            for u in range(num_nodes):
                for v in range(num_nodes):
                    if u!=v and (u,v) not in tree_set and random.random()<extra_prob:
                        G.add_edge(u,v)
            A_full = torch.tensor(nx.to_numpy_array(G, nodelist=range(num_nodes)), dtype=torch.float)
            A_tree = torch.zeros_like(A_full)
            for u,v in tree: A_tree[u,v]=1.0
            self.data.append({'adj':A_full.view(-1), 'labels':A_tree.view(-1),
                              'edge_index':list(G.edges()), 'num_nodes':num_nodes})

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]


# Collate
def collate_batch(samples):
    adjs = torch.stack([s['adj'] for s in samples])
    labs = torch.stack([s['labels'] for s in samples])
    edges = [s['edge_index'] for s in samples]
    N = samples[0]['num_nodes']
    return {'adj':adjs, 'labels':labs, 'edge_index':edges, 'num_nodes':N}


# -------------------------------------------
# Model: CNN over adjacency matrix
class CNNEdgeLabeler(nn.Module):
    def __init__(self, N, hidden_channels=32):
        super().__init__()
        self.N = N
        # Treat adjacency as 1xN xN image
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x_flat):
        # x_flat: [B, N*N]
        B = x_flat.size(0)
        x = x_flat.view(B, 1, self.N, self.N)
        h = self.encoder(x)
        out = self.decoder(h)  # [B,1,N,N]
        return out.view(B, self.N*self.N)


# -------------------------------------------
# Supervised training
def train_supervised(model, train_dl, val_dl, test_dl, dev, epochs=30, lr=1e-3):
    print("\n===== SUPERVISED TRAINING =====")
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=3)
    crit = nn.BCEWithLogitsLoss()
    best_state, best_val, patience = None, float('inf'), 0
    for e in range(1, epochs+1):
        model.train(); train_loss=0.0
        for b in train_dl:
            x,y = b['adj'].to(dev), b['labels'].to(dev)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)
        model.eval(); val_loss=0.0
        with torch.no_grad():
            for b in val_dl:
                x,y = b['adj'].to(dev), b['labels'].to(dev)
                val_loss += crit(model(x), y).item()
        val_loss /= len(val_dl)
        print(f"Epoch {e:2d} ▶ train {train_loss:.4f}   val {val_loss:.4f}")
        sched.step(val_loss)
        if val_loss < best_val:
            best_val, best_state, patience = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
        if patience >= 5:
            print("Early stopping")
            break
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), 'model_supervised_best.pth')
    # test
    crit = nn.BCEWithLogitsLoss()
    test_loss=0.0; ones=0; total=0; correct_trees=0; total_graphs=0
    with torch.no_grad():
        for b in test_dl:
            x,y = b['adj'].to(dev), b['labels'].to(dev)
            logits = model(x)
            test_loss += crit(logits,y).item()
            preds = (torch.sigmoid(logits)>0.5).float()
            ones += int(preds.sum().item()); total += preds.numel()
            matches = (preds==y).all(dim=1)
            correct_trees += int(matches.sum().item()); total_graphs += preds.shape[0]
    test_loss /= len(test_dl)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Predictions -> Ones: {ones}, Zeros: {total-ones}")
    print(f"Entirely correct trees: {correct_trees}/{total_graphs}\n")


# -------------------------------------------
# Policy training (soft or hard)
def train_policy(model, train_dl, val_dl, test_dl, dev,
                 epochs=30, lr=1e-3, entropy_coef=0.01, baseline_decay=0.9):
    mode = 'hard' if entropy_coef==0 and baseline_decay==0 else 'soft'
    print(f"\n===== POLICY ({mode.upper()}) TRAINING =====")
    opt = optim.Adam(model.parameters(), lr=lr)
    baseline=0.0
    best_val, best_state = -1.0, None
    for e in range(1, epochs+1):
        model.train()
        for b in train_dl:
            x,y = b['adj'].to(dev), b['labels'].to(dev)
            logits = model(x); probs = torch.sigmoid(logits)
            dist = torch.distributions.Bernoulli(probs)
            samples = dist.sample()
            logp = dist.log_prob(samples).sum(dim=1)
            ent = dist.entropy().sum(dim=1)
            rewards = (samples==y).float().mean(dim=1)
            avg_r = rewards.mean().item()
            baseline = baseline*baseline_decay + (1-baseline_decay)*avg_r
            adv = (rewards-baseline).detach()
            loss_pg = -(logp*adv).mean()
            loss_ent = -entropy_coef*ent.mean()
            opt.zero_grad(); (loss_pg+loss_ent).backward(); opt.step()
        model.eval(); vals=[]
        with torch.no_grad():
            for b in val_dl:
                pred = (torch.sigmoid(model(b['adj'].to(dev)))>0.5).float()
                vals.append(((pred==b['labels'].to(dev)).float().mean().item()))
        val_acc = sum(vals)/len(vals)
        print(f"Epoch {e:2d} ▶ val_acc {val_acc:.4f}")
        if val_acc > best_val:
            best_val, best_state = val_acc, copy.deepcopy(model.state_dict())
    # save best
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), f"model_policy_{mode}_best.pth")
    # test
    ones=0; total=0; correct_trees=0; total_graphs=0; tests=[]
    with torch.no_grad():
        for b in test_dl:
            logits = model(b['adj'].to(dev))
            pred = (torch.sigmoid(logits)>0.5).float()
            tests.append(((pred==b['labels'].to(dev)).float().mean().item()))
            ones += int(pred.sum().item()); total += pred.numel()
            matches = (pred==b['labels'].to(dev)).all(dim=1)
            correct_trees += int(matches.sum().item()); total_graphs += pred.shape[0]
    print(f"Test Acc: {sum(tests)/len(tests):.4f}")
    print(f"Test Predictions -> Ones: {ones}, Zeros: {total-ones}")
    print(f"Entirely correct trees: {correct_trees}/{total_graphs}\n")


# -------------------------------------------
# Main
if __name__=='__main__':
    NUM_NODES, BATCH, EPOCHS = 100, 32, 30
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds_train = SpanningTreeDataset(5000, NUM_NODES, extra_prob=0.1, seed=42)
    ds_val   = SpanningTreeDataset(1000, NUM_NODES, extra_prob=0.1, seed=43)
    ds_test  = SpanningTreeDataset(1000, NUM_NODES, extra_prob=0.1, seed=44)
    tr = DataLoader(ds_train, batch_size=BATCH, shuffle=True, collate_fn=collate_batch)
    va = DataLoader(ds_val,   batch_size=BATCH, shuffle=False, collate_fn=collate_batch)
    te = DataLoader(ds_test,  batch_size=BATCH, shuffle=False, collate_fn=collate_batch)

    # 1) Supervised
    supervised_model = CNNEdgeLabeler(NUM_NODES).to(dev)
    train_supervised(supervised_model, tr, va, te, dev, epochs=EPOCHS)

    # 2) Policy Soft
    soft_policy_model = CNNEdgeLabeler(NUM_NODES).to(dev)
    train_policy(soft_policy_model, tr, va, te, dev,
                 epochs=EPOCHS, lr=1e-3, entropy_coef=0.01, baseline_decay=0.9)

    # 3) Policy Hard
    hard_policy_model = CNNEdgeLabeler(NUM_NODES).to(dev)
    train_policy(hard_policy_model, tr, va, te, dev,
                 epochs=EPOCHS, lr=1e-3, entropy_coef=0.0, baseline_decay=0.0)
