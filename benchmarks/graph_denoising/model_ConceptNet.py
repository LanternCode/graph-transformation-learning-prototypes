import os
import subprocess
import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


def batch_encode(texts, batch_size=64):
    embs = []
    # tqdm over the batch‐offsets
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT encoding", unit="batch"):
        batch = texts[i : i + batch_size]
        inputs = {k: v.to(device) for k, v in tokenizer(batch,
                                                        return_tensors='pt',
                                                        padding=True,
                                                        truncation=True,
                                                        max_length=32).items()}
        with torch.no_grad():
            out = bert_model(**inputs).last_hidden_state.mean(dim=1)
        embs.append(out)
    return torch.cat(embs, dim=0)


def encode(df):
    # map strings -> numeric IDs for head, relation, tail
    return df.assign(
        h_id = df['h'].map(node2id),
        r_id = df['r'].map(relation2id),
        t_id = df['t'].map(node2id),
    )


class EdgeClassifierGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_rel):
        super().__init__()
        # GraphSAGE layers
        self.sage1 = SAGEConv(in_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)

        # relation embedding layer (project 768->hidden)
        self.rel_emb = torch.nn.Embedding(num_rel, in_dim)
        self.rel_linear = torch.nn.Linear(in_dim, hidden_dim)

        # final MLP: [h_src ∥ rel ∥ h_dst] -> score
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 3, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_label_index, rel_label_index):
        # two‐layer SAGE
        x = self.sage1(x, edge_index).relu()
        x = self.sage2(x, edge_index)

        h_src = x[edge_label_index[0]]    # [B×hidden]
        h_dst = x[edge_label_index[1]]    # [B×hidden]
        # relation -> project to hidden
        r_hid = self.rel_linear(self.rel_emb(rel_label_index))

        # concat and score
        h_pair = torch.cat([h_src, r_hid, h_dst], dim=1)
        return self.classifier(h_pair).squeeze()


def evaluate(model, df_eval, use_full_k=False):
    model.eval()

    # Build the batch of query edges + labels
    h_idx = torch.tensor(df_eval['h_id'].values, dtype=torch.long, device=device)
    t_idx = torch.tensor(df_eval['t_id'].values, dtype=torch.long, device=device)
    r_idx = torch.tensor(df_eval['r_id'].values, dtype=torch.long, device=device)
    labels = torch.tensor(df_eval['label'].values, dtype=torch.float, device=device)
    edge_label_index = torch.stack([h_idx, t_idx]).to(device)

    # Compute a flat index for every graph edge and every query edge
    src, dst = graph.edge_index
    N = graph.x.size(0)
    edge_flat = src * N + dst  # shape [E]
    query_flat = h_idx * N + t_idx  # shape [B]

    # Make a boolean mask that drops all query edges (and their reverse) by checking membership in query_flat
    mask = ~torch.isin(edge_flat, query_flat)
    mask &= ~torch.isin(edge_flat, t_idx * N + h_idx)

    # Build the masked edge_index
    edge_index_masked = graph.edge_index[:, mask].to(device)

    # Forward on the masked graph
    logits = model(graph.x, edge_index_masked, edge_label_index, r_idx)
    probs = torch.sigmoid(logits)

    # Compute the metrics
    preds_cpu = (probs > 0.5).cpu().numpy()
    labels_cpu = labels.cpu().numpy()
    probs_cpu = probs.detach().cpu().numpy()
    if not use_full_k:
        print(classification_report(labels_cpu, preds_cpu, digits=4))
    auc = roc_auc_score(labels_cpu, probs_cpu)
    print("AUC:", auc)

    noise_score = (1 - probs).detach()
    # choose k
    if use_full_k:
        k = len(df_noise)  # full-graph Recall@K
    else:
        k = int((labels == 0).sum().item())  # per-split Recall@k
    topk = torch.argsort(noise_score, descending=True)[:k]
    recall_at_k = (labels[topk] == 0).float().mean().item()
    print(f"Recall@{k} = {recall_at_k:.4f}")

    return auc, recall_at_k


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download GOLD if needed
    repo_dir = 'GOLD'
    if not os.path.isdir(repo_dir):
        subprocess.run(
            ["git", "clone", "https://github.com/HKUST-KnowComp/GOLD.git"],
            check=True
        )

    base = 'GOLD/dataset/conceptnet'
    noise_level = 'C-20'

    # Load positives
    df_train = pd.read_csv(os.path.join(base, 'train.txt'),
                           sep='\t', header=None,
                           names=['h', 'r', 't'])
    df_valid = pd.read_csv(os.path.join(base, 'valid.txt'),
                           sep='\t', header=None,
                           names=['h', 'r', 't'])
    df_test = pd.read_csv(os.path.join(base, 'test.txt'),
                          sep='\t', header=None,
                          names=['h', 'r', 't'])
    for df in (df_train, df_valid, df_test):
        df['label'] = 1

    # Load the ground-truth noise file and mark label=0
    err_path = os.path.join(base, 'errors', f'{noise_level}-error.txt')
    df_noise = pd.read_csv(err_path,
                           sep='\t', header=None,
                           names=['h', 'r', 't'])
    df_noise = df_noise.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_noise['label'] = 0

    # Split noise 80% train / 10% valid / 10% test
    n = len(df_noise)
    cut1, cut2 = int(0.8 * n), int(0.9 * n)
    df_train_neg = df_noise.iloc[:cut1].copy()
    df_valid_neg = df_noise.iloc[cut1:cut2].copy()
    df_test_neg = df_noise.iloc[cut2:].copy()

    # 4. Balance training positives to negatives
    df_train_pos_samp = df_train.sample(
        n=len(df_train_neg), random_state=42
    )
    df_train_all = pd.concat(
        [df_train_pos_samp, df_train_neg], ignore_index=True
    )
    df_valid_all = pd.concat([df_valid, df_valid_neg], ignore_index=True)
    df_test_all = pd.concat([df_test, df_test_neg], ignore_index=True)

    # Part 2: PLM
    MODEL_DIR = "/home/a/amm106/transformers_cache/bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    bert_model = BertModel.from_pretrained(MODEL_DIR, local_files_only=True)
    bert_model.eval()
    bert_model.to(device)

    # Make one big DataFrame of every h/t pair we'll ever use:
    df_pos_all = pd.concat([df_train[['h', 't']],
                            df_valid[['h', 't']],
                            df_test[['h', 't']]], ignore_index=True)
    df_neg_all = pd.concat([df_train_neg[['h', 't']],
                            df_valid_neg[['h', 't']],
                            df_test_neg[['h', 't']]], ignore_index=True)

    # Collect all unique nodes & relations across splits
    all_nodes = pd.concat([df_pos_all, df_neg_all]).stack().unique()
    df_pos_rels = pd.concat([df_train['r'], df_valid['r'], df_test['r']], ignore_index=True)
    df_neg_rels = df_noise['r']
    all_rels = pd.concat([df_pos_rels, df_neg_rels]).unique()

    # Fix the node order and build the map
    sorted_nodes = sorted(all_nodes)  # deterministic order
    node2id      = {n: i for i, n in enumerate(sorted_nodes)}

    # Batch‐encode in that same order
    node_feats = batch_encode(sorted_nodes)  # [N, 768]

    # Now the feature matrix lines up perfectly
    x = node_feats.to(device)

    # Do the exact same for relations
    sorted_rels = sorted(all_rels)
    relation2id = {r: i for i, r in enumerate(sorted_rels)}
    rel_feats = batch_encode(sorted_rels)  # [R, 768]

    # Merge all clean positives (train + valid + test)
    df_all_pos = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    df_all_pos_enc = encode(df_all_pos)

    # Build the full ranking set: all clean + all noise
    df_rank = pd.concat([df_all_pos, df_noise], ignore_index=True)
    df_rank_enc = encode(df_rank)

    # Build undirected edge_index over the entire clean graph
    edge_idx_full = torch.cat([
        torch.tensor(df_all_pos_enc[['h_id', 't_id']].values.T, dtype=torch.long),
        torch.tensor(df_all_pos_enc[['t_id', 'h_id']].values.T, dtype=torch.long),
    ], dim=1)

    # reate the Data object on the full graph
    graph = Data(x=x, edge_index=edge_idx_full)
    graph.edge_index = graph.edge_index.to(device)

    # Now that node2id & relation2id exist:
    df_valid_enc = encode(df_valid_all)
    df_test_enc  = encode(df_test_all)

    # Part 3: Training prep
    # TensorDataset with (h, t, r, label)
    df_train_enc = encode(df_train)
    df_train_neg_enc = encode(df_train_neg)
    h_pos = torch.tensor(df_train_enc['h_id'].values, dtype=torch.long)
    t_pos = torch.tensor(df_train_enc['t_id'].values, dtype=torch.long)
    r_pos = torch.tensor(df_train_enc['r_id'].values, dtype=torch.long)
    pos_ds = TensorDataset(h_pos, t_pos, r_pos)
    pos_loader = DataLoader(pos_ds, batch_size=256, shuffle=True)

    # Init model, copy rel2vec -> rel_emb weights
    model = EdgeClassifierGNN(in_dim=768,
                              hidden_dim=128,
                              num_rel=len(relation2id)).to(device)

    model.rel_emb.weight.data.copy_(rel_feats.to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_auc = 0.0
    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0

        for h_p, t_p, r_p in pos_loader:
            h_p = h_p.to(device)
            t_p = t_p.to(device)
            r_p = r_p.to(device)
            # sample negatives (same batch size)
            idx_n = torch.randint(0, len(df_train_neg_enc), (h_p.size(0),), device=device)
            neg = df_train_neg_enc.iloc[idx_n.cpu()]
            h_n = torch.tensor(neg.h_id.values, dtype=torch.long, device=device)
            t_n = torch.tensor(neg.t_id.values, dtype=torch.long, device=device)
            r_n = torch.tensor(neg.r_id.values, dtype=torch.long, device=device)

            # concatenate pos + neg
            h_batch = torch.cat([h_p, h_n], dim=0)
            t_batch = torch.cat([t_p, t_n], dim=0)
            r_batch = torch.cat([r_p, r_n], dim=0)
            y_batch = torch.cat([
                torch.ones_like(h_p, dtype=torch.float, device=device),
                torch.zeros_like(h_n, dtype=torch.float, device=device)
            ], dim=0)

            # forward + backward
            optimizer.zero_grad()
            edge_lab_idx = torch.stack([h_batch, t_batch]).to(device)
            logits = model(graph.x, graph.edge_index, edge_lab_idx, r_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch:02d} — Loss: {total_loss:.4f}")
        val_auc, val_rk = evaluate(model, df_valid_enc)
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), f'bert_sage_best_model_{noise_level}.pth')
            print(f"-> Saved best model (AUC={val_auc:.4f})")

    print("\n=== Final test‐set evaluation ===")
    best_path = f'bert_sage_best_model_{noise_level}.pth'
    model.load_state_dict(torch.load(best_path))
    test_auc, test_rk = evaluate(model, df_rank_enc, True)
    print(f"Test AUC: {test_auc:.4f}, Test Recall@{len(df_noise)}: {test_rk:.4f}")
