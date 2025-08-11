import os
import subprocess
import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm.auto import tqdm


def batch_encode(texts, batch_size=128):
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
    df2 = df.copy()
    df2['h_id'] = df2['h'].map(node2id)
    df2['t_id'] = df2['t'].map(node2id)
    df2['r_id'] = df2['r'].map(relation2id)

    # find rows where anything went wrong
    bad = df2[df2[['h_id','t_id','r_id']].isna().any(axis=1)]
    if not bad.empty:
        print("⚠️  encode(): dropping unmapped rows:")
        print(bad[['h','r','t']].head(10))
        df2 = df2.dropna(subset=['h_id','t_id','r_id'])
        print(f"… dropped {len(bad)} rows, continuing.")
    return df2


class EdgeClassifierGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_rel):
        super().__init__()
        # GraphSAGE layers
        self.sage1 = SAGEConv(in_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)

        # relation embedding layer (project 768→hidden)
        self.rel_emb = torch.nn.Embedding(num_rel, in_dim)
        self.rel_linear = torch.nn.Linear(in_dim, hidden_dim)

        # final MLP: [h_src ∥ rel ∥ h_dst] → score
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
        # relation → project to hidden
        r_hid = self.rel_linear(self.rel_emb(rel_label_index))

        # concat and score
        h_pair = torch.cat([h_src, r_hid, h_dst], dim=1)
        return self.classifier(h_pair).squeeze()


def evaluate(model, df_eval, use_full_k=False):
    model.eval()

    # 1) Build the batch of query edges + labels
    h_idx = torch.tensor(df_eval['h_id'].values, dtype=torch.long, device=device)
    t_idx = torch.tensor(df_eval['t_id'].values, dtype=torch.long, device=device)
    r_idx = torch.tensor(df_eval['r_id'].values, dtype=torch.long, device=device)
    labels = torch.tensor(df_eval['label'].values, dtype=torch.float, device=device)
    edge_label_index = torch.stack([h_idx, t_idx]).to(device)

    # 2) Mask out the query edges from the graph
    src, dst = graph.edge_index
    N = graph.x.size(0)
    edge_flat  = src * N + dst
    query_flat = h_idx * N + t_idx
    mask = ~torch.isin(edge_flat, query_flat)
    mask &= ~torch.isin(edge_flat, t_idx * N + h_idx)
    edge_index_masked = graph.edge_index[:, mask].to(device)

    # 3) Run the model
    logits = model(graph.x, edge_index_masked, edge_label_index, r_idx)
    probs  = torch.sigmoid(logits)

    # 4) Standard classification metrics
    preds_cpu = (probs > 0.5).cpu().numpy()
    labels_cpu = labels.cpu().numpy()
    probs_cpu  = probs.detach().cpu().numpy()
    if not use_full_k:
        print(classification_report(labels_cpu, preds_cpu, digits=4))
    auc = roc_auc_score(labels_cpu, probs_cpu)
    print(f"AUC: {auc:.4f}")

    # 5) Now the ranking / Recall@k part
    noise_score = (1 - probs).detach()   # high = more likely noise
    if use_full_k:
        k = len(df_noise)
    else:
        k = int((labels == 0).sum().item())

    # 6) Grab the top‐k by noise_score
    sorted_idx = torch.argsort(noise_score, descending=True)
    topk_idx   = sorted_idx[:k]

    # 8) compute Recall@k
    true_noise = (labels[topk_idx] == 0).sum().item()
    recall_at_k = true_noise / k if k > 0 else 0.0
    print(f"Recall@{k} = {recall_at_k:.4f}")

    return auc, recall_at_k


if __name__ == "__main__":
    # Part 1: Dataset
    TEST = False  # For debugging
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download GOLD if needed
    repo_dir = 'GOLD'
    if not os.path.isdir(repo_dir):
        subprocess.run(
            ["git", "clone", "https://github.com/HKUST-KnowComp/GOLD.git"],
            check=True
        )

    base = 'GOLD/dataset/atomic'
    noise_level = 'A-10'

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

    if TEST:
        # Only keep first 500 positives and 500 negatives
        df_train = df_train.iloc[:30000]
        df_valid = df_valid.iloc[:6000]
        df_test = df_test.iloc[:6000]
        df_noise = df_noise.iloc[:30000]

    for df in (df_train, df_valid, df_test, df_noise):
        # Strip whitespace
        for col in ('h', 'r', 't'):
            df[col] = df[col].astype(str).str.strip()
        # Drop real NaNs
        df.dropna(subset=['h', 'r', 't'], inplace=True)
        # Drop blank strings
        df = df[(df['h'] != '') & (df['r'] != '') & (df['t'] != '')]

    # Split noise 80% train / 10% valid / 10% test ===
    n = len(df_noise)
    cut1, cut2 = int(0.8 * n), int(0.9 * n)
    df_train_neg = df_noise.iloc[:cut1].copy()
    df_valid_neg = df_noise.iloc[cut1:cut2].copy()
    df_test_neg = df_noise.iloc[cut2:].copy()

    # Balance training positives to negatives ===
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

    # All clean triples (pos+neg) across train/valid/test
    df_all_pos = pd.concat([
        df_train[['h', 'r', 't', 'label']],
        df_valid[['h', 'r', 't', 'label']],
        df_test[['h', 'r', 't', 'label']],
    ], ignore_index=True)
    df_all_pos['label'] = 1

    bad = df_all_pos[df_all_pos[['h', 'r', 't']].isna().any(axis=1)]
    if len(bad):
        print("Found missing triples in df_all_pos:\n", bad)
        # Then drop them:
        df_all_pos = df_all_pos.dropna(subset=['h', 'r', 't'])
        print(f"Dropped {len(bad)} bad rows, continuing.")

    # All noise triples
    df_noise_all = df_noise

    # Unique nodes & relations from exactly those two sets
    all_nodes = pd.concat([
        df_all_pos[['h', 't']],
        df_noise_all[['h', 't']],
    ], ignore_index=True).stack().unique()

    all_rels = pd.concat([
        df_all_pos['r'],
        df_noise_all['r'],
    ], ignore_index=True).unique()

    # Build deterministic ID maps
    sorted_nodes = sorted(all_nodes)
    node2id = {n: i for i, n in enumerate(sorted_nodes)}

    sorted_rels = sorted(all_rels)
    relation2id = {r: i for i, r in enumerate(sorted_rels)}

    # BERT-encode features
    node_feats = batch_encode(sorted_nodes).to(device)  # [N×768]
    rel_feats = batch_encode(sorted_rels).to(device)  # [R×768]

    # Build PyG graph on raw positives
    df_all_pos_enc = encode(df_all_pos)
    edge_index = torch.cat([
        torch.tensor(df_all_pos_enc[['h_id', 't_id']].values.T, dtype=torch.long),
        torch.tensor(df_all_pos_enc[['t_id', 'h_id']].values.T, dtype=torch.long),
    ], dim=1).to(device)

    graph = Data(x=node_feats, edge_index=edge_index)

    # Prepare ranking set (for final eval)
    df_rank = pd.concat([df_all_pos, df_noise_all], ignore_index=True)
    df_rank_enc = encode(df_rank)

    # Part 3: Final data pre-processing and model training
    df_train_enc = encode(df_train_pos_samp)  # now has h_id, r_id, t_id
    df_train_neg_enc = encode(df_train_neg)
    df_valid_enc = encode(df_valid_all)
    df_test_enc = encode(df_test_all)
    P = len(df_train_enc)  # pool size for hard‐mining
    N_neg = len(df_train_neg_enc)

    # move all negatives on GPU once
    h_neg_all = torch.tensor(df_train_neg_enc['h_id'].values, dtype=torch.long, device=device)
    t_neg_all = torch.tensor(df_train_neg_enc['t_id'].values, dtype=torch.long, device=device)
    r_neg_all = torch.tensor(df_train_neg_enc['r_id'].values, dtype=torch.long, device=device)

    # prepare positives loader
    h_pos = torch.tensor(df_train_enc['h_id'].values, dtype=torch.long)
    t_pos = torch.tensor(df_train_enc['t_id'].values, dtype=torch.long)
    r_pos = torch.tensor(df_train_enc['r_id'].values, dtype=torch.long)
    pos_ds = TensorDataset(h_pos, t_pos, r_pos)
    pos_loader = DataLoader(pos_ds, batch_size=256, shuffle=True, pin_memory=True)

    # model + optimizer + AMP
    model = EdgeClassifierGNN(in_dim=768, hidden_dim=128, num_rel=len(relation2id)).to(device)
    model.rel_emb.weight.data.copy_(rel_feats.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()
    best_recall = 0.0

    for epoch in range(1, 31):
        # 1) mine “hard” negatives by scoring every candidate
        model.eval()
        with torch.no_grad():
            idx_pair = torch.stack([h_neg_all, t_neg_all])
            neg_logits = model(graph.x, graph.edge_index, idx_pair, r_neg_all)
            neg_probs = torch.sigmoid(neg_logits)
            hard_idx = torch.argsort(neg_probs, descending=True)[:P]
            hard_h_all = h_neg_all[hard_idx]
            hard_t_all = t_neg_all[hard_idx]
            hard_r_all = r_neg_all[hard_idx]

        # 2) train with a 25 % hard / 75 % random mix + BPR loss
        model.train()
        total_loss = 0.0
        for h_p, t_p, r_p in pos_loader:
            h_p, t_p, r_p = h_p.to(device), t_p.to(device), r_p.to(device)
            B = h_p.size(0)
            n_hard = B // 4
            n_rand = B - n_hard

            # sample hard negatives
            idx_h = torch.randint(0, P, (n_hard,), device=device)
            h_hard = hard_h_all[idx_h]
            t_hard = hard_t_all[idx_h]
            r_hard = hard_r_all[idx_h]

            # sample random negatives
            idx_r = torch.randint(0, N_neg, (n_rand,), device=device)
            h_rand = h_neg_all[idx_r]
            t_rand = t_neg_all[idx_r]
            r_rand = r_neg_all[idx_r]

            # assemble one negative batch of size B
            h_n = torch.cat([h_hard, h_rand], dim=0)
            t_n = torch.cat([t_hard, t_rand], dim=0)
            r_n = torch.cat([r_hard, r_rand], dim=0)

            # build your full (pos+neg) batch
            h_batch = torch.cat([h_p, h_n], dim=0)
            t_batch = torch.cat([t_p, t_n], dim=0)
            r_batch = torch.cat([r_p, r_n], dim=0)
            edge_lab_idx = torch.stack([h_batch, t_batch]).to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(graph.x, graph.edge_index, edge_lab_idx, r_batch)
                pos_score = logits[:B]
                neg_score = logits[B:]
                margin = 0.5
                #loss = -F.logsigmoid(pos_score - neg_score).mean() # BPR loss: −log σ( pos_score − neg_score )
                loss = F.softplus(neg_score - pos_score + margin).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        print(f"Epoch {epoch:02d} — Loss: {total_loss:.4f}")

        # checkpoint on Recall@k
        val_auc, val_rk = evaluate(model, df_valid_enc)
        if val_rk > best_recall:
            best_recall = val_rk
            torch.save(model.state_dict(), f'bert_sage_best_model_{noise_level}.pth')
            print(f"→ Saved best model (Recall@k={val_rk:.4f})")

    # final test eval (full‐graph ranking)
    print("\n=== Final test‐set evaluation ===")
    model.load_state_dict(torch.load(f'bert_sage_best_model_{noise_level}.pth'))
    test_auc, test_rk = evaluate(model, df_rank_enc, True)
    print(f"Test AUC: {test_auc:.4f}, Test Recall@{len(df_noise)}: {test_rk:.4f}")
