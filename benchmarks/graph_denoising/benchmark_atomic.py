import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch_geometric.data import Data
from sklearn.metrics import classification_report, roc_auc_score
from model import EdgeClassifierGNN

# ─── CONFIG ────────────────────────────────────────────────────────────────
base        = 'GOLD/dataset/atomic'
noise_level = 'A-20'
MODEL_DIR   = "/home/a/amm106/transformers_cache/bert-base-uncased"
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_MODE   = False
MAX_RECORDS = 500
# ─────────────────────────────────────────────────────────────────────────────

# 1) Load ATOMIC train positives (for adjacency)
df_train_pos = pd.read_csv(
    os.path.join(base, 'train.txt'),
    sep='\t', names=['h','r','t']
)
df_train_pos['label'] = 1
if TEST_MODE:
    df_train_pos = df_train_pos.sample(n=1000, random_state=42)

# 2) Load ATOMIC test positives (label=1) with optional cap
read_kw = {'sep':'\t', 'names':['h','r','t']}
if TEST_MODE:
    read_kw['nrows'] = MAX_RECORDS // 2
df_test_pos = pd.read_csv(
    os.path.join(base, 'test.txt'),
    **read_kw
)
df_test_pos['label'] = 1

# 3) Load ATOMIC-N20 injected-noise negatives (label=0) with optional cap
read_kw = {'sep':'\t', 'names':['h','r','t']}
if TEST_MODE:
    read_kw['nrows'] = MAX_RECORDS - len(df_test_pos)
df_test_neg = pd.read_csv(
    os.path.join(base, 'errors', f'{noise_level}-error.txt'),
    **read_kw
)
df_test_neg['label'] = 0

# 4) Concatenate into a single test set
df_test = pd.concat([df_test_pos, df_test_neg], ignore_index=True)

# ─── Now that all DataFrames exist, build node embeddings ─────────────
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
bert = BertModel.from_pretrained(MODEL_DIR,   local_files_only=True).to(device).eval()
gnn = EdgeClassifierGNN(in_channels=768, hidden_channels=128).to(device).eval()
gnn.load_state_dict(torch.load('bert_sage_best_model.pth', map_location=device))


@torch.no_grad()
def encode_bert(text):
    toks = tokenizer(str(text), return_tensors="pt",
                     truncation=True, padding=True, max_length=32).to(device)
    return bert(**toks).last_hidden_state.mean(dim=1).squeeze(0).cpu()


# 5) Collect every unique node string across train+test
all_nodes = pd.concat([
    df_train_pos[['h','t']],
    df_test[['h','t']]
]).stack().astype(str).unique()

# 6) Embed them and build node2id
node2vec = {n: encode_bert(n) for n in all_nodes}
nodes    = sorted(node2vec.keys())
node2id  = {n:i for i,n in enumerate(nodes)}

# 7) Build the feature matrix
x = torch.stack([node2vec[n] for n in nodes])


# 8) Helper to map DataFrame -> ids
def map_ids(df):
    return (
        df
        .assign(
            h_id = df['h'].map(node2id),
            t_id = df['t'].map(node2id)
        )
        .dropna(subset=['h_id','t_id'])
    )


# 9) Build the training graph adjacency
train_e = map_ids(df_train_pos)
edge_index = torch.cat([
    torch.tensor(train_e[['h_id','t_id']].values.T, dtype=torch.long),
    torch.tensor(train_e[['t_id','h_id']].values.T, dtype=torch.long)
], dim=1)
graph = Data(x=x, edge_index=edge_index).to(device)

# 10) Prepare test edges & labels
test_e = map_ids(df_test)
edge_label_index = torch.tensor(test_e[['h_id','t_id']].values.T, dtype=torch.long).to(device)
labels = torch.tensor(test_e['label'].values, dtype=torch.float).to(device)

# 11) Zero-shot eval
with torch.no_grad():
    logits = gnn(graph.x, graph.edge_index, edge_label_index)
    probs = torch.sigmoid(logits).cpu()
    preds = (probs > 0.5).float().cpu()

labels = labels.cpu()
print(classification_report(labels, preds, digits=4))
print("AUC:", roc_auc_score(labels, probs))
