import json
import heapq
import torch
from transformers import BertTokenizer, BertModel
from model import EdgeClassifierGNN

# ─── CONFIGURATION ─────────────────────────────────────────────────────────
MODEL_DIR       = "/home/a/amm106/transformers_cache/bert-base-uncased"
GNN_CHECKPOINT  = "bert_sage_best_model.pth"
ATOMIC10X_PATH  = "/scratch/geoai/amm106/ATOMIC10X.jsonl"
OUTPUT_CSV      = "/scratch/geoai/amm106/atomic10x_top1pct_noisy.csv"
BERT_CHUNK_SIZE = 10000       # adjust to control peak RAM
TOP_PCT         = 0.01        # top 1%

# ─── OPTIONAL TEST MODE ────────────────────────────────────────────────────
TEST_MODE   = False
MAX_RECORDS = 500             # cap for initial test run
# ─────────────────────────────────────────────────────────────────────────────

# Load models
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
bert      = BertModel.from_pretrained(MODEL_DIR, local_files_only=True).eval()
gnn       = EdgeClassifierGNN(in_channels=768, hidden_channels=128).eval()
gnn.load_state_dict(torch.load(GNN_CHECKPOINT))

@torch.no_grad()
def batch_embed(texts):
    """Embed a list of strings via mean-pooled BERT."""
    inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                       padding=True, max_length=32)
    states = bert(**inputs).last_hidden_state
    return states.mean(dim=1).cpu()

@torch.no_grad()
def score_chunk(triples):
    """
    Score a list of (h, r, t) triples:
    - Embed unique nodes
    - Build a PyG fragment
    - Run the GNN
    - Return list of (score, h, r, t)
    """
    # Embed only unique nodes in this chunk
    nodes = list({h for h,_,_ in triples} | {t for _,_,t in triples})
    embs  = batch_embed(nodes)
    idx   = {n:i for i,n in enumerate(nodes)}

    # Build edge indices
    h_ids = [idx[h] for h,_,_ in triples]
    t_ids = [idx[t] for _,_,t in triples]
    edge_index       = torch.tensor([h_ids, t_ids], dtype=torch.long)
    edge_label_index = edge_index.clone()

    # Score with GNN and detach before numpy
    raw_scores = torch.sigmoid(
        gnn(embs, edge_index, edge_label_index)
    )
    scores = raw_scores.cpu().detach().numpy()
    return list(zip(scores, [h for h,_,_ in triples],
                         [r for _,r,_ in triples],
                         [t for _,_,t in triples]))


def main():
    # Count total (with optional cap)
    total_raw = sum(1 for _ in open(ATOMIC10X_PATH))
    total     = min(total_raw, MAX_RECORDS) if TEST_MODE else total_raw
    k         = max(1, int(total * TOP_PCT))

    heap   = []  # will store (score, h, r, t)
    buffer = []

    for i, line in enumerate(open(ATOMIC10X_PATH), start=1):
        if TEST_MODE and i > MAX_RECORDS:
            break

        obj = json.loads(line)
        buffer.append((obj["head"], obj["relation"], obj["tail"]))

        if len(buffer) >= BERT_CHUNK_SIZE:
            for score, h, r, t in score_chunk(buffer):
                if len(heap) < k:
                    heapq.heappush(heap, (score, h, r, t))
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, (score, h, r, t))
            buffer.clear()

    # Process any leftover in buffer
    if buffer:
        for score, h, r, t in score_chunk(buffer):
            if len(heap) < k:
                heapq.heappush(heap, (score, h, r, t))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, h, r, t))

    # Extract and save top 1%
    top1pct = sorted(heap, key=lambda x: x[0], reverse=True)

    import csv
    with open(OUTPUT_CSV, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["noise_score","h","r","t"])
        for score, h, r, t in top1pct:
            writer.writerow([f"{score:.6f}", h, r, t])

    print(f"Processed {total:,} triples (capped)" if TEST_MODE else f"Processed {total:,} triples")
    print(f"Top 1% = {len(top1pct):,} triples -> {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
