import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import networkx as nx
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support

# Parameters
NUM_GRAPHS = 100
NUM_NODES = 150
BATCH_SIZE = 32
EPOCHS = 50
SAVE_PATH = ""


def generate_minimal_fill_dataset(num_graphs, num_nodes):
    generators = [
        lambda: nx.cycle_graph(num_nodes),
        lambda: nx.convert_node_labels_to_integers(
            nx.grid_2d_graph(int(np.floor(np.sqrt(num_nodes))), int(np.floor(np.sqrt(num_nodes))))
        ),
        lambda: nx.random_unlabeled_tree(num_nodes),
        lambda: nx.erdos_renyi_graph(num_nodes, 0.3),
        lambda: nx.barabasi_albert_graph(num_nodes, max(1, num_nodes // 10))
    ]

    data = []
    print("Generating training graphs:")
    with tqdm(total=num_graphs) as pbar:
        while len(data) < num_graphs:
            G = random.choice(generators)()
            G = nx.convert_node_labels_to_integers(G)

            if nx.is_chordal(G):
                continue

            chordal_G, _ = nx.complete_to_chordal_graph(G)
            A_raw = nx.to_numpy_array(G)
            A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            A[:A_raw.shape[0], :A_raw.shape[1]] = A_raw

            target = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            fill_edges = set(chordal_G.edges()) - set(G.edges())
            for u, v in fill_edges:
                if u < num_nodes and v < num_nodes:
                    target[u, v] = 1
                    target[v, u] = 1

            mask = np.zeros_like(target)
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if A[i, j] == 0:
                        mask[i, j] = mask[j, i] = 1

            data.append((A, target * mask, mask.astype(np.float32)))
            pbar.update(1)

    return data


class MinimalChordalDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        A, target, mask = self.graphs[idx]
        return torch.tensor(A), torch.tensor(target), torch.tensor(mask)


class MLP(nn.Module):
    def __init__(self, num_nodes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_nodes*num_nodes, 512),
            nn.ReLU(),
            nn.Linear(512, num_nodes*num_nodes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(x.size(0), x.size(1), x.size(2))


class CNN(nn.Module):
    def __init__(self, num_nodes):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.model(x).squeeze(1)


class Transformer(nn.Module):
    def __init__(self, num_nodes):
        super(Transformer, self).__init__()
        d_model = 64
        self.embedding = nn.Linear(num_nodes*num_nodes, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.decoder = nn.Linear(d_model, num_nodes*num_nodes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        x = self.decoder(x)
        return self.sigmoid(x).view(x.size(0), int(np.sqrt(x.size(1))), -1)


class Autoencoder(nn.Module):
    def __init__(self, num_nodes):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_nodes*num_nodes, 128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, num_nodes*num_nodes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(x.size(0), int(np.sqrt(x.size(1))), -1)


def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for A, target, mask in data_loader:
            output = model(A)
            loss = criterion(output * mask, target)
            total_loss += loss.item()
            preds = ((output * mask) > 0.5).int().view(-1).numpy()
            labels = target.int().view(-1).numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    avg_loss = total_loss / len(data_loader)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    count_1 = sum(all_preds)
    count_0 = len(all_preds) - count_1
    return avg_loss, precision, recall, f1, count_1, count_0


def train_and_test(model, train_loader, test_loader, model_name):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        for A, target, mask in train_loader:
            output = model(A)
            loss = criterion(output * mask, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss, precision, recall, f1, _, _ = evaluate_model(model, train_loader, criterion)
        print(f"{model_name} | Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    test_loss, precision, recall, f1, n1, n0 = evaluate_model(model, test_loader, criterion)
    print(f"{model_name} | Final Test Loss: {test_loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Predictions: 1s = {n1}, 0s = {n0}")
    torch.save(model.state_dict(), os.path.join(SAVE_PATH, f"{model_name}_best.pt"))


# Main
if __name__ == '__main__':
    all_data = generate_minimal_fill_dataset(NUM_GRAPHS, NUM_NODES)
    split = int(0.8 * NUM_GRAPHS)
    train_data = MinimalChordalDataset(all_data[:split])
    test_data = MinimalChordalDataset(all_data[split:])
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    models = {
        "MLP": MLP(NUM_NODES),
        "CNN": CNN(NUM_NODES),
        "Transformer": Transformer(NUM_NODES),
        "Autoencoder": Autoencoder(NUM_NODES)
    }

    for name, model in models.items():
        train_and_test(model, train_loader, test_loader, name)
