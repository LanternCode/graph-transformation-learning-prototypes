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
    """
    Generate graph samples for chordal-completion fill-edge prediction.

    Args:
        num_graphs: Number of non-chordal graph samples to generate.
        num_nodes: Fixed matrix size used for padded adjacency, target, and mask
            matrices.

    Returns:
        A list of tuples containing the padded adjacency matrix, the masked
        chordal-fill target matrix, and the candidate-edge mask for each sample.
    """
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
    """
    Dataset wrapper for padded chordal-completion training examples.

    Args:
        graphs: Sequence of tuples containing adjacency matrices, target fill
            matrices, and candidate masks.

    Returns:
        A PyTorch dataset whose items are tensor triples of adjacency, target,
        and mask matrices.
    """
    def __init__(self, graphs):
        """
        Store chordal-completion graph examples for indexed access.

        Args:
            graphs: Sequence of tuples containing adjacency matrices, target
                matrices, and candidate masks.

        Returns:
            None.
        """
        self.graphs = graphs

    def __len__(self):
        """
        Return the number of graph examples in the dataset.

        Args:
            None.

        Returns:
            The number of stored graph examples.
        """
        return len(self.graphs)

    def __getitem__(self, idx):
        """
        Retrieve one graph example as tensors.

        Args:
            idx: Integer index of the requested graph example.

        Returns:
            A tuple containing adjacency, target, and mask tensors.
        """
        A, target, mask = self.graphs[idx]
        return torch.tensor(A), torch.tensor(target), torch.tensor(mask)


class MLP(nn.Module):
    """
    Fully connected baseline for fixed-size chordal-fill prediction.

    Args:
        num_nodes: Number of nodes defining the square adjacency matrix size.

    Returns:
        A neural module that maps an adjacency batch to fill-edge probabilities.
    """
    def __init__(self, num_nodes):
        """
        Initialise the MLP layers.

        Args:
            num_nodes: Number of nodes defining the flattened input and output
                dimensionality.

        Returns:
            None.
        """
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_nodes*num_nodes, 512),
            nn.ReLU(),
            nn.Linear(512, num_nodes*num_nodes),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Predict fill-edge probabilities from an adjacency tensor.

        Args:
            x: Tensor of shape ``(batch_size, num_nodes, num_nodes)``.

        Returns:
            Tensor of predicted fill-edge probabilities with the same matrix
            shape as the input batch.
        """
        return self.model(x).view(x.size(0), x.size(1), x.size(2))


class CNN(nn.Module):
    """
    Convolutional baseline for fixed-size chordal-fill prediction.

    Args:
        num_nodes: Number of nodes defining the input adjacency matrix size.

    Returns:
        A neural module that maps adjacency matrices to fill-edge probabilities.
    """
    def __init__(self, num_nodes):
        """
        Initialise the convolutional layers.

        Args:
            num_nodes: Number of nodes in the input graph matrices. The value is
                accepted for interface consistency with the other models.

        Returns:
            None.
        """
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Predict fill-edge probabilities with local convolutional filters.

        Args:
            x: Tensor of shape ``(batch_size, num_nodes, num_nodes)``.

        Returns:
            Tensor of predicted fill-edge probabilities with shape
            ``(batch_size, num_nodes, num_nodes)``.
        """
        x = x.unsqueeze(1)
        return self.model(x).squeeze(1)


class Transformer(nn.Module):
    """
    Transformer encoder baseline for fixed-size chordal-fill prediction.

    Args:
        num_nodes: Number of nodes defining the flattened input and output size.

    Returns:
        A neural module that predicts chordal-fill probabilities from an
        adjacency matrix.
    """
    def __init__(self, num_nodes):
        """
        Initialise the embedding, transformer encoder, and decoder layers.

        Args:
            num_nodes: Number of nodes defining the flattened adjacency vector
                length.

        Returns:
            None.
        """
        super(Transformer, self).__init__()
        d_model = 64
        self.embedding = nn.Linear(num_nodes*num_nodes, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.decoder = nn.Linear(d_model, num_nodes*num_nodes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Predict fill-edge probabilities using a transformer over graph-level embeddings.

        Args:
            x: Tensor of shape ``(batch_size, num_nodes, num_nodes)``.

        Returns:
            Tensor of predicted fill-edge probabilities reshaped as square
            matrices for each graph in the batch.
        """
        x = x.view(x.size(0), -1)
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        x = self.decoder(x)
        return self.sigmoid(x).view(x.size(0), int(np.sqrt(x.size(1))), -1)


class Autoencoder(nn.Module):
    """
    Autoencoder baseline for fixed-size chordal-fill prediction.

    Args:
        num_nodes: Number of nodes defining the flattened input and output size.

    Returns:
        A neural module that reconstructs a fill-edge probability matrix from an
        adjacency matrix.
    """
    def __init__(self, num_nodes):
        """
        Initialise the encoder and decoder layers.

        Args:
            num_nodes: Number of nodes defining the flattened adjacency vector
                length.

        Returns:
            None.
        """
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
        """
        Encode and decode an adjacency matrix into fill-edge probabilities.

        Args:
            x: Tensor of shape ``(batch_size, num_nodes, num_nodes)``.

        Returns:
            Tensor of predicted fill-edge probabilities reshaped as square
            matrices for each graph in the batch.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(x.size(0), int(np.sqrt(x.size(1))), -1)


def evaluate_model(model, data_loader, criterion):
    """
    Evaluate a chordal-fill model on a dataset loader.

    Args:
        model: Neural model that predicts fill-edge probabilities from adjacency
            matrices.
        data_loader: Loader yielding adjacency, target, and mask tensors.
        criterion: Loss function used to compute masked prediction loss.

    Returns:
        A tuple containing average loss, precision, recall, F1 score, number of
        predicted positive entries, and number of predicted negative entries.
    """
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
    """
    Train a chordal-fill model and evaluate it on the held-out loader.

    Args:
        model: Neural model to train.
        train_loader: DataLoader yielding training adjacency, target, and mask
            tensors.
        test_loader: DataLoader yielding evaluation adjacency, target, and mask
            tensors.
        model_name: Name used in progress messages and checkpoint filename.

    Returns:
        None. The function prints metrics and saves the trained model state.
    """
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
