import pandas as pd
import torch
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch_geometric.data import Dataset, DataLoader


# Define dataset for PyTorch
class EdgeFeatureDataset(Dataset):
    """
    Dataset wrapper for tabular candidate-edge fill-in features.

    Args:
        X: Two-dimensional array-like feature matrix for candidate non-edges.
        y: One-dimensional array-like binary labels indicating fill-in membership.

    Returns:
        A PyTorch Geometric-compatible dataset yielding feature tensors and labels.
    """
    def __init__(self, X, y):
        """
        Store feature and label tensors for candidate-edge classification.

        Args:
            X: Two-dimensional array-like feature matrix.
            y: One-dimensional array-like binary label vector.

        Returns:
            None.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        """
        Return the number of candidate-edge examples in the dataset.

        Args:
            None.

        Returns:
            Number of rows stored in the feature tensor.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Return one feature-label example by index.

        Args:
            idx: Integer row index to retrieve.

        Returns:
            A tuple (features, label) for the requested candidate edge.
        """
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    """
    Feed-forward candidate-edge classifier for tabular fill-in features.

    Args:
        input_dim: Number of input feature columns.

    Returns:
        A neural classifier that outputs one probability per candidate edge.
    """
    def __init__(self, input_dim):
        """
        Initialize the MLP architecture.

        Args:
            input_dim: Number of input feature columns.

        Returns:
            None.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Predict fill-in probabilities from tabular features.

        Args:
            x: Tensor of shape (batch_size, input_dim).

        Returns:
            Tensor of shape (batch_size, 1) containing probabilities.
        """
        return self.model(x)


class CNN1D(nn.Module):
    """
    One-dimensional convolutional classifier over ordered tabular features.

    Args:
        input_dim: Number of input feature columns.

    Returns:
        A neural classifier that outputs one probability per candidate edge.
    """
    def __init__(self, input_dim):
        """
        Initialize the 1D CNN classifier.

        Args:
            input_dim: Number of input feature columns.

        Returns:
            None.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(8 * input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Predict fill-in probabilities from tabular features.

        Args:
            x: Tensor of shape (batch_size, input_dim).

        Returns:
            Tensor of shape (batch_size, 1) containing probabilities.
        """
        x = x.unsqueeze(1)
        x = self.conv(x)
        return self.fc(x)


class TransformerClassifier(nn.Module):
    """
    Transformer-style classifier for fixed-order edge feature vectors.

    Args:
        input_dim: Number of input feature columns.

    Returns:
        A neural classifier that outputs one probability per candidate edge.
    """
    def __init__(self, input_dim):
        """
        Initialize the feature embedding, transformer encoder, and output head.

        Args:
            input_dim: Number of input feature columns.

        Returns:
            None.
        """
        super().__init__()
        self.embedding = nn.Linear(input_dim, 32)
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Predict fill-in probabilities from tabular features.

        Args:
            x: Tensor of shape (batch_size, input_dim).

        Returns:
            Tensor of shape (batch_size, 1) containing probabilities.
        """
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return self.sigmoid(self.fc(x))


class AutoencoderClassifier(nn.Module):
    """
    Autoencoder-style classifier for compressed edge-feature representations.

    Args:
        input_dim: Number of input feature columns.

    Returns:
        A neural classifier that outputs one probability per candidate edge.
    """
    def __init__(self, input_dim):
        """
        Initialize the encoder, decoder, and classifier layers.

        Args:
            input_dim: Number of input feature columns.

        Returns:
            None.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, input_dim),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Predict fill-in probabilities from tabular features.

        Args:
            x: Tensor of shape (batch_size, input_dim).

        Returns:
            Tensor of shape (batch_size, 1) containing probabilities.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return self.classifier(x)


def train_model(model, name, train_loader, test_loader):
    """
    Train and evaluate one neural tabular fill-in classifier.

    Args:
        model: PyTorch module to optimize.
        name: Model name used in logs and checkpoint filename.
        train_loader: DataLoader over graph-held-out training rows.
        test_loader: DataLoader over graph-held-out test rows.

    Returns:
        A tuple (name, accuracy, report, path) summarizing the fitted model and saved checkpoint.
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\nTraining model: {name}")
    for epoch in range(20):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            preds = model(X_batch).squeeze()
            loss = criterion(preds, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch).squeeze()
            all_preds.extend((preds > 0.5).int().tolist())
            all_labels.extend(y_batch.int().tolist())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, zero_division=0)
    path = f"{name}_model_balanced.pth"
    torch.save(model.state_dict(), path)

    print(f"\nEvaluation for {name}")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)
    return name, acc, report, path


if __name__ == "__main__":
    # Load the underbalanced dataset
    balanced_path = "fillin_features_balanced.csv"
    df_balanced = pd.read_csv(balanced_path)

    feature_order = ["common_neighbors", "jaccard", "adamic_adar", "deg_u", "deg_v", "shortest_path", "cc_u", "cc_v"]
    if "graph_id" not in df_balanced.columns:
        raise ValueError(
            "fillin_features_balanced.csv must contain a graph_id column. "
            "Regenerate it with generate_minimal_fillins.py before training."
        )

    # Split by source graph so candidate edges from the same graph do not appear in both train and test.
    graph_ids = df_balanced["graph_id"].unique()
    train_graph_ids, test_graph_ids = train_test_split(graph_ids, test_size=0.15, random_state=42)
    train_df = df_balanced[df_balanced["graph_id"].isin(train_graph_ids)]
    test_df = df_balanced[df_balanced["graph_id"].isin(test_graph_ids)]

    X_train_raw = train_df[feature_order].values
    y_train = train_df["label"].values
    X_test_raw = test_df[feature_order].values
    y_test = test_df["label"].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    joblib.dump(scaler, "scaler_fillin.joblib")
    with open("feature_order.json", "w") as f:
        json.dump(feature_order, f)

    train_dataset = EdgeFeatureDataset(X_train, y_train)
    test_dataset = EdgeFeatureDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Reuse model classes and train them
    input_dim = X_train.shape[1]
    models = {
        "MLP": MLP(input_dim),
        "CNN1D": CNN1D(input_dim),
        "Transformer": TransformerClassifier(input_dim),
        "Autoencoder": AutoencoderClassifier(input_dim)
    }

    verbose_results = []
    for name, model in models.items():
        result = train_model(model, name, train_loader, test_loader)
        verbose_results.append(result)

    # Train the Random Forest model
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    # Save the RF model
    model_path = "random_forest_fillin.pkl"
    joblib.dump(clf, model_path)

    print(f"Random Forest Accuracy: {accuracy:2f}")
    print(f"Random Forest Full Report: \n{report}")
