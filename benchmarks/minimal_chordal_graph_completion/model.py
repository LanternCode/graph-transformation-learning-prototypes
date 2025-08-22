import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from torch import nn, optim
from torch_geometric.data import Dataset, DataLoader
import json


# Define dataset for PyTorch
class EdgeFeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class CNN1D(nn.Module):
    def __init__(self, input_dim):
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
        x = x.unsqueeze(1)
        x = self.conv(x)
        return self.fc(x)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 32)
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return self.sigmoid(self.fc(x))


class AutoencoderClassifier(nn.Module):
    def __init__(self, input_dim):
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
        x = self.encoder(x)
        x = self.decoder(x)
        return self.classifier(x)


def train_model(model, name):
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

    # Split features and labels
    X = df_balanced.drop(columns=["label"]).values
    y = df_balanced["label"].values

    # after building df_balanced
    feature_order = ["common_neighbors", "jaccard", "adamic_adar", "deg_u", "deg_v", "shortest_path", "cc_u", "cc_v"]

    scaler = StandardScaler()
    X = scaler.fit_transform(df_balanced[feature_order].values)
    joblib.dump(scaler, "scaler_fillin.joblib")
    with open("feature_order.json", "w") as f:
        json.dump(feature_order, f)

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(df_balanced[feature_order].values)

    joblib.dump(scaler, "scaler_fillin.joblib")
    with open("feature_order.json", "w") as f:
        json.dump(feature_order, f)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    train_dataset = EdgeFeatureDataset(X_train, y_train)
    test_dataset = EdgeFeatureDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Reuse model classes and train them
    input_dim = X.shape[1]
    models = {
        "MLP": MLP(input_dim),
        "CNN1D": CNN1D(input_dim),
        "Transformer": TransformerClassifier(input_dim),
        "Autoencoder": AutoencoderClassifier(input_dim)
    }

    # training
    verbose_results = []
    for name, model in models.items():
        result = train_model(model, name)
        verbose_results.append(result)

    # print(f"Model Training: \n{verbose_results}")

    # Train the Random Forest model
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Save the RF model
    model_path = "random_forest_fillin.pkl"
    joblib.dump(clf, model_path)

    print(f"Random Forest Accuracy: {accuracy:2f}")
    print(f"Random Forest Full Report: \n{report}")
