import copy
import joblib
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from community import community_louvain
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample


def extract_edge_features(G, edges=None, label_graph=None):
    """
    Extract structural features and community-disagreement labels for graph edges.

    Args:
        G: NetworkX graph used to compute structural edge features such as
            endpoint degree, common-neighbor count, Jaccard coefficient,
            Adamic-Adar score, and triangle participation.
        edges: Optional iterable of ``(u, v)`` node pairs to featurize. When
            omitted, all edges currently present in ``G`` are used.
        label_graph: Optional NetworkX graph containing a ``community`` node
            attribute for each endpoint. When omitted, labels are read from
            ``G`` itself.

    Returns:
        A tuple ``(features, labels)`` where ``features`` is a NumPy array with
        one row per requested edge and seven structural feature columns, and
        ``labels`` is a NumPy array whose entries are ``1`` when the two
        endpoints belong to different communities and ``0`` otherwise.
    """
    features, labels = [], []
    label_graph = G if label_graph is None else label_graph
    edge_iter = G.edges() if edges is None else edges

    for u, v in edge_iter:
        deg_u, deg_v = G.degree[u], G.degree[v]
        deg_diff = abs(deg_u - deg_v)
        common_nbrs = len(list(nx.common_neighbors(G, u, v)))
        jaccard = list(nx.jaccard_coefficient(G, [(u, v)]))[0][2]
        adamic_adar = list(nx.adamic_adar_index(G, [(u, v)]))[0][2]
        triangle = int(len(set(G[u]) & set(G[v])) > 0)
        label = int(label_graph.nodes[u]["community"] != label_graph.nodes[v]["community"])
        features.append([deg_u, deg_v, deg_diff, common_nbrs, jaccard, adamic_adar, triangle])
        labels.append(label)
    return np.array(features), np.array(labels)


# Model 3: MLP
class MLP(nn.Module):
    """
    Feed-forward neural classifier for edge community-disagreement prediction.

    Args:
        input_dim: Number of input feature columns for each edge example.

    Returns:
        An ``nn.Module`` that maps an edge-feature tensor to a probability that
        the edge connects nodes from different Louvain communities.
    """

    def __init__(self, input_dim):
        """
        Initialise the multilayer perceptron architecture.

        Args:
            input_dim: Number of scalar input features provided for each edge.

        Returns:
            None.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Compute community-disagreement probabilities for edge features.

        Args:
            x: PyTorch tensor of shape ``(num_edges, input_dim)`` containing
                edge-level feature vectors.

        Returns:
            PyTorch tensor of shape ``(num_edges, 1)`` containing predicted
            probabilities for the positive class.
        """
        return self.model(x)


if __name__ == "__main__":
    # Load graph
    G = nx.read_edgelist("facebook_combined.txt", nodetype=int)

    # Detect communities using Louvain
    partition = community_louvain.best_partition(G)
    for node, comm in partition.items():
        G.nodes[node]["community"] = comm

    # Balance edge examples before splitting, then compute structural features
    # from the training graph so held-out edges do not contribute to their own
    # structural feature values.
    all_edges = list(G.edges())
    edge_labels = np.array([
        int(G.nodes[u]["community"] != G.nodes[v]["community"])
        for u, v in all_edges
    ])
    edge_df = pd.DataFrame(all_edges, columns=["u", "v"])
    edge_df["label"] = edge_labels

    df_majority = edge_df[edge_df["label"] == 0]
    df_minority = edge_df[edge_df["label"] == 1]
    df_majority_down = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
    df_bal = pd.concat([df_majority_down, df_minority])

    balanced_edges = list(zip(df_bal["u"].values, df_bal["v"].values))
    balanced_labels = df_bal["label"].values

    train_edges, test_edges, y_train, y_test = train_test_split(
        balanced_edges,
        balanced_labels,
        test_size=0.3,
        random_state=42,
        stratify=balanced_labels
    )

    G_train_features = nx.Graph()
    G_train_features.add_nodes_from(G.nodes())
    G_train_features.add_edges_from(train_edges)

    X_train, y_train = extract_edge_features(G_train_features, train_edges, label_graph=G)
    X_test, y_test = extract_edge_features(G_train_features, test_edges, label_graph=G)

    # Model 1: Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    joblib.dump(lr, "logistic_model.pth")

    # Model 2: Random Forest
    best_f1 = -1.0
    best_model = None

    # Split training set into sub-training and validation
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    for _ in range(10):  # or use CV splits
        model = RandomForestClassifier()
        model.fit(X_subtrain, y_subtrain)
        preds = model.predict(X_val)
        score = f1_score(y_val, preds)

        if score > best_f1:
            best_f1 = score
            best_model = model

    joblib.dump(best_model, "best_random_forest.pth")

    # Predict using the best model on the test set
    rf_preds = best_model.predict(X_test)

    # Prepare tensors
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    X_subtrain_tensor = torch.tensor(X_subtrain, dtype=torch.float32)
    y_subtrain_tensor = torch.tensor(y_subtrain.reshape(-1, 1), dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)

    model = MLP(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(X_subtrain_tensor)
        loss = criterion(output, y_subtrain_tensor)
        loss.backward()
        optimizer.step()

        # Eval on val
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor).numpy().flatten()
            val_preds_bin = (val_preds > 0.5).astype(int)
            score = f1_score(y_val, val_preds_bin)

            if score > best_f1:
                best_f1 = score
                best_state = copy.deepcopy(model.state_dict())

    # Save best state
    torch.save(best_state, "mlp_model.pth")

    # Reload and evaluate
    model = MLP(X_train.shape[1])
    model.load_state_dict(torch.load("mlp_model.pth"))
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        y_pred_mlp = model(X_test_tensor).numpy().flatten()
        y_pred_mlp_bin = (y_pred_mlp > 0.5).astype(int)

    # Evaluation
    print("Logistic Regression:\n", classification_report(y_test, lr_preds))
    print("Random Forest:\n", classification_report(y_test, rf_preds))
    print("MLP Report:\n", classification_report(y_test, y_pred_mlp_bin))
