import gzip
import os
import shutil
import urllib
import networkx as nx
import pandas as pd
from community import community_louvain
from sklearn.metrics import classification_report


class EdgeDisagreementBenchmark:
    def __init__(self):
        self.graph_path = "facebook_combined.txt"
        self.feature_names = ["deg_u", "deg_v", "deg_diff", "common", "jaccard", "adamic", "triangle"]
        self._ensure_dataset()
        self._load_graph()

    @staticmethod
    def _ensure_dataset():
        url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
        compressed_file = "facebook_combined.txt.gz"
        extracted_file = "facebook_combined.txt"

        if os.path.exists(extracted_file):
            print("Dataset already present:", extracted_file)
            return

        if not os.path.exists(compressed_file):
            print("⬇Downloading dataset...")
            urllib.request.urlretrieve(url, compressed_file)
        else:
            print("Compressed file already exists.")

        print("Extracting dataset...")
        with gzip.open(compressed_file, 'rb') as f_in:
            with open(extracted_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("Extraction complete:", extracted_file)

    def _load_graph(self):
        print("Loading graph and detecting communities...")
        self.graph = nx.read_edgelist(self.graph_path, nodetype=int)
        partition = community_louvain.best_partition(self.graph)
        for node, comm in partition.items():
            self.graph.nodes[node]["community"] = comm

    def _extract_balanced_edge_data(self):
        features, labels = [], []
        for u, v in self.graph.edges():
            deg_u, deg_v = self.graph.degree[u], self.graph.degree[v]
            deg_diff = abs(deg_u - deg_v)
            common = len(list(nx.common_neighbors(self.graph, u, v)))
            jaccard = list(nx.jaccard_coefficient(self.graph, [(u, v)]))[0][2]
            adamic = list(nx.adamic_adar_index(self.graph, [(u, v)]))[0][2]
            triangle = int(len(set(self.graph[u]) & set(self.graph[v])) > 0)
            label = int(self.graph.nodes[u]["community"] != self.graph.nodes[v]["community"])
            features.append([deg_u, deg_v, deg_diff, common, jaccard, adamic, triangle])
            labels.append(label)

        # Balance the dataset
        df = pd.DataFrame(features, columns=self.feature_names)
        df["label"] = labels
        df_majority = df[df["label"] == 0]
        df_minority = df[df["label"] == 1]
        df_bal = pd.concat([
            df_majority.sample(n=len(df_minority), random_state=42),
            df_minority
        ])
        X = df_bal.drop("label", axis=1).values
        y = df_bal["label"].values
        return X, y

    def run(self, model_adapter_fn):
        X, y = self._extract_balanced_edge_data()
        y_pred = model_adapter_fn(X)
        name = model_adapter_fn.__name__.replace("_adapter", "").replace("_", " ").title()
        report = classification_report(y, y_pred, output_dict=True)
        self.pretty_print_report(name, report)

    @staticmethod
    def pretty_print_report(name, report_dict):
        print(f"\n=== {name} ===")
        print(f"{'Class':<10}{'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}")
        for label in ['0', '1']:
            row = report_dict[label]
            print(f"{label:<10}{row['precision']:10.3f} {row['recall']:10.3f} {row['f1-score']:10.3f} {row['support']:10.0f}")
        print(f"{'Accuracy':<10}{report_dict['accuracy']:>10.3f}")
        print(f"{'Macro avg':<10}{report_dict['macro avg']['precision']:10.3f} "
              f"{report_dict['macro avg']['recall']:10.3f} {report_dict['macro avg']['f1-score']:10.3f}")
        print(f"{'Weighted avg':<10}{report_dict['weighted avg']['precision']:10.3f} "
              f"{report_dict['weighted avg']['recall']:10.3f} {report_dict['weighted avg']['f1-score']:10.3f}")
