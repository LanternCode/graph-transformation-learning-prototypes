import numpy as np
import random
from tqdm import tqdm
from typing import List, Optional, Callable, Tuple, Sequence, Union, Dict
from torch.utils.data import DataLoader
import networkx as nx
from benchmark.utils.graph_generators import generate_graph
from benchmark.task_base import TaskGenerator


class BenchmarkManager:
    """
    BenchmarkManager: Task-agnostic graph benchmark pipeline.

    Generates graphs, applies a task's label generation method, and supports:
    - Feature extraction
    - Train/val/test splitting
    - Optional prepackaging into PyTorch DataLoaders
    """

    def __init__(
        self,
        task: TaskGenerator,
        num_graphs: int = 1000,
        min_nodes: int = 6,
        max_nodes: int = 140,
        graph_types: Optional[List[str]] = None,
    ):
        self.task = task
        self.num_graphs = num_graphs
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.graph_types = graph_types or [
            'erdos_renyi', 'barabasi_albert', 'watts_strogatz',
            'random_regular', 'balanced_tree'
        ]
        self.graphs: Tuple[np.ndarray, ...] = ()
        self.labels: Tuple[np.ndarray, ...] = ()

    def generate_graph(self, graph_type: str, num_nodes: int) -> nx.Graph:
        """
        Wrapper around utils.graph_generators.generate_graph for extension.
        """
        return generate_graph(graph_type, num_nodes)

    def generate_dataset(self) -> Tuple[np.ndarray, ...]:
        """
        Generate num_graphs connected graphs and compute labels via task.
        """
        graphs, labels = [], []
        print("Dataset generation begins.")
        for _ in tqdm(range(self.num_graphs), desc="Generating graphs"):
            while True:
                gt = random.choice(self.graph_types)
                nn = random.randint(self.min_nodes, self.max_nodes)
                G = self.generate_graph(gt, nn)
                if nx.is_connected(G.to_undirected()):
                    break
            A = nx.to_numpy_array(G, dtype=np.float32)
            L = self.task.generate_labels(A)
            A.setflags(write=False)
            L.setflags(write=False)
            graphs.append(A)
            labels.append(L)
        self.graphs = tuple(graphs)
        self.labels = tuple(labels)
        total_nodes = sum(g.shape[0] for g in graphs)
        total_edges = sum(np.count_nonzero(np.triu(g)) for g in graphs)
        print(f"\nGenerated {len(graphs)} graphs.")
        print(f"Total nodes: {total_nodes}.")
        print(f"Total edges: {total_edges}.")
        print("\nDataset generation concluded.")
        return self.graphs

    def extract_features(
        self,
        adj: np.ndarray,
        feature_set: Union[bool, List[str]] = False
    ) -> Dict[str, np.ndarray]:
        """
        Compute selected features for a single adjacency.
        """
        if not feature_set:
            return {}
        feature_list = ['transpose','powers','degree','triangles','clustering_coeff','top_eigs']
        if feature_set is not True:
            feature_list = feature_set
        feats: Dict[str, np.ndarray] = {}
        if 'transpose' in feature_list:
            feats['transpose'] = adj.T
        if 'powers' in feature_list:
            for k in (2,3,4,5): feats[f'power_{k}'] = np.linalg.matrix_power(adj, k)
        if 'degree' in feature_list:
            feats['degree'] = adj.sum(axis=1)
        if 'triangles' in feature_list or 'clustering_coeff' in feature_list:
            A3 = np.linalg.matrix_power(adj,3)
            tri = np.diag(A3)/2
            if 'triangles' in feature_list: feats['triangles'] = tri
            if 'clustering_coeff' in feature_list:
                deg = feats.get('degree', adj.sum(axis=1))
                possible = deg*(deg-1)/2
                with np.errstate(divide='ignore', invalid='ignore'):
                    feats['clustering_coeff'] = np.where(possible>0, tri/possible, 0.0)
        if 'top_eigs' in feature_list:
            eigs = np.linalg.eigvals(adj)
            feats['top_eigs'] = np.sort(np.real(eigs))[-3:]
        return feats

    def provide_splits(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        shuffle: bool = True,
        compute_features: Union[bool, List[str]] = False,
        prepackage_dataset: bool = False,
        batch_size: int = 64,
        collate_fn: Callable = None,
    ) -> Union[
        Tuple[Tuple[np.ndarray,...], Tuple[np.ndarray,...]],
        Tuple[Tuple[np.ndarray,...], Tuple[np.ndarray,...], List[Dict[str, np.ndarray]]],
        Tuple[DataLoader, DataLoader, DataLoader]
    ]:
        """
        Split dataset into train/val/test with options:
        - compute_features: False|True|list
        - prepackage_dataset: if True, return DataLoaders
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Splits must sum to 1.0"
        if not self.graphs:
            raise RuntimeError("No dataset generated. Call generate_dataset() first.")

        gs, ls = self.graphs, self.labels
        indices = list(range(len(gs)))
        if shuffle:
            random.shuffle(indices)
        t_end = int(train_ratio * len(gs))
        v_end = t_end + int(val_ratio * len(gs))
        train_idx, val_idx, test_idx = (
            indices[:t_end],
            indices[t_end:v_end],
            indices[v_end:],
        )

        def build(idx_list):
            gsub = tuple(gs[i] for i in idx_list)
            lsub = tuple(ls[i] for i in idx_list)
            if compute_features:
                fsub = [self.extract_features(a, compute_features) for a in gsub]
                return gsub, lsub, fsub
            return gsub, lsub

        train_split = build(train_idx)
        val_split   = build(val_idx)
        test_split  = build(test_idx)

        if not prepackage_dataset:
            return train_split, val_split, test_split

        train_data = list(zip(*train_split))
        val_data   = list(zip(*val_split))
        test_data  = list(zip(*test_split))

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        return train_loader, val_loader, test_loader

    def provide_benchmark(self, num_graphs: int = None) -> Tuple[np.ndarray, ...]:
        """
        Regenerate dataset if num_graphs provided, return adjacency matrices.

        Args:
            num_graphs: Optional number of graphs to regenerate. If None, returns existing dataset.

        Returns:
            Tuple of adjacency matrices.
        """
        if num_graphs is not None:
            self.num_graphs = num_graphs
        return self.generate_dataset()

    def evaluate(self, predictions: Sequence[np.ndarray]) -> float:
        """
        Delegate evaluation to task's evaluation method.
        """
        return self.task.evaluate(predictions, self.labels)
