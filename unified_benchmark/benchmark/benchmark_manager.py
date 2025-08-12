import numpy as np
import random
from tqdm import tqdm
from typing import List, Optional, Callable, Tuple, Sequence, Union, Dict, Any
from torch.utils.data import DataLoader
import networkx as nx
from unified_benchmark.benchmark.utils.graph_generators import generate_graph
from unified_benchmark.benchmark.task_base import TaskGenerator


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
        graph_config: Optional[Dict[str, Any]] = None,
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
        self.graph_config = graph_config or {}

    def _effective_graph_config(self) -> Dict[str, Any]:
        return dict(self.graph_config)

    def generate_graph(self, graph_type: str, num_nodes: int) -> nx.Graph:
        """
        Wrapper around utils.graph_generators.generate_graph for extension.
        """
        return generate_graph(graph_type, num_nodes, config=self._effective_graph_config())

    def generate_dataset(self) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
        """
        Generate num_graphs connected graphs and compute labels via task.
        Returns:
            (graphs_tuple, labels_tuple)
        """
        graphs: List[np.ndarray] = []
        labels: List[np.ndarray] = []

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

        print(f"\nGenerated {len(graphs)} graphs.")
        total_nodes = sum(g.shape[0] for g in graphs)
        print(f"Total nodes: {total_nodes}.")
        total_edges = sum(np.count_nonzero(np.triu(g)) for g in graphs)
        print(f"Total edges: {total_edges}.")

        print("\nDataset generation concluded.")
        return tuple(graphs), tuple(labels)

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
        Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]],
        Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], List[Dict[str, np.ndarray]]],
        Tuple[DataLoader, DataLoader, DataLoader]
    ]:
        """
        Generate a new dataset and return train/val/test splits.

        This method wraps `generate_dataset()` to:
          1. Generate fresh `(graphs, labels)` for `self.num_graphs` using the
             current `graph_types`, `min_nodes`, `max_nodes`, and `task`.
          2. Split the dataset into train/val/test sets according to the given ratios.
          3. Optionally:
             - Shuffle the dataset before splitting.
             - Compute per-graph feature dictionaries.
             - Return PyTorch DataLoaders for direct training use.

        Args:
            train_ratio (float):
                Fraction of total graphs to include in the training split.
            val_ratio (float):
                Fraction of total graphs to include in the validation split.
            test_ratio (float):
                Fraction of total graphs to include in the test split.
                The three ratios must sum to 1.0 exactly (within 1e-6 tolerance).
            shuffle (bool, default=True):
                If True, randomly shuffle graphs before splitting.
            compute_features (bool or list of str, default=False):
                - If False: no features computed; only `(graphs, labels)` returned.
                - If True: compute all available features.
                - If list: compute only the specified feature names.
                When enabled, returns a third element in each split containing
                a list of feature dictionaries (one per graph).
            prepackage_dataset (bool, default=False):
                If True, wrap each split into a PyTorch DataLoader instead of
                returning raw tuples. Useful for training pipelines.
            batch_size (int, default=64):
                Batch size to use for DataLoaders (ignored if prepackage_dataset=False).
            collate_fn (Callable, optional):
                Custom collate function for DataLoaders.

        Returns:
            One of:
              * If `compute_features=False` and `prepackage_dataset=False`:
                    (train_split, val_split, test_split)
                    where each split is `(graphs_tuple, labels_tuple)`.
              * If `compute_features=True` or list, and `prepackage_dataset=False`:
                    (train_split, val_split, test_split)
                    where each split is `(graphs_tuple, labels_tuple, features_list)`.
              * If `prepackage_dataset=True`:
                    (train_loader, val_loader, test_loader)
                    where each loader yields batches of tuples corresponding to the chosen mode.

        Notes:
            - This method does not cache graphs or labels on `self`; each call generates fresh data.
            - Feature computation is delegated to `self.extract_features`.
            - The splits are index-based; shuffling changes the graph order but not their contents.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Splits must sum to 1.0"

        gs, ls = self.generate_dataset()  # <-- direct call; no class state

        indices = list(range(len(gs)))
        if shuffle:
            random.shuffle(indices)

        t_end = int(train_ratio * len(gs))
        v_end = t_end + int(val_ratio * len(gs))
        train_idx = indices[:t_end]
        val_idx   = indices[t_end:v_end]
        test_idx  = indices[v_end:]

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

        # Package into DataLoaders; supports 2-tuple (G,L) or 3-tuple (G,L,F)
        def to_loader(split, shuffle_flag):
            items = list(zip(*split))  # list of tuples per field aligned
            return DataLoader(items, batch_size=batch_size, shuffle=shuffle_flag, collate_fn=collate_fn)

        train_loader = to_loader(train_split, True)
        val_loader   = to_loader(val_split,   False)
        test_loader  = to_loader(test_split,  False)

        return train_loader, val_loader, test_loader

    def provide_benchmark(self, num_graphs: Optional[int] = None) -> Tuple[np.ndarray, ...]:
        """
        Generate a sealed external benchmark:
          - Creates (graphs, labels), but returns ONLY graphs.
          - Stores labels internally so users cannot access them directly.
        """
        original = self.num_graphs
        if num_graphs is not None:
            self.num_graphs = num_graphs
        try:
            graphs, labels = self.generate_dataset()
        finally:
            self.num_graphs = original

        self.graphs = graphs
        self.labels = labels
        return graphs

    def evaluate(self, predictions: Sequence[np.ndarray]) -> float:
        """
        Delegate evaluation to task's evaluation method.
        """
        assert hasattr(self, 'labels') and self.labels is not None, "Run provide_benchmark() first."
        return self.task.evaluate(predictions, self.labels)
