from ..task_base import TaskGenerator
import numpy as np


class SymmetricClosureTask(TaskGenerator):
    def generate_labels(self, adj: np.ndarray) -> np.ndarray:
        return np.logical_or(adj, adj.T).astype(np.float32)
