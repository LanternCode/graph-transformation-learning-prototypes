"""
Unfinished framework prototype: shared task interface.

This file defines the abstract base class used by the prototype benchmark
framework. A task is represented by a label-generation method that converts an
input adjacency matrix into a target matrix, with an optional default evaluator
for simple prediction/label comparisons.

This framework prototype was abandoned in favour of the later standalone
implementation. It is archived for portfolio purposes to document the early
attempt to formalise graph transformation tasks behind a common interface.
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Sequence


class TaskGenerator(ABC):
    """
    Abstract base class for defining benchmark tasks.
    Each task must implement a method to generate labels for a given adjacency matrix.

    You can optionally override the evaluate method for custom metrics.
    """

    @abstractmethod
    def generate_labels(self, adj: np.ndarray) -> np.ndarray:
        """
        Generate task-specific labels given a graph adjacency matrix.
        """
        pass

    def evaluate(self, predictions: Sequence[np.ndarray], labels: Sequence[np.ndarray]) -> float:
        """
        Default evaluation method: average accuracy over all predictions.
        Override for custom scoring (e.g., F1, AUC, etc).
        """
        accs = [np.mean(p == l) for p, l in zip(predictions, labels)]
        avg = float(np.mean(accs))
        print(f"Average accuracy: {avg:.4f}")
        return avg
