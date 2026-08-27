import numpy as np
import heapq
from typing import Callable, List, Tuple

from .distance import cosine_distance


class FlatIndex:
    """
    Brute-force KNN: compares query against every stored vector.
    O(n * d) per query — exact, but does not scale.
    Used as the ground-truth baseline to measure HNSW recall.
    """

    def __init__(self, distance_fn: Callable = cosine_distance):
        self.vectors: List[np.ndarray] = []
        self.distance_fn = distance_fn

    def add(self, vector: np.ndarray) -> None:
        self.vectors.append(vector)

    def add_batch(self, vectors: np.ndarray) -> None:
        for v in vectors:
            self.add(v)

    def search(self, query: np.ndarray, k: int) -> List[Tuple[float, int]]:
        """
        Returns list of (distance, index) sorted closest-first.
        """
        if not self.vectors:
            return []

        distances = [
            (self.distance_fn(query, v), i)
            for i, v in enumerate(self.vectors)
        ]
        return heapq.nsmallest(k, distances, key=lambda x: x[0])

    def __len__(self) -> int:
        return len(self.vectors)
