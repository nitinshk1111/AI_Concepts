import math
import random
import heapq
from typing import Callable, List, Tuple

import numpy as np

from .distance import cosine_distance


class HNSWIndex:
    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        distance_fn: Callable = cosine_distance,
    ):
        self.M = M
        self.M0 = 2 * M                    # layer 0 gets denser connections
        self.ef_construction = ef_construction
        self.mL = 1.0 / math.log(M)        # controls layer assignment probability

        self.vectors: List[np.ndarray] = []
        self.levels: List[int] = []
        # graph[layer][node_id] = set of neighbour node_ids
        self.graph: List[dict] = []
        self.entry_point: int = -1
        self.max_layer: int = -1

        self.distance_fn = distance_fn

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _sample_level(self) -> int:
        # Geometric distribution: most nodes → layer 0, exponentially fewer reach higher
        return int(-math.log(random.random()) * self.mL)

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[int],
        ef: int,
        layer: int,
    ) -> List[Tuple[float, int]]:
        """
        Beam search on a single layer.
        candidates = min-heap (closest unexplored node at top)
        W          = max-heap of size ef (working result set; farthest at top)
        Stops when the closest candidate is farther than the worst node in W.
        """
        visited = set(entry_points)

        candidates = []
        W = []  # stored as (-distance, node_id) to simulate max-heap

        for ep in entry_points:
            d = self.distance_fn(query, self.vectors[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(W, (-d, ep))

        while candidates:
            c_dist, c = heapq.heappop(candidates)
            worst_in_W = -W[0][0]

            if c_dist > worst_in_W:
                break  # no candidate can improve W anymore

            for neighbour in self.graph[layer].get(c, set()):
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                d = self.distance_fn(query, self.vectors[neighbour])

                if d < -W[0][0] or len(W) < ef:
                    heapq.heappush(candidates, (d, neighbour))
                    heapq.heappush(W, (-d, neighbour))
                    if len(W) > ef:
                        heapq.heappop(W)  # evict farthest when W exceeds ef

        return sorted((-d, node) for d, node in W)

    def _select_neighbours(
        self, candidates: List[Tuple[float, int]], M: int
    ) -> List[int]:
        # Simple strategy: keep the M closest candidates
        return [node for _, node in sorted(candidates)[:M]]

    def _prune(self, layer: int, node_id: int, M_max: int) -> None:
        # If a node exceeds its connection limit after a bidirectional edge add, trim it
        neighbours = self.graph[layer][node_id]
        if len(neighbours) <= M_max:
            return
        scored = [
            (self.distance_fn(self.vectors[node_id], self.vectors[n]), n)
            for n in neighbours
        ]
        self.graph[layer][node_id] = set(self._select_neighbours(scored, M_max))

    # ── Public API ────────────────────────────────────────────────────────────

    def add(self, vector: np.ndarray) -> None:
        node_id = len(self.vectors)
        self.vectors.append(vector)

        level = self._sample_level()
        self.levels.append(level)

        # Extend graph list to accommodate new level
        while len(self.graph) <= level:
            self.graph.append({})

        for lc in range(level + 1):
            self.graph[lc][node_id] = set()

        # First node — becomes entry point by default
        if self.entry_point == -1:
            self.entry_point = node_id
            self.max_layer = level
            return

        ep = [self.entry_point]

        # Phase 1: greedy descent from max_layer to level+1 (ef=1, fast)
        for lc in range(self.max_layer, level, -1):
            if lc < len(self.graph):
                results = self._search_layer(vector, ep, ef=1, layer=lc)
                ep = [results[0][1]]

        # Phase 2: beam search from min(level, max_layer) down to layer 0
        for lc in range(min(level, self.max_layer), -1, -1):
            results = self._search_layer(vector, ep, ef=self.ef_construction, layer=lc)

            M_max = self.M0 if lc == 0 else self.M
            neighbours = self._select_neighbours(results, M_max)

            for neighbour in neighbours:
                self.graph[lc][node_id].add(neighbour)
                self.graph[lc][neighbour].add(node_id)
                self._prune(lc, neighbour, M_max)

            ep = [node for _, node in results]

        if level > self.max_layer:
            self.entry_point = node_id
            self.max_layer = level

    def add_batch(self, vectors: np.ndarray) -> None:
        for v in vectors:
            self.add(v)

    def search(
        self, query: np.ndarray, k: int, ef: int = 50
    ) -> List[Tuple[float, int]]:
        """
        Returns list of (distance, index) sorted closest-first.
        ef controls recall vs latency — higher ef = better recall, slower query.
        """
        if self.entry_point == -1:
            return []

        ep = [self.entry_point]

        # Greedy descent to layer 1
        for lc in range(self.max_layer, 0, -1):
            if lc < len(self.graph):
                results = self._search_layer(query, ep, ef=1, layer=lc)
                ep = [results[0][1]]

        # Full beam search at layer 0
        results = self._search_layer(query, ep, ef=max(ef, k), layer=0)
        return results[:k]

    def __len__(self) -> int:
        return len(self.vectors)
