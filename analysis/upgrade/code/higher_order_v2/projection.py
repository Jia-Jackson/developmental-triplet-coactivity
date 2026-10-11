from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class Projection:
    edges: np.ndarray
    edge_values: np.ndarray
    node_values: np.ndarray


def triplet_index(n_nodes: int) -> np.ndarray:
    if int(n_nodes) < 3:
        raise ValueError("at least three nodes are required")
    return np.asarray(list(combinations(range(int(n_nodes)), 3)), dtype=int)


def incidence_projection(values: np.ndarray, triplets: np.ndarray, n_nodes: int) -> Projection:
    """Mean-incidence projection of triplet values to constituent edges/nodes."""
    values = np.asarray(values, dtype=float).reshape(-1)
    triplets = np.asarray(triplets, dtype=int)
    if triplets.shape != (len(values), 3):
        raise ValueError("triplets must have shape (len(values), 3)")
    edges = np.asarray(list(combinations(range(int(n_nodes)), 2)), dtype=int)
    edge_lookup = {tuple(edge): index for index, edge in enumerate(edges)}
    edge_sum = np.zeros(len(edges), dtype=float)
    edge_count = np.zeros(len(edges), dtype=int)
    node_sum = np.zeros(int(n_nodes), dtype=float)
    node_count = np.zeros(int(n_nodes), dtype=int)
    for value, (i, j, k) in zip(values, triplets):
        for edge in ((i, j), (i, k), (j, k)):
            index = edge_lookup[tuple(sorted(edge))]
            edge_sum[index] += value
            edge_count[index] += 1
        for node in (i, j, k):
            node_sum[node] += value
            node_count[node] += 1
    if np.any(edge_count == 0) or np.any(node_count == 0):
        raise ValueError("projection contains an unrepresented edge or node")
    return Projection(edges, edge_sum / edge_count, node_sum / node_count)

