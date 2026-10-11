from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Mapping

import gudhi
import numpy as np


SupportMap = dict[int, dict[tuple[int, ...], int]]


def _row_bitsets(x: np.ndarray) -> list[int]:
    packed = np.packbits(np.asarray(x, dtype=np.uint8), axis=1, bitorder="little")
    return [int.from_bytes(row.tobytes(), byteorder="little") for row in packed]


def simplex_supports(x: np.ndarray, *, max_order: int = 3) -> SupportMap:
    """Count support for observed simplices up to ``max_order`` vertices.

    The bitset implementation makes all 41,664 possible triplets tractable for
    a 64-electrode, 12,000-bin recording without expanding every co-active bin.
    Zero-support simplices are omitted.
    """
    data = np.asarray(x, dtype=np.uint8)
    if data.ndim != 2:
        raise ValueError("x must be a 2D channel-by-bin matrix")
    if max_order < 1:
        raise ValueError("max_order must be at least 1")
    data = (data > 0).astype(np.uint8, copy=False)
    n_nodes = data.shape[0]
    bits = _row_bitsets(data)
    supports: SupportMap = {order: {} for order in range(1, max_order + 1)}

    for order in range(1, min(max_order, n_nodes) + 1):
        for simplex in combinations(range(n_nodes), order):
            mask = bits[simplex[0]]
            for node in simplex[1:]:
                mask &= bits[node]
                if mask == 0:
                    break
            count = mask.bit_count()
            if count:
                supports[order][simplex] = count
    return supports


def persistence_summary_from_supports(
    supports: Mapping[int, Mapping[tuple[int, ...], int]],
    *,
    n_bins: int,
) -> dict[str, float]:
    """Build a support filtration and summarize H0/H1 persistence.

    Filtration time is ``1 - support / n_bins``. Because a face cannot have
    lower support than its coface, this ordering satisfies simplicial closure.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    tree = gudhi.SimplexTree()
    for order in sorted(supports):
        for simplex, count in supports[order].items():
            filtration = 1.0 - float(count) / float(n_bins)
            tree.insert(simplex, filtration=filtration)
    tree.make_filtration_non_decreasing()
    tree.persistence(
        homology_coeff_field=2,
        min_persistence=0.0,
        persistence_dim_max=True,
    )

    output: dict[str, float] = {}
    for dim in (0, 1):
        intervals = tree.persistence_intervals_in_dimension(dim)
        if intervals.size == 0:
            finite_lifetimes = np.array([], dtype=float)
            infinite_count = 0
            capped_lifetimes = np.array([], dtype=float)
        else:
            finite = np.isfinite(intervals[:, 1])
            finite_lifetimes = intervals[finite, 1] - intervals[finite, 0]
            infinite_count = int(np.sum(~finite))
            capped_deaths = np.where(finite, intervals[:, 1], 1.0)
            capped_lifetimes = np.maximum(0.0, capped_deaths - intervals[:, 0])
        output[f"h{dim}_finite_count"] = int(finite_lifetimes.size)
        output[f"h{dim}_infinite_count"] = infinite_count
        output[f"h{dim}_total_persistence"] = float(finite_lifetimes.sum())
        output[f"h{dim}_total_persistence_capped"] = float(capped_lifetimes.sum())
        output[f"h{dim}_mean_persistence"] = (
            float(finite_lifetimes.mean()) if finite_lifetimes.size else 0.0
        )
    output["n_simplices"] = int(tree.num_simplices())
    output["complex_dimension"] = int(tree.dimension())
    return output


def normalized_simplex_density(
    supports: Mapping[int, Mapping[tuple[int, ...], int]],
    *,
    n_nodes: int,
    order: int,
    min_support: int = 1,
) -> float:
    """Fraction of possible ``order``-node simplices meeting support cutoff."""
    denominator = comb(n_nodes, order) if n_nodes >= order else 0
    if denominator == 0:
        return 0.0
    numerator = sum(v >= min_support for v in supports.get(order, {}).values())
    return float(numerator / denominator)
