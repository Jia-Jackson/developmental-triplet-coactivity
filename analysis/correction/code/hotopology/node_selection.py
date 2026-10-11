from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def rank_fixed_longitudinal_nodes(
    binary_by_div: Mapping[int, np.ndarray],
    *,
    min_active_bins: int = 20,
    min_inactive_bins: int = 20,
) -> pd.DataFrame:
    """Rank nodes common to all ages by maximin activity, then total activity."""
    if not binary_by_div:
        raise ValueError("binary_by_div is empty")
    ordered = sorted((int(div), np.asarray(x, dtype=np.uint8)) for div, x in binary_by_div.items())
    n_nodes = ordered[0][1].shape[0]
    if any(x.ndim != 2 or x.shape[0] != n_nodes for _, x in ordered):
        raise ValueError("all binary matrices must be two-dimensional with the same node count")
    active_counts = np.column_stack([x.sum(axis=1, dtype=np.int64) for _, x in ordered])
    inactive_counts = np.column_stack([x.shape[1] - x.sum(axis=1, dtype=np.int64) for _, x in ordered])
    eligible = (active_counts >= int(min_active_bins)).all(axis=1) & (
        inactive_counts >= int(min_inactive_bins)
    ).all(axis=1)
    indices = np.flatnonzero(eligible)
    table = pd.DataFrame(
        {
            "electrode_index_zero_based": indices,
            "electrode_index_one_based": indices + 1,
            "minimum_active_bin_count": active_counts[indices].min(axis=1),
            "total_active_bin_count": active_counts[indices].sum(axis=1),
            "minimum_inactive_bin_count": inactive_counts[indices].min(axis=1),
            "n_available_ages": len(ordered),
        }
    )
    table = table.sort_values(
        [
            "minimum_active_bin_count",
            "total_active_bin_count",
            "electrode_index_zero_based",
        ],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    table.insert(0, "fixed_rank", np.arange(1, len(table) + 1))
    return table


def select_fixed_node_sets(
    binary_by_div: Mapping[int, np.ndarray],
    *,
    node_counts: Sequence[int],
    min_active_bins: int = 20,
    min_inactive_bins: int = 20,
) -> dict[int, np.ndarray]:
    ranking = rank_fixed_longitudinal_nodes(
        binary_by_div,
        min_active_bins=min_active_bins,
        min_inactive_bins=min_inactive_bins,
    )
    maximum = max(int(value) for value in node_counts)
    if len(ranking) < maximum:
        raise ValueError(f"only {len(ranking)} common eligible nodes; {maximum} required")
    return {
        int(count): ranking["electrode_index_zero_based"].to_numpy(dtype=int)[: int(count)]
        for count in node_counts
    }
