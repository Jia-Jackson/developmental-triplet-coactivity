from __future__ import annotations

import numpy as np


def test_fixed_nodes_are_common_eligible_and_deterministic():
    from hotopology.node_selection import rank_fixed_longitudinal_nodes

    x14 = np.zeros((5, 20), dtype=np.uint8)
    x21 = np.zeros((5, 20), dtype=np.uint8)
    counts14 = [8, 7, 6, 5, 1]
    counts21 = [6, 7, 8, 5, 9]
    for node, count in enumerate(counts14):
        x14[node, :count] = 1
    for node, count in enumerate(counts21):
        x21[node, :count] = 1

    ranking = rank_fixed_longitudinal_nodes(
        {14: x14, 21: x21}, min_active_bins=3, min_inactive_bins=3
    )
    assert ranking["electrode_index_zero_based"].tolist() == [1, 0, 2, 3]
    assert 4 not in set(ranking["electrode_index_zero_based"])
    repeated = rank_fixed_longitudinal_nodes(
        {14: x14, 21: x21}, min_active_bins=3, min_inactive_bins=3
    )
    assert ranking.equals(repeated)


def test_select_fixed_node_sets_are_nested():
    from hotopology.node_selection import select_fixed_node_sets

    rng = np.random.default_rng(5)
    arrays = {div: (rng.random((24, 100)) < 0.3).astype(np.uint8) for div in [14, 21, 28]}
    selected = select_fixed_node_sets(
        arrays, node_counts=[12, 16, 20], min_active_bins=5, min_inactive_bins=5
    )
    assert set(selected[12]).issubset(set(selected[16]))
    assert set(selected[16]).issubset(set(selected[20]))
    assert len(selected[20]) == 20
