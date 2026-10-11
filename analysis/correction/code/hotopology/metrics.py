from __future__ import annotations

from math import comb

import numpy as np

from .simplicial import persistence_summary_from_supports, simplex_supports


def compute_direct_metrics(
    x: np.ndarray,
    *,
    min_active_bins: int = 1,
) -> tuple[dict[str, float], dict[str, object]]:
    """Compute well-level observed higher-order co-activity metrics."""
    data = (np.asarray(x) > 0).astype(np.uint8)
    if data.ndim != 2:
        raise ValueError("x must be channel-by-bin")
    n_bins = data.shape[1]
    active_mask = data.sum(axis=1) >= min_active_bins
    active_nodes = np.flatnonzero(active_mask)
    active = data[active_mask]
    n_nodes = active.shape[0]
    k = active.sum(axis=0) if n_nodes else np.zeros(n_bins, dtype=int)
    supports = simplex_supports(active, max_order=3) if n_nodes else {1: {}, 2: {}, 3: {}}
    persistence = (
        persistence_summary_from_supports(supports, n_bins=n_bins)
        if n_nodes
        else {
            "h0_finite_count": 0,
            "h0_infinite_count": 0,
            "h0_total_persistence": 0.0,
            "h0_total_persistence_capped": 0.0,
            "h0_mean_persistence": 0.0,
            "h1_finite_count": 0,
            "h1_infinite_count": 0,
            "h1_total_persistence": 0.0,
            "h1_total_persistence_capped": 0.0,
            "h1_mean_persistence": 0.0,
            "n_simplices": 0,
            "complex_dimension": -1,
        }
    )

    possible_pairs = comb(n_nodes, 2) if n_nodes >= 2 else 0
    possible_triplets = comb(n_nodes, 3) if n_nodes >= 3 else 0
    pair_counts = np.fromiter(supports.get(2, {}).values(), dtype=float)
    triplet_counts = np.fromiter(supports.get(3, {}).values(), dtype=float)

    metrics: dict[str, float] = {
        "n_analysis_nodes": int(n_nodes),
        "n_bins": int(n_bins),
        "mean_population_count": float(k.mean()) if n_bins else 0.0,
        "median_population_count": float(np.median(k)) if n_bins else 0.0,
        "max_population_count": int(k.max()) if n_bins else 0,
        "fraction_bins_k_ge_3": float(np.mean(k >= 3)) if n_bins else 0.0,
        "fraction_bins_k_ge_4": float(np.mean(k >= 4)) if n_bins else 0.0,
        "pair_occurrence_density": (
            float(pair_counts.size / possible_pairs) if possible_pairs else 0.0
        ),
        "triplet_occurrence_density": (
            float(triplet_counts.size / possible_triplets) if possible_triplets else 0.0
        ),
        "triplet_repeated5_density": (
            float(np.sum(triplet_counts >= 5) / possible_triplets)
            if possible_triplets
            else 0.0
        ),
        "pair_support_mass": (
            float(pair_counts.sum() / (n_bins * possible_pairs))
            if n_bins and possible_pairs
            else 0.0
        ),
        "triplet_support_mass": (
            float(triplet_counts.sum() / (n_bins * possible_triplets))
            if n_bins and possible_triplets
            else 0.0
        ),
        "median_observed_triplet_support": (
            float(np.median(triplet_counts) / n_bins)
            if n_bins and triplet_counts.size
            else 0.0
        ),
    }
    metrics.update(persistence)
    detail = {
        "active_nodes": active_nodes,
        "population_count": k,
        "supports": supports,
    }
    return metrics, detail
