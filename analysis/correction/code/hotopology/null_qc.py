from __future__ import annotations

import numpy as np


def null_moment_qc(
    observed: np.ndarray,
    simulated: np.ndarray,
    *,
    require_exact_k: bool = True,
) -> dict[str, float | int | bool]:
    observed_data = (np.asarray(observed) > 0).astype(np.float64)
    simulated_data = (np.asarray(simulated) > 0).astype(np.float64)
    if observed_data.ndim != 2 or simulated_data.ndim != 3:
        raise ValueError("observed must be node-by-time and simulated realization-by-node-by-time")
    if simulated_data.shape[1:] != observed_data.shape:
        raise ValueError("observed and simulated shapes are incompatible")
    n_realizations, n_nodes, n_bins = simulated_data.shape
    observed_firing = observed_data.mean(axis=1)
    simulated_firing = simulated_data.mean(axis=(0, 2))
    firing_rmse = float(np.sqrt(np.mean(np.square(observed_firing - simulated_firing))))

    observed_pair = observed_data @ observed_data.T / n_bins
    simulated_pair = np.zeros((n_nodes, n_nodes), dtype=float)
    for realization in simulated_data:
        simulated_pair += realization @ realization.T
    simulated_pair /= n_realizations * n_bins
    upper = np.triu_indices(n_nodes, 1)
    pair_rmse = float(
        np.sqrt(np.mean(np.square(observed_pair[upper] - simulated_pair[upper])))
    )
    observed_k = observed_data.sum(axis=0).astype(int)
    simulated_k = simulated_data.sum(axis=1).astype(int)
    mismatch = int(np.sum(simulated_k != observed_k[None, :]))
    if require_exact_k and mismatch:
        raise ValueError(f"simulated null has {mismatch} population-count mismatches")
    return {
        "firing_probability_rmse": firing_rmse,
        "simultaneous_pair_moment_rmse": pair_rmse,
        "exact_k_mismatch_count": mismatch,
        "exact_k_preserved": mismatch == 0,
        "n_realizations": int(n_realizations),
    }
