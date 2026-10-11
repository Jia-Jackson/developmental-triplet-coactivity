from __future__ import annotations

import numpy as np


def test_moment_qc_uses_floating_point_and_is_exact_for_identical_data():
    from hotopology.null_qc import null_moment_qc

    observed = np.array([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=np.uint8)
    simulated = np.repeat(observed[None, :, :], 3, axis=0)
    qc = null_moment_qc(observed, simulated)
    assert qc["firing_probability_rmse"] == 0.0
    assert qc["simultaneous_pair_moment_rmse"] == 0.0
    assert qc["exact_k_mismatch_count"] == 0


def test_moment_qc_detects_pairwise_difference_even_if_firing_matches():
    from hotopology.null_qc import null_moment_qc

    observed = np.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=np.uint8)
    alternative = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    simulated = alternative[None, :, :]
    qc = null_moment_qc(observed, simulated, require_exact_k=False)
    assert qc["firing_probability_rmse"] == 0.0
    assert qc["simultaneous_pair_moment_rmse"] > 0
