from __future__ import annotations

import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_observed_benchmark", ROOT / "code/calibration/run_observed_benchmark.py"
)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def test_lower_median_selection_tie_and_ineligible_candidate_are_explicit():
    rows = pd.DataFrame(
        [
            {"culture": "P1", "DIV": 14, "recording": "r3", "eligible": True, "mean_population_count": 2.0, "reason": ""},
            {"culture": "P1", "DIV": 14, "recording": "r1", "eligible": True, "mean_population_count": 1.0, "reason": ""},
            {"culture": "P1", "DIV": 14, "recording": "r2", "eligible": True, "mean_population_count": 2.0, "reason": ""},
            {"culture": "P1", "DIV": 14, "recording": "missing", "eligible": False, "mean_population_count": np.nan, "reason": "eligible_nodes=14<16"},
        ]
    )
    out = BENCH.select_lower_median_candidates(rows)
    assert out.loc[out.selected, "recording"].tolist() == ["r2"]
    assert out["candidate_n"].unique().tolist() == [3]
    assert out.loc[out.recording.eq("missing"), "exclusion_reason"].item() == "eligible_nodes=14<16"


def test_theta_mapping_matches_native_cpp_negative_energy_logit():
    factors = np.array([0.2, -0.3, 0.7, 1.1, -0.4, 0.9])
    theta = BENCH.theta_from_toolbox_factors(factors, 3)
    states = np.array([[0, 0, 0], [1, 0, 1], [1, 1, 1]], dtype=float)
    # IsingEnergy.cpp returns +sum(lambda*f) as energy, while
    # maxent_functions.cpp getLogProbability returns -energy + logZ.
    native_centered_logit = -np.array(
        [0.0, 0.2 + 0.7 - 0.4, 0.2 - 0.3 + 0.7 + 1.1 - 0.4 + 0.9]
    )
    pair = np.column_stack([states[:, i] * states[:, j] for i, j in combinations(range(3), 2)])
    assert np.allclose(np.column_stack((states, pair)) @ theta, native_centered_logit)


def test_empirical_zeta_support_matches_direct_triplet_loop():
    rng = np.random.default_rng(7)
    x = rng.integers(0, 2, size=(6, 31), dtype=np.uint8)
    expected = np.array([np.count_nonzero(x[i] * x[j] * x[k]) / x.shape[1]
                         for i, j, k in combinations(range(6), 3)])
    counts, support = BENCH.empirical_triplet_support_zeta(x)
    assert counts.dtype.kind in "iu"
    assert np.array_equal(counts, np.rint(expected * x.shape[1]).astype(int))
    assert np.allclose(support, expected)


def test_named_crc32_seed_is_stable_and_arm_specific():
    assert BENCH.stable_seed("record", "_exact") == BENCH.stable_seed("record", "_exact")
    assert BENCH.stable_seed("record", "_exact") != BENCH.stable_seed("record", "_refit")
    assert BENCH.stable_seed("record", "_exact") == 3901557337


def test_small_n_exact_arm_preserves_k_and_has_complete_cost_outputs():
    x = np.array([[0, 1, 1, 0, 1], [0, 0, 1, 1, 1], [0, 1, 0, 1, 1]], dtype=np.uint8)
    result = BENCH.evaluate_exact_arm(x, np.zeros(6), n_realizations=7, seed=11)
    assert result["null_supports"].shape == (7, 1)
    assert result["simulated"].shape == (7, 3, 5)
    assert result["exact_k_mismatch_count"] == 0
    assert set(["observed_mass", "null_mass_mean", "null_mass_sd", "p_upper", "z", "status"]) <= set(result)
