from __future__ import annotations

import numpy as np
from scipy.io import savemat


def test_bin_spikes_uses_half_open_bins_and_binary_occupancy():
    from hotopology.encoding import bin_spikes

    times = np.array([0.000, 0.010, 0.0249, 0.0250, 0.0499, 0.0500])
    channels = np.array([0, 0, 1, 1, 1, 2])
    x = bin_spikes(times, channels, n_channels=3, duration_s=0.075, bin_s=0.025)

    expected = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(x, expected)


def test_simplex_supports_count_edges_and_triangles_exactly():
    from hotopology.simplicial import simplex_supports

    x = np.array(
        [
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    supports = simplex_supports(x, max_order=3)

    assert supports[1][(0,)] == 3
    assert supports[2][(0, 1)] == 2
    assert supports[2][(0, 2)] == 2
    assert supports[2][(1, 2)] == 2
    assert supports[3][(0, 1, 2)] == 1


def test_persistence_detects_boundary_cycle_and_triangle_fills_it():
    from hotopology.simplicial import persistence_summary_from_supports

    boundary = {
        1: {(0,): 10, (1,): 10, (2,): 10},
        2: {(0, 1): 8, (0, 2): 8, (1, 2): 8},
        3: {},
    }
    filled = {
        **boundary,
        3: {(0, 1, 2): 7},
    }

    boundary_summary = persistence_summary_from_supports(boundary, n_bins=10)
    filled_summary = persistence_summary_from_supports(filled, n_bins=10)

    assert boundary_summary["h1_infinite_count"] == 1
    assert filled_summary["h1_infinite_count"] == 0
    assert filled_summary["h1_total_persistence"] > 0
    assert np.isclose(boundary_summary["h1_total_persistence_capped"], 0.8)
    assert np.isclose(filled_summary["h1_total_persistence_capped"], 0.1)


def test_independent_surrogate_is_seed_reproducible():
    from hotopology.null_models import sample_independent

    probabilities = np.array([0.1, 0.5, 0.9])
    a = sample_independent(probabilities, n_bins=100, seed=17)
    b = sample_independent(probabilities, n_bins=100, seed=17)
    np.testing.assert_array_equal(a, b)


def test_k_conditioned_pairwise_sampler_preserves_each_population_count():
    from hotopology.null_models import sample_pairwise_conditioned_on_k

    h = np.zeros(4)
    j = np.zeros((4, 4))
    k_sequence = np.array([0, 1, 2, 3, 4, 2, 1])
    samples = sample_pairwise_conditioned_on_k(
        h, j, k_sequence, sweeps_per_bin=8, seed=5
    )

    np.testing.assert_array_equal(samples.sum(axis=0), k_sequence)


def test_pool_conditioning_preserves_population_count_and_seed():
    from hotopology.null_models import resample_pool_conditioned_on_k

    pool = np.array(
        [
            [0, 1, 0, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    desired = np.array([0, 1, 2, 3, 1, 2])
    a = resample_pool_conditioned_on_k(pool, desired, n_surrogates=2, seed=9)
    b = resample_pool_conditioned_on_k(pool, desired, n_surrogates=2, seed=9)

    assert a.shape == (2, 3, 6)
    np.testing.assert_array_equal(a.sum(axis=1), np.tile(desired, (2, 1)))
    np.testing.assert_array_equal(a, b)


def test_pool_conditioning_uses_fixed_k_fallback_for_missing_rare_counts():
    from hotopology.null_models import resample_pool_conditioned_on_k

    pool = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.uint8)
    desired = np.array([3, 1, 3])
    output = resample_pool_conditioned_on_k(
        pool,
        desired,
        n_surrogates=2,
        seed=4,
        fallback_h=np.zeros(3),
        fallback_j=np.zeros((3, 3)),
        fallback_sweeps=4,
    )
    np.testing.assert_array_equal(output.sum(axis=1), np.tile(desired, (2, 1)))


def test_decode_ising_factors_converts_toolbox_energy_to_log_potential():
    from hotopology.null_models import decode_ising_factors

    h, j = decode_ising_factors(np.array([1, 2, 3, 10, 20, 30], dtype=float), 3)
    np.testing.assert_array_equal(h, np.array([-1.0, -2.0, -3.0]))
    np.testing.assert_array_equal(
        j,
        np.array([[0, -10, -20], [-10, 0, -30], [-20, -30, 0]], dtype=float),
    )


def test_pairwise_pseudolikelihood_recovers_small_synthetic_moments():
    from hotopology.null_models import fit_pairwise_pseudolikelihood, sample_pairwise

    rng = np.random.default_rng(11)
    h_true = np.array([-1.0, -0.7, -0.9])
    j_true = np.array(
        [
            [0.0, 0.65, 0.0],
            [0.65, 0.0, 0.45],
            [0.0, 0.45, 0.0],
        ]
    )
    from hotopology.null_models import sample_pairwise as draw

    train = draw(h_true, j_true, n_bins=12000, burn_in=2000, thin=2, seed=11)
    fit = fit_pairwise_pseudolikelihood(train, c=5.0, max_iter=2000)
    sim = sample_pairwise(
        fit.h, fit.j, n_bins=12000, burn_in=2000, thin=2, seed=12
    )

    np.testing.assert_allclose(sim.mean(axis=1), train.mean(axis=1), atol=0.06)
    train_pairs = (train @ train.T) / train.shape[1]
    sim_pairs = (sim @ sim.T) / sim.shape[1]
    np.testing.assert_allclose(sim_pairs, train_pairs, atol=0.07)


def test_null_standardization_and_empirical_p_are_finite():
    from hotopology.null_models import standardize_against_null

    result = standardize_against_null(5.0, np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.isfinite(result.z)
    assert 0 < result.p_two_sided <= 1


def test_irreducible_triplet_summary_uses_leave_one_out_null():
    from hotopology.null_models import irreducible_triplet_summary

    observed = np.array([5.0, 1.0, 2.0])
    null_blocks = np.array(
        [
            [1.0, 1.0, 2.0],
            [2.0, 1.0, 2.0],
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
        ]
    )
    result = irreducible_triplet_summary(observed, null_blocks, n_bins=10)

    assert result["positive_residual_mass"] > 0
    assert result["positive_residual_z"] > 0
    assert result["null_positive_residual_mass"].shape == (4,)


def test_direct_metrics_have_expected_triplet_mass():
    from hotopology.metrics import compute_direct_metrics

    x = np.array(
        [
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    metrics, _ = compute_direct_metrics(x, min_active_bins=1)

    assert metrics["n_analysis_nodes"] == 3
    assert metrics["fraction_bins_k_ge_3"] == 0.25
    assert metrics["triplet_support_mass"] == 0.25
    assert metrics["triplet_occurrence_density"] == 1.0


def test_network_burst_detector_returns_intervals_and_bin_mask():
    from hotopology.bursts import detect_network_bursts

    times = np.linspace(1.0, 1.045, 10)
    channels = np.tile(np.array([0, 1, 2, 3, 4]), 2)
    result = detect_network_bursts(
        times,
        channels,
        duration_s=5.0,
        fs=12500.0,
        min_spikes=10,
        min_channels=3,
    )

    assert result.intervals.shape == (1, 2)
    assert result.intervals[0, 0] <= 1.0
    assert result.intervals[0, 1] >= 1.045
    mask = result.bin_mask(bin_s=0.025, n_bins=200)
    assert mask.any()


def test_load_recording_converts_matlab_channel_indices_to_zero_based(tmp_path):
    from hotopology.io import load_recording_mat

    path = tmp_path / "WT_DIV14_P1_A1_arrays.mat"
    savemat(
        path,
        {
            "spike_time_s": np.array([[0.1], [0.2]]),
            "spike_channel_index": np.array([[1], [64]]),
            "duration_s": np.array([[300.0]]),
            "fs": np.array([[12500.0]]),
            "recording": np.array(["WT_DIV14_P1_A1"], dtype=object),
            "divValue": np.array([[14]]),
            "plateIndex": np.array([[1]]),
            "wellLabel": np.array(["A1"], dtype=object),
        },
    )
    rec = load_recording_mat(path)

    np.testing.assert_array_equal(rec.channel_indices, np.array([0, 63]))
    assert rec.recording == "WT_DIV14_P1_A1"
    assert rec.div == 14


def test_load_recording_uses_filename_when_matlab_string_metadata_is_unreliable(tmp_path):
    from hotopology.io import load_recording_mat

    path = tmp_path / "WT_DIV21_P2_B3_arrays.mat"
    savemat(
        path,
        {
            "spike_time_s": np.array([[0.1]]),
            "spike_channel_index": np.array([[2]]),
            "duration_s": np.array([[300.0]]),
            "fs": np.array([[12500.0]]),
            "recording": np.array(["unreliable"], dtype=object),
            "divValue": np.array([[21]]),
            "plateIndex": np.array([[2]]),
            "wellLabel": np.array(["unreliable"], dtype=object),
        },
    )
    rec = load_recording_mat(path)

    assert rec.recording == "WT_DIV21_P2_B3"
    assert rec.well == "B3"


def test_publication_export_writes_all_required_formats(tmp_path):
    import matplotlib.pyplot as plt

    from hotopology.plotting import export_publication_figure

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = export_publication_figure(fig, tmp_path / "test_figure")

    assert {path.suffix for path in paths} == {".svg", ".pdf", ".png", ".tiff"}
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)


def test_benjamini_hochberg_matches_known_values():
    from hotopology.statistics import benjamini_hochberg

    adjusted = benjamini_hochberg(np.array([0.01, 0.04, 0.03]))
    np.testing.assert_allclose(adjusted, np.array([0.03, 0.04, 0.04]))


def test_adjacent_paired_tests_use_subject_intersection():
    import pandas as pd

    from hotopology.statistics import adjacent_paired_wilcoxon

    table = pd.DataFrame(
        {
            "subject_id": ["a", "b", "c", "a", "b", "d"],
            "DIV": [14, 14, 14, 21, 21, 21],
            "value": [1.0, 2.0, 9.0, 3.0, 4.0, 10.0],
        }
    )
    result = adjacent_paired_wilcoxon(table, metric="value", divs=[14, 21])

    assert result.loc[0, "n_pairs"] == 2
    assert result.loc[0, "median_change"] == 2.0


def test_repeated_friedman_uses_only_complete_subjects():
    import pandas as pd

    from hotopology.statistics import repeated_friedman

    table = pd.DataFrame(
        {
            "subject_id": ["a", "b", "c"] * 3,
            "DIV": [14] * 3 + [21] * 3 + [28] * 3,
            "value": [1.0, 2.0, 3.0, 2.0, 3.0, np.nan, 3.0, 4.0, 5.0],
        }
    )
    result = repeated_friedman(table, metric="value", divs=[14, 21, 28])

    assert result["n_complete_subjects"] == 2
    assert result["df"] == 2
    assert np.isfinite(result["raw_p"])


def test_vectorized_pairwise_chains_recover_small_model_moments():
    from hotopology.null_models import sample_pairwise_chains

    h = np.array([-0.4, 0.2])
    j = np.array([[0.0, 0.8], [0.8, 0.0]])
    states = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    log_weights = states @ h + 0.5 * np.einsum("bi,ij,bj->b", states, j, states)
    probabilities = np.exp(log_weights - log_weights.max())
    probabilities /= probabilities.sum()
    expected_mean = probabilities @ states
    expected_pair = probabilities[-1]

    samples = sample_pairwise_chains(
        h,
        j,
        n_chains=12,
        n_bins=2000,
        burn_in_sweeps=300,
        thin_sweeps=1,
        seed=304,
    )
    pooled = samples.transpose(1, 0, 2).reshape(2, -1)

    np.testing.assert_allclose(pooled.mean(axis=1), expected_mean, atol=0.025)
    assert abs(np.mean(pooled[0] * pooled[1]) - expected_pair) < 0.025


def test_joint_pseudolikelihood_fit_recovers_synthetic_moments():
    from hotopology.null_models import (
        fit_pairwise_joint_pseudolikelihood,
        sample_pairwise,
    )

    h = np.array([-0.5, -0.2, 0.1, -0.4])
    j = np.array(
        [
            [0.0, 0.7, -0.2, 0.1],
            [0.7, 0.0, 0.4, 0.0],
            [-0.2, 0.4, 0.0, 0.5],
            [0.1, 0.0, 0.5, 0.0],
        ]
    )
    data = sample_pairwise(h, j, n_bins=6000, burn_in=800, seed=120)
    fit = fit_pairwise_joint_pseudolikelihood(data, l2=1e-5, max_iter=200)
    generated = sample_pairwise(fit.h, fit.j, n_bins=12000, burn_in=800, seed=121)
    data_float = data.astype(float)
    generated_float = generated.astype(float)
    np.testing.assert_allclose(generated_float.mean(1), data_float.mean(1), atol=0.05)
    data_pair = data_float @ data_float.T / data.shape[1]
    generated_pair = generated_float @ generated_float.T / generated.shape[1]
    upper = np.triu_indices(4, 1)
    np.testing.assert_allclose(generated_pair[upper], data_pair[upper], atol=0.05)


def test_one_sample_wilcoxon_by_div_adjusts_across_ages():
    import pandas as pd

    from hotopology.statistics import one_sample_wilcoxon_by_div

    table = pd.DataFrame(
        {
            "DIV": [14] * 4 + [21] * 4,
            "value": [1.0, 2.0, 3.0, 4.0, 1.0, 1.5, 2.0, 2.5],
        }
    )
    result = one_sample_wilcoxon_by_div(
        table, metric="value", divs=[14, 21], null_value=0.0, alternative="greater"
    )

    assert list(result["n"]) == [4, 4]
    assert np.all(result["adjusted_p"] >= result["raw_p"])
