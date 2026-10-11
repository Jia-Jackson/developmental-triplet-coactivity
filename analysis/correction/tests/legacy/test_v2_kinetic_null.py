from __future__ import annotations

import numpy as np


def test_conditional_bernoulli_sampler_preserves_k_and_seed():
    from hotopology.kinetic_null import sample_conditional_bernoulli_fixed_k

    logits = np.array([[0.2, -0.5, 1.0, 0.1], [-0.2, 0.4, 0.3, -1.0]])
    rng1 = np.random.default_rng(13)
    rng2 = np.random.default_rng(13)
    first = sample_conditional_bernoulli_fixed_k(logits, 2, rng1)
    second = sample_conditional_bernoulli_fixed_k(logits, 2, rng2)
    assert np.array_equal(first, second)
    assert np.array_equal(first.sum(axis=1), [2, 2])


def test_conditional_sampler_matches_small_exact_subset_probabilities():
    from hotopology.kinetic_null import sample_conditional_bernoulli_fixed_k

    log_odds = np.log(np.array([1.0, 2.0, 4.0]))
    logits = np.repeat(log_odds[None, :], 30000, axis=0)
    samples = sample_conditional_bernoulli_fixed_k(
        logits, 2, np.random.default_rng(17)
    )
    observed = {
        tuple(np.flatnonzero(row)): count / len(samples)
        for row, count in zip(*np.unique(samples, axis=0, return_counts=True))
    }
    expected_weights = {(0, 1): 2.0, (0, 2): 4.0, (1, 2): 8.0}
    total = sum(expected_weights.values())
    for subset, weight in expected_weights.items():
        assert abs(observed[subset] - weight / total) < 0.015


def test_kinetic_simulation_preserves_entire_population_count_sequence():
    from hotopology.kinetic_null import simulate_kinetic_conditioned_k

    intercept = np.zeros(4)
    weights = np.eye(4) * 1.2
    k = np.array([1, 2, 0, 3, 1, 4, 2])
    initial = np.array([1, 0, 0, 0], dtype=np.uint8)
    samples = simulate_kinetic_conditioned_k(
        intercept,
        weights,
        k,
        initial_state=initial,
        n_realizations=12,
        seed=29,
    )
    assert samples.shape == (12, 4, len(k))
    assert np.array_equal(samples.sum(axis=1), np.repeat(k[None, :], 12, axis=0))


def test_fitted_kinetic_model_beats_rate_only_on_persistent_data():
    from hotopology.kinetic_null import fit_kinetic_pairwise

    rng = np.random.default_rng(31)
    x = np.zeros((3, 1800), dtype=np.uint8)
    x[:, 0] = rng.integers(0, 2, size=3)
    for t in range(1, x.shape[1]):
        stay = rng.random(3) < 0.88
        refresh = rng.random(3) < np.array([0.2, 0.4, 0.6])
        x[:, t] = np.where(stay, x[:, t - 1], refresh)
    fit = fit_kinetic_pairwise(x, c_grid=[0.1, 1.0], n_splits=3, max_iter=400)
    assert fit.weights.shape == (3, 3)
    assert fit.heldout_log_loss < fit.rate_only_log_loss
    assert np.diag(fit.weights).mean() > 0
