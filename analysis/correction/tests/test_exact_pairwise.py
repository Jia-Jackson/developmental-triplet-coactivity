from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "code" / "calibration" / "exact_pairwise.py"
SPEC = importlib.util.spec_from_file_location("exact_pairwise", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
exact_pairwise = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exact_pairwise)
ExactPairwise = exact_pairwise.ExactPairwise
empirical_triplet_support = exact_pairwise.empirical_triplet_support
triplet_residual_score = exact_pairwise.triplet_residual_score


def test_n3_uniform_conditional_k2_has_expected_moments_and_triplet_support():
    model = ExactPairwise(3)
    prob = model.probabilities(np.zeros(6), k_weights=np.array([0, 0, 1, 0]))

    np.testing.assert_allclose(model.moments(prob)[:3], 2 / 3)
    np.testing.assert_allclose(model.moments(prob)[3:], 1 / 3)
    np.testing.assert_allclose(model.triplet_support(prob), [0.0])


@pytest.mark.parametrize("k, expected", [(0, 0), (4, 1)])
def test_conditional_k_extremes_are_deterministic(k, expected):
    model = ExactPairwise(4)
    weights = np.zeros(5)
    weights[k] = 1
    prob = model.probabilities(np.zeros(model.n_features), k_weights=weights)

    assert prob.sum() == pytest.approx(1.0)
    assert np.count_nonzero(prob) == 1
    assert np.all(model.states[np.argmax(prob)] == expected)


def test_simulation_preserves_each_requested_count_and_seed_is_repeatable():
    model = ExactPairwise(5)
    theta = np.linspace(-0.4, 0.5, model.n_features)
    k_sequence = np.array([0, 1, 4, 5, 2, 2, 3])

    first = model.simulate(theta, k_sequence, n_realizations=4, seed=17)
    second = model.simulate(theta, k_sequence, n_realizations=4, seed=17)

    assert first.dtype == np.uint8
    assert first.shape == (4, 5, k_sequence.size)
    np.testing.assert_array_equal(first.sum(axis=1), np.tile(k_sequence, (4, 1)))
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("conditional", [False, True])
def test_loss_gradient_matches_central_difference(conditional):
    model = ExactPairwise(4)
    theta = np.linspace(-0.3, 0.4, model.n_features)
    target = np.linspace(0.12, 0.48, model.n_features)
    weights = np.array([0.08, 0.20, 0.35, 0.27, 0.10]) if conditional else None
    _, analytic = model.loss_grad(theta, target, k_weights=weights)
    epsilon = 1e-6
    numeric = np.empty_like(theta)
    for index in range(theta.size):
        step = np.zeros_like(theta)
        step[index] = epsilon
        numeric[index] = (
            model.loss_grad(theta + step, target, k_weights=weights)[0]
            - model.loss_grad(theta - step, target, k_weights=weights)[0]
        ) / (2 * epsilon)

    np.testing.assert_allclose(analytic, numeric, atol=2e-8, rtol=2e-7)


def test_exact_triplet_support_agrees_with_direct_empirical_histogram():
    x = np.array(
        [[1, 1, 0, 1, 0], [1, 0, 1, 1, 0], [1, 1, 1, 0, 0], [0, 1, 1, 1, 0]],
        dtype=np.uint8,
    )
    model = ExactPairwise(4)
    state_index = np.sum(x.astype(np.uint64) << np.arange(4, dtype=np.uint64)[:, None], axis=0)
    probability = np.bincount(state_index.astype(int), minlength=16) / x.shape[1]

    np.testing.assert_allclose(model.triplet_support(probability), empirical_triplet_support(x))


def test_fitted_unconditional_model_matches_low_order_synthetic_moments():
    generating = ExactPairwise(3)
    theta = np.array([-0.4, 0.15, 0.35, 0.2, -0.25, 0.1])
    rng = np.random.default_rng(91)
    indices = rng.choice(generating.n_states, size=40_000, p=generating.probabilities(theta))
    x = generating.states[indices].T

    fitted = generating.fit(x, gtol=1e-7, maxiter=1000)
    empirical = generating.features[indices].mean(axis=0)
    recovered = generating.moments(generating.probabilities(fitted["theta"]))

    assert fitted["success"]
    np.testing.assert_allclose(recovered, empirical, atol=2e-7)


@pytest.mark.parametrize(
    "bad",
    [np.zeros(5), np.zeros((3, 0)), np.array([[0, 1], [1, 2]]), np.array([[0.0, np.nan]])],
)
def test_empirical_input_validation_rejects_bad_shape_empty_nonbinary_or_nan(bad):
    with pytest.raises(ValueError):
        empirical_triplet_support(bad)


def test_fit_validates_node_count():
    with pytest.raises(ValueError):
        ExactPairwise(3).fit(np.zeros((4, 10), dtype=np.uint8))


def test_absolute_residual_score_not_positive_mass_for_asymmetric_residuals():
    observed = np.array([5.0, 0.0])
    nulls = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

    result = triplet_residual_score(observed, nulls)

    assert result["absolute_residual_mass"] == pytest.approx(2.5)
    assert result["score"] == pytest.approx(2.5)
    assert result["absolute_residual_mass"] != pytest.approx(2.0)


def test_zero_variance_score_is_explicit_without_infinite_z_and_keeps_empirical_p():
    tied = triplet_residual_score(np.array([1.0, 1.0]), np.ones((4, 2)))
    separated = triplet_residual_score(np.array([2.0, 2.0]), np.ones((4, 2)))

    assert tied["null_sd"] == 0.0
    assert tied["z"] == 0.0
    assert tied["zero_variance"]
    assert tied["status"] == "zero_variance_tied"
    assert tied["p_upper"] == 1.0
    assert np.isnan(separated["z"])
    assert separated["status"] == "zero_variance_separated"
    assert separated["p_upper"] == pytest.approx(1 / 5)


@pytest.mark.parametrize("value, n_null", [(0.03, 20), (0.1, 4)])
def test_decimal_constant_supports_remain_exact_ties(value, n_null):
    result = triplet_residual_score(np.array([value]), np.full((n_null, 1), value))

    assert result["score"] == 0.0
    np.testing.assert_array_equal(result["null_scores"], 0.0)
    assert result["null_sd"] == 0.0
    assert result["z"] == 0.0
    assert result["p_upper"] == 1.0
    assert result["status"] == "zero_variance_tied"


def test_one_ulp_support_separation_is_not_masked_as_a_tie():
    baseline = 0.03
    observed = np.nextafter(baseline, np.inf)
    result = triplet_residual_score(np.array([observed]), np.full((20, 1), baseline))

    assert result["score"] == observed - baseline
    np.testing.assert_array_equal(result["null_scores"], 0.0)
    assert result["p_upper"] == pytest.approx(1 / 21)
    assert np.isnan(result["z"])
    assert result["status"] == "zero_variance_separated"
