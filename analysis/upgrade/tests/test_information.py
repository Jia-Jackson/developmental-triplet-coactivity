import numpy as np

from higher_order_v2.information import (
    binary_entropy,
    o_information_from_binary,
    o_information_many_binary,
    triplet_state_probabilities,
)


def _all_binary_states(n_nodes: int) -> np.ndarray:
    values = np.arange(2**n_nodes, dtype=np.uint16)
    return ((values[:, None] >> np.arange(n_nodes)) & 1).T.astype(np.uint8)


def test_independent_uniform_triplet_has_zero_o_information():
    states = _all_binary_states(3)
    omega = o_information_from_binary(states)
    np.testing.assert_allclose(omega, [0.0], atol=1e-12)


def test_xor_triplet_is_synergy_dominated():
    x = np.array([0, 0, 1, 1], dtype=np.uint8)
    y = np.array([0, 1, 0, 1], dtype=np.uint8)
    z = x ^ y
    omega = o_information_from_binary(np.vstack([x, y, z]))
    np.testing.assert_allclose(omega, [-1.0], atol=1e-12)


def test_three_redundant_copies_are_redundancy_dominated():
    x = np.array([0, 1] * 4, dtype=np.uint8)
    omega = o_information_from_binary(np.vstack([x, x, x]))
    np.testing.assert_allclose(omega, [1.0], atol=1e-12)


def test_triplet_state_probabilities_are_valid():
    p = triplet_state_probabilities(
        np.array([0.5]), np.array([0.5]), np.array([0.5]),
        np.array([0.25]), np.array([0.25]), np.array([0.25]),
        np.array([0.125]),
    )
    assert p.shape == (1, 8)
    np.testing.assert_allclose(p, np.full((1, 8), 0.125), atol=1e-12)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)


def test_binary_entropy_handles_boundaries():
    np.testing.assert_allclose(binary_entropy(np.array([0.0, 1.0])), [0.0, 0.0])


def test_many_realizations_matches_one_at_a_time():
    rng = np.random.default_rng(3)
    x = (rng.random((4, 5, 100)) < 0.25).astype(np.uint8)
    expected = np.vstack([o_information_from_binary(item) for item in x])
    np.testing.assert_allclose(o_information_many_binary(x), expected, atol=1e-12)
