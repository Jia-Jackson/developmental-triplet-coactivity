from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class PairwiseFit:
    h: np.ndarray
    j: np.ndarray
    converged: bool


@dataclass(frozen=True)
class NullComparison:
    observed: float
    null_mean: float
    null_sd: float
    z: float
    p_two_sided: float


def sample_independent(probabilities: np.ndarray, *, n_bins: int, seed: int) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    return (rng.random((probabilities.size, n_bins)) < probabilities[:, None]).astype(
        np.uint8
    )


def sample_pairwise(
    h: np.ndarray,
    j: np.ndarray,
    *,
    n_bins: int,
    burn_in: int = 1000,
    thin: int = 1,
    seed: int,
) -> np.ndarray:
    """Draw binary states by single-site Gibbs sampling."""
    h = np.asarray(h, dtype=float).reshape(-1)
    j = np.asarray(j, dtype=float)
    if j.shape != (h.size, h.size):
        raise ValueError("j must be square and match h")
    rng = np.random.default_rng(seed)
    n_nodes = h.size
    state = (rng.random(n_nodes) < expit(h)).astype(np.uint8)
    samples = np.zeros((n_nodes, n_bins), dtype=np.uint8)
    total_sweeps = burn_in + n_bins * thin
    out = 0
    for sweep in range(total_sweeps):
        for node in rng.permutation(n_nodes):
            logit = h[node] + float(j[node] @ state)
            state[node] = rng.random() < expit(logit)
        if sweep >= burn_in and (sweep - burn_in) % thin == 0:
            samples[:, out] = state
            out += 1
    return samples


def sample_pairwise_chains(
    h: np.ndarray,
    j: np.ndarray,
    *,
    n_chains: int,
    n_bins: int,
    burn_in_sweeps: int = 500,
    thin_sweeps: int = 1,
    seed: int,
    initial_states: np.ndarray | None = None,
) -> np.ndarray:
    """Draw several pairwise-model chains in parallel, returning chain×node×bin."""
    h = np.asarray(h, dtype=float).reshape(-1)
    j = np.asarray(j, dtype=float)
    if j.shape != (h.size, h.size):
        raise ValueError("j must be square and match h")
    if n_chains <= 0 or n_bins <= 0 or thin_sweeps <= 0 or burn_in_sweeps < 0:
        raise ValueError("invalid chain or sampling dimensions")
    rng = np.random.default_rng(seed)
    if initial_states is None:
        states = (rng.random((n_chains, h.size)) < expit(h)[None, :]).astype(np.uint8)
    else:
        states = (np.asarray(initial_states) > 0).astype(np.uint8)
        if states.shape != (n_chains, h.size):
            raise ValueError("initial_states must have shape (n_chains, n_nodes)")
    samples = np.empty((n_chains, h.size, n_bins), dtype=np.uint8)
    total_sweeps = burn_in_sweeps + n_bins * thin_sweeps
    output_bin = 0
    for sweep in range(total_sweeps):
        for node in rng.permutation(h.size):
            logits = h[node] + states @ j[node]
            states[:, node] = rng.random(n_chains) < expit(logits)
        if sweep >= burn_in_sweeps and (sweep - burn_in_sweeps) % thin_sweeps == 0:
            samples[:, :, output_bin] = states
            output_bin += 1
    return samples


def fit_pairwise_pseudolikelihood(
    x: np.ndarray,
    *,
    c: float = 1.0,
    max_iter: int = 1000,
) -> PairwiseFit:
    """Fit a symmetric pairwise binary model by node-wise pseudolikelihood."""
    data = (np.asarray(x) > 0).astype(np.uint8)
    if data.ndim != 2:
        raise ValueError("x must be channel-by-bin")
    n_nodes = data.shape[0]
    directed = np.zeros((n_nodes, n_nodes), dtype=float)
    h = np.zeros(n_nodes, dtype=float)
    converged = True
    for node in range(n_nodes):
        others = np.arange(n_nodes) != node
        y = data[node]
        if np.unique(y).size < 2:
            p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
            h[node] = np.log(p / (1 - p))
            continue
        model = LogisticRegression(
            C=c,
            solver="lbfgs",
            max_iter=max_iter,
            fit_intercept=True,
        )
        model.fit(data[others].T, y)
        h[node] = float(model.intercept_[0])
        directed[node, others] = model.coef_[0]
        converged &= int(model.n_iter_[0]) < max_iter
    j = 0.5 * (directed + directed.T)
    np.fill_diagonal(j, 0.0)
    return PairwiseFit(h=h, j=j, converged=bool(converged))


def fit_pairwise_joint_pseudolikelihood(
    x: np.ndarray,
    *,
    l2: float = 1e-5,
    max_iter: int = 300,
) -> PairwiseFit:
    """Fit one symmetric pairwise model by maximizing joint pseudolikelihood."""
    data = (np.asarray(x) > 0).astype(float).T
    if data.ndim != 2:
        raise ValueError("x must be channel-by-bin")
    n_bins, n_nodes = data.shape
    if n_nodes < 2 or n_bins < 2 or l2 < 0:
        raise ValueError("invalid data dimensions or regularization")
    pair_i, pair_j = np.triu_indices(n_nodes, 1)
    initial = fit_pairwise_pseudolikelihood(data.T, c=0.05, max_iter=max_iter)
    theta0 = np.concatenate([initial.h, initial.j[pair_i, pair_j]])

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        h = theta[:n_nodes]
        couplings = theta[n_nodes:]
        j = np.zeros((n_nodes, n_nodes), dtype=float)
        j[pair_i, pair_j] = couplings
        j[pair_j, pair_i] = couplings
        eta = data @ j + h
        loss = float(np.mean(np.logaddexp(0.0, eta) - data * eta))
        if couplings.size:
            loss += 0.5 * l2 * float(np.mean(couplings**2))
        residual = expit(eta) - data
        grad_h = residual.mean(axis=0) / n_nodes
        grad_pairs = (
            np.sum(residual[:, pair_i] * data[:, pair_j], axis=0)
            + np.sum(residual[:, pair_j] * data[:, pair_i], axis=0)
        ) / (n_bins * n_nodes)
        if couplings.size:
            grad_pairs += l2 * couplings / couplings.size
        return loss, np.concatenate([grad_h, grad_pairs])

    fitted = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iter), "ftol": 1e-10, "gtol": 1e-6},
    )
    values = fitted.x
    h = values[:n_nodes].copy()
    j = np.zeros((n_nodes, n_nodes), dtype=float)
    j[pair_i, pair_j] = values[n_nodes:]
    j[pair_j, pair_i] = values[n_nodes:]
    return PairwiseFit(h=h, j=j, converged=bool(fitted.success))


def sample_pairwise_conditioned_on_k(
    h: np.ndarray,
    j: np.ndarray,
    k_sequence: np.ndarray,
    *,
    sweeps_per_bin: int = 20,
    seed: int,
) -> np.ndarray:
    """Sample states while exactly preserving each observed population count.

    Each bin is sampled with active/inactive swap proposals under the fitted
    pairwise energy. This is a conservative population-count-conditioned null.
    """
    h = np.asarray(h, dtype=float).reshape(-1)
    j = np.asarray(j, dtype=float)
    counts = np.asarray(k_sequence, dtype=int).reshape(-1)
    n_nodes = h.size
    if j.shape != (n_nodes, n_nodes):
        raise ValueError("j must be square and match h")
    if np.any((counts < 0) | (counts > n_nodes)):
        raise ValueError("k_sequence contains invalid population counts")
    rng = np.random.default_rng(seed)
    output = np.zeros((n_nodes, counts.size), dtype=np.uint8)
    for col, k in enumerate(counts):
        if k == 0:
            continue
        if k == n_nodes:
            output[:, col] = 1
            continue
        state = np.zeros(n_nodes, dtype=np.uint8)
        state[rng.choice(n_nodes, size=k, replace=False)] = 1
        for _ in range(max(1, sweeps_per_bin) * n_nodes):
            active = np.flatnonzero(state)
            inactive = np.flatnonzero(1 - state)
            remove = int(rng.choice(active))
            add = int(rng.choice(inactive))
            delta = (
                h[add]
                - h[remove]
                + float(j[add] @ state)
                - float(j[remove] @ state)
                - j[add, remove]
            )
            if delta >= 0 or np.log(rng.random()) < delta:
                state[remove] = 0
                state[add] = 1
        output[:, col] = state
    return output


def resample_pool_conditioned_on_k(
    sample_pool: np.ndarray,
    k_sequence: np.ndarray,
    *,
    n_surrogates: int,
    seed: int,
    fallback_h: np.ndarray | None = None,
    fallback_j: np.ndarray | None = None,
    fallback_sweeps: int = 20,
) -> np.ndarray:
    """Resample pairwise-model states conditional on the exact observed K(t).

    A sufficiently large pool drawn from ``P2(x)`` is partitioned by population
    count. Drawing from the corresponding partition approximates ``P2(x|K)``
    and preserves every requested population count exactly.
    """
    pool = (np.asarray(sample_pool) > 0).astype(np.uint8)
    desired = np.asarray(k_sequence, dtype=int).reshape(-1)
    if pool.ndim != 2:
        raise ValueError("sample_pool must be node-by-sample")
    if n_surrogates <= 0:
        raise ValueError("n_surrogates must be positive")
    pool_k = pool.sum(axis=0)
    buckets = {int(k): np.flatnonzero(pool_k == k) for k in np.unique(pool_k)}
    missing = sorted(set(int(k) for k in np.unique(desired)) - set(buckets))
    if missing and (fallback_h is None or fallback_j is None):
        raise ValueError(f"sample pool has no states for population counts {missing}")
    rng = np.random.default_rng(seed)
    output = np.empty((n_surrogates, pool.shape[0], desired.size), dtype=np.uint8)
    for k in sorted(set(int(k) for k in np.unique(desired)) & set(buckets)):
        positions = np.flatnonzero(desired == k)
        chosen = rng.choice(buckets[int(k)], size=(n_surrogates, positions.size), replace=True)
        for surrogate in range(n_surrogates):
            output[surrogate][:, positions] = pool[:, chosen[surrogate]]
    for k in missing:
        positions = np.flatnonzero(desired == k)
        missing_sequence = np.full(positions.size, k, dtype=int)
        for surrogate in range(n_surrogates):
            fallback = sample_pairwise_conditioned_on_k(
                np.asarray(fallback_h),
                np.asarray(fallback_j),
                missing_sequence,
                sweeps_per_bin=fallback_sweeps,
                seed=seed + 100000 + 1000 * surrogate + k,
            )
            output[surrogate][:, positions] = fallback
    return output


def decode_ising_factors(factors: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode toolbox energy factors into log-potential fields/couplings.

    ``IsingEnergy::getEnergy`` returns ``sum(lambda*f)``, while the native
    toolbox probability and sampler use ``exp(-energy)``.  The Python samplers
    consume positive log-potential coefficients, hence the sign inversion.
    """
    values = -np.asarray(factors, dtype=float).reshape(-1)
    expected = n_nodes * (n_nodes + 1) // 2
    if values.size != expected:
        raise ValueError(f"expected {expected} factors, found {values.size}")
    h = values[:n_nodes].copy()
    j = np.zeros((n_nodes, n_nodes), dtype=float)
    index = n_nodes
    for first in range(n_nodes):
        for second in range(first + 1, n_nodes):
            j[first, second] = values[index]
            j[second, first] = values[index]
            index += 1
    return h, j


def standardize_against_null(observed: float, null_values: np.ndarray) -> NullComparison:
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("at least two finite null values are required")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    if sd == 0:
        z = 0.0 if observed == mean else float(np.sign(observed - mean) * 1e12)
    else:
        z = float((observed - mean) / sd)
    centered_obs = abs(observed - mean)
    p = float((1 + np.sum(np.abs(values - mean) >= centered_obs)) / (values.size + 1))
    return NullComparison(
        observed=float(observed),
        null_mean=mean,
        null_sd=sd,
        z=z,
        p_two_sided=p,
    )


def irreducible_triplet_summary(
    observed_counts: np.ndarray,
    null_block_counts: np.ndarray,
    *,
    n_bins: int,
) -> dict[str, object]:
    """Summarize triplet residuals with a leave-one-null-block calibration."""
    observed = np.asarray(observed_counts, dtype=float).reshape(-1)
    null = np.asarray(null_block_counts, dtype=float)
    if null.ndim != 2 or null.shape[1] != observed.size:
        raise ValueError("null_block_counts must be block-by-triplet")
    if null.shape[0] < 3 or n_bins <= 0:
        raise ValueError("at least three null blocks and positive n_bins are required")
    expected = null.mean(axis=0)
    residual = (observed - expected) / n_bins
    positive = float(np.mean(np.clip(residual, 0.0, None)))
    negative = float(np.mean(np.clip(-residual, 0.0, None)))
    absolute = float(np.mean(np.abs(residual)))
    rms = float(np.sqrt(np.mean(residual**2)))

    total = null.sum(axis=0)
    loo_positive = np.empty(null.shape[0], dtype=float)
    loo_negative = np.empty(null.shape[0], dtype=float)
    loo_absolute = np.empty(null.shape[0], dtype=float)
    for block in range(null.shape[0]):
        loo_mean = (total - null[block]) / (null.shape[0] - 1)
        block_residual = (null[block] - loo_mean) / n_bins
        loo_positive[block] = np.mean(np.clip(block_residual, 0.0, None))
        loo_negative[block] = np.mean(np.clip(-block_residual, 0.0, None))
        loo_absolute[block] = np.mean(np.abs(block_residual))
    pos_cmp = standardize_against_null(positive, loo_positive)
    neg_cmp = standardize_against_null(negative, loo_negative)
    abs_cmp = standardize_against_null(absolute, loo_absolute)
    return {
        "expected_counts": expected,
        "residual_probability": residual,
        "positive_residual_mass": positive,
        "negative_residual_mass": negative,
        "absolute_residual_mass": absolute,
        "rms_residual": rms,
        "positive_residual_z": pos_cmp.z,
        "negative_residual_z": neg_cmp.z,
        "absolute_residual_z": abs_cmp.z,
        "positive_residual_p": pos_cmp.p_two_sided,
        "negative_residual_p": neg_cmp.p_two_sided,
        "absolute_residual_p": abs_cmp.p_two_sided,
        "null_positive_residual_mass": loo_positive,
        "null_negative_residual_mass": loo_negative,
        "null_absolute_residual_mass": loo_absolute,
    }
