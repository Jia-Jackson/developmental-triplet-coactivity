"""Exact finite-state pairwise maximum-entropy benchmark for small binary systems.

The model uses the positive log potential ``theta @ features(x)``.  In the
conditional model, probabilities are normalized separately within each
population count K and then mixed using the supplied K weights.  Conditioning
introduces gauge nonidentifiability: common shifts of all node fields and of
all pair couplings are constant within each K stratum.  No regularizer or
triplet term is added to resolve that gauge because neither changes the fitted
conditional distribution.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def _validate_binary_x(x: np.ndarray, *, n: int | None = None) -> np.ndarray:
    data = np.asarray(x)
    if data.ndim != 2:
        raise ValueError("x must have shape N x T")
    if data.shape[0] < 1 or data.shape[1] < 1:
        raise ValueError("x must be nonempty")
    if n is not None and data.shape[0] != n:
        raise ValueError(f"x must have {n} rows")
    if not np.issubdtype(data.dtype, np.number) or not np.all(np.isfinite(data)):
        raise ValueError("x must contain finite binary values")
    if not np.all((data == 0) | (data == 1)):
        raise ValueError("x must contain only 0 and 1")
    return data.astype(np.uint8, copy=False)


class ExactPairwise:
    """Enumerate and fit a binary pairwise model for 1 <= N <= 16."""

    def __init__(self, n: int):
        if not isinstance(n, (int, np.integer)) or not 1 <= int(n) <= 16:
            raise ValueError("n must be an integer in [1, 16]")
        self.n = int(n)
        self.n_states = 1 << self.n
        self.pairs = tuple(combinations(range(self.n), 2))
        self.triplets = tuple(combinations(range(self.n), 3))
        state_ids = np.arange(self.n_states, dtype=np.uint32)
        self.states = ((state_ids[:, None] >> np.arange(self.n)) & 1).astype(np.uint8)
        self.counts = self.states.sum(axis=1, dtype=np.int16)
        pair_features = (
            np.column_stack([self.states[:, i] * self.states[:, j] for i, j in self.pairs])
            if self.pairs
            else np.empty((self.n_states, 0), dtype=np.uint8)
        )
        self.features = np.column_stack((self.states, pair_features)).astype(np.float64)
        self.n_features = self.features.shape[1]

    def _theta(self, theta: np.ndarray) -> np.ndarray:
        values = np.asarray(theta, dtype=np.float64)
        if values.shape != (self.n_features,) or not np.all(np.isfinite(values)):
            raise ValueError(f"theta must be a finite vector of length {self.n_features}")
        return values

    def _weights(self, k_weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(k_weights, dtype=np.float64)
        if weights.shape != (self.n + 1,) or not np.all(np.isfinite(weights)):
            raise ValueError(f"k_weights must be a finite vector of length {self.n + 1}")
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("k_weights must be nonnegative with positive total mass")
        return weights / weights.sum()

    def probabilities(self, theta: np.ndarray, k_weights: np.ndarray | None = None) -> np.ndarray:
        values = self._theta(theta)
        log_potential = self.features @ values
        if k_weights is None:
            return np.exp(log_potential - logsumexp(log_potential))
        weights = self._weights(k_weights)
        probability = np.zeros(self.n_states, dtype=np.float64)
        for k, weight in enumerate(weights):
            if weight == 0:
                continue
            selected = self.counts == k
            probability[selected] = weight * np.exp(
                log_potential[selected] - logsumexp(log_potential[selected])
            )
        probability /= probability.sum()
        return probability

    def moments(self, prob: np.ndarray) -> np.ndarray:
        probability = self._probability(prob)
        return probability @ self.features

    def _probability(self, prob: np.ndarray) -> np.ndarray:
        probability = np.asarray(prob, dtype=np.float64)
        if probability.shape != (self.n_states,) or not np.all(np.isfinite(probability)):
            raise ValueError(f"prob must be a finite vector of length {self.n_states}")
        if np.any(probability < 0) or not np.isclose(probability.sum(), 1.0, atol=1e-12):
            raise ValueError("prob must be nonnegative and sum to one")
        return probability

    def triplet_support(self, prob: np.ndarray) -> np.ndarray:
        """Return P(X_i=X_j=X_k=1) via an in-place superset zeta transform."""
        superset_mass = self._probability(prob).copy()
        state_ids = np.arange(self.n_states, dtype=np.uint32)
        for bit in range(self.n):
            absent = (state_ids & (1 << bit)) == 0
            superset_mass[absent] += superset_mass[state_ids[absent] | (1 << bit)]
        masks = np.fromiter(
            ((1 << i) | (1 << j) | (1 << k) for i, j, k in self.triplets),
            dtype=np.uint32,
            count=len(self.triplets),
        )
        return superset_mass[masks].copy()

    def loss_grad(
        self,
        theta: np.ndarray,
        target_moments: np.ndarray,
        k_weights: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        values = self._theta(theta)
        target = np.asarray(target_moments, dtype=np.float64)
        if target.shape != (self.n_features,) or not np.all(np.isfinite(target)):
            raise ValueError(f"target_moments must be finite and length {self.n_features}")
        log_potential = self.features @ values
        if k_weights is None:
            log_partition = float(logsumexp(log_potential))
            probability = np.exp(log_potential - log_partition)
        else:
            weights = self._weights(k_weights)
            probability = np.zeros(self.n_states, dtype=np.float64)
            log_partition = 0.0
            for k, weight in enumerate(weights):
                if weight == 0:
                    continue
                selected = self.counts == k
                log_z_k = float(logsumexp(log_potential[selected]))
                log_partition += weight * log_z_k
                probability[selected] = weight * np.exp(log_potential[selected] - log_z_k)
        model_moments = probability @ self.features
        loss = log_partition - float(values @ target)
        return float(loss), model_moments - target

    def fit(
        self,
        x: np.ndarray,
        conditional: bool = False,
        theta0: np.ndarray | None = None,
        gtol: float = 1e-7,
        maxiter: int = 2000,
    ) -> dict[str, object]:
        data = _validate_binary_x(x, n=self.n)
        if not np.isfinite(gtol) or gtol <= 0 or maxiter <= 0:
            raise ValueError("gtol and maxiter must be positive")
        samples = data.T.astype(np.float64)
        pair_columns = (
            np.column_stack([samples[:, i] * samples[:, j] for i, j in self.pairs])
            if self.pairs
            else np.empty((samples.shape[0], 0), dtype=np.float64)
        )
        target = np.column_stack((samples, pair_columns)).mean(axis=0)
        weights = None
        if conditional:
            weights = np.bincount(data.sum(axis=0), minlength=self.n + 1).astype(np.float64)
            weights /= data.shape[1]
        initial = np.zeros(self.n_features, dtype=np.float64) if theta0 is None else self._theta(theta0).copy()
        result = minimize(
            lambda theta: self.loss_grad(theta, target, weights),
            initial,
            method="L-BFGS-B",
            jac=True,
            options={"gtol": float(gtol), "maxiter": int(maxiter), "ftol": 1e-15},
        )
        objective, gradient = self.loss_grad(result.x, target, weights)
        max_abs_gradient = float(np.max(np.abs(gradient)))
        gradient_ok = max_abs_gradient <= float(gtol)
        success = bool(result.success and gradient_ok)
        message = str(result.message)
        if not gradient_ok:
            message += f"; gradient tolerance not met ({max_abs_gradient:.3g} > {gtol:.3g})"
        fitted_moments = self.moments(self.probabilities(result.x, weights))
        node_error = fitted_moments[: self.n] - target[: self.n]
        pair_error = fitted_moments[self.n :] - target[self.n :]
        return {
            "theta": np.asarray(result.x, dtype=np.float64),
            "success": success,
            "nit": int(result.nit),
            "message": message,
            "max_abs_gradient": max_abs_gradient,
            "firing_rmse": float(np.sqrt(np.mean(node_error**2))),
            "pair_rmse": float(np.sqrt(np.mean(pair_error**2))) if pair_error.size else 0.0,
            "objective": float(objective),
        }

    def simulate(
        self,
        theta: np.ndarray,
        k_sequence: np.ndarray,
        n_realizations: int,
        seed: int,
    ) -> np.ndarray:
        values = self._theta(theta)
        raw_counts = np.asarray(k_sequence)
        if raw_counts.ndim != 1 or raw_counts.size == 0 or not np.all(np.isfinite(raw_counts)):
            raise ValueError("k_sequence must be a finite nonempty vector")
        if not np.all(raw_counts == np.floor(raw_counts)):
            raise ValueError("k_sequence must contain integers")
        counts = raw_counts.astype(int)
        if np.any((counts < 0) | (counts > self.n)):
            raise ValueError(f"k_sequence values must lie in [0, {self.n}]")
        if not isinstance(n_realizations, (int, np.integer)) or n_realizations <= 0:
            raise ValueError("n_realizations must be a positive integer")
        log_potential = self.features @ values
        rng = np.random.default_rng(seed)
        output = np.empty((int(n_realizations), self.n, counts.size), dtype=np.uint8)
        for k in np.unique(counts):
            state_indices = np.flatnonzero(self.counts == k)
            conditional_probability = np.exp(
                log_potential[state_indices] - logsumexp(log_potential[state_indices])
            )
            positions = np.flatnonzero(counts == k)
            draws = rng.choice(
                state_indices,
                size=(int(n_realizations), positions.size),
                p=conditional_probability,
            )
            output[:, :, positions] = self.states[draws].transpose(0, 2, 1)
        return output


def empirical_triplet_support(x: np.ndarray) -> np.ndarray:
    data = _validate_binary_x(x)
    triplets = tuple(combinations(range(data.shape[0]), 3))
    return np.fromiter(
        (np.mean(data[i] * data[j] * data[k]) for i, j, k in triplets),
        dtype=np.float64,
        count=len(triplets),
    )


def triplet_residual_score(observed_support: np.ndarray, null_supports: np.ndarray) -> dict[str, object]:
    """Calibrate mean absolute triplet residual using leave-one-null-out scores."""
    observed = np.asarray(observed_support, dtype=np.float64)
    null = np.asarray(null_supports, dtype=np.float64)
    if observed.ndim != 1 or observed.size == 0 or not np.all(np.isfinite(observed)):
        raise ValueError("observed_support must be a finite nonempty vector")
    if null.ndim != 2 or null.shape[1] != observed.size or null.shape[0] < 3:
        raise ValueError("null_supports must have at least three rows matching observed_support")
    if not np.all(np.isfinite(null)):
        raise ValueError("null_supports must be finite")
    # Center before averaging so a constant decimal-valued ensemble has an
    # exactly zero residual.  The LOO identity below uses the same centered
    # arithmetic for every null realization, avoiding a separately rounded
    # sum/subtract path without imposing a tolerance on genuine differences.
    anchor = null[0]
    centered_null = null - anchor
    centered_mean = centered_null.mean(axis=0)
    observed_residual = (observed - anchor) - centered_mean
    score = float(np.mean(np.abs(observed_residual)))
    loo_residuals = (null.shape[0] / (null.shape[0] - 1)) * (
        centered_null - centered_mean
    )
    null_scores = np.mean(np.abs(loo_residuals), axis=1)
    null_mean = float(null_scores.mean())
    null_sd = float(null_scores.std(ddof=1))
    p_upper = float((1 + np.count_nonzero(null_scores >= score)) / (null_scores.size + 1))
    if null_sd == 0.0:
        zero_variance = True
        if score == null_mean:
            z = 0.0
            status = "zero_variance_tied"
        else:
            z = float("nan")
            status = "zero_variance_separated"
    else:
        zero_variance = False
        z = float((score - null_mean) / null_sd)
        status = "ok"
    return {
        "score": score,
        "absolute_residual_mass": score,
        "null_scores": null_scores,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "z": z,
        "p_upper": p_upper,
        "zero_variance": zero_variance,
        "status": status,
    }
