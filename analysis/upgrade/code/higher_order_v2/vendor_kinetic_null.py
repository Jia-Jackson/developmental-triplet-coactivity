from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class KineticFit:
    intercepts: np.ndarray
    weights: np.ndarray
    selected_c: np.ndarray
    heldout_log_loss: float
    rate_only_log_loss: float
    n_splits: int
    n_bins: int


def sample_conditional_bernoulli_fixed_k(
    logits: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exact conditional-Bernoulli samples with a fixed number of successes.

    For independent Bernoulli odds w_i, conditioning on sum(x)=k gives
    P(S) proportional to the product of w_i over the selected subset S. The
    elementary-symmetric-polynomial recursion samples this distribution exactly.
    """
    values = np.asarray(logits, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError("logits must be one- or two-dimensional")
    n_rows, n_nodes = values.shape
    k = int(k)
    if k < 0 or k > n_nodes:
        raise ValueError(f"k must lie in [0, {n_nodes}]")
    if k == 0:
        return np.zeros((n_rows, n_nodes), dtype=np.uint8)
    if k == n_nodes:
        return np.ones((n_rows, n_nodes), dtype=np.uint8)

    shifted = values - np.max(values, axis=1, keepdims=True)
    odds = np.exp(np.clip(shifted, -700, 0))
    suffix = np.zeros((n_rows, n_nodes + 1, k + 1), dtype=float)
    suffix[:, n_nodes, 0] = 1.0
    for node in range(n_nodes - 1, -1, -1):
        suffix[:, node, 0] = 1.0
        maximum_order = min(k, n_nodes - node)
        for order in range(1, maximum_order + 1):
            suffix[:, node, order] = (
                suffix[:, node + 1, order]
                + odds[:, node] * suffix[:, node + 1, order - 1]
            )

    samples = np.zeros((n_rows, n_nodes), dtype=np.uint8)
    remaining = np.full(n_rows, k, dtype=int)
    uniforms = rng.random((n_nodes, n_rows))
    row_index = np.arange(n_rows)
    for node in range(n_nodes):
        active = remaining > 0
        if not np.any(active):
            break
        forced = remaining >= (n_nodes - node)
        denominator = suffix[row_index, node, np.minimum(remaining, k)]
        numerator = np.zeros(n_rows, dtype=float)
        valid = active & ~forced
        numerator[valid] = (
            odds[valid, node]
            * suffix[valid, node + 1, remaining[valid] - 1]
        )
        probability = np.zeros(n_rows, dtype=float)
        probability[forced] = 1.0
        probability[valid] = np.divide(
            numerator[valid],
            denominator[valid],
            out=np.zeros(np.sum(valid), dtype=float),
            where=denominator[valid] > 0,
        )
        choose = active & (uniforms[node] < np.clip(probability, 0.0, 1.0))
        samples[choose, node] = 1
        remaining[choose] -= 1
    if np.any(remaining != 0):
        raise RuntimeError("conditional sampler failed to allocate exactly k successes")
    return samples


def simulate_kinetic_conditioned_k(
    intercepts: np.ndarray,
    weights: np.ndarray,
    k_sequence: np.ndarray,
    *,
    initial_state: np.ndarray,
    n_realizations: int,
    seed: int,
) -> np.ndarray:
    intercepts = np.asarray(intercepts, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float)
    k_sequence = np.asarray(k_sequence, dtype=int).reshape(-1)
    initial = np.asarray(initial_state, dtype=np.uint8).reshape(-1)
    n_nodes = len(intercepts)
    if weights.shape != (n_nodes, n_nodes):
        raise ValueError("weights must be n_nodes by n_nodes")
    if len(initial) != n_nodes:
        raise ValueError("initial_state has the wrong length")
    if len(k_sequence) == 0:
        raise ValueError("k_sequence is empty")
    rng = np.random.default_rng(seed)
    samples = np.zeros((int(n_realizations), n_nodes, len(k_sequence)), dtype=np.uint8)
    if int(initial.sum()) == int(k_sequence[0]):
        samples[:, :, 0] = initial[None, :]
    else:
        samples[:, :, 0] = sample_conditional_bernoulli_fixed_k(
            np.repeat(intercepts[None, :], int(n_realizations), axis=0),
            int(k_sequence[0]),
            rng,
        )
    previous = samples[:, :, 0]
    for time_index in range(1, len(k_sequence)):
        logits = intercepts[None, :] + previous @ weights.T
        current = sample_conditional_bernoulli_fixed_k(
            logits, int(k_sequence[time_index]), rng
        )
        samples[:, :, time_index] = current
        previous = current
    return samples


def _safe_log_loss(y_true: np.ndarray, probability: np.ndarray) -> float:
    return float(log_loss(y_true, np.clip(probability, 1e-8, 1 - 1e-8), labels=[0, 1]))


def fit_kinetic_pairwise(
    x: np.ndarray,
    *,
    c_grid: Sequence[float] = (0.01, 0.1, 1.0),
    n_splits: int = 3,
    max_iter: int = 500,
) -> KineticFit:
    data = (np.asarray(x) > 0).astype(np.uint8)
    if data.ndim != 2 or data.shape[1] < 50:
        raise ValueError("x must be node-by-time with at least 50 bins")
    predictors = data[:, :-1].T.astype(float)
    responses = data[:, 1:].T.astype(np.uint8)
    splitter = TimeSeriesSplit(n_splits=int(n_splits))
    splits = list(splitter.split(predictors))
    n_nodes = data.shape[0]
    intercepts = np.zeros(n_nodes, dtype=float)
    weights = np.zeros((n_nodes, n_nodes), dtype=float)
    selected_c = np.zeros(n_nodes, dtype=float)
    kinetic_losses = []
    rate_losses = []

    for node in range(n_nodes):
        target = responses[:, node]
        c_scores = {}
        for c_value in c_grid:
            fold_losses = []
            for train, test in splits:
                if np.unique(target[train]).size < 2:
                    rate = float(np.mean(target[train]))
                    probability = np.full(len(test), rate)
                else:
                    model = LogisticRegression(
                        C=float(c_value),
                        solver="liblinear",
                        max_iter=int(max_iter),
                        random_state=0,
                    )
                    model.fit(predictors[train], target[train])
                    probability = model.predict_proba(predictors[test])[:, 1]
                fold_losses.append(_safe_log_loss(target[test], probability))
            c_scores[float(c_value)] = float(np.mean(fold_losses))
        best_c = min(c_scores, key=lambda value: (c_scores[value], value))
        selected_c[node] = best_c
        kinetic_losses.append(c_scores[best_c])
        node_rate_losses = []
        for train, test in splits:
            rate = float(np.clip(np.mean(target[train]), 1e-8, 1 - 1e-8))
            node_rate_losses.append(
                _safe_log_loss(target[test], np.full(len(test), rate))
            )
        rate_losses.append(float(np.mean(node_rate_losses)))

        final = LogisticRegression(
            C=best_c,
            solver="liblinear",
            max_iter=int(max_iter),
            random_state=0,
        )
        final.fit(predictors, target)
        intercepts[node] = float(final.intercept_[0])
        weights[node] = final.coef_[0]

    return KineticFit(
        intercepts=intercepts,
        weights=weights,
        selected_c=selected_c,
        heldout_log_loss=float(np.mean(kinetic_losses)),
        rate_only_log_loss=float(np.mean(rate_losses)),
        n_splits=int(n_splits),
        n_bins=int(data.shape[1]),
    )


def lag1_moment_rmse(observed: np.ndarray, simulated: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    simulated = np.asarray(simulated, dtype=float)
    observed_moment = observed[:, 1:] @ observed[:, :-1].T / (observed.shape[1] - 1)
    simulated_moment = simulated[:, 1:] @ simulated[:, :-1].T / (simulated.shape[1] - 1)
    return float(np.sqrt(np.mean(np.square(observed_moment - simulated_moment))))
