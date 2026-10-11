from __future__ import annotations

from itertools import combinations

import numpy as np


def support_counts_from_binary(
    x: np.ndarray, triplets: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dense first-, second-, and third-order counts using Python bitsets."""
    data = (np.asarray(x) > 0).astype(np.uint8)
    if data.ndim != 2 or data.shape[0] < 3:
        raise ValueError("x must be node-by-time with at least three nodes")
    n_nodes = data.shape[0]
    pairs = np.asarray(list(combinations(range(n_nodes), 2)), dtype=int)
    if triplets is None:
        triplets = np.asarray(list(combinations(range(n_nodes), 3)), dtype=int)
    else:
        triplets = np.asarray(triplets, dtype=int)
    packed = np.packbits(data, axis=1, bitorder="little")
    bits = [int.from_bytes(row.tobytes(), byteorder="little") for row in packed]
    singles = np.asarray([value.bit_count() for value in bits], dtype=np.int64)
    pair_counts = np.asarray([(bits[i] & bits[j]).bit_count() for i, j in pairs], dtype=np.int64)
    triplet_counts = np.asarray(
        [(bits[i] & bits[j] & bits[k]).bit_count() for i, j, k in triplets], dtype=np.int64
    )
    return singles, pairs, pair_counts, triplets, triplet_counts


def o_information_from_support_counts(
    single_counts: np.ndarray,
    pairs: np.ndarray,
    pair_counts: np.ndarray,
    triplets: np.ndarray,
    triplet_counts: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    singles = np.asarray(single_counts, dtype=float) / float(n_bins)
    pairs = np.asarray(pairs, dtype=int)
    pair_counts = np.asarray(pair_counts, dtype=float) / float(n_bins)
    triplets = np.asarray(triplets, dtype=int)
    triplet_counts = np.asarray(triplet_counts, dtype=float) / float(n_bins)
    pair_lookup = {tuple(pair): value for pair, value in zip(pairs, pair_counts)}
    i, j, k = triplets.T
    pij = np.asarray([pair_lookup[(int(a), int(b))] for a, b in triplets[:, [0, 1]]])
    pik = np.asarray([pair_lookup[(int(a), int(b))] for a, b in triplets[:, [0, 2]]])
    pjk = np.asarray([pair_lookup[(int(a), int(b))] for a, b in triplets[:, [1, 2]]])
    return o_information_from_moments(
        singles[i], singles[j], singles[k], pij, pik, pjk, triplet_counts
    )


def entropy(probabilities: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy in bits, with the convention 0 log(0) = 0."""
    p = np.asarray(probabilities, dtype=float)
    if np.any(p < -1e-10) or np.any(p > 1 + 1e-10):
        raise ValueError("probabilities lie outside [0, 1]")
    p = np.clip(p, 0.0, 1.0)
    terms = np.zeros_like(p)
    positive = p > 0
    terms[positive] = -p[positive] * np.log2(p[positive])
    return terms.sum(axis=axis)


def binary_entropy(probability_one: np.ndarray) -> np.ndarray:
    p = np.asarray(probability_one, dtype=float)
    return entropy(np.stack([1.0 - p, p], axis=-1), axis=-1)


def pair_state_probabilities(pi: np.ndarray, pj: np.ndarray, pij: np.ndarray) -> np.ndarray:
    pi, pj, pij = np.broadcast_arrays(
        np.asarray(pi, dtype=float), np.asarray(pj, dtype=float), np.asarray(pij, dtype=float)
    )
    values = np.stack([1.0 - pi - pj + pij, pj - pij, pi - pij, pij], axis=-1)
    if np.any(values < -1e-10):
        raise ValueError("inconsistent first- and second-order moments")
    values = np.clip(values, 0.0, 1.0)
    values /= values.sum(axis=-1, keepdims=True)
    return values


def triplet_state_probabilities(
    pi: np.ndarray,
    pj: np.ndarray,
    pk: np.ndarray,
    pij: np.ndarray,
    pik: np.ndarray,
    pjk: np.ndarray,
    pijk: np.ndarray,
) -> np.ndarray:
    """Return probabilities in order 000,001,010,011,100,101,110,111."""
    pi, pj, pk, pij, pik, pjk, pijk = np.broadcast_arrays(
        *[np.asarray(v, dtype=float) for v in (pi, pj, pk, pij, pik, pjk, pijk)]
    )
    values = np.stack(
        [
            1.0 - pi - pj - pk + pij + pik + pjk - pijk,
            pk - pik - pjk + pijk,
            pj - pij - pjk + pijk,
            pjk - pijk,
            pi - pij - pik + pijk,
            pik - pijk,
            pij - pijk,
            pijk,
        ],
        axis=-1,
    )
    if np.any(values < -1e-10) or np.any(values > 1 + 1e-10):
        minimum = float(values.min())
        maximum = float(values.max())
        raise ValueError(f"inconsistent triplet moments: range {minimum:g} to {maximum:g}")
    values = np.clip(values, 0.0, 1.0)
    normalizer = values.sum(axis=-1, keepdims=True)
    if np.any(normalizer <= 0):
        raise ValueError("triplet state probabilities have zero total mass")
    return values / normalizer


def o_information_from_moments(
    pi: np.ndarray,
    pj: np.ndarray,
    pk: np.ndarray,
    pij: np.ndarray,
    pik: np.ndarray,
    pjk: np.ndarray,
    pijk: np.ndarray,
) -> np.ndarray:
    """Third-order O-information in bits.

    Omega_3 = H(X_i)+H(X_j)+H(X_k)-H(X_i,X_j)-H(X_i,X_k)
              -H(X_j,X_k)+H(X_i,X_j,X_k).
    Positive values are redundancy dominated and negative values are synergy
    dominated in the information-theoretic sense.
    """
    h_single = binary_entropy(pi) + binary_entropy(pj) + binary_entropy(pk)
    h_pair = (
        entropy(pair_state_probabilities(pi, pj, pij))
        + entropy(pair_state_probabilities(pi, pk, pik))
        + entropy(pair_state_probabilities(pj, pk, pjk))
    )
    h_triplet = entropy(triplet_state_probabilities(pi, pj, pk, pij, pik, pjk, pijk))
    return h_single - h_pair + h_triplet


def o_information_from_binary(x: np.ndarray) -> np.ndarray:
    data = (np.asarray(x) > 0).astype(np.uint8)
    if data.ndim != 2 or data.shape[0] < 3 or data.shape[1] == 0:
        raise ValueError("x must be node-by-time with at least three nodes")
    singles, pairs, pair_counts, triplets, triplet_counts = support_counts_from_binary(data)
    return o_information_from_support_counts(
        singles, pairs, pair_counts, triplets, triplet_counts, data.shape[1]
    )


def o_information_many_binary(simulations: np.ndarray, triplets: np.ndarray | None = None) -> np.ndarray:
    """O-information for realization-by-node-by-time binary arrays."""
    data = (np.asarray(simulations) > 0).astype(np.uint8)
    if data.ndim != 3:
        raise ValueError("simulations must be realization-by-node-by-time")
    n_realizations, n_nodes, n_bins = data.shape
    if triplets is None:
        triplets = np.asarray(list(combinations(range(n_nodes), 3)), dtype=int)
    else:
        triplets = np.asarray(triplets, dtype=int)
    output = np.empty((n_realizations, len(triplets)), dtype=float)
    for realization in range(n_realizations):
        singles, pairs, pair_counts, _, triplet_counts = support_counts_from_binary(
            data[realization], triplets
        )
        output[realization] = o_information_from_support_counts(
            singles, pairs, pair_counts, triplets, triplet_counts, n_bins
        )
    return output


def summarize_o_information(values: np.ndarray, axis: int = -1) -> dict[str, np.ndarray]:
    omega = np.asarray(values, dtype=float)
    return {
        "oinfo_mean": np.mean(omega, axis=axis),
        "oinfo_positive_mass": np.mean(np.clip(omega, 0.0, None), axis=axis),
        "oinfo_negative_mass": np.mean(np.clip(-omega, 0.0, None), axis=axis),
        "oinfo_redundant_fraction": np.mean(omega > 0, axis=axis),
    }
