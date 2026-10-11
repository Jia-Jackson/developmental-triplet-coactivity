from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    value = spearmanr(np.asarray(first, dtype=float), np.asarray(second, dtype=float)).statistic
    return float(value) if np.isfinite(value) else np.nan


def rank_similarity_matrix(profiles: np.ndarray) -> np.ndarray:
    """Pairwise Spearman correlations after ranking each profile once."""
    values = np.asarray(profiles, dtype=float)
    if values.ndim != 2:
        raise ValueError("profiles must be recording-by-feature")
    if not np.isfinite(values).all():
        raise ValueError("profiles contain non-finite values")
    ranks = rankdata(values, axis=1, method="average")
    centered = ranks - ranks.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = np.divide(centered, norm, out=np.full_like(centered, np.nan), where=norm > 0)
    similarity = normalized @ normalized.T
    return np.clip(similarity, -1.0, 1.0)


def retrieval_from_similarity(similarity: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    similarity = np.asarray(similarity, dtype=float)
    metadata = metadata.reset_index(drop=True).copy()
    required = {"culture", "well", "subject_id", "DIV"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"metadata are missing {sorted(required - set(metadata.columns))}")
    if similarity.shape != (len(metadata), len(metadata)):
        raise ValueError("similarity matrix does not match metadata")
    rows: list[dict[str, object]] = []
    for query_index, query in enumerate(metadata.itertuples()):
        target_ages = sorted(metadata.loc[
            (metadata["culture"] == query.culture) & (metadata["DIV"] != query.DIV), "DIV"
        ].unique())
        for target_div in target_ages:
            candidate_indices = metadata.index[
                (metadata["culture"] == query.culture) & (metadata["DIV"] == target_div)
            ].to_numpy()
            values = similarity[query_index, candidate_indices]
            finite_for_rank = np.where(np.isfinite(values), values, -np.inf)
            ranks = rankdata(-finite_for_rank, method="min")
            n_candidates = len(candidate_indices)
            for target_index, value, rank in zip(candidate_indices, values, ranks):
                candidate = metadata.iloc[int(target_index)]
                normalized = 1.0 if n_candidates == 1 else 1.0 - (float(rank) - 1.0) / (n_candidates - 1.0)
                rows.append({
                    "query_culture": query.culture,
                    "target_culture": candidate["culture"],
                    "query_subject": query.subject_id,
                    "target_subject": candidate["subject_id"],
                    "query_div": int(query.DIV),
                    "target_div": int(target_div),
                    "similarity": float(value),
                    "true_match": query.subject_id == candidate["subject_id"],
                    "rank": float(rank),
                    "normalized_rank": normalized,
                    "n_candidates": n_candidates,
                    "top1": bool(query.subject_id == candidate["subject_id"] and rank == 1),
                })
    return pd.DataFrame(rows)


def build_retrieval_queries(profiles: pd.DataFrame) -> pd.DataFrame:
    required = {"culture", "well", "subject_id", "DIV", "profile"}
    if not required.issubset(profiles.columns):
        raise ValueError(f"profiles are missing {sorted(required - set(profiles.columns))}")
    rows: list[dict[str, object]] = []
    source = profiles.reset_index(drop=True)
    for query in source.itertuples():
        target_ages = sorted(source.loc[
            (source["culture"] == query.culture) & (source["DIV"] != query.DIV), "DIV"
        ].unique())
        for target_div in target_ages:
            candidates = source[(source["culture"] == query.culture) & (source["DIV"] == target_div)]
            similarities = np.asarray([
                _spearman(query.profile, candidate.profile) for candidate in candidates.itertuples()
            ])
            finite_for_rank = np.where(np.isfinite(similarities), similarities, -np.inf)
            ranks = rankdata(-finite_for_rank, method="min")
            n_candidates = len(candidates)
            for candidate, similarity, rank in zip(candidates.itertuples(), similarities, ranks):
                normalized = 1.0 if n_candidates == 1 else 1.0 - (float(rank) - 1.0) / (n_candidates - 1.0)
                rows.append({
                    "query_culture": query.culture,
                    "target_culture": candidate.culture,
                    "query_subject": query.subject_id,
                    "target_subject": candidate.subject_id,
                    "query_div": int(query.DIV),
                    "target_div": int(target_div),
                    "similarity": similarity,
                    "true_match": query.subject_id == candidate.subject_id,
                    "rank": float(rank),
                    "normalized_rank": normalized,
                    "n_candidates": n_candidates,
                    "top1": bool(query.subject_id == candidate.subject_id and rank == 1),
                })
    return pd.DataFrame(rows)


def identification_summary(queries: pd.DataFrame) -> dict[str, float]:
    true = queries[queries["true_match"]].copy()
    if true.empty:
        raise ValueError("queries contain no true matches")
    return {
        "n_queries": float(len(true)),
        "top1_accuracy": float(true["top1"].mean()),
        "mean_normalized_rank": float(true["normalized_rank"].mean()),
        "mean_true_similarity": float(true["similarity"].mean()),
    }


def permuted_top1_accuracy(
    profiles: pd.DataFrame, *, n_permutations: int = 1000, seed: int = 0
) -> np.ndarray:
    queries = build_retrieval_queries(profiles)
    top = queries.sort_values(
        ["query_culture", "query_subject", "query_div", "target_div", "rank", "target_subject"]
    ).groupby(["query_culture", "query_subject", "query_div", "target_div"], as_index=False).first()
    rng = np.random.default_rng(seed)
    groups = {
        key: sorted(group["subject_id"].unique())
        for key, group in profiles.groupby(["culture", "DIV"])
    }
    output = np.empty(int(n_permutations), dtype=float)
    for iteration in range(int(n_permutations)):
        maps: dict[tuple[str, int], dict[str, str]] = {}
        for key, subjects in groups.items():
            shuffled = rng.permutation(subjects)
            maps[(str(key[0]), int(key[1]))] = dict(zip(subjects, shuffled))
        correct = []
        for row in top.itertuples():
            assigned = maps[(row.target_culture, int(row.target_div))][row.target_subject]
            correct.append(assigned == row.query_subject)
        output[iteration] = np.mean(correct)
    return output
