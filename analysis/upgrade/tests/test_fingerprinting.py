import numpy as np
import pandas as pd

from higher_order_v2.fingerprinting import (
    build_retrieval_queries,
    identification_summary,
    permuted_top1_accuracy,
    rank_similarity_matrix,
    retrieval_from_similarity,
)


def _toy_profiles():
    rows = []
    for culture in ("P1", "P2"):
        for well_index, well in enumerate(("A1", "A2")):
            subject = f"{culture}_{well}"
            base = np.array([1.0, 0.0, 0.0]) if well_index == 0 else np.array([0.0, 1.0, 0.0])
            for div in (14, 21):
                rows.append({"culture": culture, "well": well, "subject_id": subject,
                             "DIV": div, "profile": base + div * 1e-4})
    return pd.DataFrame(rows)


def test_candidates_are_same_culture_and_different_age():
    queries = build_retrieval_queries(_toy_profiles())
    assert len(queries) == 16
    assert queries.groupby(
        ["query_culture", "query_subject", "query_div", "target_div"]
    ).ngroups == 8
    assert all(row.query_culture == row.target_culture for row in queries.itertuples())
    assert all(row.query_div != row.target_div for row in queries.itertuples())


def test_perfect_profiles_reidentify_wells():
    queries = build_retrieval_queries(_toy_profiles())
    summary = identification_summary(queries)
    assert summary["top1_accuracy"] == 1.0
    assert summary["mean_normalized_rank"] == 1.0


def test_label_permutation_is_reproducible():
    profiles = _toy_profiles()
    first = permuted_top1_accuracy(profiles, n_permutations=50, seed=123)
    second = permuted_top1_accuracy(profiles, n_permutations=50, seed=123)
    np.testing.assert_array_equal(first, second)


def test_precomputed_rank_similarity_and_retrieval():
    profiles = _toy_profiles().reset_index(drop=True)
    matrix = np.vstack(profiles["profile"])
    similarity = rank_similarity_matrix(matrix)
    np.testing.assert_allclose(np.diag(similarity), 1.0)
    outcomes = retrieval_from_similarity(similarity, profiles.drop(columns="profile"))
    assert identification_summary(outcomes)["top1_accuracy"] == 1.0
