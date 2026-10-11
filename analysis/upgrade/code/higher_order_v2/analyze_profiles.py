from __future__ import annotations

from itertools import combinations
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata, spearmanr

from .contract import audit_fixed12_contract
from .fingerprinting import identification_summary, rank_similarity_matrix, retrieval_from_similarity
from .io_utils import atomic_json


def _complete_metadata(source: Path) -> pd.DataFrame:
    keys = pd.read_csv(source / "tables/computational_closure_v2/v2_recording_key_manifest.csv")
    manifest = pd.read_csv(source / "tables/computational_closure_v2/v2_fixed_node_manifest_long.csv")
    audit = audit_fixed12_contract(manifest, keys)
    return keys[keys["subject_id"].isin(audit.complete_subjects)].sort_values(
        ["culture", "well", "DIV"]
    ).reset_index(drop=True)


def _load_full_matrices(output: Path, metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    matrices: dict[str, list[np.ndarray]] = {
        "firing_node": [],
        "triplet_node_projection": [],
        "oinfo_node_projection": [],
        "pair_edge": [],
        "triplet_edge_projection": [],
        "oinfo_edge_projection": [],
        "triplet_native": [],
        "oinfo_native": [],
    }
    folder = output / "intermediate/full_array_profiles"
    for recording in metadata["recording"]:
        with np.load(folder / f"{recording}.npz", allow_pickle=False) as a:
            matrices["firing_node"].append(a["single_probability"])
            matrices["triplet_node_projection"].append(a["triplet_node_projection"])
            matrices["oinfo_node_projection"].append(a["oinfo_node_projection"])
            matrices["pair_edge"].append(a["pair_support_probability"])
            matrices["triplet_edge_projection"].append(a["triplet_edge_projection"])
            matrices["oinfo_edge_projection"].append(a["oinfo_edge_projection"])
            matrices["triplet_native"].append(a["triplet_support_probability"])
            matrices["oinfo_native"].append(a["oinfo_bits"])
    return {name: np.vstack(values) for name, values in matrices.items()}


def _query_level(queries: pd.DataFrame) -> pd.DataFrame:
    keys = ["query_culture", "query_subject", "query_div", "target_div"]
    true = queries[queries["true_match"]].copy()
    other_mean = (
        queries[~queries["true_match"]].groupby(keys, as_index=False)["similarity"]
        .mean().rename(columns={"similarity": "mean_other_similarity"})
    )
    true = true.merge(other_mean, on=keys, how="left")
    true["similarity_margin"] = true["similarity"] - true["mean_other_similarity"]
    return true


def _bootstrap_subject_means(
    query_level: pd.DataFrame, column: str, *, n_bootstrap: int, seed: int
) -> np.ndarray:
    subject = query_level.groupby(["query_culture", "query_subject"])[column].mean().reset_index()
    rng = np.random.default_rng(seed)
    sampled_cultures = []
    for culture in sorted(subject["query_culture"].unique()):
        group = subject[subject["query_culture"] == culture][column].to_numpy(float)
        indices = rng.integers(0, len(group), size=(n_bootstrap, len(group)))
        sampled_cultures.append(group[indices])
    return np.nanmean(np.concatenate(sampled_cultures, axis=1), axis=1)


def _permutation_accuracy(
    queries: pd.DataFrame, *, n_permutations: int, seed: int
) -> np.ndarray:
    keys = ["query_culture", "query_subject", "query_div", "target_div"]
    top = queries.sort_values(keys + ["rank", "target_subject"]).groupby(keys, as_index=False).first()
    rng = np.random.default_rng(seed)
    correct_counts = np.zeros(n_permutations, dtype=float)
    for (culture, target_div), group in top.groupby(["target_culture", "target_div"]):
        subjects = sorted(set(group["query_subject"]) | set(group["target_subject"]))
        lookup = {subject: index for index, subject in enumerate(subjects)}
        random_order = np.argsort(rng.random((n_permutations, len(subjects))), axis=1)
        predicted_index = np.asarray([lookup[value] for value in group["target_subject"]], dtype=int)
        query_index = np.asarray([lookup[value] for value in group["query_subject"]], dtype=int)
        assigned = random_order[:, predicted_index]
        correct_counts += np.sum(assigned == query_index[None, :], axis=1)
    return correct_counts / len(top)


def run_fingerprinting(source: Path, output: Path) -> None:
    table_dir = output / "tables/upgrade_v2"
    table_dir.mkdir(parents=True, exist_ok=True)
    metadata = _complete_metadata(source)
    matrices = _load_full_matrices(output, metadata)
    similarities = {
        name: rank_similarity_matrix(values) for name, values in matrices.items()
    }
    similarities["pair_plus_oinfo_edge"] = np.nanmean(
        np.stack([similarities["pair_edge"], similarities["oinfo_edge_projection"]]), axis=0
    )
    similarities["pair_plus_triplet_edge"] = np.nanmean(
        np.stack([similarities["pair_edge"], similarities["triplet_edge_projection"]]), axis=0
    )

    all_queries, all_summary, all_permutations, all_bootstrap, age_pair_rows = [], [], [], [], []
    for metric_index, (metric, similarity) in enumerate(similarities.items()):
        queries = retrieval_from_similarity(similarity, metadata)
        queries.insert(0, "metric", metric)
        all_queries.append(queries)
        query_level = _query_level(queries)
        summary = identification_summary(queries)
        summary["metric"] = metric
        summary["analytic_chance_top1"] = float(np.mean(1.0 / query_level["n_candidates"]))
        summary["mean_similarity_margin"] = float(query_level["similarity_margin"].mean())
        permutation = _permutation_accuracy(
            queries, n_permutations=10000, seed=9200 + metric_index
        )
        summary["permutation_p_top1"] = float(
            (1 + np.sum(permutation >= summary["top1_accuracy"])) / (len(permutation) + 1)
        )
        for iteration, value in enumerate(permutation):
            all_permutations.append({"metric": metric, "iteration": iteration, "top1_accuracy": value})
        for column in ("top1", "normalized_rank", "similarity_margin"):
            bootstrap = _bootstrap_subject_means(
                query_level, column, n_bootstrap=5000, seed=13000 + 10 * metric_index + len(column)
            )
            summary[f"{column}_ci_low"] = float(np.nanquantile(bootstrap, 0.025))
            summary[f"{column}_ci_high"] = float(np.nanquantile(bootstrap, 0.975))
            for iteration, value in enumerate(bootstrap):
                all_bootstrap.append({
                    "metric": metric, "statistic": column, "iteration": iteration, "value": value
                })
        all_summary.append(summary)
        for (query_div, target_div), group in query_level.groupby(["query_div", "target_div"]):
            age_pair_rows.append({
                "metric": metric,
                "query_DIV": int(query_div),
                "target_DIV": int(target_div),
                "n_queries": int(len(group)),
                "top1_accuracy": float(group["top1"].mean()),
                "mean_normalized_rank": float(group["normalized_rank"].mean()),
                "mean_similarity_margin": float(group["similarity_margin"].mean()),
            })
    pd.concat(all_queries, ignore_index=True).to_csv(
        table_dir / "v2_longitudinal_identification_candidates.csv", index=False
    )
    pd.DataFrame(all_summary).to_csv(table_dir / "v2_longitudinal_identification_summary.csv", index=False)
    pd.DataFrame(age_pair_rows).to_csv(table_dir / "v2_longitudinal_identification_age_pairs.csv", index=False)
    pd.DataFrame(all_permutations).to_csv(table_dir / "v2_longitudinal_identification_permutations.csv", index=False)
    pd.DataFrame(all_bootstrap).to_csv(table_dir / "v2_longitudinal_identification_bootstrap.csv", index=False)

    pair_query = _query_level(all_queries[3])  # pair_edge by insertion order
    pair_key = ["query_culture", "query_subject", "query_div", "target_div"]
    comparison_rows = []
    query_frames = {frame["metric"].iloc[0]: _query_level(frame) for frame in all_queries}
    for metric, frame in query_frames.items():
        if metric == "pair_edge":
            continue
        merged = pair_query[pair_key + ["top1", "normalized_rank", "similarity_margin"]].merge(
            frame[pair_key + ["top1", "normalized_rank", "similarity_margin"]],
            on=pair_key, suffixes=("_pair", "_candidate"), validate="one_to_one"
        )
        for statistic in ("top1", "normalized_rank", "similarity_margin"):
            difference = merged[f"{statistic}_candidate"].astype(float) - merged[f"{statistic}_pair"].astype(float)
            merged_for_boot = merged[pair_key[:2]].copy()
            merged_for_boot["difference"] = difference
            boot = _bootstrap_subject_means(
                merged_for_boot.rename(columns={"query_culture": "query_culture", "query_subject": "query_subject"}),
                "difference", n_bootstrap=5000, seed=18000 + len(comparison_rows)
            )
            comparison_rows.append({
                "candidate_metric": metric, "reference_metric": "pair_edge", "statistic": statistic,
                "observed_difference": float(difference.mean()),
                "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
                "n_query_age_pairs": int(len(merged)),
            })
    pd.DataFrame(comparison_rows).to_csv(
        table_dir / "v2_identification_incremental_comparisons.csv", index=False
    )
    np.savez_compressed(output / "intermediate" / "v2_full_array_similarity_matrices.npz", **similarities)


def run_taxonomy(source: Path, output: Path) -> None:
    table_dir = output / "tables/upgrade_v2"
    edge = pd.read_csv(table_dir / "v2_fixed12_edge_projections.csv")
    metrics = [
        "pair_support_probability",
        "triplet_support_projection",
        "residual_projection",
        "absolute_residual_projection",
        "oinfo_projection_bits",
        "oinfo_deviation_projection_bits",
    ]
    rows = []
    for recording, group in edge.groupby("recording"):
        group = group.sort_values(["local_i", "local_j"])
        for first, second in combinations(metrics, 2):
            rho = spearmanr(group[first], group[second]).statistic
            rows.append({
                "recording": recording, "DIV": int(group["DIV"].iloc[0]),
                "culture": group["culture"].iloc[0], "subject_id": group["subject_id"].iloc[0],
                "metric_1": first, "metric_2": second,
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
            })
    record_correlations = pd.DataFrame(rows)
    record_correlations.to_csv(table_dir / "v2_cross_metric_recording_correlations.csv", index=False)
    matrix = pd.DataFrame(np.eye(len(metrics)), index=metrics, columns=metrics)
    for (first, second), group in record_correlations.groupby(["metric_1", "metric_2"]):
        value = float(group["spearman_rho"].median())
        matrix.loc[first, second] = value
        matrix.loc[second, first] = value
    matrix.to_csv(table_dir / "v2_cross_metric_median_correlation_matrix.csv")
    distance = np.sqrt(np.clip(2.0 * (1.0 - matrix.to_numpy(float)), 0.0, None))
    np.fill_diagonal(distance, 0.0)
    linkage_matrix = linkage(squareform(distance, checks=False), method="average")
    order = [metrics[index] for index in leaves_list(linkage_matrix)]
    atomic_json(output / "tables/upgrade_v2/v2_cross_metric_cluster_order.json", {
        "metric_order": order, "distance": "sqrt(2*(1-median Spearman rho))", "linkage": "average"
    })

    keys = pd.read_csv(source / "tables/computational_closure_v2/v2_recording_key_manifest.csv")
    complete = _complete_metadata(source)["subject_id"].unique()
    edge = edge[edge["subject_id"].isin(complete)]
    continuity_metrics = metrics
    similarity_by_subject: dict[tuple[str, str], np.ndarray] = {}
    observed_rows = []
    for subject, subject_data in edge.groupby("subject_id"):
        divs = sorted(subject_data["DIV"].unique())
        for metric in continuity_metrics:
            profiles = np.vstack([
                subject_data[subject_data["DIV"] == div].sort_values(["local_i", "local_j"])[metric]
                for div in divs
            ])
            similarity = rank_similarity_matrix(profiles)
            similarity_by_subject[(subject, metric)] = similarity
            adjacent = [similarity[index, index + 1] for index in range(len(divs) - 1)]
            observed_rows.append({"subject_id": subject, "metric": metric, "mean_adjacent_similarity": np.nanmean(adjacent)})
    observed_frame = pd.DataFrame(observed_rows)
    rng = np.random.default_rng(24680)
    n_permutations = 10000
    continuity_rows = []
    for metric in continuity_metrics:
        observed = float(observed_frame[observed_frame["metric"] == metric]["mean_adjacent_similarity"].mean())
        subjects = sorted(observed_frame[observed_frame["metric"] == metric]["subject_id"])
        subject_nulls = []
        for subject in subjects:
            order = np.argsort(rng.random((n_permutations, 8)), axis=1)
            similarity = similarity_by_subject[(subject, metric)]
            adjacent_values = similarity[order[:, :-1], order[:, 1:]]
            subject_nulls.append(np.nanmean(adjacent_values, axis=1))
        null = np.nanmean(np.column_stack(subject_nulls), axis=1)
        continuity_rows.append({
            "metric": metric, "observed_mean_adjacent_similarity": observed,
            "permuted_mean": float(np.nanmean(null)),
            "permuted_ci_low": float(np.nanquantile(null, 0.025)),
            "permuted_ci_high": float(np.nanquantile(null, 0.975)),
            "empirical_p_greater": float((1 + np.sum(null >= observed)) / (n_permutations + 1)),
            "n_complete_wells": len(subjects), "n_permutations": n_permutations,
        })
    observed_frame.to_csv(table_dir / "v2_fixed12_subject_temporal_continuity.csv", index=False)
    pd.DataFrame(continuity_rows).to_csv(table_dir / "v2_fixed12_temporal_continuity_tests.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["fingerprint", "taxonomy", "all"], default="all")
    args = parser.parse_args()
    if args.phase in {"fingerprint", "all"}:
        run_fingerprinting(args.source.resolve(), args.output.resolve())
    if args.phase in {"taxonomy", "all"}:
        run_taxonomy(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
