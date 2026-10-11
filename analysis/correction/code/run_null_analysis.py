from __future__ import annotations

import argparse
import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from hotopology.toolbox_cache import can_reuse_energy_sign_cache, archive_invalid_cache

from hotopology.metrics import compute_direct_metrics
from hotopology.null_models import (
    decode_ising_factors,
    irreducible_triplet_summary,
    resample_pool_conditioned_on_k,
    sample_independent,
    standardize_against_null,
)


PIPELINE_VERSION = 4
PRIMARY_METRICS = (
    "mean_population_count",
    "fraction_bins_k_ge_3",
    "triplet_support_mass",
    "triplet_repeated5_density",
    "h1_total_persistence_capped",
    "h1_infinite_count",
)


def dense_triplet_counts(
    supports: dict[int, dict[tuple[int, ...], int]],
    triplets: list[tuple[int, int, int]],
) -> np.ndarray:
    values = supports.get(3, {})
    return np.fromiter((values.get(t, 0) for t in triplets), dtype=np.float64)


def _model_metrics(block: np.ndarray) -> tuple[dict[str, float], dict[str, object]]:
    return compute_direct_metrics(block, min_active_bins=1)


def process_one(model_path: str, result_dir: str) -> dict[str, object]:
    result = Path(result_dir)
    model_file = Path(model_path)
    recording = model_file.name.replace("_maxent.mat", "")
    checkpoint_dir = result / "data_intermediate" / "null_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{recording}.json"
    loaded = loadmat(model_file, squeeze_me=True, struct_as_record=False)
    if bool(loaded.get("skipped", False)):
        payload = {
            "pipeline_version": PIPELINE_VERSION,
            "recording": recording,
            "qc_rows": [{"recording": recording, "model_qc_pass": False, "skip_reason": str(loaded.get("skip_reason", "skipped"))}],
            "metric_rows": [],
            "irreducible_rows": [],
            "top_rows": [],
        }
        checkpoint.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    fit = loaded["fit"]
    active_nodes = np.asarray(loaded["active_nodes"], dtype=int).reshape(-1) - 1
    binary_path = result / "data_intermediate" / "binary_25ms" / f"{recording}_25ms.npz"
    binary = np.load(binary_path)
    x_full = np.asarray(binary["x"], dtype=np.uint8)
    observed = x_full[active_nodes]
    n_nodes, n_bins = observed.shape
    pool = np.asarray(fit.pairwise_samples, dtype=np.uint8)
    if pool.ndim == 1:
        pool = pool.reshape(n_nodes, -1)
    n_blocks = int(fit.n_blocks)
    if pool.shape != (n_nodes, n_blocks * n_bins):
        raise ValueError(f"unexpected pairwise pool shape {pool.shape} for {recording}")
    if checkpoint.exists():
        existing = json.loads(checkpoint.read_text(encoding="utf-8"))
        if existing.get('pipeline_version') == PIPELINE_VERSION:
            if can_reuse_energy_sign_cache(existing, pool, observed.sum(axis=0)):
                return existing
            archive_invalid_cache(checkpoint, result)

    div = int(binary["div"])
    plate = int(binary["plate"])
    well = str(binary["well"])
    subject_id = f"P{plate}_{well}"
    seed = int(zlib.crc32(recording.encode("utf-8")))
    pairwise_firing_rmse = float(fit.pairwise_firing_rmse)
    pairwise_pair_rmse = float(fit.pairwise_pair_rmse)
    pairwise_k_rmse = float(fit.pairwise_k_rmse)
    model_qc_pass = (
        bool(fit.pairwise_converged)
        and pairwise_firing_rmse <= 0.02
        and pairwise_pair_rmse <= 0.02
    )
    qc_rows = [
        {
            "recording": recording,
            "DIV": div,
            "plate": plate,
            "well": well,
            "subject_id": subject_id,
            "n_model_nodes": n_nodes,
            "model_node_selection": "16 electrodes with the highest active-bin counts among eligible electrodes",
            "n_bins": n_bins,
            "pairwise_solver_converged": bool(fit.pairwise_converged),
            "pairwise_firing_rmse": pairwise_firing_rmse,
            "pairwise_pair_rmse": pairwise_pair_rmse,
            "pairwise_k_rmse": pairwise_k_rmse,
            "k_distribution_used_as_qc": False,
            "model_qc_pass": bool(model_qc_pass),
            "conditioned_fallback_k": "",
        }
    ]
    if not model_qc_pass:
        payload = {
            "pipeline_version": PIPELINE_VERSION,
            "recording": recording,
            "qc_rows": qc_rows,
            "metric_rows": [],
            "irreducible_rows": [],
            "top_rows": [],
        }
        checkpoint.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    observed_metrics, observed_detail = _model_metrics(observed)
    pair_blocks = pool.reshape(n_nodes, n_blocks, n_bins).transpose(1, 0, 2)
    desired_k = observed.sum(axis=0)
    pool_k_values = set(int(k) for k in np.unique(pool.sum(axis=0)))
    missing_k = sorted(set(int(k) for k in np.unique(desired_k)) - pool_k_values)
    qc_rows[0]["conditioned_fallback_k"] = ",".join(str(k) for k in missing_k)
    h, j = decode_ising_factors(np.asarray(fit.pairwise_model.factors), n_nodes)
    conditioned = resample_pool_conditioned_on_k(
        pool,
        desired_k,
        n_surrogates=n_blocks,
        seed=seed + 1,
        fallback_h=h,
        fallback_j=j,
        fallback_sweeps=20,
    )

    probabilities = observed.mean(axis=1)
    independent_blocks = [
        sample_independent(probabilities, n_bins=n_bins, seed=seed + 100 + block)
        for block in range(10)
    ]
    null_blocks: dict[str, list[np.ndarray] | np.ndarray] = {
        "independent": independent_blocks,
        "pairwise": pair_blocks,
        "pairwise_conditioned_k": conditioned,
    }

    metric_rows: list[dict[str, object]] = []
    null_metric_cache: dict[str, list[dict[str, float]]] = {}
    for model_name, blocks in null_blocks.items():
        block_metrics = [_model_metrics(np.asarray(block))[0] for block in blocks]
        null_metric_cache[model_name] = block_metrics
        for metric in PRIMARY_METRICS:
            values = np.asarray([row[metric] for row in block_metrics], dtype=float)
            comparison = standardize_against_null(observed_metrics[metric], values)
            metric_rows.append(
                {
                    "recording": recording,
                    "DIV": div,
                    "plate": plate,
                    "well": well,
                    "subject_id": subject_id,
                    "model": model_name,
                    "metric": metric,
                    "observed": comparison.observed,
                    "null_mean": comparison.null_mean,
                    "null_sd": comparison.null_sd,
                    "z": comparison.z,
                    "empirical_p": comparison.p_two_sided,
                    "n_null_blocks": int(values.size),
                }
            )

    triplets = list(combinations(range(n_nodes), 3))
    observed_counts = dense_triplet_counts(observed_detail["supports"], triplets)
    irreducible_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    for model_name, blocks in (
        ("pairwise", pair_blocks),
        ("pairwise_conditioned_k", conditioned),
    ):
        block_counts = np.empty((len(blocks), len(triplets)), dtype=np.float32)
        for block_idx, block in enumerate(blocks):
            _, detail = _model_metrics(np.asarray(block))
            block_counts[block_idx] = dense_triplet_counts(detail["supports"], triplets)
        summary = irreducible_triplet_summary(
            observed_counts, block_counts, n_bins=n_bins
        )
        irreducible_rows.append(
            {
                "recording": recording,
                "DIV": div,
                "plate": plate,
                "well": well,
                "subject_id": subject_id,
                "model": model_name,
                "n_model_nodes": n_nodes,
                "n_triplets": len(triplets),
                "positive_residual_mass": summary["positive_residual_mass"],
                "negative_residual_mass": summary["negative_residual_mass"],
                "absolute_residual_mass": summary["absolute_residual_mass"],
                "rms_residual": summary["rms_residual"],
                "positive_residual_z": summary["positive_residual_z"],
                "negative_residual_z": summary["negative_residual_z"],
                "absolute_residual_z": summary["absolute_residual_z"],
                "positive_residual_p": summary["positive_residual_p"],
                "negative_residual_p": summary["negative_residual_p"],
                "absolute_residual_p": summary["absolute_residual_p"],
            }
        )
        if model_name == "pairwise_conditioned_k":
            expected = np.asarray(summary["expected_counts"])
            residual = np.asarray(summary["residual_probability"])
            sd = block_counts.std(axis=0, ddof=1)
            z_triplet = np.divide(
                observed_counts - expected,
                sd,
                out=np.zeros_like(expected, dtype=float),
                where=sd > 0,
            )
            ranked = np.argsort(np.abs(residual))[::-1][:400]
            for index in ranked:
                local = triplets[int(index)]
                original = active_nodes[np.asarray(local)]
                top_rows.append(
                    {
                        "recording": recording,
                        "DIV": div,
                        "plate": plate,
                        "well": well,
                        "subject_id": subject_id,
                        "node_i_zero_based": int(original[0]),
                        "node_j_zero_based": int(original[1]),
                        "node_k_zero_based": int(original[2]),
                        "observed_count": float(observed_counts[index]),
                        "conditioned_expected_count": float(expected[index]),
                        "residual_probability": float(residual[index]),
                        "triplet_z_exploratory": float(z_triplet[index]),
                    }
                )

    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "toolbox_logpotential_version": 2,
        "recording": recording,
        "qc_rows": qc_rows,
        "metric_rows": metric_rows,
        "irreducible_rows": irreducible_rows,
        "top_rows": top_rows,
    }
    checkpoint.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    args = parser.parse_args()
    result = Path(args.result_dir)
    model_dir = result / "data_intermediate" / "maxent_models_top16"
    files = sorted(str(path) for path in model_dir.glob("*_maxent.mat"))
    from requested_node_eligibility import expected_keys
    eligibility = pd.read_csv(result / 'tables/computational_closure_v2/requested_node_eligibility.csv')
    if {Path(p).stem.removesuffix('_maxent') for p in files} != expected_keys(eligibility, 'dynamic16'):
        raise RuntimeError('Legacy-comparison models differ from explicit dynamic16 eligibility')

    payloads: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(process_one, path, str(result)): path for path in files}
        for done, future in enumerate(as_completed(future_map), start=1):
            path = future_map[future]
            try:
                payloads.append(future.result())
                print(f"[{done:03d}/{len(files):03d}] {Path(path).stem}", flush=True)
            except Exception as exc:  # pragma: no cover
                failures.append({"path": path, "error": repr(exc)})
                print(f"FAILED {path}: {exc!r}", flush=True)

    tables = result / "tables"
    logs = result / "logs"
    tables.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    for key, filename in (
        ("qc_rows", "maxent_model_qc.csv"),
        ("metric_rows", "topology_null_comparisons.csv"),
        ("irreducible_rows", "irreducible_triplet_summary.csv"),
        ("top_rows", "exploratory_top_triplet_residuals.csv"),
    ):
        rows = [row for payload in payloads for row in payload[key]]
        pd.DataFrame(rows).to_csv(tables / filename, index=False)
    (logs / "null_analysis_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"{len(failures)} null analyses failed")


if __name__ == "__main__":
    main()
