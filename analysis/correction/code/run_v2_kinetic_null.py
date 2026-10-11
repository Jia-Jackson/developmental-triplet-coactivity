from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
import json
import os
from pathlib import Path
import time
import zlib

import numpy as np
import pandas as pd
from scipy.io import loadmat

from hotopology.kinetic_null import (
    fit_kinetic_pairwise,
    lag1_moment_rmse,
    simulate_kinetic_conditioned_k,
)
from hotopology.metrics import compute_direct_metrics
from hotopology.null_models import irreducible_triplet_summary, standardize_against_null
from hotopology.endpoint_validity import mark_undefined_endpoint_z


PIPELINE_VERSION = 1
N_NULL = 100
MINIMUM_VALID = 90
PRIMARY_METRICS = (
    "mean_population_count",
    "fraction_bins_k_ge_3",
    "triplet_support_mass",
    "triplet_repeated5_density",
    "h1_total_persistence_capped",
    "h1_infinite_count",
)


def _dense_triplet_counts(supports, triplets) -> np.ndarray:
    values = supports.get(3, {})
    return np.fromiter((values.get(item, 0) for item in triplets), dtype=np.float64)


def _simultaneous_pair_rmse(observed: np.ndarray, simulated: np.ndarray) -> float:
    observed_float = observed.astype(float)
    observed_moment = observed_float @ observed_float.T / observed.shape[1]
    total = np.zeros_like(observed_moment)
    for realization in simulated:
        data = realization.astype(float)
        total += data @ data.T
    simulated_moment = total / (simulated.shape[0] * simulated.shape[2])
    upper = np.triu_indices(observed.shape[0], 1)
    return float(np.sqrt(np.mean(np.square(observed_moment[upper] - simulated_moment[upper]))))


def _aggregate_lag1_rmse(observed: np.ndarray, simulated: np.ndarray) -> float:
    observed_float = observed.astype(float)
    observed_moment = observed_float[:, 1:] @ observed_float[:, :-1].T / (observed.shape[1] - 1)
    total = np.zeros_like(observed_moment)
    for realization in simulated:
        data = realization.astype(float)
        total += data[:, 1:] @ data[:, :-1].T
    simulated_moment = total / (simulated.shape[0] * (simulated.shape[2] - 1))
    return float(np.sqrt(np.mean(np.square(observed_moment - simulated_moment))))


def process_one(model_path: str, result_dir: str) -> dict[str, object]:
    started = time.perf_counter()
    root = Path(result_dir)
    model_path_obj = Path(model_path)
    recording = model_path_obj.name.removesuffix("_maxent.mat")
    checkpoint_dir = root / "data_intermediate" / "kinetic_null_checkpoints_top16"
    fit_dir = root / "data_intermediate" / "kinetic_models_top16"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fit_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{recording}.json"
    if checkpoint.exists():
        existing = json.loads(checkpoint.read_text(encoding="utf-8"))
        if existing.get("pipeline_version") == PIPELINE_VERSION:
            return existing

    model_file = loadmat(model_path_obj, squeeze_me=True, struct_as_record=False)
    active_nodes = np.asarray(model_file["active_nodes"], dtype=int).reshape(-1) - 1
    binary_path = root / "data_intermediate" / "binary_25ms" / f"{recording}_25ms.npz"
    with np.load(binary_path) as archive:
        full = np.asarray(archive["x"], dtype=np.uint8)
        div = int(archive["div"])
        plate = int(archive["plate"])
        well = str(archive["well"])
    observed = full[active_nodes]
    subject_id = f"P{plate}_{well}"
    seed = int(zlib.crc32((recording + "_kinetic_v2").encode("utf-8")))

    fit = fit_kinetic_pairwise(
        observed,
        c_grid=(0.01, 0.1, 1.0),
        n_splits=3,
        max_iter=500,
    )
    desired_k = observed.sum(axis=0, dtype=np.int32)
    simulated = simulate_kinetic_conditioned_k(
        fit.intercepts,
        fit.weights,
        desired_k,
        initial_state=observed[:, 0],
        n_realizations=N_NULL,
        seed=seed,
    )
    k_mismatch_count = int(
        np.sum(simulated.sum(axis=1, dtype=np.int32) != desired_k[None, :])
    )
    observed_firing = observed.mean(axis=1)
    simulated_firing = simulated.mean(axis=(0, 2))
    firing_rmse = float(np.sqrt(np.mean(np.square(observed_firing - simulated_firing))))
    pair_rmse = _simultaneous_pair_rmse(observed, simulated)
    lag_rmse = _aggregate_lag1_rmse(observed, simulated)

    np.savez_compressed(
        fit_dir / f"{recording}_kinetic_fit.npz",
        active_nodes_zero_based=active_nodes,
        intercepts=fit.intercepts,
        weights=fit.weights,
        selected_c=fit.selected_c,
        heldout_log_loss=fit.heldout_log_loss,
        rate_only_log_loss=fit.rate_only_log_loss,
        seed=seed,
    )

    observed_metrics, observed_detail = compute_direct_metrics(observed, min_active_bins=1)
    triplets = list(combinations(range(observed.shape[0]), 3))
    observed_counts = _dense_triplet_counts(observed_detail["supports"], triplets)
    null_metric_rows = []
    block_counts = np.empty((N_NULL, len(triplets)), dtype=np.float32)
    failures = []
    for index, realization in enumerate(simulated):
        try:
            metrics, detail = compute_direct_metrics(realization, min_active_bins=1)
            null_metric_rows.append(metrics)
            block_counts[index] = _dense_triplet_counts(detail["supports"], triplets)
        except Exception as exc:  # pragma: no cover - retained as explicit null failure
            failures.append({"realization": index, "error": f"{type(exc).__name__}: {exc}"})
            block_counts[index] = np.nan
    valid_mask = np.isfinite(block_counts).all(axis=1)
    valid_count = int(valid_mask.sum())
    inferential_valid = valid_count >= MINIMUM_VALID and k_mismatch_count == 0

    comparison_rows = []
    if inferential_valid:
        for metric in PRIMARY_METRICS:
            values = np.asarray([row[metric] for row in null_metric_rows], dtype=float)
            comparison = standardize_against_null(observed_metrics[metric], values)
            comparison_rows.append(
                {
                    "recording": recording,
                    "DIV": div,
                    "plate": plate,
                    "culture": f"P{plate}",
                    "well": well,
                    "subject_id": subject_id,
                    "model": "kinetic_conditioned_k",
                    "metric": metric,
                    "observed": comparison.observed,
                    "null_mean": comparison.null_mean,
                    "null_sd": comparison.null_sd,
                    "z": comparison.z,
                    "empirical_p": comparison.p_two_sided,
                    "n_null_blocks": valid_count,
                    "monte_carlo_se_mean": float(values.std(ddof=1) / np.sqrt(valid_count)),
                }
            )
        residual = irreducible_triplet_summary(
            observed_counts, block_counts[valid_mask], n_bins=observed.shape[1]
        )
        residual_row = {
            "recording": recording,
            "DIV": div,
            "plate": plate,
            "culture": f"P{plate}",
            "well": well,
            "subject_id": subject_id,
            "model": "kinetic_conditioned_k",
            "n_model_nodes": observed.shape[0],
            "n_triplets": len(triplets),
            **{
                key: float(residual[key])
                for key in [
                    "positive_residual_mass",
                    "negative_residual_mass",
                    "absolute_residual_mass",
                    "rms_residual",
                    "positive_residual_z",
                    "negative_residual_z",
                    "absolute_residual_z",
                    "positive_residual_p",
                    "negative_residual_p",
                    "absolute_residual_p",
                ]
            },
        }
    else:
        residual_row = {}

    qc_row = {
        "recording": recording,
        "DIV": div,
        "plate": plate,
        "culture": f"P{plate}",
        "well": well,
        "subject_id": subject_id,
        "n_model_nodes": observed.shape[0],
        "n_bins": observed.shape[1],
        "n_requested_nulls": N_NULL,
        "n_valid_nulls": valid_count,
        "inferential_valid": bool(inferential_valid),
        "exact_k_mismatch_count": k_mismatch_count,
        "firing_probability_rmse": firing_rmse,
        "simultaneous_pair_moment_rmse_diagnostic": pair_rmse,
        "lag1_moment_rmse": lag_rmse,
        "heldout_kinetic_log_loss": fit.heldout_log_loss,
        "heldout_rate_only_log_loss": fit.rate_only_log_loss,
        "heldout_log_loss_improvement": fit.rate_only_log_loss - fit.heldout_log_loss,
        "median_selected_c": float(np.median(fit.selected_c)),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "recording": recording,
        "qc_row": qc_row,
        "comparison_rows": comparison_rows,
        "residual_row": residual_row,
        "null_failures": failures,
    }
    checkpoint.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--max-files", type=int, default=143)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    model_dir = root / "data_intermediate" / "maxent_models_top16"
    files = sorted(model_dir.glob("*_maxent.mat"))[: args.max_files]
    payloads = []
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(process_one, str(path), str(root)): path for path in files
        }
        for done, future in enumerate(as_completed(future_map), start=1):
            path = future_map[future]
            try:
                payload = future.result()
                payloads.append(payload)
                print(
                    f"[{done:03d}/{len(files):03d}] {payload['recording']} "
                    f"{payload['qc_row']['runtime_seconds']:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
                print(f"FAILED {path.name}: {exc}", flush=True)

    out = root / "tables" / "computational_closure_v2"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([payload["qc_row"] for payload in payloads]).to_csv(
        out / "v2_kinetic_null_qc.csv", index=False
    )
    mark_undefined_endpoint_z(pd.DataFrame(
        [row for payload in payloads for row in payload["comparison_rows"]]
    )).to_csv(out / "v2_kinetic_null_comparisons.csv", index=False)
    pd.DataFrame(
        [payload["residual_row"] for payload in payloads if payload["residual_row"]]
    ).to_csv(out / "v2_kinetic_triplet_residuals.csv", index=False)
    (root / "logs" / "v2_kinetic_null_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"{len(failures)} kinetic-null recordings failed")


if __name__ == "__main__":
    main()
