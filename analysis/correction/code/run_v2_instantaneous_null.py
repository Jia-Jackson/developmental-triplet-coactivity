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

from hotopology.metrics import compute_direct_metrics
from hotopology.null_models import (
    decode_ising_factors,
    irreducible_triplet_summary,
    resample_pool_conditioned_on_k,
    standardize_against_null,
)
from hotopology.null_qc import null_moment_qc
from requested_node_eligibility import expected_keys
from hotopology.toolbox_cache import can_reuse_energy_sign_cache, archive_invalid_cache


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


def process_one(model_path: str, result_dir: str, output_prefix: str) -> dict[str, object]:
    started = time.perf_counter()
    root = Path(result_dir)
    path = Path(model_path)
    recording = path.name.removesuffix("_maxent.mat")
    checkpoint_folder = ("instantaneous_null100_checkpoints_top16"
                         if output_prefix == "instantaneous_null100"
                         else f"{output_prefix}_checkpoints")
    checkpoint_dir = root / "data_intermediate" / checkpoint_folder
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{recording}.json"
    loaded = loadmat(path, squeeze_me=True, struct_as_record=False)
    fit = loaded["fit"]
    active_nodes = np.asarray(loaded["active_nodes"], dtype=int).reshape(-1) - 1
    with np.load(root / "data_intermediate" / "binary_25ms" / f"{recording}_25ms.npz") as archive:
        full = np.asarray(archive["x"], dtype=np.uint8)
        div = int(archive["div"])
        plate = int(archive["plate"])
        well = str(archive["well"])
    observed = full[active_nodes]
    n_nodes, n_bins = observed.shape
    pool = np.asarray(fit.pairwise_samples, dtype=np.uint8).reshape(n_nodes, -1)
    h, j = decode_ising_factors(np.asarray(fit.pairwise_model.factors), n_nodes)
    desired_k = observed.sum(axis=0, dtype=np.int32)
    if checkpoint.exists():
        existing = json.loads(checkpoint.read_text(encoding="utf-8"))
        if existing.get('pipeline_version') == PIPELINE_VERSION:
            if can_reuse_energy_sign_cache(existing, pool, desired_k):
                return existing
            archive_invalid_cache(checkpoint, root)
    seed = int(zlib.crc32((recording + "_instantaneous100_v2").encode("utf-8")))
    simulated = resample_pool_conditioned_on_k(
        pool,
        desired_k,
        n_surrogates=N_NULL,
        seed=seed,
        fallback_h=h,
        fallback_j=j,
        fallback_sweeps=20,
    )
    qc = null_moment_qc(observed, simulated, require_exact_k=True)
    observed_metrics, observed_detail = compute_direct_metrics(observed, min_active_bins=1)
    triplets = list(combinations(range(n_nodes), 3))
    observed_counts = _dense_triplet_counts(observed_detail["supports"], triplets)
    metric_values = {metric: [] for metric in PRIMARY_METRICS}
    block_counts = np.empty((N_NULL, len(triplets)), dtype=np.float32)
    failures = []
    for index, realization in enumerate(simulated):
        try:
            metrics, detail = compute_direct_metrics(realization, min_active_bins=1)
            for metric in PRIMARY_METRICS:
                metric_values[metric].append(metrics[metric])
            block_counts[index] = _dense_triplet_counts(detail["supports"], triplets)
        except Exception as exc:  # pragma: no cover
            failures.append({"realization": index, "error": f"{type(exc).__name__}: {exc}"})
            block_counts[index] = np.nan
    valid = np.isfinite(block_counts).all(axis=1)
    valid_count = int(valid.sum())
    inferential_valid = valid_count >= MINIMUM_VALID
    comparison_rows = []
    if inferential_valid:
        for metric in PRIMARY_METRICS:
            values = np.asarray(metric_values[metric], dtype=float)
            comparison = standardize_against_null(observed_metrics[metric], values)
            comparison_rows.append(
                {
                    "recording": recording,
                    "DIV": div,
                    "plate": plate,
                    "culture": f"P{plate}",
                    "well": well,
                    "subject_id": f"P{plate}_{well}",
                    "model": output_prefix,
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
            observed_counts, block_counts[valid], n_bins=n_bins
        )
        residual_row = {
            "recording": recording,
            "DIV": div,
            "plate": plate,
            "culture": f"P{plate}",
            "well": well,
            "subject_id": f"P{plate}_{well}",
            "model": output_prefix,
            "n_model_nodes": n_nodes,
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
        "subject_id": f"P{plate}_{well}",
        "n_model_nodes": n_nodes,
        "n_bins": n_bins,
        "n_requested_nulls": N_NULL,
        "n_valid_nulls": valid_count,
        "inferential_valid": inferential_valid,
        **qc,
        "pairwise_fit_firing_rmse": float(fit.pairwise_firing_rmse),
        "pairwise_fit_pair_rmse": float(fit.pairwise_pair_rmse),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "toolbox_logpotential_version": 2,
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
    parser.add_argument("--model-folder", default="maxent_models_top16")
    parser.add_argument("--output-prefix", default="instantaneous_null100_top16_dynamic")
    args = parser.parse_args()
    root = args.result_dir.resolve()
    files = sorted((root / "data_intermediate" / args.model_folder).glob("*_maxent.mat"))[
        : args.max_files
    ]
    analysis = ('dynamic16' if args.model_folder == 'maxent_models_top16'
                else 'fixed' + args.model_folder.rsplit('_n', 1)[1])
    eligibility = pd.read_csv(root / 'tables/computational_closure_v2/requested_node_eligibility.csv')
    if {p.stem.removesuffix('_maxent') for p in files} != expected_keys(eligibility, analysis):
        raise ValueError('Model files differ from requested-N eligibility manifest')
    payloads = []
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(process_one, str(path), str(root), args.output_prefix): path for path in files
        }
        for done, future in enumerate(as_completed(future_map), start=1):
            path = future_map[future]
            try:
                payload = future.result()
                if args.output_prefix == 'instantaneous_null100':
                    # Canonical metadata only; preserve all numerical source values.
                    for row in payload['comparison_rows']:
                        row['model'] = 'pairwise_conditioned_k_100'
                    if payload['residual_row']:
                        payload['residual_row']['model'] = 'pairwise_conditioned_k_100'
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
    pd.DataFrame([payload["qc_row"] for payload in payloads]).to_csv(
        out / f"v2_{args.output_prefix}_qc.csv", index=False
    )
    pd.DataFrame(
        [row for payload in payloads for row in payload["comparison_rows"]]
    ).to_csv(out / f"v2_{args.output_prefix}_comparisons.csv", index=False)
    pd.DataFrame(
        [payload["residual_row"] for payload in payloads if payload["residual_row"]]
    ).to_csv(out / f"v2_{args.output_prefix}_triplet_residuals.csv", index=False)
    (root / "logs" / f"v2_{args.output_prefix}_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"{len(failures)} instantaneous-null recordings failed")


if __name__ == "__main__":
    main()
