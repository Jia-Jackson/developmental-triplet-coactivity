from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from hotopology.encoding import bin_spikes
from hotopology.io import load_recording_mat
from hotopology.metrics import compute_direct_metrics
from hotopology.stability import temporal_windows
from requested_node_eligibility import expected_keys


BIN_WIDTHS = [10, 25, 50, 100]
KEY_METRICS = [
    "mean_population_count",
    "fraction_bins_k_ge_3",
    "triplet_occurrence_density",
    "triplet_repeated5_density",
    "triplet_support_mass",
    "h1_total_persistence_capped",
    "h1_infinite_count",
]


def process_one(mat_path: str, nodes: list[int]) -> dict[str, object]:
    rec = load_recording_mat(mat_path)
    base = {
        "recording": rec.recording,
        "DIV": rec.div,
        "plate": rec.plate,
        "culture": f"P{rec.plate}",
        "well": rec.well,
        "subject_id": f"P{rec.plate}_{rec.well}",
        "node_selection": "fixed_longitudinal_maximin_n16",
        "n_nodes_requested": 16,
    }
    bin_rows = []
    x25 = None
    for bin_ms in BIN_WIDTHS:
        x = bin_spikes(
            rec.spike_times_s,
            rec.channel_indices,
            n_channels=64,
            duration_s=rec.duration_s,
            bin_s=bin_ms / 1000.0,
        )[np.asarray(nodes, dtype=int)]
        metrics, _ = compute_direct_metrics(x, min_active_bins=1)
        bin_rows.append({**base, "bin_ms": bin_ms, **metrics})
        if bin_ms == 25:
            x25 = x
    if x25 is None:
        raise RuntimeError("25-ms representation was not generated")
    segment_rows = []
    for label, window in temporal_windows(n_bins=x25.shape[1], bin_ms=25).items():
        segment = x25[:, window]
        metrics, _ = compute_direct_metrics(segment, min_active_bins=1)
        if label.startswith("half_"):
            kind = "half"
        elif label.startswith("block"):
            kind = "block_50s"
        else:
            kind = "cumulative"
        segment_rows.append(
            {
                **base,
                "segment_kind": kind,
                "segment_label": label,
                "segment_start_bin": int(window.start or 0),
                "segment_stop_bin": int(window.stop or x25.shape[1]),
                "segment_duration_s": float((window.stop - (window.start or 0)) * 0.025),
                **metrics,
            }
        )
    return {"bin_rows": bin_rows, "segment_rows": segment_rows}


def _spearman(a: pd.Series, b: pd.Series) -> tuple[float, float, int]:
    paired = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(paired) < 3 or paired["a"].nunique() < 2 or paired["b"].nunique() < 2:
        return np.nan, np.nan, len(paired)
    rho, p = stats.spearmanr(paired["a"], paired["b"])
    return float(rho), float(p), len(paired)


def _summaries(bin_table: pd.DataFrame, segment_table: pd.DataFrame):
    bin_wide = bin_table.pivot(index="recording", columns="bin_ms", values=KEY_METRICS)
    bin_agreement = []
    for metric in KEY_METRICS:
        for bin_ms in [10, 50, 100]:
            rho, p, n = _spearman(bin_wide[(metric, 25)], bin_wide[(metric, bin_ms)])
            bin_agreement.append(
                {
                    "metric": metric,
                    "reference_bin_ms": 25,
                    "comparison_bin_ms": bin_ms,
                    "n_recordings": n,
                    "spearman_rho": rho,
                    "raw_p": p,
                }
            )

    transition_rows = []
    for bin_ms, subset in bin_table.groupby("bin_ms"):
        for metric in KEY_METRICS:
            left = subset.loc[subset["DIV"] == 14, ["subject_id", metric]].rename(columns={metric: "first"})
            right = subset.loc[subset["DIV"] == 21, ["subject_id", metric]].rename(columns={metric: "second"})
            paired = left.merge(right, on="subject_id", how="inner").dropna()
            change = paired["second"] - paired["first"]
            transition_rows.append(
                {
                    "bin_ms": bin_ms,
                    "metric": metric,
                    "n_pairs": len(paired),
                    "mean_change": float(change.mean()),
                    "median_change": float(change.median()),
                    "fraction_positive": float(np.mean(change > 0)),
                }
            )

    halves = segment_table.loc[segment_table["segment_kind"] == "half"]
    half_wide = halves.pivot(index="recording", columns="segment_label", values=KEY_METRICS)
    reliability_rows = []
    for metric in KEY_METRICS:
        rho, p, n = _spearman(half_wide[(metric, "half_first")], half_wide[(metric, "half_second")])
        reliability_rows.append(
            {
                "metric": metric,
                "n_recordings": n,
                "split_half_spearman_rho": rho,
                "raw_p": p,
            }
        )

    cumulative = segment_table.loc[segment_table["segment_kind"] == "cumulative"].copy()
    cumulative["duration_s"] = cumulative["segment_label"].str.extract(r"(\d+)").astype(int)
    duration_rows = []
    for metric in KEY_METRICS:
        full = cumulative.loc[cumulative["duration_s"] == 300, ["recording", metric]].rename(columns={metric: "full"})
        for duration, subset in cumulative.groupby("duration_s"):
            paired = subset[["recording", metric]].merge(full, on="recording", how="inner").dropna()
            rho, p, n = _spearman(paired[metric], paired["full"])
            scale = np.maximum(np.abs(paired["full"].to_numpy(float)), 1e-12)
            relative = np.abs(paired[metric].to_numpy(float) - paired["full"].to_numpy(float)) / scale
            duration_rows.append(
                {
                    "metric": metric,
                    "duration_s": int(duration),
                    "n_recordings": n,
                    "spearman_vs_300s": rho,
                    "raw_p": p,
                    "median_absolute_relative_error": float(np.median(relative)),
                }
            )
    return (
        pd.DataFrame(bin_agreement),
        pd.DataFrame(transition_rows),
        pd.DataFrame(reliability_rows),
        pd.DataFrame(duration_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    args = parser.parse_args()
    root = args.result_dir.resolve()
    manifest = pd.read_csv(
        root / "tables" / "computational_closure_v2" / "v2_fixed_node_manifest_long.csv"
    )
    manifest = manifest.loc[manifest["n_nodes"] == 16].copy()
    node_map = {
        recording: subset.sort_values("fixed_rank")["electrode_index_zero_based"].astype(int).tolist()
        for recording, subset in manifest.groupby("recording")
    }
    paths = sorted((root / "data_intermediate" / "spike_arrays").glob("*_arrays.mat"))
    eligibility = pd.read_csv(root / 'tables/computational_closure_v2/requested_node_eligibility.csv')
    required = expected_keys(eligibility, 'fixed16')
    if set(node_map) != required:
        raise ValueError('Fixed16 node manifest differs from explicit eligibility')
    paths = [path for path in paths if path.stem.removesuffix('_arrays') in required]
    if {path.stem.removesuffix('_arrays') for path in paths} != required:
        raise ValueError('Missing required stability input')
    payloads = []
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_map = {}
        for path in paths:
            recording = path.stem.removesuffix("_arrays")
            future_map[pool.submit(process_one, str(path), node_map[recording])] = path
        for done, future in enumerate(as_completed(future_map), start=1):
            path = future_map[future]
            try:
                payloads.append(future.result())
                print(f"[{done:03d}/{len(paths):03d}] {path.stem}", flush=True)
            except Exception as exc:
                failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})

    if failures:
        (root / "logs" / "v2_stability_failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"{len(failures)} stability recordings failed")
    bin_table = pd.DataFrame([row for payload in payloads for row in payload["bin_rows"]])
    segment_table = pd.DataFrame([row for payload in payloads for row in payload["segment_rows"]])
    bin_table = bin_table.sort_values(["bin_ms", "DIV", "plate", "well"])
    segment_table = segment_table.sort_values(["segment_kind", "segment_label", "DIV", "plate", "well"])
    summaries = _summaries(bin_table, segment_table)
    out = root / "tables" / "computational_closure_v2"
    bin_table.to_csv(out / "v2_fixed_n16_bin_width_metrics.csv", index=False)
    segment_table.to_csv(out / "v2_fixed_n16_segment_metrics.csv", index=False)
    for table, filename in zip(
        summaries,
        [
            "v2_bin_width_rank_agreement.csv",
            "v2_bin_width_div14_21_changes.csv",
            "v2_split_half_reliability.csv",
            "v2_duration_convergence.csv",
        ],
    ):
        table.to_csv(out / filename, index=False)
    print(
        json.dumps(
            {
                "n_recordings": len(paths),
                "n_bin_rows": len(bin_table),
                "n_segment_rows": len(segment_table),
                "failures": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
