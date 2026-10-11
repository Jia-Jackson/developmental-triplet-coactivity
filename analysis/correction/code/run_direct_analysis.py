from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

from hotopology.bursts import detect_network_bursts
from hotopology.encoding import bin_spikes
from hotopology.io import load_recording_mat
from hotopology.metrics import compute_direct_metrics


BIN_SIZES_MS = (10, 25, 50, 100)
PRIMARY_BIN_MS = 25
PIPELINE_VERSION = 3


def _json_ready(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def process_one(mat_path: str, result_dir: str) -> dict[str, object]:
    result = Path(result_dir)
    checkpoint_dir = result / "data_intermediate" / "direct_checkpoints"
    binary_dir = result / "data_intermediate" / "binary_25ms"
    triplet_dir = result / "data_intermediate" / "triplets_25ms"
    for directory in (checkpoint_dir, binary_dir, triplet_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rec = load_recording_mat(mat_path)
    checkpoint = checkpoint_dir / f"{rec.recording}.json"
    if checkpoint.exists():
        existing = json.loads(checkpoint.read_text(encoding="utf-8"))
        if existing.get("pipeline_version") == PIPELINE_VERSION:
            return existing

    burst = detect_network_bursts(
        rec.spike_times_s,
        rec.channel_indices,
        duration_s=rec.duration_s,
        fs=rec.fs,
    )
    base = {
        "recording": rec.recording,
        "DIV": rec.div,
        "plate": rec.plate,
        "well": rec.well,
        "subject_id": f"P{rec.plate}_{rec.well}",
        "duration_s": rec.duration_s,
        "fs": rec.fs,
        "n_spikes": int(rec.spike_times_s.size),
        "network_burst_threshold_s": burst.threshold_s,
        "n_network_bursts": int(burst.intervals.shape[0]),
        "fraction_spikes_in_network_bursts": (
            float(burst.spike_mask.mean()) if burst.spike_mask.size else 0.0
        ),
        "mean_channels_per_network_burst": (
            float(burst.n_channels.mean()) if burst.n_channels.size else 0.0
        ),
    }
    rows: list[dict[str, object]] = []
    burst_rows: list[dict[str, object]] = []
    for bin_ms in BIN_SIZES_MS:
        bin_s = bin_ms / 1000.0
        x = bin_spikes(
            rec.spike_times_s,
            rec.channel_indices,
            n_channels=64,
            duration_s=rec.duration_s,
            bin_s=bin_s,
        )
        minimum_active_bins = max(1, int(np.ceil(0.01 * rec.duration_s)))
        metrics, detail = compute_direct_metrics(x, min_active_bins=minimum_active_bins)
        row = dict(base)
        row.update({"bin_ms": bin_ms})
        row.update(metrics)
        row["fraction_nonempty_bins"] = float(np.mean(x.sum(axis=0) > 0))
        row["fraction_bins_k_ge_half_nodes"] = (
            float(np.mean(detail["population_count"] >= max(1, metrics["n_analysis_nodes"] / 2)))
            if metrics["n_analysis_nodes"]
            else 0.0
        )
        row["n_model_eligible_nodes_20bins"] = int(np.sum(x.sum(axis=1) >= 20))
        rows.append(row)

        if bin_ms == PRIMARY_BIN_MS:
            np.savez_compressed(
                binary_dir / f"{rec.recording}_25ms.npz",
                x=x,
                recording=rec.recording,
                div=rec.div,
                plate=rec.plate,
                well=rec.well,
                duration_s=rec.duration_s,
                fs=rec.fs,
            )
            savemat(
                binary_dir / f"{rec.recording}_25ms.mat",
                {
                    "x": x,
                    "recording": rec.recording,
                    "DIV": rec.div,
                    "plate": rec.plate,
                    "well": rec.well,
                    "duration_s": rec.duration_s,
                    "fs": rec.fs,
                },
                do_compression=True,
            )
            active_nodes = np.asarray(detail["active_nodes"], dtype=np.int16)
            triplets = detail["supports"].get(3, {})
            if triplets:
                compressed_nodes = np.asarray(list(triplets.keys()), dtype=np.int16)
                original_nodes = active_nodes[compressed_nodes]
                counts = np.asarray(list(triplets.values()), dtype=np.int32)
            else:
                original_nodes = np.empty((0, 3), dtype=np.int16)
                counts = np.empty(0, dtype=np.int32)
            np.savez_compressed(
                triplet_dir / f"{rec.recording}_triplets_25ms.npz",
                nodes=original_nodes,
                counts=counts,
                active_nodes=active_nodes,
                n_bins=x.shape[1],
            )

            burst_bin_mask = burst.bin_mask(bin_s=bin_s, n_bins=x.shape[1])
            for label, mask in (
                ("inside_burst", burst_bin_mask),
                ("outside_burst", ~burst_bin_mask),
            ):
                if int(mask.sum()) == 0:
                    continue
                stratum_metrics, _ = compute_direct_metrics(
                    x[:, mask], min_active_bins=1
                )
                stratum_row = dict(base)
                stratum_row.update(
                    {
                        "bin_ms": bin_ms,
                        "stratum": label,
                        "n_stratum_bins": int(mask.sum()),
                    }
                )
                stratum_row.update(stratum_metrics)
                burst_rows.append(stratum_row)

    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "recording": rec.recording,
        "direct_rows": rows,
        "burst_rows": burst_rows,
    }
    checkpoint.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_ready),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    args = parser.parse_args()
    result = Path(args.result_dir)
    spike_dir = result / "data_intermediate" / "spike_arrays"
    files = sorted(str(path) for path in spike_dir.glob("*_arrays.mat"))
    if len(files) != 143:
        raise RuntimeError(f"Expected 143 retained recordings, found {len(files)}")

    payloads: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(process_one, path, str(result)): path for path in files}
        for completed, future in enumerate(as_completed(future_map), start=1):
            path = future_map[future]
            try:
                payloads.append(future.result())
                print(f"[{completed:03d}/{len(files):03d}] completed {Path(path).stem}", flush=True)
            except Exception as exc:  # pragma: no cover - integration failure log
                failures.append({"path": path, "error": repr(exc)})
                print(f"FAILED {path}: {exc!r}", flush=True)

    table_dir = result / "tables"
    log_dir = result / "logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    direct_rows = [row for payload in payloads for row in payload["direct_rows"]]
    burst_rows = [row for payload in payloads for row in payload["burst_rows"]]
    direct = pd.DataFrame(direct_rows).sort_values(["DIV", "plate", "well", "bin_ms"])
    burst_table = pd.DataFrame(burst_rows).sort_values(
        ["DIV", "plate", "well", "stratum"]
    )
    direct.to_csv(table_dir / "direct_higher_order_metrics_all_bins.csv", index=False)
    burst_table.to_csv(table_dir / "burst_stratified_higher_order_metrics_25ms.csv", index=False)
    (log_dir / "direct_analysis_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"{len(failures)} recordings failed; see failure log")


if __name__ == "__main__":
    main()
