from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


IDENTIFIER_COLUMNS = ["SourceRecording", "SourceBatch", "DIV", "PlateIndex", "Plate", "Well"]
FEATURE_COLUMNS = [
    "mean_firing_rate_hz_all", "active_electrode_fraction", "network_burst_rate_per_min",
    "mean_network_burst_duration_s", "frac_spikes_in_network_bursts",
    "cv_inter_network_burst_interval", "ControlEnergyMeanPerElectrode", "EdgeWeight",
    "NodeStrength", "NetworkDensity", "NModules", "Modularity", "SmallWorldnessSigma",
    "SmallWorldnessOmega", "NMFComponents", "EffectiveRank", "NC7_UltraPeripheral",
    "NC7_Peripheral", "NC7_NonHubConnector", "NC7_NonHubKinless", "NC7_ProvincialHub",
    "NC7_ConnectorHub", "network_burst_threshold_s", "median_network_burst_duration_s",
    "mean_channels_in_network_bursts", "mean_isi_within_network_bursts_ms",
    "mean_isi_outside_network_bursts_ms",
]
EXCLUDED_RECORDING = "WT_DIV35_P2_A1"
PREPROCESSING_VERSION = "original27_corrected_globalmedian_compat_v1"
RECORDING_RE = re.compile(r"^WT_DIV(14|21|28|35|42|49|56|63)_P([123])_([AB][123])$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_count(path: Path) -> int | None:
    if path.suffix.lower() == ".csv":
        return int(len(pd.read_csv(path)))
    return None


def _expected_keys(include_excluded: bool) -> set[str]:
    keys = {
        f"WT_DIV{div}_P{plate}_{well}"
        for div in (14, 21, 28, 35, 42, 49, 56, 63)
        for plate in (1, 2, 3)
        for well in ("A1", "A2", "A3", "B1", "B2", "B3")
    }
    if not include_excluded:
        keys.remove(EXCLUDED_RECORDING)
    return keys


def _validate_keys(frame: pd.DataFrame, expected: set[str], label: str) -> None:
    if "SourceRecording" not in frame:
        raise ValueError(f"{label} lacks SourceRecording")
    keys = frame["SourceRecording"].astype(str)
    duplicates = sorted(keys[keys.duplicated()].unique())
    missing = sorted(expected - set(keys))
    unexpected = sorted(set(keys) - expected)
    if duplicates or missing or unexpected or len(frame) != len(expected):
        raise ValueError(
            f"{label} key closure failed: duplicates={duplicates}, missing={missing}, unexpected={unexpected}"
        )


def _validate_metadata(frame: pd.DataFrame) -> None:
    for row in frame.itertuples(index=False):
        match = RECORDING_RE.fullmatch(str(row.SourceRecording))
        if match is None:
            raise ValueError(f"invalid SourceRecording: {row.SourceRecording}")
        div, plate, well = int(match.group(1)), int(match.group(2)), match.group(3)
        expected_batch = "NewWT_P3" if plate == 3 else "PreviousWT_P1P2"
        if (
            int(row.DIV) != div or int(row.PlateIndex) != plate or str(row.Plate) != f"P{plate}"
            or str(row.Well) != well or str(row.SourceBatch) != expected_batch
        ):
            raise ValueError(f"metadata mismatch for {row.SourceRecording}")


def _input_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "row_count": _row_count(path)}


def build_corrected_baseline(
    *, all_metrics: Path, old: Path, provenance: Path, role: Path, boundary: Path,
    control: Path, upstream_imputation: Path, output_dir: Path,
) -> dict[str, Any]:
    paths = {
        "all_metrics": Path(all_metrics), "old": Path(old), "provenance": Path(provenance),
        "role": Path(role), "boundary": Path(boundary), "control": Path(control),
        "upstream_imputation": Path(upstream_imputation),
    }
    input_hashes = {name: _input_record(path) for name, path in paths.items()}
    source = pd.read_csv(paths["all_metrics"])
    old_frame = pd.read_csv(paths["old"])
    provenance_frame = pd.read_csv(paths["provenance"])
    if "SourceRecording" not in provenance_frame and "recording" in provenance_frame:
        provenance_frame = provenance_frame.rename(columns={"recording": "SourceRecording"})
    _validate_keys(provenance_frame, _expected_keys(include_excluded=True), "provenance")
    _validate_keys(source, _expected_keys(include_excluded=False), "all_metrics")
    if old_frame.columns[:6].tolist() != IDENTIFIER_COLUMNS:
        raise ValueError("old schema source does not begin with the six frozen identifiers")
    old_features = [column for column in old_frame.columns if column not in IDENTIFIER_COLUMNS]
    if old_features != FEATURE_COLUMNS:
        raise ValueError("old schema source does not contain the exact ordered original 27 features")
    missing_columns = [column for column in [*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS] if column not in source]
    if missing_columns:
        raise ValueError(f"all_metrics missing required columns: {missing_columns}")

    pre = source[[*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS]].copy()
    _validate_metadata(pre)
    pre["DIV"] = pd.to_numeric(pre["DIV"], errors="raise").astype(int)
    pre["PlateIndex"] = pd.to_numeric(pre["PlateIndex"], errors="raise").astype(int)
    for feature in FEATURE_COLUMNS:
        pre[feature] = pd.to_numeric(pre[feature], errors="coerce")
    pre = pre.sort_values(["DIV", "PlateIndex", "Well"], kind="stable").reset_index(drop=True)
    compat = pre.copy()
    imputation_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for position, feature in enumerate(FEATURE_COLUMNS, start=1):
        values = pre[feature].to_numpy(dtype=float)
        finite = np.isfinite(values)
        median = float(np.median(values[finite])) if finite.any() else 0.0
        for index in np.flatnonzero(~finite):
            value = values[index]
            status = "missing" if np.isnan(value) else ("positive_infinity" if value > 0 else "negative_infinity")
            imputation_rows.append({
                "SourceRecording": pre.iloc[index]["SourceRecording"], "feature": feature,
                "input_status": status, "imputed_value": median, "feature_median": median,
                "method": "global_finite_median" if finite.any() else "all_nonfinite_fallback_zero",
            })
        compat.loc[~finite, feature] = median
        manifest_rows.append({
            "position": position, "feature": feature, "source_column": feature,
            "source_file": str(paths["all_metrics"].resolve()), "finite_count": int(finite.sum()),
            "nonfinite_count": int((~finite).sum()), "global_finite_median": median,
            "imputed_count": int((~finite).sum()), "preprocessing_version": PREPROCESSING_VERSION,
        })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pre_path = output_dir / "WT_development_original27_corrected_filtered_preimputation.csv"
    compat_path = output_dir / "WT_development_original27_corrected_globalmedian_compat.csv"
    manifest_path = output_dir / "baseline_feature_manifest.csv"
    audit_path = output_dir / "baseline_imputation_audit.csv"
    key_path = output_dir / "baseline_key_audit.json"
    pre.to_csv(pre_path, index=False)
    compat.to_csv(compat_path, index=False)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    pd.DataFrame(imputation_rows, columns=[
        "SourceRecording", "feature", "input_status", "imputed_value", "feature_median", "method"
    ]).to_csv(audit_path, index=False)

    old_by_key = old_frame.set_index("SourceRecording")
    pre_by_key = pre.set_index("SourceRecording")
    p12_keys = pre.loc[pre["PlateIndex"].isin([1, 2]), "SourceRecording"]
    role_features = [feature for feature in FEATURE_COLUMNS if feature.startswith("NC7_")]
    shared_role_changes = sum(
        not np.isclose(float(old_by_key.loc[key, feature]), float(pre_by_key.loc[key, feature]), equal_nan=True)
        for key in p12_keys for feature in role_features
    )
    key_audit = {
        "status": "pass", "preprocessing_version": PREPROCESSING_VERSION,
        "compatibility_limitation": (
            "Global feature medians use all 143 retained recordings; corrected masks are pooled and DIV-dependent. "
            "This bridge preserves original preprocessing compatibility and is not leakage-free predictive validation."
        ),
        "input_provenance_rows": int(len(provenance_frame)), "retained_rows": int(len(pre)),
        "excluded_recordings": [EXCLUDED_RECORDING],
        "plate_counts": {key: int(value) for key, value in pre["Plate"].value_counts().sort_index().items()},
        "unique_source_recordings": int(pre["SourceRecording"].nunique()), "feature_count": len(FEATURE_COLUMNS),
        "all_compatibility_values_finite": bool(np.isfinite(compat[FEATURE_COLUMNS].to_numpy()).all()),
        "p1p2_change_categories": {
            "shared_role_reclassification_cells": int(shared_role_changes),
            "pooled_mask_nonfinite_cells": int((~np.isfinite(pre.loc[pre["PlateIndex"].isin([1, 2]), FEATURE_COLUMNS])).sum().sum()),
            "changed_global_median_features": int(sum(
                not np.isclose(
                    pd.to_numeric(old_frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).median(),
                    pd.to_numeric(pre[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).median(),
                    equal_nan=True,
                ) for feature in FEATURE_COLUMNS
            )),
        },
    }
    key_path.write_text(json.dumps(key_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs = {
        "compatibility_bridge": compat_path, "filtered_preimputation_bridge": pre_path,
        "feature_manifest": manifest_path, "imputation_audit": audit_path, "key_audit": key_path,
    }
    hash_path = output_dir / "baseline_input_hashes.json"
    hash_payload = {
        "preprocessing_version": PREPROCESSING_VERSION, "inputs": input_hashes,
        "outputs": {name: _input_record(path) for name, path in outputs.items()},
    }
    hash_path.write_text(json.dumps(hash_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"n_rows": len(pre), "n_features": len(FEATURE_COLUMNS), "n_imputed_cells": len(imputation_rows),
            "compatibility_bridge": str(compat_path.resolve()), "sha256": _sha256(compat_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the audited corrected original-27 baseline bridge.")
    parser.add_argument("--all-metrics", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--role", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--upstream-imputation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_corrected_baseline(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
