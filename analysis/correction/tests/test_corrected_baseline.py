from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from build_corrected_baseline import (
    EXCLUDED_RECORDING,
    FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    build_corrected_baseline,
)
from run_v2_activity_adjustment import _prior_feature_matrix as activity_prior_feature_matrix
from run_v2_incremental_value import _prior_feature_matrix as incremental_prior_feature_matrix


def _recordings() -> list[str]:
    return [
        f"WT_DIV{div}_P{plate}_{well}"
        for div in (14, 21, 28, 35, 42, 49, 56, 63)
        for plate in (1, 2, 3)
        for well in ("A1", "A2", "A3", "B1", "B2", "B3")
        if f"WT_DIV{div}_P{plate}_{well}" != EXCLUDED_RECORDING
    ]


def _all_metrics() -> pd.DataFrame:
    rows = []
    for index, recording in enumerate(_recordings()):
        _, div_token, plate_token, well = recording.split("_")
        div = int(div_token.removeprefix("DIV"))
        plate_index = int(plate_token.removeprefix("P"))
        row = {
            "SourceRecording": recording,
            "SourceBatch": "NewWT_P3" if plate_index == 3 else "PreviousWT_P1P2",
            "DIV": div,
            "PlateIndex": plate_index,
            "Plate": f"P{plate_index}",
            "Well": well,
            **{feature: float(index + offset + 1) for offset, feature in enumerate(FEATURE_COLUMNS)},
            "new_pca_only": 999.0,
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.loc[frame["SourceRecording"] == "WT_DIV14_P1_A1", "NModules"] = np.nan
    frame.loc[frame["SourceRecording"] == "WT_DIV21_P2_B3", "NModules"] = np.inf
    return frame


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    all_metrics = _all_metrics()
    old = all_metrics[[*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS]].copy()
    old.loc[:, FEATURE_COLUMNS] = old[FEATURE_COLUMNS].fillna(0).replace(np.inf, 0)
    provenance = pd.DataFrame({"SourceRecording": [*_recordings(), EXCLUDED_RECORDING]})
    role = all_metrics[["SourceRecording", "NC7_UltraPeripheral"]]
    boundary = pd.DataFrame({"n_nodes": [8760], "seed": [20260912]})
    control = all_metrics[["SourceRecording", "ControlEnergyMeanPerElectrode"]]
    upstream_imputation = pd.DataFrame({"feature": ["placeholder"], "median": [1.0]})
    paths = {}
    for name, frame in {
        "all_metrics": all_metrics,
        "old": old,
        "provenance": provenance,
        "role": role,
        "boundary": boundary,
        "control": control,
        "upstream_imputation": upstream_imputation,
    }.items():
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths


def test_builds_exact_original27_bridge_and_audits_nonfinite_cells(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    out = tmp_path / "bridge"

    summary = build_corrected_baseline(output_dir=out, **paths)

    pre = pd.read_csv(out / "WT_development_original27_corrected_filtered_preimputation.csv")
    compat = pd.read_csv(out / "WT_development_original27_corrected_globalmedian_compat.csv")
    assert pre.columns.tolist() == [*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS]
    assert compat.columns.tolist() == [*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS]
    assert len(pre) == pre["SourceRecording"].nunique() == 143
    assert np.isfinite(compat[FEATURE_COLUMNS].to_numpy(dtype=float)).all()
    expected_median = float(
        pd.to_numeric(_all_metrics()["NModules"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .median()
    )
    assert compat.loc[compat["SourceRecording"] == "WT_DIV14_P1_A1", "NModules"].item() == expected_median
    assert compat.loc[compat["SourceRecording"] == "WT_DIV21_P2_B3", "NModules"].item() == expected_median
    audit = pd.read_csv(out / "baseline_imputation_audit.csv")
    assert audit[["SourceRecording", "feature"]].to_dict("records") == [
        {"SourceRecording": "WT_DIV14_P1_A1", "feature": "NModules"},
        {"SourceRecording": "WT_DIV21_P2_B3", "feature": "NModules"},
    ]
    assert set(audit["input_status"]) == {"missing", "positive_infinity"}
    assert audit["imputed_value"].eq(expected_median).all()
    key_audit = json.loads((out / "baseline_key_audit.json").read_text())
    assert key_audit["input_provenance_rows"] == 144
    assert key_audit["excluded_recordings"] == [EXCLUDED_RECORDING]
    assert key_audit["plate_counts"] == {"P1": 48, "P2": 47, "P3": 48}
    assert summary["n_features"] == 27
    assert summary["n_imputed_cells"] == 2


def test_rejects_missing_or_unexpected_provenance_keys(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    provenance = pd.read_csv(paths["provenance"])
    provenance.loc[0, "SourceRecording"] = "WT_DIV14_P4_A1"
    provenance.to_csv(paths["provenance"], index=False)

    with pytest.raises(ValueError, match="provenance key closure"):
        build_corrected_baseline(output_dir=tmp_path / "bridge", **paths)


def test_hash_manifest_covers_every_declared_input_and_output(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    out = tmp_path / "bridge"
    build_corrected_baseline(output_dir=out, **paths)

    hashes = json.loads((out / "baseline_input_hashes.json").read_text())
    assert set(hashes["inputs"]) == set(paths)
    assert all(item["sha256"] and item["row_count"] is not None for item in hashes["inputs"].values())
    assert set(hashes["outputs"]) == {
        "compatibility_bridge",
        "filtered_preimputation_bridge",
        "feature_manifest",
        "imputation_audit",
        "key_audit",
    }


def test_all_consumers_resolve_prior_matrix_from_result_config(tmp_path: Path) -> None:
    configured = tmp_path / "bridge.csv"
    (tmp_path / "analysis_config.yaml").write_text(
        "computational_closure_v2:\n  prior_feature_matrix: '" + configured.as_posix() + "'\n",
        encoding="utf-8",
    )
    assert activity_prior_feature_matrix(tmp_path) == configured
    assert incremental_prior_feature_matrix(tmp_path) == configured


def test_accepts_recording_key_name_used_by_input_provenance(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    provenance = pd.read_csv(paths["provenance"]).rename(columns={"SourceRecording": "recording"})
    provenance.to_csv(paths["provenance"], index=False)
    summary = build_corrected_baseline(output_dir=tmp_path / "bridge", **paths)
    assert summary["n_rows"] == 143
