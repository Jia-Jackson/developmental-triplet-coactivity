from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from hotopology.v2_contract import (
    audit_recording_table,
    audited_one_to_one_merge,
    expected_recording_grid,
    file_sha256,
    normalize_recording_keys,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    config = yaml.safe_load((root / "analysis_config.yaml").read_text(encoding="utf-8"))
    v2 = config["computational_closure_v2"]

    expected = expected_recording_grid(
        divs=config["design"]["div"],
        cultures=range(1, int(config["design"]["plates"]) + 1),
        wells=["A1", "A2", "A3", "B1", "B2", "B3"],
        excluded_recording=config["design"]["excluded_recording"],
    )
    direct = pd.read_csv(root / "tables" / "direct_higher_order_metrics_all_bins.csv")
    primary = direct.loc[direct["bin_ms"] == config["encoding"]["primary_bin_ms"]].copy()
    baseline = pd.read_csv(Path(v2["prior_feature_matrix"]))

    direct_audit = audit_recording_table(primary, expected)
    baseline_audit = audit_recording_table(baseline, expected)
    merged = audited_one_to_one_merge(primary, baseline, suffixes=("_higher", "_prior"))

    legacy_report = Path("D:/Published paper/high-order-topology/Data_processing/Result/Higher_order_topology_analysis_report.docx")
    actual_hash = file_sha256(legacy_report)
    expected_hash = str(v2["legacy_report_sha256"]).lower()
    if actual_hash.lower() != expected_hash:
        raise RuntimeError(
            f"legacy report fingerprint changed: expected {expected_hash}, got {actual_hash}"
        )

    qa_dir = root / "qa"
    table_dir = root / "tables" / "computational_closure_v2"
    qa_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    normalized_baseline = normalize_recording_keys(baseline)
    baseline_features = [
        column
        for column in baseline.columns
        if column
        not in {
            "SourceRecording",
            "SourceBatch",
            "DIV",
            "PlateIndex",
            "Plate",
            "Well",
        }
    ]
    if len(baseline_features) != 27:
        raise RuntimeError(f"expected 27 prior features, found {len(baseline_features)}")

    key_table = normalized_baseline[
        ["recording", "DIV", "plate", "culture", "well", "subject_id"]
    ].sort_values(["DIV", "plate", "well"])
    key_table.to_csv(table_dir / "v2_recording_key_manifest.csv", index=False)
    pd.DataFrame({"feature": baseline_features, "baseline": "previous_27"}).to_csv(
        table_dir / "v2_prior_27_feature_manifest.csv", index=False
    )

    audit = {
        "status": "pass",
        "result_root": str(root),
        "direct_primary": direct_audit,
        "prior_27_features": baseline_audit,
        "one_to_one_merge_rows": int(len(merged)),
        "n_prior_features": len(baseline_features),
        "cultures_are_independent_preparations": bool(
            v2["culture_is_independent_preparation"]
        ),
        "legacy_report_sha256": actual_hash,
    }
    (qa_dir / "v2_preflight_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
