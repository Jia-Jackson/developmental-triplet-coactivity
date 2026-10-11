from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from .io_utils import atomic_json
from .vendor_hotopology.incremental import (
    compare_feature_sets,
    prediction_metrics,
)


ALPHAS = np.logspace(-3, 3, 13).tolist()
IDENTIFIERS = {"SourceRecording", "SourceBatch", "DIV", "PlateIndex", "Plate", "Well"}
LEGACY_HIGHER = [
    "log1p_absolute_residual_mass",
    "absolute_residual_z",
    "positive_residual_z",
    "h1_persistence_deviation_z",
    "h1_infinite_deviation_z",
    "direct_triplet_repeated5_density",
    "direct_h1_total_persistence_capped",
]


def clustered_delta_mae_bootstrap_fast(
    predictions: pd.DataFrame,
    *,
    baseline_name: str,
    augmented_name: str,
    subject_col: str = "subject_id",
    culture_col: str = "culture",
    n_bootstrap: int = 5000,
    seed: int = 0,
) -> dict[str, float | int]:
    index_columns = [
        column for column in ("row_index", subject_col, culture_col, "observed")
        if column in predictions.columns
    ]
    wide = predictions.pivot_table(
        index=index_columns, columns="model_set", values="prediction", aggfunc="first"
    ).reset_index()
    if not {baseline_name, augmented_name}.issubset(wide.columns):
        raise ValueError("baseline or augmented predictions are missing")
    wide["baseline_error"] = np.abs(wide["observed"] - wide[baseline_name])
    wide["augmented_error"] = np.abs(wide["observed"] - wide[augmented_name])
    observed_delta = float(wide["baseline_error"].mean() - wide["augmented_error"].mean())
    subject = (
        wide.groupby([culture_col, subject_col])
        .agg(
            baseline_sum=("baseline_error", "sum"),
            augmented_sum=("augmented_error", "sum"),
            n=("observed", "size"),
        )
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    baseline_sum = np.zeros(n_bootstrap, dtype=float)
    augmented_sum = np.zeros(n_bootstrap, dtype=float)
    count = np.zeros(n_bootstrap, dtype=float)
    for culture in sorted(subject[culture_col].unique()):
        group = subject[subject[culture_col] == culture].reset_index(drop=True)
        indices = rng.integers(0, len(group), size=(n_bootstrap, len(group)))
        baseline_sum += group["baseline_sum"].to_numpy(float)[indices].sum(axis=1)
        augmented_sum += group["augmented_sum"].to_numpy(float)[indices].sum(axis=1)
        count += group["n"].to_numpy(float)[indices].sum(axis=1)
    delta = baseline_sum / count - augmented_sum / count
    return {
        "n_bootstrap": int(n_bootstrap),
        "delta_mae_baseline_minus_augmented": observed_delta,
        "ci95_low": float(np.quantile(delta, 0.025)),
        "ci95_high": float(np.quantile(delta, 0.975)),
        "probability_improvement": float(np.mean(delta > 0)),
    }


def _assemble(source: Path, output: Path) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    baseline_path = (
        source / "data_intermediate/baseline_bridge/WT_development_original27_corrected_filtered_preimputation.csv"
    )
    baseline = pd.read_csv(baseline_path)
    baseline_features = [column for column in baseline.columns if column not in IDENTIFIERS]
    baseline = baseline.rename(columns={"SourceRecording": "recording", "Plate": "culture", "Well": "well"})
    baseline["subject_id"] = baseline["culture"].astype(str) + "_" + baseline["well"].astype(str)

    legacy_source = pd.read_csv(
        source / "tables/computational_closure_v2/v2_incremental_analysis_table.csv"
    )[["recording", *LEGACY_HIGHER]]

    oinfo = pd.read_csv(output / "tables/upgrade_v2/v2_dynamic16_oinfo_analysis_table.csv")
    observed = (
        oinfo.drop_duplicates(["recording", "metric"])
        .pivot(index="recording", columns="metric", values="observed")
        .add_prefix("oinfo_observed_")
    )
    deviations = (
        oinfo.pivot(index="recording", columns=["model", "metric"], values="null_deviation")
    )
    deviations.columns = [f"oinfo_deviation_{model}_{metric}" for model, metric in deviations.columns]
    oinfo_wide = observed.join(deviations).reset_index()
    oinfo_features = [column for column in oinfo_wide.columns if column != "recording"]

    table = baseline.merge(legacy_source, on="recording", how="inner", validate="one_to_one")
    table = table.merge(oinfo_wide, on="recording", how="inner", validate="one_to_one")
    for column in [*baseline_features, *LEGACY_HIGHER, *oinfo_features]:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if len(table) != 142 or table["recording"].duplicated().any():
        raise ValueError(f"expected 142 unique dynamic16 recordings, found {len(table)}")
    return table, baseline_features, LEGACY_HIGHER, oinfo_features


def _fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(
        ["analysis_range", "validation", "model_set", "outer_fold"], sort=True
    ):
        values = prediction_metrics(group)
        values.update({
            "analysis_range": keys[0], "validation": keys[1], "model_set": keys[2],
            "outer_fold": int(keys[3]), "held_out_group": group["held_out_group"].iloc[0],
        })
        rows.append(values)
    return pd.DataFrame(rows)


def run(source: Path, output: Path) -> None:
    table_dir = output / "tables/upgrade_v2"
    table, baseline, legacy, oinfo = _assemble(source, output)
    feature_sets = {
        "traditional_27": baseline,
        "legacy_higher_order_only": legacy,
        "oinfo_only": oinfo,
        "traditional_27_plus_legacy": [*baseline, *legacy],
        "traditional_27_plus_oinfo": [*baseline, *oinfo],
        "traditional_27_plus_all_higher_order": [*baseline, *legacy, *oinfo],
    }
    table.to_csv(table_dir / "v2_prediction_analysis_table_preimputation.csv", index=False)
    pd.DataFrame([
        {"model_set": model_set, "feature": feature}
        for model_set, features in feature_sets.items() for feature in features
    ]).to_csv(table_dir / "v2_prediction_feature_manifest.csv", index=False)

    predictions_all, metrics_all, audits_all, bootstrap_rows = [], [], [], []
    settings = [
        ("DIV14_63", table),
        ("DIV21_63", table[table["DIV"] >= 21].copy()),
    ]
    validations = [
        ("leave_one_culture_out", "culture", "subject_id"),
        ("leave_one_well_out", "subject_id", "subject_id"),
    ]
    for analysis_range, subset in settings:
        for validation, outer_group, inner_group in validations:
            predictions, metrics, audit = compare_feature_sets(
                subset, feature_sets=feature_sets, target="DIV",
                outer_group=outer_group, inner_group=inner_group, alphas=ALPHAS,
            )
            for frame in (predictions, metrics, audit):
                frame.insert(0, "validation", validation)
                frame.insert(0, "analysis_range", analysis_range)
            predictions_all.append(predictions)
            metrics_all.append(metrics)
            audits_all.append(audit)
            for augmented in (
                "traditional_27_plus_legacy",
                "traditional_27_plus_oinfo",
                "traditional_27_plus_all_higher_order",
            ):
                comparison_predictions = predictions[
                    predictions["model_set"].isin(["traditional_27", augmented])
                ]
                result = clustered_delta_mae_bootstrap_fast(
                    comparison_predictions,
                    baseline_name="traditional_27",
                    augmented_name=augmented,
                    n_bootstrap=5000,
                    seed=20260913 + len(bootstrap_rows),
                )
                result.update({
                    "analysis_range": analysis_range, "validation": validation,
                    "baseline_name": "traditional_27", "augmented_name": augmented,
                })
                bootstrap_rows.append(result)
    predictions = pd.concat(predictions_all, ignore_index=True)
    metrics = pd.concat(metrics_all, ignore_index=True)
    audits = pd.concat(audits_all, ignore_index=True)
    bootstrap = pd.DataFrame(bootstrap_rows)
    predictions.to_csv(table_dir / "v2_prediction_predictions.csv", index=False)
    metrics.to_csv(table_dir / "v2_prediction_metrics.csv", index=False)
    _fold_metrics(predictions).to_csv(table_dir / "v2_prediction_fold_metrics.csv", index=False)
    bootstrap.to_csv(table_dir / "v2_prediction_bootstrap.csv", index=False)
    audits.to_csv(table_dir / "v2_prediction_fold_audit.csv", index=False)

    gates = []
    for row in bootstrap.itertuples():
        augmented_metric = metrics[
            (metrics["analysis_range"] == row.analysis_range)
            & (metrics["validation"] == row.validation)
            & (metrics["model_set"] == row.augmented_name)
        ]
        r2 = float(augmented_metric["r2"].iloc[0])
        passes = bool(
            row.validation == "leave_one_culture_out"
            and row.ci95_low > 0
            and r2 > 0
        )
        gates.append({
            "analysis_range": row.analysis_range, "validation": row.validation,
            "augmented_name": row.augmented_name, "mae_improvement": row.delta_mae_baseline_minus_augmented,
            "ci95_low": row.ci95_low, "ci95_high": row.ci95_high,
            "augmented_r2": r2, "main_text_gate_pass": passes,
            "criterion": "LOCO clustered-bootstrap MAE improvement CI excludes zero and augmented pooled R2 is positive",
        })
    pd.DataFrame(gates).to_csv(table_dir / "v2_prediction_claim_gate.csv", index=False)
    audit_payload = {
        "input_file": str(source / "data_intermediate/baseline_bridge/WT_development_original27_corrected_filtered_preimputation.csv"),
        "n_recordings": len(table), "n_baseline_features": len(baseline),
        "n_legacy_higher_order_features": len(legacy), "n_oinfo_features": len(oinfo),
        "missing_baseline_cells_before_fold_local_imputation": int(table[baseline].isna().sum().sum()),
        "all_outer_groups_disjoint": bool(audits["outer_groups_disjoint"].all()),
        "remaining_limitation": "Some imported baseline features inherit upstream DIV-dependent masks; fold-local imputation and scaling cannot undo those upstream masks.",
    }
    atomic_json(output / "logs/v2_prediction_preprocessing_audit.json", audit_payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
