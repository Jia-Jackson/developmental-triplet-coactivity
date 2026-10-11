from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hotopology.hierarchical import adjacent_contrasts_from_fit, fit_hierarchical_div_model
from hotopology.incremental import (
    clustered_delta_mae_bootstrap,
    compare_feature_sets,
    cross_fitted_residuals,
    prediction_metrics,
)
from hotopology.v2_contract import normalize_recording_keys
from requested_node_eligibility import expected_keys


ALPHAS = np.logspace(-3, 3, 13).tolist()
DIVS = [14, 21, 28, 35, 42, 49, 56, 63]


def _prior_feature_matrix(root: Path) -> Path:
    config = yaml.safe_load((root / "analysis_config.yaml").read_text(encoding="utf-8"))
    return Path(config["computational_closure_v2"]["prior_feature_matrix"])


def _assemble(root: Path) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    baseline_path = _prior_feature_matrix(root)
    raw_baseline = pd.read_csv(baseline_path)
    baseline = normalize_recording_keys(raw_baseline)
    identifier_columns = {
        "SourceRecording",
        "SourceBatch",
        "DIV",
        "PlateIndex",
        "Plate",
        "Well",
    }
    baseline_features = [column for column in raw_baseline.columns if column not in identifier_columns]
    compact_features = [
        "mean_firing_rate_hz_all",
        "active_electrode_fraction",
        "network_burst_rate_per_min",
        "frac_spikes_in_network_bursts",
        "EdgeWeight",
        "NetworkDensity",
    ]

    high = pd.read_csv(
        root / "tables" / "computational_closure_v2" / "v2_activity_adjustment_analysis_table.csv"
    )
    high = normalize_recording_keys(high)
    if set(high['model']) != {'pairwise_conditioned_k_100'} or not high['n_null_realizations'].eq(100).all():
        raise ValueError('Recompute activity-adjustment inputs using 100 null realizations before prediction')
    direct = pd.read_csv(root / "tables" / "direct_higher_order_metrics_all_bins.csv")
    direct = normalize_recording_keys(direct.loc[direct["bin_ms"] == 25].copy())
    direct_keep = direct[
        ["recording", "triplet_repeated5_density", "h1_total_persistence_capped"]
    ].rename(
        columns={
            "triplet_repeated5_density": "direct_triplet_repeated5_density",
            "h1_total_persistence_capped": "direct_h1_total_persistence_capped",
        }
    )
    higher_features = [
        "log1p_absolute_residual_mass",
        "absolute_residual_z",
        "positive_residual_z",
        "h1_persistence_deviation_z",
        "h1_infinite_deviation_z",
        "direct_triplet_repeated5_density",
        "direct_h1_total_persistence_capped",
    ]

    high_keep = high[
        [
            "recording",
            "log1p_absolute_residual_mass",
            "absolute_residual_z",
            "positive_residual_z",
            "h1_persistence_deviation_z",
            "h1_infinite_deviation_z",
        ]
    ]
    table = baseline.merge(high_keep, on="recording", how="inner", validate="one_to_one")
    table = table.merge(direct_keep, on="recording", how="inner", validate="one_to_one")
    table = normalize_recording_keys(table)
    eligibility = pd.read_csv(root / 'tables/computational_closure_v2/requested_node_eligibility.csv')
    required = expected_keys(eligibility, 'dynamic16')
    if set(high.recording) != required or set(table.recording) != required:
        raise ValueError('Prediction cohort differs from explicit dynamic16 eligibility')
    table[['recording', 'DIV', 'culture', 'subject_id']].to_csv(
        root / 'tables/computational_closure_v2/v2_prediction_cohort_manifest.csv', index=False)
    for column in [*baseline_features, *higher_features]:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    return table, baseline_features, compact_features, higher_features


def _fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (analysis, validation, model_set, fold), subset in predictions.groupby(
        ["analysis_range", "validation", "model_set", "outer_fold"], sort=True
    ):
        metric = prediction_metrics(subset)
        metric.update(
            {
                "analysis_range": analysis,
                "validation": validation,
                "model_set": model_set,
                "outer_fold": fold,
                "held_out_group": subset["held_out_group"].iloc[0],
            }
        )
        rows.append(metric)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    out = root / "tables" / "computational_closure_v2"
    out.mkdir(parents=True, exist_ok=True)
    table, baseline_features, compact_features, higher_features = _assemble(root)
    table.to_csv(out / "v2_incremental_analysis_table.csv", index=False)

    feature_manifest = []
    for model_set, features in {
        "compact_baseline": compact_features,
        "prior_27_baseline": baseline_features,
        "prior_27_plus_higher_order": [*baseline_features, *higher_features],
    }.items():
        for feature in features:
            feature_manifest.append({"model_set": model_set, "feature": feature})
    pd.DataFrame(feature_manifest).to_csv(
        out / "v2_incremental_feature_manifest.csv", index=False
    )

    prediction_outputs = []
    metric_outputs = []
    audit_outputs = []
    bootstrap_outputs = []
    settings = [
        ("DIV14_63", table),
        ("DIV21_63", table.loc[table["DIV"] >= 21].copy()),
    ]
    validations = [
        ("leave_one_culture_out", "culture", "subject_id"),
        ("leave_one_well_out", "subject_id", "subject_id"),
    ]
    feature_sets = {
        "compact_baseline": compact_features,
        "prior_27_baseline": baseline_features,
        "prior_27_plus_higher_order": [*baseline_features, *higher_features],
    }
    for analysis_range, subset in settings:
        for validation, outer_group, inner_group in validations:
            predictions, metrics, audit = compare_feature_sets(
                subset,
                feature_sets=feature_sets,
                target="DIV",
                outer_group=outer_group,
                inner_group=inner_group,
                alphas=ALPHAS,
            )
            for frame in (predictions, metrics, audit):
                frame.insert(0, "validation", validation)
                frame.insert(0, "analysis_range", analysis_range)
            prediction_outputs.append(predictions)
            metric_outputs.append(metrics)
            audit_outputs.append(audit)
            for baseline_name in ("compact_baseline", "prior_27_baseline"):
                boot = clustered_delta_mae_bootstrap(
                    predictions,
                    baseline_name=baseline_name,
                    augmented_name="prior_27_plus_higher_order",
                    n_bootstrap=2000,
                    seed=20260828,
                )
                boot.update(
                    {
                        "analysis_range": analysis_range,
                        "validation": validation,
                        "baseline_name": baseline_name,
                        "augmented_name": "prior_27_plus_higher_order",
                    }
                )
                bootstrap_outputs.append(boot)

    predictions_all = pd.concat(prediction_outputs, ignore_index=True)
    metrics_all = pd.concat(metric_outputs, ignore_index=True)
    audits_all = pd.concat(audit_outputs, ignore_index=True)
    predictions_all.to_csv(out / "v2_incremental_predictions.csv", index=False)
    metrics_all.to_csv(out / "v2_incremental_metrics.csv", index=False)
    audits_all.to_json(
        out / "v2_incremental_fold_audit.json",
        orient="records",
        indent=2,
    )
    _fold_metrics(predictions_all).to_csv(
        out / "v2_incremental_fold_metrics.csv", index=False
    )
    pd.DataFrame(bootstrap_outputs).to_csv(
        out / "v2_incremental_bootstrap.csv", index=False
    )

    residual_outputs = []
    residual_contrasts = []
    for feature in higher_features:
        residuals = cross_fitted_residuals(
            table,
            feature_columns=baseline_features,
            target=feature,
            outer_group="culture",
            inner_group="subject_id",
            alphas=ALPHAS,
        )
        residuals["higher_order_feature"] = feature
        residual_outputs.append(residuals)
        fit_table = residuals.rename(columns={"residual": "conditional_residual"})
        fit = fit_hierarchical_div_model(
            fit_table,
            outcome="conditional_residual",
            divs=DIVS,
        )
        contrast = adjacent_contrasts_from_fit(fit)
        contrast.insert(0, "higher_order_feature", feature)
        residual_contrasts.append(contrast)
    pd.concat(residual_outputs, ignore_index=True).to_csv(
        out / "v2_higher_order_cross_fitted_residuals.csv", index=False
    )
    pd.concat(residual_contrasts, ignore_index=True).to_csv(
        out / "v2_higher_order_conditional_div_contrasts.csv", index=False
    )

    summary = {
        "n_recordings": int(len(table)),
        "n_baseline_features": len(baseline_features),
        "n_higher_order_features": len(higher_features),
        "n_prediction_rows": int(len(predictions_all)),
        "all_outer_folds_disjoint": bool(audits_all["outer_groups_disjoint"].all()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
