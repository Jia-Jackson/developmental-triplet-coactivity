from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd

from .prediction import clustered_delta_mae_bootstrap_fast


COMPARISONS = [
    ("traditional_27", "traditional_27_plus_legacy"),
    ("traditional_27", "traditional_27_plus_oinfo"),
    ("traditional_27", "traditional_27_plus_all_higher_order"),
    ("traditional_27_plus_legacy", "traditional_27_plus_all_higher_order"),
    ("traditional_27_plus_oinfo", "traditional_27_plus_all_higher_order"),
]


def run(output: Path) -> None:
    table_dir = output / "tables/upgrade_v2"
    predictions = pd.read_csv(table_dir / "v2_prediction_predictions.csv")
    metrics = pd.read_csv(table_dir / "v2_prediction_metrics.csv")
    rows = []
    for index, ((analysis_range, validation), group) in enumerate(
        predictions.groupby(["analysis_range", "validation"], sort=True)
    ):
        for comparison_index, (baseline, augmented) in enumerate(COMPARISONS):
            subset = group[group["model_set"].isin([baseline, augmented])]
            result = clustered_delta_mae_bootstrap_fast(
                subset, baseline_name=baseline, augmented_name=augmented,
                n_bootstrap=5000, seed=20260950 + 10 * index + comparison_index,
            )
            result.update({
                "analysis_range": analysis_range, "validation": validation,
                "baseline_name": baseline, "augmented_name": augmented,
            })
            rows.append(result)
    bootstrap = pd.DataFrame(rows)
    bootstrap.to_csv(table_dir / "v2_prediction_bootstrap.csv", index=False)
    gates = []
    for row in bootstrap.itertuples():
        augmented_metric = metrics[
            (metrics["analysis_range"] == row.analysis_range)
            & (metrics["validation"] == row.validation)
            & (metrics["model_set"] == row.augmented_name)
        ]
        r2 = float(augmented_metric["r2"].iloc[0])
        passes = bool(row.validation == "leave_one_culture_out" and row.ci95_low > 0 and r2 > 0)
        gates.append({
            "analysis_range": row.analysis_range, "validation": row.validation,
            "baseline_name": row.baseline_name, "augmented_name": row.augmented_name,
            "mae_improvement": row.delta_mae_baseline_minus_augmented,
            "ci95_low": row.ci95_low, "ci95_high": row.ci95_high,
            "augmented_r2": r2, "main_text_gate_pass": passes,
            "criterion": "LOCO clustered-bootstrap MAE improvement CI excludes zero and augmented pooled R2 is positive",
        })
    pd.DataFrame(gates).to_csv(table_dir / "v2_prediction_claim_gate.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()

