from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from .io_utils import atomic_json
from .vendor_hotopology.hierarchical import (
    adjacent_contrasts_from_fit,
    coefficient_table,
    culture_specific_adjacent_changes,
    fit_hierarchical_div_model,
)


DIVS = (14, 21, 28, 35, 42, 49, 56, 63)


def _age_summary(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in data.groupby(["model", "metric", "DIV"]):
        for column in columns:
            values = group[column].dropna().to_numpy(float)
            rows.append({
                "model": keys[0], "metric": keys[1], "DIV": int(keys[2]), "quantity": column,
                "n_recordings": int(len(values)), "n_wells": int(group.loc[group[column].notna(), "subject_id"].nunique()),
                "n_cultures": int(group.loc[group[column].notna(), "culture"].nunique()),
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "sem": float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan,
                "median": float(np.median(values)) if len(values) else np.nan,
                "q25": float(np.quantile(values, 0.25)) if len(values) else np.nan,
                "q75": float(np.quantile(values, 0.75)) if len(values) else np.nan,
            })
    return pd.DataFrame(rows)


def run(source: Path, output: Path) -> None:
    table_dir = output / "tables/upgrade_v2"
    data = pd.read_csv(table_dir / "v2_dynamic16_oinfo_null_comparisons.csv")
    data["null_deviation"] = data["observed"] - data["null_mean"]
    data["absolute_null_deviation"] = data["null_deviation"].abs()
    data.to_csv(table_dir / "v2_dynamic16_oinfo_analysis_table.csv", index=False)
    _age_summary(data, ["observed", "null_mean", "null_deviation", "z"]).to_csv(
        table_dir / "v2_dynamic16_oinfo_age_summary.csv", index=False
    )

    coefficient_rows, contrast_rows, culture_rows, diagnostics = [], [], [], []
    for (model, metric), group in data.groupby(["model", "metric"]):
        for outcome in ("observed", "null_deviation"):
            work = group.copy()
            unique_outcome = f"{outcome}_value"
            work[unique_outcome] = work[outcome]
            try:
                fit = fit_hierarchical_div_model(work, outcome=unique_outcome, divs=DIVS)
                coefficients = coefficient_table(fit)
                coefficients.insert(0, "model", model)
                coefficients.insert(1, "metric", metric)
                coefficients.insert(2, "quantity", outcome)
                coefficient_rows.append(coefficients)
                contrasts = adjacent_contrasts_from_fit(fit)
                contrasts.insert(0, "model", model)
                contrasts.insert(1, "metric", metric)
                contrasts.insert(2, "quantity", outcome)
                contrast_rows.append(contrasts)
                culture = culture_specific_adjacent_changes(
                    work, outcome=unique_outcome, divs=DIVS
                )
                culture.insert(0, "model", model)
                culture.insert(1, "metric", metric)
                culture.insert(2, "quantity", outcome)
                culture_rows.append(culture)
                diagnostics.append({
                    "model": model, "metric": metric, "quantity": outcome,
                    "fit_method": fit.method, "converged": fit.converged,
                    "n_observations": fit.n_observations, "n_subjects": fit.n_subjects,
                    "n_cultures": fit.n_cultures, "diagnostics": fit.diagnostics,
                })
            except Exception as exc:
                diagnostics.append({
                    "model": model, "metric": metric, "quantity": outcome,
                    "fit_method": None, "converged": False, "error": f"{type(exc).__name__}: {exc}",
                })
    coefficients = pd.concat(coefficient_rows, ignore_index=True)
    contrasts = pd.concat(contrast_rows, ignore_index=True)
    cultures = pd.concat(culture_rows, ignore_index=True)
    coefficients.to_csv(table_dir / "v2_dynamic16_oinfo_hierarchical_coefficients.csv", index=False)
    contrasts.to_csv(table_dir / "v2_dynamic16_oinfo_adjacent_contrasts.csv", index=False)
    cultures.to_csv(table_dir / "v2_dynamic16_oinfo_culture_changes.csv", index=False)
    atomic_json(output / "logs/v2_dynamic16_oinfo_model_diagnostics.json", diagnostics)

    claim_rows = []
    for metric in sorted(data["metric"].unique()):
        model_results = []
        for model in sorted(data["model"].unique()):
            row = contrasts[
                (contrasts["model"] == model)
                & (contrasts["metric"] == metric)
                & (contrasts["quantity"] == "null_deviation")
                & (contrasts["DIV_first"] == 14)
                & (contrasts["DIV_second"] == 21)
            ]
            culture = cultures[
                (cultures["model"] == model)
                & (cultures["metric"] == metric)
                & (cultures["quantity"] == "null_deviation")
                & (cultures["DIV_first"] == 14)
                & (cultures["DIV_second"] == 21)
            ]
            if len(row) != 1:
                continue
            estimate = float(row["estimate"].iloc[0])
            directions = np.sign(culture["mean_change"].dropna().to_numpy(float))
            model_results.append({
                "model": model, "estimate": estimate,
                "adjusted_p": float(row["adjusted_p"].iloc[0]),
                "all_cultures_same_direction": bool(len(directions) == 3 and np.all(directions == np.sign(estimate))),
            })
        signs = [np.sign(item["estimate"]) for item in model_results]
        passes = bool(
            len(model_results) == 2
            and len(set(signs)) == 1
            and all(item["adjusted_p"] < 0.05 for item in model_results)
            and all(item["all_cultures_same_direction"] for item in model_results)
        )
        claim_rows.append({
            "metric": metric,
            "main_text_gate_pass": passes,
            "criterion": "DIV14-to-DIV21 null-deviation contrast adjusted P<0.05, same sign under both nulls, and same direction in all three cultures",
            "model_results_json": json.dumps(model_results),
        })
    pd.DataFrame(claim_rows).to_csv(table_dir / "v2_dynamic16_oinfo_claim_gate.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()

