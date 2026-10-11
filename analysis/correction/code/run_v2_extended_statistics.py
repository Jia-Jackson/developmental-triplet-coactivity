from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from hotopology.hierarchical import (
    adjacent_contrasts_from_fit,
    coefficient_table,
    culture_specific_adjacent_changes,
    fit_hierarchical_div_model,
)
from hotopology.statistics import benjamini_hochberg, one_sample_wilcoxon_by_div
from hotopology.v2_contract import normalize_recording_keys


DIVS = [14, 21, 28, 35, 42, 49, 56, 63]


def _paired_null_comparison(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    subset = table.loc[table["metric"] == metric]
    wide = subset.pivot_table(
        index=["recording", "DIV", "subject_id"], columns="model", values="z", aggfunc="first"
    ).reset_index()
    rows = []
    first_model = "pairwise_conditioned_k_100"
    second_model = "kinetic_conditioned_k"
    for div in DIVS:
        paired = wide.loc[wide["DIV"] == div, [first_model, second_model]].dropna()
        difference = paired[second_model] - paired[first_model]
        if len(difference) == 0:
            statistic = p_value = np.nan
        elif np.allclose(difference, 0):
            statistic, p_value = 0.0, 1.0
        else:
            test = stats.wilcoxon(paired[second_model], paired[first_model], alternative="two-sided")
            statistic, p_value = float(test.statistic), float(test.pvalue)
        rows.append(
            {
                "metric": metric,
                "DIV": div,
                "n_pairs": len(paired),
                "median_z_difference_kinetic_minus_instantaneous": float(difference.median()) if len(difference) else np.nan,
                "wilcoxon_statistic": statistic,
                "raw_p": p_value,
            }
        )
    result = pd.DataFrame(rows)
    result["adjusted_p"] = benjamini_hochberg(result["raw_p"].to_numpy())
    return result


def _legacy_new_agreement(root: Path, instant: pd.DataFrame, residual: pd.DataFrame):
    old_topology = pd.read_csv(root / "tables" / "topology_null_comparisons.csv")
    old_topology = old_topology.loc[old_topology["model"] == "pairwise_conditioned_k"]
    old_residual = pd.read_csv(root / "tables" / "irreducible_triplet_summary.csv")
    old_residual = old_residual.loc[old_residual["model"] == "pairwise_conditioned_k"]
    rows = []
    for metric in sorted(set(old_topology["metric"]) & set(instant["metric"])):
        left = old_topology.loc[old_topology["metric"] == metric, ["recording", "z"]].rename(columns={"z": "z_20"})
        right = instant.loc[instant["metric"] == metric, ["recording", "z"]].rename(columns={"z": "z_100"})
        paired = left.merge(right, on="recording", validate="one_to_one").dropna()
        rho, p = stats.spearmanr(paired["z_20"], paired["z_100"])
        rows.append(
            {
                "family": "topology",
                "metric": metric,
                "n": len(paired),
                "spearman_rho_20_vs_100": float(rho),
                "spearman_p": float(p),
                "median_absolute_z_difference": float(np.median(np.abs(paired["z_20"] - paired["z_100"]))),
            }
        )
    for metric in ["absolute_residual_mass", "absolute_residual_z", "positive_residual_z"]:
        paired = old_residual[["recording", metric]].rename(columns={metric: "value_20"}).merge(
            residual[["recording", metric]].rename(columns={metric: "value_100"}),
            on="recording",
            validate="one_to_one",
        ).dropna()
        rho, p = stats.spearmanr(paired["value_20"], paired["value_100"])
        rows.append(
            {
                "family": "triplet_residual",
                "metric": metric,
                "n": len(paired),
                "spearman_rho_20_vs_100": float(rho),
                "spearman_p": float(p),
                "median_absolute_z_difference": float(np.median(np.abs(paired["value_20"] - paired["value_100"]))),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    out = root / "tables" / "computational_closure_v2"
    instant = normalize_recording_keys(pd.read_csv(out / "v2_instantaneous_null100_comparisons.csv"))
    kinetic = normalize_recording_keys(pd.read_csv(out / "v2_kinetic_null_comparisons.csv"))
    instant_residual = normalize_recording_keys(
        pd.read_csv(out / "v2_instantaneous_null100_triplet_residuals.csv")
    )
    kinetic_residual = normalize_recording_keys(pd.read_csv(out / "v2_kinetic_triplet_residuals.csv"))
    combined = pd.concat([instant, kinetic], ignore_index=True)
    combined_residual = pd.concat([instant_residual, kinetic_residual], ignore_index=True)

    one_sample = []
    for model, model_table in combined.groupby("model"):
        for metric in ["triplet_repeated5_density", "h1_total_persistence_capped", "h1_infinite_count"]:
            subset = model_table.loc[model_table["metric"] == metric]
            result = one_sample_wilcoxon_by_div(
                subset,
                metric="z",
                divs=DIVS,
                null_value=0.0,
                alternative="greater",
            )
            result.insert(0, "source_metric", metric)
            result.insert(0, "model", model)
            one_sample.append(result)
    for model, model_table in combined_residual.groupby("model"):
        result = one_sample_wilcoxon_by_div(
            model_table,
            metric="absolute_residual_z",
            divs=DIVS,
            null_value=0.0,
            alternative="greater",
        )
        result.insert(0, "source_metric", "absolute_residual_z")
        result.insert(0, "model", model)
        one_sample.append(result)
    pd.concat(one_sample, ignore_index=True).to_csv(
        out / "v2_null_deviation_one_sample_statistics.csv", index=False
    )

    paired = [
        _paired_null_comparison(combined, metric)
        for metric in ["triplet_repeated5_density", "h1_total_persistence_capped", "h1_infinite_count"]
    ]
    pd.concat(paired, ignore_index=True).to_csv(
        out / "v2_instantaneous_vs_kinetic_paired_statistics.csv", index=False
    )

    coefficients = []
    contrasts = []
    culture_rows = []
    diagnostics = []
    for model, model_table in combined.groupby("model"):
        wide = model_table.pivot_table(
            index=["recording", "DIV", "plate", "culture", "well", "subject_id"],
            columns="metric",
            values="z",
            aggfunc="first",
        ).reset_index()
        for metric in ["triplet_repeated5_density", "h1_total_persistence_capped", "h1_infinite_count"]:
            fit = fit_hierarchical_div_model(wide, outcome=metric, divs=DIVS)
            coef = coefficient_table(fit)
            coef.insert(0, "model", model)
            contrast = adjacent_contrasts_from_fit(fit)
            contrast.insert(0, "model", model)
            culture = culture_specific_adjacent_changes(wide, outcome=metric, divs=DIVS)
            culture.insert(0, "model", model)
            coefficients.append(coef)
            contrasts.append(contrast)
            culture_rows.append(culture)
            diagnostics.append({"model": model, "outcome": metric, "method": fit.method, "details": fit.diagnostics})
    for model, model_table in combined_residual.groupby("model"):
        model_table = model_table.copy()
        model_table["log1p_absolute_residual_mass"] = np.log1p(model_table["absolute_residual_mass"] * 1e6)
        for metric in ["log1p_absolute_residual_mass", "absolute_residual_z"]:
            fit = fit_hierarchical_div_model(model_table, outcome=metric, divs=DIVS)
            coef = coefficient_table(fit)
            coef.insert(0, "model", model)
            contrast = adjacent_contrasts_from_fit(fit)
            contrast.insert(0, "model", model)
            culture = culture_specific_adjacent_changes(model_table, outcome=metric, divs=DIVS)
            culture.insert(0, "model", model)
            coefficients.append(coef)
            contrasts.append(contrast)
            culture_rows.append(culture)
            diagnostics.append({"model": model, "outcome": metric, "method": fit.method, "details": fit.diagnostics})
    pd.concat(coefficients, ignore_index=True).to_csv(out / "v2_extended_null_hierarchical_coefficients.csv", index=False)
    pd.concat(contrasts, ignore_index=True).to_csv(out / "v2_extended_null_hierarchical_contrasts.csv", index=False)
    pd.concat(culture_rows, ignore_index=True).to_csv(out / "v2_extended_null_culture_changes.csv", index=False)
    (out / "v2_extended_null_model_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _legacy_new_agreement(root, instant, instant_residual).to_csv(
        out / "v2_null20_vs_null100_agreement.csv", index=False
    )
    qc_frames = []
    for family, filename in [
        ("instantaneous_pairwise_conditioned_k", "v2_instantaneous_null100_qc.csv"),
        ("kinetic_conditioned_k", "v2_kinetic_null_qc.csv"),
    ]:
        qc = pd.read_csv(out / filename)
        numeric = [column for column in qc.select_dtypes(include=[np.number]).columns if column != "DIV"]
        summary = qc.groupby("DIV")[numeric].agg(["count", "mean", "std", "median"]).reset_index()
        summary.columns = ["_".join(str(part) for part in column if str(part)) for column in summary.columns]
        summary.insert(0, "null_family", family)
        qc_frames.append(summary)
    pd.concat(qc_frames, ignore_index=True).to_csv(out / "v2_null_qc_div_summary.csv", index=False)
    print(json.dumps({"status": "complete", "n_instant": instant.recording.nunique(), "n_kinetic": kinetic.recording.nunique()}, indent=2))


if __name__ == "__main__":
    main()
