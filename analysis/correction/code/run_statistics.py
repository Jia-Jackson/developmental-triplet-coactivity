from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from hotopology.statistics import (
    adjacent_paired_wilcoxon,
    benjamini_hochberg,
    one_sample_wilcoxon_by_div,
    repeated_friedman,
    summarize_by_div,
)


DIVS = [14, 21, 28, 35, 42, 49, 56, 63]
DIRECT_METRICS = [
    "n_analysis_nodes",
    "mean_population_count",
    "fraction_bins_k_ge_3",
    "triplet_occurrence_density",
    "triplet_repeated5_density",
    "triplet_support_mass",
    "h1_total_persistence_capped",
    "h1_infinite_count",
]


def burst_inside_outside_tests(table: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in metrics:
        for div in DIVS:
            subset = table.loc[table["DIV"] == div, ["recording", "stratum", metric]]
            wide = subset.pivot(index="recording", columns="stratum", values=metric).dropna()
            if wide.empty:
                p = np.nan
                statistic = np.nan
                median_change = np.nan
            else:
                change = wide["inside_burst"] - wide["outside_burst"]
                median_change = float(change.median())
                if np.allclose(change, 0):
                    statistic, p = 0.0, 1.0
                else:
                    test = stats.wilcoxon(wide["outside_burst"], wide["inside_burst"])
                    statistic, p = float(test.statistic), float(test.pvalue)
            rows.append(
                {
                    "metric": metric,
                    "DIV": div,
                    "n_pairs": len(wide),
                    "median_inside_minus_outside": median_change,
                    "wilcoxon_statistic": statistic,
                    "raw_p": p,
                }
            )
    output = pd.DataFrame(rows)
    output["adjusted_p"] = np.nan
    for metric, indices in output.groupby("metric").groups.items():
        output.loc[indices, "adjusted_p"] = benjamini_hochberg(
            output.loc[indices, "raw_p"].to_numpy()
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    root = Path(args.result_dir)
    tables = root / "tables"
    direct = pd.read_csv(tables / "direct_higher_order_metrics_all_bins.csv")
    primary = direct.loc[direct["bin_ms"] == 25].copy()

    summaries = pd.concat(
        [summarize_by_div(primary, metric) for metric in DIRECT_METRICS],
        ignore_index=True,
    )
    adjacent = pd.concat(
        [adjacent_paired_wilcoxon(primary, metric=metric, divs=DIVS) for metric in DIRECT_METRICS],
        ignore_index=True,
    )
    omnibus_rows = [repeated_friedman(primary, metric=metric, divs=DIVS) for metric in DIRECT_METRICS]
    omnibus_rows.extend(
        repeated_friedman(primary, metric=metric, divs=DIVS[1:]) for metric in DIRECT_METRICS
    )
    omnibus = pd.DataFrame(omnibus_rows)
    omnibus["adjusted_p"] = np.nan
    for _, indices in omnibus.groupby("DIV_range").groups.items():
        omnibus.loc[indices, "adjusted_p"] = benjamini_hochberg(
            omnibus.loc[indices, "raw_p"].to_numpy()
        )

    sensitivity = (
        direct.groupby(["bin_ms", "DIV"])[DIRECT_METRICS]
        .agg(["count", "mean", "sem", "median"])
    )
    sensitivity.columns = ["__".join(col) for col in sensitivity.columns]
    sensitivity = sensitivity.reset_index()

    burst = pd.read_csv(tables / "burst_stratified_higher_order_metrics_25ms.csv")
    burst_tests = burst_inside_outside_tests(
        burst,
        [
            "fraction_bins_k_ge_3",
            "triplet_support_mass",
            "triplet_repeated5_density",
            "h1_total_persistence_capped",
        ],
    )

    summaries.to_csv(tables / "direct_metrics_div_summary.csv", index=False)
    adjacent.to_csv(tables / "direct_metrics_adjacent_wilcoxon.csv", index=False)
    omnibus.to_csv(tables / "direct_metrics_friedman_omnibus.csv", index=False)
    sensitivity.to_csv(tables / "bin_size_sensitivity_summary.csv", index=False)
    burst_tests.to_csv(tables / "burst_inside_outside_wilcoxon.csv", index=False)

    workbook = tables / "higher_order_statistical_results.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        summaries.to_excel(writer, sheet_name="DIV summary", index=False)
        adjacent.to_excel(writer, sheet_name="Adjacent Wilcoxon", index=False)
        omnibus.to_excel(writer, sheet_name="Friedman omnibus", index=False)
        sensitivity.to_excel(writer, sheet_name="Bin sensitivity", index=False)
        burst_tests.to_excel(writer, sheet_name="Burst strata", index=False)

    irreducible_path = tables / "irreducible_triplet_summary.csv"
    if irreducible_path.exists() and irreducible_path.stat().st_size > 0:
        irreducible = pd.read_csv(irreducible_path)
        metrics = [
            "positive_residual_mass",
            "absolute_residual_mass",
            "positive_residual_z",
            "absolute_residual_z",
        ]
        irr_summary_rows = []
        irr_adjacent_rows = []
        for model, subset in irreducible.groupby("model"):
            for metric in metrics:
                summary = summarize_by_div(subset, metric)
                summary["model"] = model
                irr_summary_rows.append(summary)
                tests = adjacent_paired_wilcoxon(subset, metric=metric, divs=DIVS)
                tests["model"] = model
                irr_adjacent_rows.append(tests)
        irr_summary = pd.concat(irr_summary_rows, ignore_index=True)
        irr_adjacent = pd.concat(irr_adjacent_rows, ignore_index=True)
        irr_summary.to_csv(tables / "irreducible_div_summary.csv", index=False)
        irr_adjacent.to_csv(tables / "irreducible_adjacent_wilcoxon.csv", index=False)

        null_tests = []
        for model, subset in irreducible.groupby("model"):
            for metric in ["positive_residual_z", "absolute_residual_z"]:
                tested = one_sample_wilcoxon_by_div(
                    subset,
                    metric=metric,
                    divs=DIVS,
                    null_value=0.0,
                    alternative="greater",
                )
                tested["model"] = model
                tested["analysis_family"] = "irreducible_triplet_residual"
                null_tests.append(tested)

        topology_path = tables / "topology_null_comparisons.csv"
        if topology_path.exists() and topology_path.stat().st_size > 0:
            topology = pd.read_csv(topology_path)
            for (model, metric), subset in topology.groupby(["model", "metric"]):
                tested = one_sample_wilcoxon_by_div(
                    subset,
                    metric="z",
                    divs=DIVS,
                    null_value=0.0,
                    alternative="greater",
                )
                tested["model"] = model
                tested["source_metric"] = metric
                tested["analysis_family"] = "topology_null_deviation"
                null_tests.append(tested)
        pd.concat(null_tests, ignore_index=True).to_csv(
            tables / "null_deviation_one_sample_wilcoxon.csv", index=False
        )


if __name__ == "__main__":
    main()
