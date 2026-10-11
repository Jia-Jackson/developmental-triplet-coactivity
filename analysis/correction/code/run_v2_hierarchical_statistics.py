from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from hotopology.hierarchical import (
    adjacent_contrasts_from_fit,
    coefficient_table,
    culture_specific_adjacent_changes,
    fit_hierarchical_div_model,
)
from hotopology.v2_contract import normalize_recording_keys
from hotopology.current_null import load_instantaneous_null100


DIVS = [14, 21, 28, 35, 42, 49, 56, 63]


def _assemble_datasets(root: Path) -> dict[str, tuple[pd.DataFrame, list[str]]]:
    direct = pd.read_csv(root / "tables" / "direct_higher_order_metrics_all_bins.csv")
    direct = normalize_recording_keys(direct.loc[direct["bin_ms"] == 25].copy())
    direct_metrics = [
        "mean_population_count",
        "fraction_bins_k_ge_3",
        "pair_occurrence_density",
        "triplet_occurrence_density",
        "triplet_repeated5_density",
        "pair_support_mass",
        "triplet_support_mass",
        "h0_total_persistence_capped",
        "h1_total_persistence_capped",
        "h1_infinite_count",
    ]

    irreducible, topology_long = load_instantaneous_null100(root)
    irreducible["log1p_absolute_residual_mass"] = np.log1p(
        irreducible["absolute_residual_mass"].astype(float) * 1e6
    )
    irreducible_metrics = [
        "positive_residual_mass",
        "negative_residual_mass",
        "absolute_residual_mass",
        "log1p_absolute_residual_mass",
        "rms_residual",
        "positive_residual_z",
        "negative_residual_z",
        "absolute_residual_z",
    ]

    topology = topology_long.pivot_table(
        index=["recording", "DIV", "plate", "culture", "well", "subject_id"],
        columns="metric",
        values=["observed", "null_mean", "z"],
        aggfunc="first",
    )
    topology.columns = [f"{stat}_{metric}" for stat, metric in topology.columns]
    topology = topology.reset_index()
    topology_metrics = [
        column
        for column in topology.columns
        if column
        in {
            "z_h1_total_persistence_capped",
            "z_h1_infinite_count",
            "z_triplet_repeated5_density",
            "z_triplet_support_mass",
        }
    ]

    burst = pd.read_csv(root / "tables" / "burst_stratified_higher_order_metrics_25ms.csv")
    burst = normalize_recording_keys(burst)
    burst_metrics = [
        "mean_population_count",
        "fraction_bins_k_ge_3",
        "triplet_occurrence_density",
        "triplet_repeated5_density",
        "triplet_support_mass",
        "h1_total_persistence_capped",
    ]
    burst_wide = burst.pivot_table(
        index=["recording", "DIV", "plate", "culture", "well", "subject_id"],
        columns="stratum",
        values=burst_metrics,
        aggfunc="first",
    )
    burst_wide.columns = [f"{metric}_{stratum}" for metric, stratum in burst_wide.columns]
    burst_wide = burst_wide.reset_index()
    difference_metrics = []
    for metric in burst_metrics:
        inside = f"{metric}_inside_burst"
        outside = f"{metric}_outside_burst"
        if inside in burst_wide and outside in burst_wide:
            name = f"inside_minus_outside_{metric}"
            burst_wide[name] = burst_wide[inside] - burst_wide[outside]
            difference_metrics.append(name)

    return {
        "direct_25ms": (direct, direct_metrics),
        "conditioned_triplet_residual": (irreducible, irreducible_metrics),
        "conditioned_topology_deviation": (topology, topology_metrics),
        "burst_inside_outside_difference": (burst_wide, difference_metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    out = root / "tables" / "computational_closure_v2"
    out.mkdir(parents=True, exist_ok=True)

    all_coefficients = []
    all_contrasts = []
    all_culture = []
    diagnostics: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for family, (table, metrics) in _assemble_datasets(root).items():
        for metric in metrics:
            if metric not in table or table[metric].notna().sum() < 20:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = fit_hierarchical_div_model(
                        table, outcome=metric, divs=DIVS
                    )
                coefficients = coefficient_table(fit)
                coefficients.insert(0, "family", family)
                contrasts = adjacent_contrasts_from_fit(fit)
                contrasts.insert(0, "family", family)
                culture = culture_specific_adjacent_changes(
                    table, outcome=metric, divs=DIVS
                )
                culture.insert(0, "family", family)
                all_coefficients.append(coefficients)
                all_contrasts.append(contrasts)
                all_culture.append(culture)
                diagnostics.append(
                    {
                        "family": family,
                        "outcome": metric,
                        "method": fit.method,
                        "converged": fit.converged,
                        "residual_scale": fit.residual_scale,
                        "n_observations": fit.n_observations,
                        "n_subjects": fit.n_subjects,
                        "n_cultures": fit.n_cultures,
                        "details": fit.diagnostics,
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "family": family,
                        "outcome": metric,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    pd.concat(all_coefficients, ignore_index=True).to_csv(
        out / "v2_hierarchical_coefficients.csv", index=False
    )
    pd.concat(all_contrasts, ignore_index=True).to_csv(
        out / "v2_hierarchical_adjacent_contrasts.csv", index=False
    )
    pd.concat(all_culture, ignore_index=True).to_csv(
        out / "v2_culture_specific_adjacent_changes.csv", index=False
    )
    (out / "v2_hierarchical_model_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(failures, columns=["family", "outcome", "error"]).to_csv(
        out / "v2_hierarchical_failures.csv", index=False
    )
    print(
        json.dumps(
            {
                "models_fit": len(diagnostics),
                "failures": len(failures),
                "mixedlm": sum(item["method"] == "mixedlm" for item in diagnostics),
                "gee_fallback": sum(item["method"] == "gee" for item in diagnostics),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
