from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import yaml
from scipy.io import loadmat

from hotopology.activity_adjustment import (
    fit_activity_adjusted_model,
    select_activity_form,
)
from hotopology.hierarchical import adjacent_contrasts_from_fit, coefficient_table
from hotopology.v2_contract import normalize_recording_keys
from hotopology.current_null import load_instantaneous_null100


DIVS = [14, 21, 28, 35, 42, 49, 56, 63]


def _prior_feature_matrix(root: Path) -> Path:
    config = yaml.safe_load((root / "analysis_config.yaml").read_text(encoding="utf-8"))
    return Path(config["computational_closure_v2"]["prior_feature_matrix"])


def _top16_mean_k(root: Path) -> pd.DataFrame:
    rows = []
    model_dir = root / "data_intermediate" / "maxent_models_top16"
    binary_dir = root / "data_intermediate" / "binary_25ms"
    for model_path in sorted(model_dir.glob("*_maxent.mat")):
        recording = model_path.stem.removesuffix("_maxent")
        model = loadmat(model_path, squeeze_me=True, struct_as_record=False)
        nodes = np.asarray(model["active_nodes"], dtype=int).reshape(-1) - 1
        binary_path = binary_dir / f"{recording}_25ms.npz"
        with np.load(binary_path) as archive:
            x = np.asarray(archive["x"], dtype=np.uint8)
        selected = x[nodes]
        k = selected.sum(axis=0, dtype=np.int32)
        rows.append(
            {
                "recording": recording,
                "mean_k_top16": float(k.mean()),
                "fraction_k_ge_3_top16": float(np.mean(k >= 3)),
                "max_k_top16": int(k.max()),
                "n_top16_nodes": int(len(nodes)),
            }
        )
    return normalize_recording_keys(pd.DataFrame(rows))


def _analysis_table(root: Path) -> pd.DataFrame:
    residual, topology = load_instantaneous_null100(root)
    residual["log1p_absolute_residual_mass"] = np.log1p(
        residual["absolute_residual_mass"].astype(float) * 1e6
    )

    topology = topology.loc[
        topology["metric"].isin(["h1_total_persistence_capped", "h1_infinite_count"])
    ].copy()
    topology = normalize_recording_keys(topology)
    topology = topology.pivot(
        index="recording", columns="metric", values="z"
    ).rename(
        columns={
            "h1_total_persistence_capped": "h1_persistence_deviation_z",
            "h1_infinite_count": "h1_infinite_deviation_z",
        }
    ).reset_index()

    baseline_path = _prior_feature_matrix(root)
    baseline = normalize_recording_keys(pd.read_csv(baseline_path))
    activity_columns = [
        "recording",
        "mean_firing_rate_hz_all",
        "network_burst_rate_per_min",
        "frac_spikes_in_network_bursts",
        "active_electrode_fraction",
    ]
    table = residual.merge(topology, on="recording", how="inner", validate="one_to_one")
    table = table.merge(
        baseline[activity_columns], on="recording", how="inner", validate="one_to_one"
    )
    top16 = _top16_mean_k(root)
    table = table.merge(
        top16[["recording", "mean_k_top16", "fraction_k_ge_3_top16", "max_k_top16"]],
        on="recording",
        how="inner",
        validate="one_to_one",
    )
    return normalize_recording_keys(table)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    out = root / "tables" / "computational_closure_v2"
    out.mkdir(parents=True, exist_ok=True)
    table = _analysis_table(root)
    table.to_csv(out / "v2_activity_adjustment_analysis_table.csv", index=False)

    outcomes = [
        "log1p_absolute_residual_mass",
        "absolute_residual_z",
        "positive_residual_z",
        "h1_persistence_deviation_z",
        "h1_infinite_deviation_z",
    ]
    tier2_covariates = [
        "mean_firing_rate_hz_all",
        "network_burst_rate_per_min",
        "frac_spikes_in_network_bursts",
    ]
    forms = []
    coefficients = []
    contrasts = []
    diagnostics = []
    for outcome in outcomes:
        selected, scores = select_activity_form(
            table, outcome=outcome, activity_col="mean_k_top16", n_splits=6
        )
        scores.insert(0, "outcome", outcome)
        forms.append(scores)
        for tier, covariates in [("tier1_mean_k", []), ("tier2_activity_burst", tier2_covariates)]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = fit_activity_adjusted_model(
                    table,
                    outcome=outcome,
                    divs=DIVS,
                    activity_col="mean_k_top16",
                    activity_form=selected,
                    additional_covariates=covariates,
                )
            coef = coefficient_table(fit)
            coef.insert(0, "adjustment_tier", tier)
            coef.insert(0, "selected_activity_form", selected)
            contrast = adjacent_contrasts_from_fit(fit)
            contrast.insert(0, "adjustment_tier", tier)
            contrast.insert(0, "selected_activity_form", selected)
            coefficients.append(coef)
            contrasts.append(contrast)
            diagnostics.append(
                {
                    "outcome": outcome,
                    "tier": tier,
                    "method": fit.method,
                    "selected_activity_form": selected,
                    "n_observations": fit.n_observations,
                    "details": fit.diagnostics,
                }
            )

    pd.concat(forms, ignore_index=True).to_csv(
        out / "v2_activity_form_cv.csv", index=False
    )
    pd.concat(coefficients, ignore_index=True).to_csv(
        out / "v2_activity_adjusted_coefficients.csv", index=False
    )
    pd.concat(contrasts, ignore_index=True).to_csv(
        out / "v2_activity_adjusted_adjacent_contrasts.csv", index=False
    )
    (out / "v2_activity_adjustment_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "n_recordings": len(table),
                "n_outcomes": len(outcomes),
                "models_fit": len(diagnostics),
                "selected_forms": {
                    outcome: str(
                        pd.concat(forms).loc[
                            lambda frame: (frame["outcome"] == outcome)
                            & (frame.groupby("outcome")["cv_mae"].transform("min") == frame["cv_mae"]),
                            "activity_form",
                        ].iloc[0]
                    )
                    for outcome in outcomes
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
