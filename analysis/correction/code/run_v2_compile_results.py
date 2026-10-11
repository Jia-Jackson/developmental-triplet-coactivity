from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hotopology.claim_gates import incremental_value_gate, null_technical_gate, robustness_gate
from requested_node_eligibility import expected_keys


def require_qc_cohort(qc, keys, label):
    if qc.recording.duplicated().any() or set(qc.recording) != keys:
        raise ValueError(f'{label}: QC cohort differs from explicit eligible keys')
    if not qc.n_requested_nulls.eq(100).all():
        raise ValueError(f'{label}: requested ensemble is not 100')
    return dict(n_expected=len(keys), n_observed=len(qc),
                n_valid100=int(qc.n_valid_nulls.eq(100).sum()))


def require_age_model_grid(frame):
    wanted={(model,div) for model in ['pairwise_conditioned_k_100','kinetic_conditioned_k']
            for div in [14,21,28,35,42,49,56,63]}
    if frame.duplicated(['model','DIV']).any() or set(zip(frame.model,frame.DIV)) != wanted:
        raise ValueError('Incomplete or duplicated model by age comparison grid')


def _row(frame: pd.DataFrame, **criteria) -> pd.Series:
    selected = frame.copy()
    for column, value in criteria.items():
        selected = selected.loc[selected[column] == value]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {criteria}, found {len(selected)}")
    return selected.iloc[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _status_from_count(count: int, total: int) -> str:
    if count == total:
        return "supported"
    if count > 0:
        return "partially_supported"
    return "not_supported"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    out = root / "tables" / "computational_closure_v2"

    hierarchical = pd.read_csv(out / "v2_hierarchical_adjacent_contrasts.csv")
    culture = pd.read_csv(out / "v2_culture_specific_adjacent_changes.csv")
    adjusted = pd.read_csv(out / "v2_activity_adjusted_adjacent_contrasts.csv")
    bootstrap = pd.read_csv(out / "v2_incremental_bootstrap.csv")
    fold = pd.read_csv(out / "v2_incremental_fold_metrics.csv")
    one_sample = pd.read_csv(out / "v2_null_deviation_one_sample_statistics.csv")
    instant_qc = pd.read_csv(out / "v2_instantaneous_null100_qc.csv")
    kinetic_qc = pd.read_csv(out / "v2_kinetic_null_qc.csv")
    node_effects = pd.read_csv(out / "v2_node_specification_hierarchical_contrasts.csv")
    bin_changes = pd.read_csv(out / "v2_bin_width_div14_21_changes.csv")
    split = pd.read_csv(out / "v2_split_half_reliability.csv")
    duration = pd.read_csv(out / "v2_duration_convergence.csv")
    fixed_qc = pd.read_csv(out / "v2_fixed_node_null_qc_all.csv")
    eligibility = pd.read_csv(out / 'requested_node_eligibility.csv')
    cohort_counts = {}
    for label, frame in [('instantaneous', instant_qc), ('kinetic', kinetic_qc)]:
        cohort_counts[label] = require_qc_cohort(frame, expected_keys(eligibility, 'dynamic16'), label)
    for n in [12,16,20]:
        label = f'fixed{n}'
        frame = fixed_qc.loc[fixed_qc.node_specification.eq(f'fixed_n{n}')]
        cohort_counts[label] = require_qc_cohort(frame, expected_keys(eligibility, label), label)

    gates: list[dict[str, object]] = []
    findings: list[dict[str, object]] = []

    direct = _row(
        hierarchical,
        family="direct_25ms",
        outcome="triplet_support_mass",
        DIV_first=14,
        DIV_second=21,
    )
    directions = culture.loc[
        (culture["family"] == "direct_25ms")
        & (culture["outcome"] == "triplet_support_mass")
        & (culture["DIV_first"] == 14)
        & (culture["DIV_second"] == 21),
        "median_change",
    ].to_numpy(float)
    if len(directions) != 3:
        raise ValueError('Expected all three independent culture directions')
    developmental_supported = bool(direct.adjusted_p < 0.05 and np.all(directions > 0))
    gates.append(
        {
            "claim": "Developmental reorganization of direct higher-order coactivity",
            "status": "supported" if developmental_supported else "not_supported",
            "criterion": "Hierarchical DIV14-to-DIV21 contrast significant and same direction in P1-P3",
            "evidence": f"standardized effect={direct.standardized_effect:.3f}; adjusted P={direct.adjusted_p:.3g}; culture directions={np.sign(directions).astype(int).tolist()}",
            "interpretation": "Direct triplet support increased at the early transition, but its total mass remains algebraically tied to K(t).",
        }
    )
    findings.append(
        {
            "analysis": "Hierarchical direct triplet support",
            "contrast": "DIV14 to DIV21",
            "estimate": direct.estimate,
            "standardized_effect": direct.standardized_effect,
            "adjusted_p": direct.adjusted_p,
            "note": f"Positive median change in {int(np.sum(directions > 0))}/3 independent cultures",
        }
    )

    activity_outcomes = [
        "absolute_residual_z",
        "h1_persistence_deviation_z",
        "h1_infinite_deviation_z",
    ]
    activity_rows = adjusted.loc[
        (adjusted["adjustment_tier"] == "tier2_activity_burst")
        & (adjusted["DIV_first"] == 14)
        & (adjusted["DIV_second"] == 21)
        & adjusted["outcome"].isin(activity_outcomes)
    ].copy()
    n_activity_supported = int((activity_rows["adjusted_p"] < 0.05).sum())
    gates.append(
        {
            "claim": "Activity-independent developmental reorganization",
            "status": _status_from_count(n_activity_supported, len(activity_outcomes)),
            "criterion": "Tier-2 activity/burst-adjusted standardized residual or H1 outcomes retain DIV14-to-DIV21 association",
            "evidence": f"{n_activity_supported}/{len(activity_outcomes)} prespecified standardized outcomes passed adjusted P<0.05",
            "interpretation": "The early developmental effect is outcome-dependent after adjustment; absolute residual mass alone is not treated as sufficient evidence.",
        }
    )
    for _, row in activity_rows.iterrows():
        findings.append(
            {
                "analysis": f"Tier-2 adjusted {row.outcome}",
                "contrast": "DIV14 to DIV21",
                "estimate": row.estimate,
                "standardized_effect": row.standardized_effect,
                "adjusted_p": row.adjusted_p,
                "note": "Adjusted for mean K, firing, network-burst rate, and burst participation",
            }
        )

    for analysis_range, label in [("DIV14_63", "full developmental range"), ("DIV21_63", "post-transition range")]:
        boot = _row(
            bootstrap,
            analysis_range=analysis_range,
            validation="leave_one_culture_out",
            baseline_name="prior_27_baseline",
        )
        subset = fold.loc[
            (fold["analysis_range"] == analysis_range)
            & (fold["validation"] == "leave_one_culture_out")
            & fold["model_set"].isin(["prior_27_baseline", "prior_27_plus_higher_order"])
        ].pivot(index="held_out_group", columns="model_set", values="mae")
        improved = int(
            (subset["prior_27_plus_higher_order"] < subset["prior_27_baseline"]).sum()
        )
        gate = incremental_value_gate(
            float(boot.delta_mae_baseline_minus_augmented),
            float(boot.ci95_low),
            float(boot.ci95_high),
            improved,
            len(subset),
        )
        gates.append(
            {
                "claim": f"Incremental developmental information: {label}",
                "status": gate["status"],
                "criterion": "Augmented model improves leave-one-culture-out MAE beyond original 27 features with inherited global baseline preprocessing",
                "evidence": f"MAE improvement={boot.delta_mae_baseline_minus_augmented:.3f} DIV days (95% CI {boot.ci95_low:.3f} to {boot.ci95_high:.3f}); improved {improved}/{len(subset)} cultures",
                "interpretation": "Positive values favor adding higher-order features.",
            }
        )
        findings.append(
            {
                "analysis": f"Incremental prediction ({analysis_range}, culture-out)",
                "contrast": "prior 27 minus augmented MAE",
                "estimate": boot.delta_mae_baseline_minus_augmented,
                "standardized_effect": np.nan,
                "adjusted_p": np.nan,
                "note": f"95% CI {boot.ci95_low:.3f} to {boot.ci95_high:.3f}; {improved}/{len(subset)} culture folds improved",
            }
        )

    residual_test = one_sample.loc[one_sample["source_metric"] == "absolute_residual_z"]
    require_age_model_grid(residual_test)
    residual_by_model = residual_test.groupby("model").apply(
        lambda frame: bool(((frame["median"] > 0) & (frame["adjusted_p"] < 0.05)).all()),
        include_groups=False,
    )
    residual_supported = bool(residual_by_model.all() and len(residual_by_model) == 2)
    gates.append(
        {
            "claim": "Triplet-specific deviations beyond complementary lower-order nulls",
            "status": "supported" if residual_supported else "not_supported",
            "criterion": "Absolute residual z is positive and significant at every DIV under both null families",
            "evidence": "; ".join(f"{model}: {value}" for model, value in residual_by_model.items()),
            "interpretation": "Deviations refer to the two fitted references; their interpretation requires the separate fitting and sampling calibration.",
        }
    )
    h1 = one_sample.loc[one_sample["source_metric"] == "h1_total_persistence_capped"]
    require_age_model_grid(h1)
    h1_by_model = h1.groupby("model").apply(
        lambda frame: int(((frame["median"] > 0) & (frame["adjusted_p"] < 0.05)).sum()),
        include_groups=False,
    )
    h1_supported = bool((h1_by_model == len(pd.unique(h1["DIV"]))).all())
    gates.append(
        {
            "claim": "H1 topological excess beyond complementary lower-order nulls",
            "status": "supported" if h1_supported else "not_supported",
            "criterion": "Positive significant H1 persistence deviation at every DIV under both null families",
            "evidence": "; ".join(f"{model}: {count}/8 DIV" for model, count in h1_by_model.items()),
            "interpretation": "H1 excess is null-model dependent and should not be described as temporally irreducible.",
        }
    )

    for name, qc, pair_column in [
        ("Instantaneous pairwise + exact K(t)", instant_qc, "simultaneous_pair_moment_rmse"),
        ("Lag-1 kinetic + exact K(t)", kinetic_qc, "lag1_moment_rmse"),
    ]:
        qc_pass = bool(
            qc["inferential_valid"].all()
            and (qc["exact_k_mismatch_count"] == 0).all()
            and qc["firing_probability_rmse"].max() < 0.02
            and qc[pair_column].max() < 0.02
        )
        technical = null_technical_gate(
            valid_nulls=int(qc["n_valid_nulls"].min()), minimum_valid=90, qc_pass=qc_pass
        )
        gates.append(
            {
                "claim": f"Technical adequacy: {name}",
                "status": technical["status"],
                "criterion": "At least 90 valid nulls, exact K(t), and maximum fitted moment RMSE below 0.02",
                "evidence": f"minimum valid nulls={qc.n_valid_nulls.min():.0f}; max firing RMSE={qc.firing_probability_rmse.max():.4f}; max target-moment RMSE={qc[pair_column].max():.4f}",
                "interpretation": "Technical criteria passed." if qc_pass else "At least one technical criterion failed; inspect record-level QC.",
            }
        )

    node_selected = node_effects.loc[
        (node_effects["DIV_first"] == 14)
        & (node_effects["DIV_second"] == 21)
        & (node_effects["analysis_family"] == "residual")
        & (node_effects["outcome"] == "absolute_residual_z")
    ]
    node_gate = robustness_gate(node_selected["standardized_effect"].to_numpy(float))
    bin_selected = bin_changes.loc[bin_changes["metric"].isin([
        "triplet_repeated5_density", "triplet_support_mass", "h1_total_persistence_capped"
    ])]
    bin_gate = robustness_gate(bin_selected["median_change"].to_numpy(float))
    reliability_pass = bool(
        split["split_half_spearman_rho"].median() >= 0.70
        and duration.loc[duration["duration_s"] >= 180, "spearman_vs_300s"].median() >= 0.75
    )
    fixed_qc_pass = bool(
        fixed_qc["inferential_valid"].all()
        and (fixed_qc["exact_k_mismatch_count"] == 0).all()
        and fixed_qc["firing_probability_rmse"].max() < 0.02
        and fixed_qc["simultaneous_pair_moment_rmse"].max() < 0.02
    )
    robustness_supported = bool(
        node_gate["status"] == "supported"
        and bin_gate["status"] == "supported"
        and reliability_pass
        and fixed_qc_pass
    )
    gates.append(
        {
            "claim": "Robustness to node, bin-width, segment, and duration specifications",
            "status": "supported" if robustness_supported else "partially_supported",
            "criterion": "At least 75% directional agreement plus reliability and fixed-node null QC gates",
            "evidence": f"node direction agreement={node_gate.get('direction_agreement', np.nan):.2f}; bin direction agreement={bin_gate.get('direction_agreement', np.nan):.2f}; median split-half rho={split.split_half_spearman_rho.median():.2f}; fixed-node QC={fixed_qc_pass}",
            "interpretation": "Robustness concerns direction and ranking, not equality of raw metric values across specifications.",
        }
    )

    gates_frame = pd.DataFrame(gates)
    findings_frame = pd.DataFrame(findings)
    gates_frame.to_csv(out / "v2_claim_gate_summary.csv", index=False)
    findings_frame.to_csv(out / "v2_key_statistical_findings.csv", index=False)

    required = [
        out / "v2_claim_gate_summary.csv",
        out / "v2_key_statistical_findings.csv",
        out / "v2_hierarchical_adjacent_contrasts.csv",
        out / "v2_activity_adjusted_adjacent_contrasts.csv",
        out / "v2_incremental_metrics.csv",
        out / "v2_null_deviation_one_sample_statistics.csv",
        out / "v2_node_specification_hierarchical_contrasts.csv",
    ]
    manifest = {
        "status": "complete",
        "n_full_array_recordings": len(expected_keys(eligibility, 'full_array')),
        "n_source_longitudinal_wells": 18,
        "analysis_specific_qc_counts": cohort_counts,
        "n_stability_recordings": len(expected_keys(eligibility, 'fixed16')),
        "prediction_preprocessing": "Original27 compatibility global median plus pooled DIV-dependent masks; not leakage-free external validation",
        "n_independent_cultures": 3,
        "primary_bin_ms": 25,
        "null_realizations": 100,
        "outputs": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in required
        ],
    }
    (out / "v2_analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(gates_frame[["claim", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
