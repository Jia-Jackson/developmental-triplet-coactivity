from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat

from hotopology.hierarchical import adjacent_contrasts_from_fit, fit_hierarchical_div_model
from hotopology.metrics import compute_direct_metrics
from hotopology.v2_contract import normalize_recording_keys
from requested_node_eligibility import expected_keys


DIVS = [14, 21, 28, 35, 42, 49, 56, 63]


def require_recording_grid(table, required, label, metric_col=None):
    groups = table.groupby(metric_col) if metric_col else [(label, table)]
    if table.empty:
        raise ValueError(f"Empty {label}")
    for metric, subset in groups:
        if subset.recording.duplicated().any() or set(subset.recording) != set(required):
            raise ValueError(f"Recording grid mismatch: {label}/{metric}")


def _dynamic_direct(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    overlaps = []
    manifest = pd.read_csv(
        root / "tables" / "computational_closure_v2" / "v2_fixed_node_manifest_long.csv"
    )
    fixed16 = {
        recording: set(subset["electrode_index_zero_based"].astype(int))
        for recording, subset in manifest.loc[manifest["n_nodes"] == 16].groupby("recording")
    }
    for path in sorted((root / "data_intermediate" / "maxent_models_top16").glob("*_maxent.mat")):
        recording = path.stem.removesuffix("_maxent")
        loaded = loadmat(path, squeeze_me=True, struct_as_record=False)
        dynamic = np.asarray(loaded["active_nodes"], dtype=int).reshape(-1) - 1
        with np.load(root / "data_intermediate" / "binary_25ms" / f"{recording}_25ms.npz") as archive:
            x = np.asarray(archive["x"], dtype=np.uint8)
            div = int(archive["div"])
            plate = int(archive["plate"])
            well = str(archive["well"])
        metrics, _ = compute_direct_metrics(x[dynamic], min_active_bins=1)
        rows.append(
            {
                "recording": recording,
                "DIV": div,
                "plate": plate,
                "culture": f"P{plate}",
                "well": well,
                "subject_id": f"P{plate}_{well}",
                "node_specification": "dynamic_n16",
                **metrics,
            }
        )
        fixed = fixed16.get(recording)
        dynamic_set = set(int(value) for value in dynamic)
        overlaps.append(
            {
                "recording": recording,
                "DIV": div,
                "plate": plate,
                "culture": f"P{plate}",
                "well": well,
                "subject_id": f"P{plate}_{well}",
                "intersection_n": len(dynamic_set & fixed) if fixed is not None else np.nan,
                "union_n": len(dynamic_set | fixed) if fixed is not None else np.nan,
                "jaccard": len(dynamic_set & fixed) / len(dynamic_set | fixed) if fixed is not None else np.nan,
                "overlap_status": "available" if fixed is not None else "fixed16_unavailable",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(overlaps)


def _rank_agreement(table: pd.DataFrame, value: str) -> pd.DataFrame:
    wide = table.pivot(index="recording", columns="node_specification", values=value)
    columns = list(wide.columns)
    rows = []
    for i, first in enumerate(columns):
        for second in columns[i + 1 :]:
            paired = wide[[first, second]].dropna()
            rho, p = stats.spearmanr(paired[first], paired[second])
            rows.append(
                {
                    "metric": value,
                    "specification_first": first,
                    "specification_second": second,
                    "n": len(paired),
                    "spearman_rho": float(rho),
                    "raw_p": float(p),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    out = root / "tables" / "computational_closure_v2"
    eligibility = pd.read_csv(out / "requested_node_eligibility.csv")
    dynamic_direct, overlap = _dynamic_direct(root)
    require_recording_grid(dynamic_direct, expected_keys(eligibility, 'dynamic16'), 'dynamic direct')
    fixed_direct = normalize_recording_keys(pd.read_csv(out / "v2_fixed_node_direct_metrics_25ms.csv"))
    fixed_direct["node_specification"] = "fixed_n" + fixed_direct["n_nodes_requested"].astype(str)
    for n in (12, 16, 20):
        require_recording_grid(fixed_direct.loc[fixed_direct.n_nodes_requested.eq(n)],
                               expected_keys(eligibility, f'fixed{n}'), f'fixed{n} direct')
    direct = pd.concat([dynamic_direct, fixed_direct], ignore_index=True)
    direct.to_csv(out / "v2_node_specification_direct_metrics.csv", index=False)
    overlap.to_csv(out / "v2_dynamic_fixed_n16_overlap.csv", index=False)

    residual_frames = []
    comparison_frames = []
    dynamic_residual = normalize_recording_keys(pd.read_csv(out / "v2_instantaneous_null100_triplet_residuals.csv"))
    dynamic_residual["node_specification"] = "dynamic_n16"
    require_recording_grid(dynamic_residual, expected_keys(eligibility, 'dynamic16'), 'dynamic residual')
    residual_frames.append(dynamic_residual)
    dynamic_comparison = normalize_recording_keys(pd.read_csv(out / "v2_instantaneous_null100_comparisons.csv"))
    dynamic_comparison["node_specification"] = "dynamic_n16"
    require_recording_grid(dynamic_comparison, expected_keys(eligibility, 'dynamic16'), 'dynamic comparison', 'metric')
    comparison_frames.append(dynamic_comparison)
    for n_nodes in [12, 16, 20]:
        prefix = f"v2_instantaneous_null100_fixed_n{n_nodes}"
        residual = normalize_recording_keys(pd.read_csv(out / f"{prefix}_triplet_residuals.csv"))
        residual["node_specification"] = f"fixed_n{n_nodes}"
        comparison = normalize_recording_keys(pd.read_csv(out / f"{prefix}_comparisons.csv"))
        comparison["node_specification"] = f"fixed_n{n_nodes}"
        required = expected_keys(eligibility, f'fixed{n_nodes}')
        require_recording_grid(residual, required, f'fixed{n_nodes} residual')
        require_recording_grid(comparison, required, f'fixed{n_nodes} comparison', 'metric')
        residual_frames.append(residual)
        comparison_frames.append(comparison)
    residual_all = pd.concat(residual_frames, ignore_index=True)
    comparison_all = pd.concat(comparison_frames, ignore_index=True)
    residual_all.to_csv(out / "v2_node_specification_triplet_residuals.csv", index=False)
    comparison_all.to_csv(out / "v2_node_specification_null_comparisons.csv", index=False)

    effect_rows = []
    outcome_tables = []
    for specification, subset in direct.groupby("node_specification"):
        for metric in ["triplet_repeated5_density", "triplet_support_mass", "h1_total_persistence_capped"]:
            outcome_tables.append((specification, "direct", metric, subset))
    for specification, subset in residual_all.groupby("node_specification"):
        subset = subset.copy()
        subset["log1p_absolute_residual_mass"] = np.log1p(subset["absolute_residual_mass"] * 1e6)
        for metric in ["log1p_absolute_residual_mass", "absolute_residual_z"]:
            outcome_tables.append((specification, "residual", metric, subset))
    comparison_wide = comparison_all.pivot_table(
        index=["recording", "DIV", "plate", "culture", "well", "subject_id", "node_specification"],
        columns="metric",
        values="z",
        aggfunc="first",
    ).reset_index()
    for specification, subset in comparison_wide.groupby("node_specification"):
        for metric in ["h1_total_persistence_capped", "h1_infinite_count"]:
            outcome_tables.append((specification, "null_deviation_z", metric, subset))
    for specification, family, metric, table in outcome_tables:
        fit = fit_hierarchical_div_model(table, outcome=metric, divs=DIVS)
        contrast = adjacent_contrasts_from_fit(fit)
        contrast.insert(0, "analysis_family", family)
        contrast.insert(0, "node_specification", specification)
        effect_rows.append(contrast)
    effects = pd.concat(effect_rows, ignore_index=True)
    effects.to_csv(out / "v2_node_specification_hierarchical_contrasts.csv", index=False)

    agreements = []
    for metric in ["triplet_repeated5_density", "triplet_support_mass", "h1_total_persistence_capped"]:
        agreements.append(_rank_agreement(direct, metric))
    for metric in ["absolute_residual_mass", "absolute_residual_z"]:
        agreements.append(_rank_agreement(residual_all, metric))
    h1_long = comparison_all.loc[comparison_all["metric"] == "h1_total_persistence_capped"].rename(columns={"z": "h1_deviation_z"})
    agreements.append(_rank_agreement(h1_long, "h1_deviation_z"))
    pd.concat(agreements, ignore_index=True).to_csv(out / "v2_node_specification_rank_agreement.csv", index=False)

    qc_frames = []
    for n_nodes in [12, 16, 20]:
        qc = pd.read_csv(out / f"v2_instantaneous_null100_fixed_n{n_nodes}_qc.csv")
        qc["node_specification"] = f"fixed_n{n_nodes}"
        require_recording_grid(qc, expected_keys(eligibility, f'fixed{n_nodes}'), f'fixed{n_nodes} QC')
        qc_frames.append(qc)
    pd.concat(qc_frames, ignore_index=True).to_csv(out / "v2_fixed_node_null_qc_all.csv", index=False)
    print(
        json.dumps(
            {
                "n_recordings": direct.recording.nunique(),
                "node_specifications": sorted(direct.node_specification.unique()),
                "mean_dynamic_fixed_n16_jaccard": float(overlap.jaccard.mean()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
