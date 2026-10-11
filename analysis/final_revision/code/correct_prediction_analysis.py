from __future__ import annotations

import argparse
import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


ALPHAS = np.logspace(-3, 3, 13).tolist()
BASELINE_SET = "traditional_27"
RESIDUAL_TOPOLOGY_SET = "legacy_higher_order_only"
OINFO_SET = "oinfo_only"
BASELINE_PLUS_RESIDUAL = "traditional_27_plus_legacy"
BASELINE_PLUS_OINFO = "traditional_27_plus_oinfo"
BASELINE_PLUS_ALL = "traditional_27_plus_all_higher_order"

CORRECTED_RESIDUAL_TOPOLOGY = [
    "log1p_absolute_residual_mass",
    "absolute_residual_z",
    "h1_persistence_deviation_z",
    "direct_triplet_repeated5_density",
    "direct_h1_total_persistence_capped",
]

EXCLUDED_FEATURES = {
    "positive_residual_z": (
        "Numerically redundant with absolute_residual_z under exact K(t); "
        "the two columns differ only at floating-point roundoff."
    ),
    "h1_infinite_deviation_z": (
        "Endpoint-count z is mathematically undefined when the null SD is zero. "
        "This occurs for most recordings, so the feature was excluded rather than imputed."
    ),
}

MODEL_ORDER = [
    BASELINE_SET,
    BASELINE_PLUS_RESIDUAL,
    BASELINE_PLUS_OINFO,
    BASELINE_PLUS_ALL,
]
MODEL_LABELS = {
    BASELINE_SET: "Conventional 27",
    BASELINE_PLUS_RESIDUAL: "+ residual/topological (5)",
    BASELINE_PLUS_OINFO: "+ O-information (12)",
    BASELINE_PLUS_ALL: "+ both groups",
}

BLUE = "#2F6FB0"
ORANGE = "#E68632"
TEAL = "#2A9D8F"
PURPLE = "#7B5AB6"
GRAY = "#6C757D"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_imports(v2_code: Path, source_code: Path) -> None:
    sys.path.insert(0, str(v2_code))
    sys.path.insert(0, str(source_code))


def bootstrap_delta(
    predictions: pd.DataFrame,
    *,
    baseline_name: str,
    augmented_name: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, float | int], np.ndarray]:
    index_columns = [
        column
        for column in ("row_index", "subject_id", "culture", "observed")
        if column in predictions.columns
    ]
    wide = predictions.pivot_table(
        index=index_columns,
        columns="model_set",
        values="prediction",
        aggfunc="first",
    ).reset_index()
    if not {baseline_name, augmented_name}.issubset(wide.columns):
        raise ValueError("baseline or augmented predictions are missing")
    wide["baseline_error"] = np.abs(wide["observed"] - wide[baseline_name])
    wide["augmented_error"] = np.abs(wide["observed"] - wide[augmented_name])
    observed_delta = float(
        wide["baseline_error"].mean() - wide["augmented_error"].mean()
    )
    subject = (
        wide.groupby(["culture", "subject_id"])
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
    for culture in sorted(subject["culture"].unique()):
        group = subject[subject["culture"] == culture].reset_index(drop=True)
        indices = rng.integers(0, len(group), size=(n_bootstrap, len(group)))
        baseline_sum += group["baseline_sum"].to_numpy(float)[indices].sum(axis=1)
        augmented_sum += group["augmented_sum"].to_numpy(float)[indices].sum(axis=1)
        count += group["n"].to_numpy(float)[indices].sum(axis=1)
    delta = baseline_sum / count - augmented_sum / count
    # Positive delta favors the augmented model. The add-one lower-tail probability
    # is reported descriptively; the prespecified decision rule uses the percentile CI.
    p_one_sided = float((1 + np.sum(delta <= 0)) / (n_bootstrap + 1))
    result: dict[str, float | int] = {
        "n_bootstrap": int(n_bootstrap),
        "delta_mae_baseline_minus_augmented": observed_delta,
        "ci95_low": float(np.quantile(delta, 0.025)),
        "ci95_high": float(np.quantile(delta, 0.975)),
        "probability_improvement": float(np.mean(delta > 0)),
        "empirical_p_one_sided_no_improvement": p_one_sided,
        "seed": int(seed),
    }
    return result, delta


def fold_metrics(predictions: pd.DataFrame, prediction_metrics) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(
        ["analysis_range", "validation", "model_set", "outer_fold"], sort=True
    ):
        values = prediction_metrics(group)
        values.update(
            {
                "analysis_range": keys[0],
                "validation": keys[1],
                "model_set": keys[2],
                "outer_fold": int(keys[3]),
                "held_out_group": group["held_out_group"].iloc[0],
            }
        )
        rows.append(values)
    return pd.DataFrame(rows)


def correct_endpoint_tables(source: Path, table_dir: Path) -> dict[str, object]:
    source_tables = source / "tables" / "computational_closure_v2"
    frames = []
    summaries = []
    for family, filename in (
        ("instantaneous", "v2_instantaneous_null100_comparisons.csv"),
        ("kinetic", "v2_kinetic_null_comparisons.csv"),
    ):
        frame = pd.read_csv(source_tables / filename)
        endpoint = frame[frame["metric"].eq("h1_infinite_count")].copy()
        zero_variance = endpoint["null_sd"].eq(0)
        exact_match = np.isclose(endpoint["observed"], endpoint["null_mean"])
        endpoint["z_original"] = endpoint["z"]
        endpoint["z"] = endpoint["z"].where(~zero_variance, np.nan)
        endpoint["z_available"] = endpoint["z"].notna()
        endpoint["z_unavailable_reason"] = np.where(
            zero_variance & exact_match,
            "zero_null_variance_observed_equals_null",
            np.where(
                zero_variance & ~exact_match,
                "zero_null_variance_observed_differs",
                "",
            ),
        )
        endpoint.insert(0, "null_family", family)
        frames.append(endpoint)
        for div, group in endpoint.groupby("DIV", sort=True):
            summaries.append(
                {
                    "null_family": family,
                    "DIV": int(div),
                    "n_recordings": int(len(group)),
                    "n_z_available": int(group["z"].notna().sum()),
                    "n_z_undefined_zero_variance": int(group["null_sd"].eq(0).sum()),
                    "n_zero_variance_exact_match": int(
                        (group["null_sd"].eq(0) & np.isclose(group["observed"], group["null_mean"])).sum()
                    ),
                    "n_zero_variance_observed_differs": int(
                        (group["null_sd"].eq(0) & ~np.isclose(group["observed"], group["null_mean"])).sum()
                    ),
                }
            )
    corrected = pd.concat(frames, ignore_index=True)
    corrected.to_csv(table_dir / "corrected_endpoint_h1_null_comparisons.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(table_dir / "corrected_endpoint_h1_sample_sizes.csv", index=False)

    # Age-specific signed-rank tests are retained only as an audit. With nearly all
    # z values undefined, most age/model cells contain too few observations for
    # meaningful inference and are not used for manuscript claims.
    from hotopology.statistics import one_sample_wilcoxon_by_div

    tests = []
    for family, group in corrected.groupby("null_family", sort=True):
        test = one_sample_wilcoxon_by_div(
            group,
            metric="z",
            divs=[14, 21, 28, 35, 42, 49, 56, 63],
            null_value=0.0,
            alternative="greater",
        )
        test.insert(0, "null_family", family)
        tests.append(test)
    endpoint_tests = pd.concat(tests, ignore_index=True)
    endpoint_tests.to_csv(table_dir / "corrected_endpoint_h1_one_sample_tests_audit.csv", index=False)
    return {
        "instantaneous_total": int((corrected["null_family"] == "instantaneous").sum()),
        "instantaneous_z_available": int(
            corrected.loc[corrected["null_family"] == "instantaneous", "z"].notna().sum()
        ),
        "kinetic_total": int((corrected["null_family"] == "kinetic").sum()),
        "kinetic_z_available": int(
            corrected.loc[corrected["null_family"] == "kinetic", "z"].notna().sum()
        ),
        "all_zero_variance_marked_missing": bool(
            corrected.loc[corrected["null_sd"].eq(0), "z"].isna().all()
        ),
    }


def export_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    tiff_path = base.with_suffix(".tiff")
    # Render to an in-memory PNG first because Matplotlib's TIFF writer may encode
    # RGBA despite a white canvas. The explicit conversion guarantees RGB output.
    buffer = BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    buffer.seek(0)
    rgb_path = tiff_path.with_name(f"{tiff_path.stem}.rgb.tmp.tiff")
    with Image.open(buffer) as image:
        rgb = image.convert("RGB")
        rgb.save(rgb_path, compression="tiff_lzw", dpi=(600, 600))
        rgb.close()
    buffer.close()
    if tiff_path.exists():
        tiff_path.unlink()
    rgb_path.replace(tiff_path)


def set_figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    sns.set_style("ticks")


def draw_mae_panel(ax: plt.Axes, metrics: pd.DataFrame, analysis_range: str) -> None:
    selected = (
        metrics[
            (metrics["analysis_range"] == analysis_range)
            & (metrics["validation"] == "leave_one_culture_out")
        ]
        .set_index("model_set")
        .loc[MODEL_ORDER]
    )
    colors = [GRAY, BLUE, PURPLE, ORANGE]
    x = np.arange(len(MODEL_ORDER))
    bars = ax.bar(x, selected["mae"], color=colors, width=0.70)
    for bar, value in zip(bars, selected["mae"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(selected["mae"]) * 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    ax.set_xticks(x, [MODEL_LABELS[name] for name in MODEL_ORDER], rotation=25, ha="right")
    ax.set_ylabel("LOCO MAE (days)")
    ax.set_title("DIV14–63" if analysis_range == "DIV14_63" else "DIV21–63")
    ax.set_ylim(0, max(selected["mae"]) * 1.18)
    sns.despine(ax=ax)


def make_figures(output: Path) -> None:
    set_figure_style()
    table_dir = output / "tables"
    metrics = pd.read_csv(table_dir / "corrected_prediction_metrics.csv")
    boot = pd.read_csv(table_dir / "corrected_prediction_bootstrap.csv")
    predictions = pd.read_csv(table_dir / "corrected_prediction_predictions.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    draw_mae_panel(axes[0, 0], metrics, "DIV14_63")
    draw_mae_panel(axes[0, 1], metrics, "DIV21_63")

    selected = boot[
        (boot["analysis_range"] == "DIV14_63")
        & (boot["validation"] == "leave_one_culture_out")
        & (boot["baseline_name"] == BASELINE_SET)
    ].set_index("augmented_name").loc[MODEL_ORDER[1:]]
    y = np.arange(3)[::-1]
    values = selected["delta_mae_baseline_minus_augmented"].to_numpy(float)
    low = values - selected["ci95_low"].to_numpy(float)
    high = selected["ci95_high"].to_numpy(float) - values
    axes[1, 0].errorbar(
        values,
        y,
        xerr=np.vstack([low, high]),
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        capsize=3,
        ms=4.5,
    )
    axes[1, 0].axvline(0, color="#777777", ls="--", lw=0.8)
    axes[1, 0].set_yticks(y, [MODEL_LABELS[name] for name in MODEL_ORDER[1:]])
    axes[1, 0].set_xlabel("MAE improvement vs conventional 27 (days)")
    axes[1, 0].set_title("Culture-well bootstrap, DIV14–63")
    sns.despine(ax=axes[1, 0])

    selected_predictions = predictions[
        (predictions["analysis_range"] == "DIV14_63")
        & (predictions["validation"] == "leave_one_culture_out")
        & (predictions["model_set"] == BASELINE_PLUS_ALL)
    ]
    for culture, group in selected_predictions.groupby("culture", sort=True):
        axes[1, 1].scatter(
            group["observed"],
            group["prediction"],
            s=13,
            alpha=0.68,
            label=culture,
            color={"P1": BLUE, "P2": ORANGE, "P3": TEAL}[culture],
        )
    raw_bounds = [
        min(14, float(selected_predictions["prediction"].min())),
        max(63, float(selected_predictions["prediction"].max())),
    ]
    padding = 0.04 * (raw_bounds[1] - raw_bounds[0])
    bounds = [raw_bounds[0] - padding, raw_bounds[1] + padding]
    axes[1, 1].plot(bounds, bounds, color="#777777", ls="--", lw=0.8)
    axes[1, 1].set_xlim(bounds)
    axes[1, 1].set_ylim(bounds)
    axes[1, 1].set_xlabel("Observed DIV")
    axes[1, 1].set_ylabel("Predicted DIV")
    axes[1, 1].set_title("All corrected feature groups, LOCO")
    axes[1, 1].legend(frameon=False, title="Culture")
    sns.despine(ax=axes[1, 1])

    for label, ax in zip("ABCD", axes.flat):
        ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=11, weight="bold", va="top")
    fig.tight_layout()
    figure_dir = output / "figures"
    export_figure(fig, figure_dir / "Figure_S12_corrected_prediction")
    plt.close(fig)

    # Standalone panels use the same data and rendering backend as the composite.
    for label, drawer in (
        ("A", lambda ax: draw_mae_panel(ax, metrics, "DIV14_63")),
        ("B", lambda ax: draw_mae_panel(ax, metrics, "DIV21_63")),
    ):
        panel_fig, panel_ax = plt.subplots(figsize=(3.5, 3.0))
        drawer(panel_ax)
        panel_fig.tight_layout()
        export_figure(panel_fig, figure_dir / f"Figure_S12{label}_corrected")
        plt.close(panel_fig)

    panel_fig, panel_ax = plt.subplots(figsize=(3.7, 3.0))
    panel_ax.errorbar(
        values,
        y,
        xerr=np.vstack([low, high]),
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        capsize=3,
        ms=4.5,
    )
    panel_ax.axvline(0, color="#777777", ls="--", lw=0.8)
    panel_ax.set_yticks(y, [MODEL_LABELS[name] for name in MODEL_ORDER[1:]])
    panel_ax.set_xlabel("MAE improvement vs conventional 27 (days)")
    panel_ax.set_title("Culture-well bootstrap, DIV14–63")
    sns.despine(ax=panel_ax)
    panel_fig.tight_layout()
    export_figure(panel_fig, figure_dir / "Figure_S12C_corrected")
    plt.close(panel_fig)

    panel_fig, panel_ax = plt.subplots(figsize=(3.5, 3.0))
    for culture, group in selected_predictions.groupby("culture", sort=True):
        panel_ax.scatter(
            group["observed"],
            group["prediction"],
            s=13,
            alpha=0.68,
            label=culture,
            color={"P1": BLUE, "P2": ORANGE, "P3": TEAL}[culture],
        )
    panel_ax.plot(bounds, bounds, color="#777777", ls="--", lw=0.8)
    panel_ax.set_xlim(bounds)
    panel_ax.set_ylim(bounds)
    panel_ax.set_xlabel("Observed DIV")
    panel_ax.set_ylabel("Predicted DIV")
    panel_ax.set_title("All corrected feature groups, LOCO")
    panel_ax.legend(frameon=False, title="Culture")
    sns.despine(ax=panel_ax)
    panel_fig.tight_layout()
    export_figure(panel_fig, figure_dir / "Figure_S12D_corrected")
    plt.close(panel_fig)

    # Verify raster export metadata now, while the renderer context is explicit.
    tiff_meta = {}
    for path in sorted(figure_dir.glob("*.tiff")):
        with Image.open(path) as image:
            tiff_meta[path.name] = {
                "pixels": [int(image.width), int(image.height)],
                "dpi": [float(value) for value in image.info.get("dpi", (np.nan, np.nan))],
                "mode": image.mode,
            }
    (output / "qa" / "figure_metadata.json").write_text(
        json.dumps(tiff_meta, indent=2), encoding="utf-8"
    )


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    table_dir = output / "tables"
    qa_dir = output / "qa"
    table_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    configure_imports(args.v2_code.resolve(), args.source_code.resolve())
    from higher_order_v2.vendor_hotopology.incremental import (
        compare_feature_sets,
        prediction_metrics,
    )

    original_table_path = (
        args.v2_root.resolve()
        / "tables"
        / "upgrade_v2"
        / "v2_prediction_analysis_table_preimputation.csv"
    )
    original_manifest_path = (
        args.v2_root.resolve()
        / "tables"
        / "upgrade_v2"
        / "v2_prediction_feature_manifest.csv"
    )
    table = pd.read_csv(original_table_path)
    original_manifest = pd.read_csv(original_manifest_path)
    if len(table) != 142 or table["recording"].duplicated().any():
        raise ValueError("Expected 142 unique prediction records")

    baseline = original_manifest.loc[
        original_manifest["model_set"] == BASELINE_SET, "feature"
    ].tolist()
    oinfo = original_manifest.loc[
        original_manifest["model_set"] == OINFO_SET, "feature"
    ].tolist()
    if len(baseline) != 27 or len(oinfo) != 12:
        raise ValueError(f"Unexpected feature counts: baseline={len(baseline)}, oinfo={len(oinfo)}")

    duplicate_max_abs_difference = float(
        np.nanmax(
            np.abs(
                table["absolute_residual_z"].to_numpy(float)
                - table["positive_residual_z"].to_numpy(float)
            )
        )
    )
    if duplicate_max_abs_difference > 1e-12:
        raise ValueError("Expected residual-z duplicate was not reproduced")

    missing_required = [
        feature
        for feature in [*baseline, *CORRECTED_RESIDUAL_TOPOLOGY, *oinfo]
        if feature not in table.columns
    ]
    if missing_required:
        raise ValueError(f"Missing required features: {missing_required}")

    endpoint_audit = correct_endpoint_tables(args.source.resolve(), table_dir)
    corrected_table = table.drop(columns=list(EXCLUDED_FEATURES)).copy()
    corrected_table.to_csv(
        table_dir / "corrected_prediction_analysis_table_preimputation.csv", index=False
    )
    exclusion_audit = pd.DataFrame(
        [
            {
                "feature": feature,
                "decision": "excluded_from_prediction",
                "reason": reason,
                "original_nonmissing": int(table[feature].notna().sum()),
            }
            for feature, reason in EXCLUDED_FEATURES.items()
        ]
    )
    exclusion_audit.to_csv(table_dir / "corrected_prediction_excluded_features.csv", index=False)

    feature_sets = {
        BASELINE_SET: baseline,
        RESIDUAL_TOPOLOGY_SET: CORRECTED_RESIDUAL_TOPOLOGY,
        OINFO_SET: oinfo,
        BASELINE_PLUS_RESIDUAL: [*baseline, *CORRECTED_RESIDUAL_TOPOLOGY],
        BASELINE_PLUS_OINFO: [*baseline, *oinfo],
        BASELINE_PLUS_ALL: [*baseline, *CORRECTED_RESIDUAL_TOPOLOGY, *oinfo],
    }
    manifest = pd.DataFrame(
        [
            {"model_set": model_set, "feature": feature}
            for model_set, features in feature_sets.items()
            for feature in features
        ]
    )
    manifest.to_csv(table_dir / "corrected_prediction_feature_manifest.csv", index=False)

    predictions_all = []
    metrics_all = []
    audits_all = []
    settings = [
        ("DIV14_63", corrected_table),
        ("DIV21_63", corrected_table[corrected_table["DIV"] >= 21].copy()),
    ]
    validations = [
        ("leave_one_culture_out", "culture", "subject_id"),
        ("leave_one_well_out", "subject_id", "subject_id"),
    ]
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
            predictions_all.append(predictions)
            metrics_all.append(metrics)
            audits_all.append(audit)

    predictions = pd.concat(predictions_all, ignore_index=True)
    metrics = pd.concat(metrics_all, ignore_index=True)
    audits = pd.concat(audits_all, ignore_index=True)
    predictions.to_csv(table_dir / "corrected_prediction_predictions.csv", index=False)
    metrics.to_csv(table_dir / "corrected_prediction_metrics.csv", index=False)
    fold_metrics(predictions, prediction_metrics).to_csv(
        table_dir / "corrected_prediction_fold_metrics.csv", index=False
    )
    audits.to_csv(table_dir / "corrected_prediction_fold_audit.csv", index=False)

    comparisons = [
        (BASELINE_SET, BASELINE_PLUS_RESIDUAL),
        (BASELINE_SET, BASELINE_PLUS_OINFO),
        (BASELINE_SET, BASELINE_PLUS_ALL),
        (BASELINE_PLUS_RESIDUAL, BASELINE_PLUS_ALL),
        (BASELINE_PLUS_OINFO, BASELINE_PLUS_ALL),
    ]
    bootstrap_rows = []
    sample_rows = []
    for group_index, ((analysis_range, validation), group) in enumerate(
        predictions.groupby(["analysis_range", "validation"], sort=True)
    ):
        for comparison_index, (baseline_name, augmented_name) in enumerate(comparisons):
            seed = 20260950 + 10 * group_index + comparison_index
            result, samples = bootstrap_delta(
                group[group["model_set"].isin([baseline_name, augmented_name])],
                baseline_name=baseline_name,
                augmented_name=augmented_name,
                n_bootstrap=5000,
                seed=seed,
            )
            result.update(
                {
                    "analysis_range": analysis_range,
                    "validation": validation,
                    "baseline_name": baseline_name,
                    "augmented_name": augmented_name,
                }
            )
            bootstrap_rows.append(result)
            for draw, value in enumerate(samples):
                sample_rows.append(
                    {
                        "analysis_range": analysis_range,
                        "validation": validation,
                        "baseline_name": baseline_name,
                        "augmented_name": augmented_name,
                        "seed": seed,
                        "draw": draw,
                        "delta_mae_baseline_minus_augmented": float(value),
                    }
                )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(table_dir / "corrected_prediction_bootstrap.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(
        table_dir / "corrected_prediction_bootstrap_draws.csv", index=False
    )

    gates = []
    for row in bootstrap.itertuples():
        augmented_metric = metrics[
            (metrics["analysis_range"] == row.analysis_range)
            & (metrics["validation"] == row.validation)
            & (metrics["model_set"] == row.augmented_name)
        ]
        r2 = float(augmented_metric["r2"].iloc[0])
        passes = bool(
            row.validation == "leave_one_culture_out" and row.ci95_low > 0 and r2 > 0
        )
        gates.append(
            {
                "analysis_range": row.analysis_range,
                "validation": row.validation,
                "baseline_name": row.baseline_name,
                "augmented_name": row.augmented_name,
                "mae_improvement": row.delta_mae_baseline_minus_augmented,
                "ci95_low": row.ci95_low,
                "ci95_high": row.ci95_high,
                "empirical_p_one_sided_no_improvement": row.empirical_p_one_sided_no_improvement,
                "augmented_r2": r2,
                "main_text_gate_pass": passes,
                "criterion": (
                    "LOCO clustered-bootstrap MAE improvement CI excludes zero "
                    "and augmented pooled R2 is positive"
                ),
            }
        )
    pd.DataFrame(gates).to_csv(
        table_dir / "corrected_prediction_claim_gate.csv", index=False
    )

    audit_payload = {
        "source_prediction_table": str(original_table_path),
        "source_prediction_table_sha256": sha256(original_table_path),
        "n_recordings_full_range": int(len(corrected_table)),
        "n_recordings_late_range": int((corrected_table["DIV"] >= 21).sum()),
        "n_baseline_features": len(baseline),
        "n_residual_topological_features_after_correction": len(CORRECTED_RESIDUAL_TOPOLOGY),
        "n_oinfo_features": len(oinfo),
        "duplicate_max_abs_difference": duplicate_max_abs_difference,
        "excluded_features": EXCLUDED_FEATURES,
        "endpoint_audit": endpoint_audit,
        "all_outer_groups_disjoint": bool(audits["outer_groups_disjoint"].all()),
        "missing_baseline_cells_before_fold_local_imputation": int(
            corrected_table[baseline].isna().sum().sum()
        ),
        "remaining_limitation": (
            "Some imported baseline features inherit upstream DIV-dependent masks; "
            "fold-local imputation and scaling cannot undo those upstream masks."
        ),
        "figure_contract": {
            "core_conclusion": (
                "After removing redundant and mathematically undefined predictors, "
                "residual/topological features are evaluated for culture-held-out age information."
            ),
            "evidence_chain": [
                "Full-range LOCO MAE across matched feature sets",
                "Late-range LOCO MAE across matched feature sets",
                "Culture-well bootstrap uncertainty for full-range MAE improvements",
                "Held-out predictions for the corrected all-feature model",
            ],
            "archetype": "quantitative grid",
            "backend": "Python/matplotlib only",
            "review_risks": [
                "Only three independent cultures",
                "Imported conventional features retain upstream age-dependent masks",
                "Full-range performance may be driven by separation of DIV14 from later ages",
            ],
        },
    }
    (qa_dir / "correction_audit.json").write_text(
        json.dumps(audit_payload, indent=2), encoding="utf-8"
    )
    make_figures(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--source-code", type=Path, required=True)
    parser.add_argument("--v2-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
