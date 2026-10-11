from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats

from hotopology.plotting import (
    PALETTE,
    add_panel_label,
    apply_publication_style,
    export_publication_figure,
)


DIVS = [14, 21, 28, 35, 42, 49, 56, 63]
COLORS = {
    "P1": "#484878",
    "P2": "#42949E",
    "P3": "#D9544D",
    "instantaneous": "#3775BA",
    "kinetic": "#9A4D8E",
    "baseline": "#7884B4",
    "augmented": "#D9544D",
    "neutral": "#A8A8A8",
    "positive": "#2E9E44",
    "negative": "#E53935",
}


def _save_source(frame: pd.DataFrame, source_dir: Path, filename: str) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(source_dir / filename, index=False)


def _box(ax, xy, width, height, text, *, color, fontsize=6.5, linewidth=0.8):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor=color,
        edgecolor="#4D4D4D",
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#272727",
    )


def _arrow(ax, start, end, *, color="#767676"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=color,
        )
    )


def figure_schematic(root: Path, figure_dir: Path, source_dir: Path):
    apply_publication_style(7)
    fig = plt.figure(figsize=(7.2, 5.0))
    grid = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.16)
    axes = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(2)]
    for ax in axes:
        ax.set_axis_off()

    ax = axes[0]
    add_panel_label(ax, "a", x=-0.02, y=1.0)
    ax.text(0.04, 0.92, "Longitudinal biological hierarchy", transform=ax.transAxes, fontweight="bold")
    for row, culture in enumerate(["P1", "P2", "P3"]):
        y = 0.68 - row * 0.22
        _box(ax, (0.04, y), 0.18, 0.13, f"{culture}\nculture", color="#E0E0F0", fontsize=5.7)
        _arrow(ax, (0.23, y + 0.065), (0.34, y + 0.065))
        _box(ax, (0.35, y), 0.24, 0.13, "6 longitudinal\nwells", color="#E0F0F0", fontsize=5.7)
        _arrow(ax, (0.60, y + 0.065), (0.69, y + 0.065))
        _box(ax, (0.70, y), 0.26, 0.13, "DIV14 → … → 63", color="#F0E0D0", fontsize=5.7)
    ax.text(0.04, 0.05, "143 recordings; one predefined DIV35 exclusion", transform=ax.transAxes, fontsize=6, color="#606060")

    ax = axes[1]
    add_panel_label(ax, "b", x=-0.02, y=1.0)
    ax.text(0.04, 0.92, "Higher-order representation", transform=ax.transAxes, fontweight="bold")
    _box(ax, (0.04, 0.59), 0.23, 0.20, "Electrode × time\nbinary activity", color="#E0F0F0", fontsize=5.7)
    _arrow(ax, (0.28, 0.69), (0.38, 0.69))
    _box(ax, (0.39, 0.59), 0.22, 0.20, "Pair and triplet\nsupport", color="#E4CCD8", fontsize=5.7)
    _arrow(ax, (0.62, 0.69), (0.72, 0.69))
    _box(ax, (0.73, 0.59), 0.23, 0.20, "Support filtration\nH0 / H1", color="#F0C0CC", fontsize=5.7)
    _box(ax, (0.10, 0.18), 0.32, 0.18, "Triplet-specific\nresidual redistribution", color="#DDF3DE")
    _box(ax, (0.58, 0.18), 0.32, 0.18, "Topology beyond\nmatched null", color="#DDF3DE")
    _arrow(ax, (0.50, 0.58), (0.29, 0.37))
    _arrow(ax, (0.60, 0.58), (0.73, 0.37))

    ax = axes[2]
    add_panel_label(ax, "c", x=-0.02, y=1.0)
    ax.text(0.04, 0.92, "Complementary lower-order nulls", transform=ax.transAxes, fontweight="bold")
    _box(ax, (0.05, 0.52), 0.40, 0.28, "Instantaneous pairwise\n+ exact K(t)\n\nFirst/second moments", color="#DCE8F5", fontsize=5.4)
    _box(ax, (0.55, 0.52), 0.40, 0.28, "Lag-1 kinetic\n+ exact K(t)\n\nPersistence + propagation", color="#E8DCEC", fontsize=5.4)
    _arrow(ax, (0.25, 0.50), (0.38, 0.28), color=COLORS["instantaneous"])
    _arrow(ax, (0.75, 0.50), (0.62, 0.28), color=COLORS["kinetic"])
    _box(ax, (0.29, 0.10), 0.42, 0.17, "Convergent residual\nand H1 evidence?", color="#DDF3DE", fontsize=5.5)

    ax = axes[3]
    add_panel_label(ax, "d", x=-0.02, y=1.0)
    ax.text(0.04, 0.92, "Computational closure", transform=ax.transAxes, fontweight="bold")
    items = [
        ("Hierarchical inference", "culture + repeated well", "#E0E0F0"),
        ("Activity adjustment", "mean K + firing + bursts", "#E0F0F0"),
        ("Incremental value", "27-feature baseline + held-out CV", "#F0E0D0"),
        ("Robustness", "fixed N, bin width, segment, duration", "#E4CCD8"),
    ]
    for index, (title, detail, color) in enumerate(items):
        y = 0.68 - index * 0.18
        _box(ax, (0.08, y), 0.84, 0.13, f"{title}:  {detail}", color=color)

    fig.suptitle("Computational closure of longitudinal higher-order MEA analysis", fontsize=9, fontweight="bold", y=0.995)
    export_publication_figure(fig, figure_dir / "Figure_S1_computational_closure_schematic")
    source = pd.DataFrame(
        {
            "item": ["cultures", "wells", "recordings", "div_stages", "null_realizations"],
            "value": [3, 18, 143, 8, 100],
        }
    )
    _save_source(source, source_dir, "Figure_S1_source_data.csv")


def figure_hierarchical(root: Path, figure_dir: Path, source_dir: Path):
    apply_publication_style(6.5)
    out = root / "tables" / "computational_closure_v2"
    contrasts = pd.read_csv(out / "v2_hierarchical_adjacent_contrasts.csv")
    culture = pd.read_csv(out / "v2_culture_specific_adjacent_changes.csv")
    direct = pd.read_csv(root / "tables" / "direct_higher_order_metrics_all_bins.csv")
    direct = direct.loc[direct["bin_ms"] == 25]
    selected = {
        "triplet_support_mass": "Triplet support mass",
        "triplet_repeated5_density": "Repeated-triplet density",
        "h1_total_persistence_capped": "H1 persistence",
        "log1p_absolute_residual_mass": "Absolute residual mass (log1p)",
        "absolute_residual_z": "Absolute residual z",
        "z_h1_total_persistence_capped": "H1 deviation z",
    }
    effect = contrasts.loc[
        (contrasts["DIV_first"] == 14)
        & (contrasts["DIV_second"] == 21)
        & contrasts["outcome"].isin(selected)
    ].copy()
    effect["label"] = effect["outcome"].map(selected)
    effect = effect.drop_duplicates("outcome").sort_values("standardized_effect")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), gridspec_kw={"hspace": 0.44, "wspace": 0.68})
    ax = axes[0, 0]
    metric = "triplet_support_mass"
    for culture_name, subset in direct.groupby(direct["plate"].map(lambda value: f"P{int(value)}")):
        summary = subset.groupby("DIV")[metric].mean().reindex(DIVS)
        ax.plot(DIVS, summary, marker="o", ms=3, lw=1.1, color=COLORS[culture_name], label=culture_name)
    ax.set_yscale("log")
    ax.set_xlabel("DIV")
    ax.set_ylabel("Triplet support mass")
    ax.legend(ncol=3, fontsize=5.5)
    add_panel_label(ax, "a")

    ax = axes[0, 1]
    y = np.arange(len(effect))
    ax.axvline(0, color="#A8A8A8", lw=0.8, ls="--")
    ax.errorbar(
        effect["standardized_effect"],
        y,
        xerr=[
            effect["standardized_effect"] - effect["ci95_low"] / effect["estimate"] * effect["standardized_effect"],
            effect["ci95_high"] / effect["estimate"] * effect["standardized_effect"] - effect["standardized_effect"],
        ],
        fmt="o",
        color=COLORS["augmented"],
        ecolor=COLORS["augmented"],
        ms=3,
        lw=0.8,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(effect["label"])
    ax.set_xlabel("Standardized DIV14→21 effect")
    ax.set_xscale("symlog", linthresh=1)
    add_panel_label(ax, "b")

    ax = axes[1, 0]
    heat_metrics = ["triplet_support_mass", "h1_total_persistence_capped", "absolute_residual_z"]
    heat = culture.loc[
        (culture["DIV_first"] == 14)
        & (culture["DIV_second"] == 21)
        & culture["outcome"].isin(heat_metrics)
    ].pivot_table(index="outcome", columns="culture", values="median_change", aggfunc="first")
    heat = heat.reindex(heat_metrics)
    standardized = heat.div(heat.abs().max(axis=1).replace(0, np.nan), axis=0)
    im = ax.imshow(standardized, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(standardized.columns)), standardized.columns)
    ax.set_yticks(range(len(standardized.index)), [selected.get(item, item) for item in standardized.index])
    ax.set_title("Culture-specific median change")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025, label="Row-normalized direction")
    add_panel_label(ax, "c")

    ax = axes[1, 1]
    burst = contrasts.loc[
        (contrasts["family"] == "burst_inside_outside_difference")
        & (contrasts["DIV_first"] == 14)
        & (contrasts["DIV_second"] == 21)
    ].sort_values("standardized_effect")
    burst_labels = {
        "inside_minus_outside_fraction_bins_k_ge_3": "Fraction K≥3",
        "inside_minus_outside_h1_total_persistence_capped": "H1 persistence",
        "inside_minus_outside_mean_population_count": "Mean K",
        "inside_minus_outside_triplet_support_mass": "Triplet support",
        "inside_minus_outside_triplet_repeated5_density": "Repeated triplets",
        "inside_minus_outside_triplet_occurrence_density": "Triplet occurrence",
    }
    labels = [burst_labels.get(item, item) for item in burst["outcome"]]
    colors = [COLORS["positive"] if value > 0 else COLORS["negative"] for value in burst["standardized_effect"]]
    ax.barh(np.arange(len(burst)), burst["standardized_effect"], color=colors, alpha=0.8)
    ax.axvline(0, color="#767676", lw=0.8)
    ax.set_yticks(np.arange(len(burst)), labels)
    ax.set_xlabel("Standardized change in inside–outside contrast")
    add_panel_label(ax, "d")
    fig.suptitle("Hierarchical reanalysis of the original higher-order results", fontsize=9, fontweight="bold")
    export_publication_figure(fig, figure_dir / "Figure_6_hierarchical_reanalysis")
    _save_source(effect, source_dir, "Figure_6b_hierarchical_effects.csv")
    _save_source(culture, source_dir, "Figure_6c_culture_changes.csv")


def figure_activity(root: Path, figure_dir: Path, source_dir: Path):
    apply_publication_style(6.5)
    out = root / "tables" / "computational_closure_v2"
    table = pd.read_csv(out / "v2_activity_adjustment_analysis_table.csv")
    adjusted = pd.read_csv(out / "v2_activity_adjusted_adjacent_contrasts.csv")
    form_cv = pd.read_csv(out / "v2_activity_form_cv.csv")
    unadjusted = pd.read_csv(out / "v2_hierarchical_adjacent_contrasts.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), gridspec_kw={"hspace": 0.42, "wspace": 0.42})
    ax = axes[0, 0]
    scatter = ax.scatter(
        table["mean_k_top16"],
        table["log1p_absolute_residual_mass"],
        c=table["DIV"],
        cmap="viridis",
        s=12,
        alpha=0.75,
        edgecolors="none",
    )
    rho, p = stats.spearmanr(table["mean_k_top16"], table["log1p_absolute_residual_mass"])
    ax.text(0.03, 0.95, f"Spearman ρ={rho:.2f}\nP={p:.1e}", transform=ax.transAxes, va="top", fontsize=5.5)
    ax.set_xlabel("Mean K in modeled 16-node subnetwork")
    ax.set_ylabel("log1p absolute residual mass")
    fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.03, label="DIV")
    add_panel_label(ax, "a")

    ax = axes[0, 1]
    outcomes = [
        "log1p_absolute_residual_mass",
        "absolute_residual_z",
        "h1_persistence_deviation_z",
        "h1_infinite_deviation_z",
    ]
    labels = ["Residual mass", "Residual z", "H1 persistence z", "H1 infinite z"]
    effect = adjusted.loc[
        (adjusted["DIV_first"] == 14)
        & (adjusted["DIV_second"] == 21)
        & (adjusted["adjustment_tier"] == "tier2_activity_burst")
        & adjusted["outcome"].isin(outcomes)
    ].set_index("outcome").reindex(outcomes)
    y = np.arange(len(outcomes))
    colors = [COLORS["positive"] if value > 0 else COLORS["negative"] for value in effect["standardized_effect"]]
    ax.barh(y, effect["standardized_effect"], color=colors, alpha=0.8)
    ax.axvline(0, color="#767676", lw=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Tier-2 adjusted standardized effect")
    for yi, pval in zip(y, effect["adjusted_p"]):
        ax.text(ax.get_xlim()[1], yi, f" P={pval:.3g}", va="center", ha="left", fontsize=5)
    add_panel_label(ax, "b")

    ax = axes[1, 0]
    for outcome, color in zip(outcomes, ["#484878", "#3775BA", "#42949E", "#9A4D8E"]):
        subset = form_cv.loc[form_cv["outcome"] == outcome].copy()
        subset["mae_above_best_percent"] = 100 * (subset["cv_mae"] / subset["cv_mae"].min() - 1)
        ax.plot(subset["activity_form"], subset["mae_above_best_percent"], marker="o", ms=3, lw=1, label=outcome.replace("_", " "), color=color)
    ax.set_ylabel("Group-CV MAE above best (%)")
    ax.set_xlabel("Activity functional form")
    ax.legend(fontsize=4.8)
    add_panel_label(ax, "c")

    ax = axes[1, 1]
    selected = adjusted.loc[
        (adjusted["DIV_first"] == 14)
        & (adjusted["DIV_second"] == 21)
        & adjusted["outcome"].isin(outcomes)
    ].copy()
    pivot = selected.pivot(index="outcome", columns="adjustment_tier", values="standardized_effect").reindex(outcomes)
    x = np.arange(len(outcomes))
    width = 0.35
    ax.bar(x - width / 2, pivot["tier1_mean_k"], width, label="Mean K adjusted", color="#7884B4")
    ax.bar(x + width / 2, pivot["tier2_activity_burst"], width, label="K + firing/burst adjusted", color="#D9544D")
    ax.axhline(0, color="#767676", lw=0.8)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Standardized DIV14→21 effect")
    ax.legend(fontsize=5.2)
    add_panel_label(ax, "d")
    fig.suptitle("Activity-adjusted developmental effects", fontsize=9, fontweight="bold")
    export_publication_figure(fig, figure_dir / "Figure_7_activity_adjusted")
    _save_source(table, source_dir, "Figure_7a_activity_table.csv")
    _save_source(adjusted, source_dir, "Figure_7bd_adjusted_effects.csv")
    _save_source(form_cv, source_dir, "Figure_7c_activity_form_cv.csv")


def figure_incremental(root: Path, figure_dir: Path, source_dir: Path):
    apply_publication_style(6.5)
    out = root / "tables" / "computational_closure_v2"
    metrics = pd.read_csv(out / "v2_incremental_metrics.csv")
    bootstrap = pd.read_csv(out / "v2_incremental_bootstrap.csv")
    predictions = pd.read_csv(out / "v2_incremental_predictions.csv")
    conditional = pd.read_csv(out / "v2_higher_order_conditional_div_contrasts.csv")
    fig = plt.figure(figsize=(7.2, 5.5))
    grid = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.45)
    ax_schematic = fig.add_subplot(grid[0, 0])
    ax_schematic.set_axis_off()
    add_panel_label(ax_schematic, "a", x=-0.03, y=1.0)
    ax_schematic.text(0.03, 0.93, "Leakage-free validation", transform=ax_schematic.transAxes, fontweight="bold")
    _box(ax_schematic, (0.04, 0.62), 0.38, 0.18, "Train cultures\nfit + tune", color="#DCE8F5", fontsize=5.4)
    _arrow(ax_schematic, (0.42, 0.71), (0.58, 0.71))
    _box(ax_schematic, (0.58, 0.62), 0.38, 0.18, "Held-out culture\npredict", color="#F0E0D0", fontsize=5.4)
    _box(ax_schematic, (0.09, 0.25), 0.82, 0.18, "Repeat P1 / P2 / P3\n+ leave-one-well-out", color="#E0F0F0", fontsize=5.4)
    _arrow(ax_schematic, (0.77, 0.60), (0.52, 0.44))

    ax = fig.add_subplot(grid[0, 1])
    add_panel_label(ax, "b")
    subset = metrics.loc[metrics["validation"] == "leave_one_culture_out"].copy()
    order = ["compact_baseline", "prior_27_baseline", "prior_27_plus_higher_order"]
    labels = ["Compact", "Prior 27", "27 + high-order"]
    x = np.arange(2)
    width = 0.22
    for index, (model, label, color) in enumerate(zip(order, labels, ["#B4C0E4", "#7884B4", "#D9544D"])):
        values = [subset.loc[(subset["analysis_range"] == analysis) & (subset["model_set"] == model), "mae"].iloc[0] for analysis in ["DIV14_63", "DIV21_63"]]
        ax.bar(x + (index - 1) * width, values, width, label=label, color=color)
    ax.set_xticks(x, ["DIV14–63", "DIV21–63"])
    ax.set_ylabel("Held-out MAE (DIV days)")
    ax.legend(fontsize=5)

    ax = fig.add_subplot(grid[0, 2])
    add_panel_label(ax, "c")
    boot = bootstrap.loc[bootstrap["baseline_name"] == "prior_27_baseline"].copy()
    boot["label"] = boot["analysis_range"].str.replace("_", "–") + "\n" + boot["validation"].map({"leave_one_culture_out": "culture-out", "leave_one_well_out": "well-out"})
    y = np.arange(len(boot))
    ax.axvline(0, color="#767676", lw=0.8, ls="--")
    ax.errorbar(
        boot["delta_mae_baseline_minus_augmented"],
        y,
        xerr=[boot["delta_mae_baseline_minus_augmented"] - boot["ci95_low"], boot["ci95_high"] - boot["delta_mae_baseline_minus_augmented"]],
        fmt="o",
        color="#D9544D",
        ecolor="#D9544D",
        ms=3,
        lw=0.8,
    )
    ax.set_yticks(y, boot["label"])
    ax.set_xlabel("MAE improvement (prior 27 − augmented)")

    ax = fig.add_subplot(grid[1, :2])
    add_panel_label(ax, "d", x=-0.07)
    pred = predictions.loc[
        (predictions["analysis_range"] == "DIV14_63")
        & (predictions["validation"] == "leave_one_culture_out")
        & (predictions["model_set"] == "prior_27_plus_higher_order")
    ]
    for culture_name, group in pred.groupby("culture"):
        ax.scatter(group["observed"], group["prediction"], s=12, alpha=0.75, color=COLORS[culture_name], label=culture_name)
    ax.plot([14, 63], [14, 63], color="#767676", lw=0.8, ls="--")
    ax.set_xlabel("Observed DIV")
    ax.set_ylabel("Held-out predicted DIV")
    ax.legend(ncol=3, fontsize=5.5)

    ax = fig.add_subplot(grid[1, 2])
    add_panel_label(ax, "e")
    cond = conditional.loc[(conditional["DIV_first"] == 14) & (conditional["DIV_second"] == 21)].copy()
    cond = cond.sort_values("standardized_effect")
    colors = [COLORS["positive"] if value > 0 else COLORS["negative"] for value in cond["standardized_effect"]]
    ax.barh(np.arange(len(cond)), cond["standardized_effect"], color=colors, alpha=0.8)
    ax.axvline(0, color="#767676", lw=0.8)
    conditional_labels = {
        "positive_residual_z": "Positive residual z",
        "absolute_residual_z": "Absolute residual z",
        "direct_h1_total_persistence_capped": "Direct H1 persistence",
        "h1_infinite_deviation_z": "H1 infinite z",
        "log1p_absolute_residual_mass": "Residual mass (log1p)",
        "direct_triplet_repeated5_density": "Repeated triplets",
        "h1_persistence_deviation_z": "H1 persistence z",
    }
    ax.set_yticks(np.arange(len(cond)), [conditional_labels.get(item, item) for item in cond["higher_order_feature"]], fontsize=5)
    ax.set_xlabel("Cross-fitted residual DIV14→21 effect")
    fig.suptitle("Incremental information beyond the previous 27 features", fontsize=9, fontweight="bold")
    export_publication_figure(fig, figure_dir / "Figure_8_incremental_value")
    _save_source(metrics, source_dir, "Figure_8b_incremental_metrics.csv")
    _save_source(bootstrap, source_dir, "Figure_8c_incremental_bootstrap.csv")
    _save_source(pred, source_dir, "Figure_8d_heldout_predictions.csv")
    _save_source(cond, source_dir, "Figure_8e_conditional_residual_effects.csv")


def _median_iqr(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    return frame.groupby(["model", "DIV"])[value].agg(
        median="median",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
    ).reset_index()


def figure_nulls(root: Path, figure_dir: Path, source_dir: Path):
    apply_publication_style(6.2)
    out = root / "tables" / "computational_closure_v2"
    instant_qc = pd.read_csv(out / "v2_instantaneous_null100_qc.csv")
    kinetic_qc = pd.read_csv(out / "v2_kinetic_null_qc.csv")
    instant_comp = pd.read_csv(out / "v2_instantaneous_null100_comparisons.csv")
    kinetic_comp = pd.read_csv(out / "v2_kinetic_null_comparisons.csv")
    instant_res = pd.read_csv(out / "v2_instantaneous_null100_triplet_residuals.csv")
    kinetic_res = pd.read_csv(out / "v2_kinetic_triplet_residuals.csv")
    instant_comp["model"] = "instantaneous"
    kinetic_comp["model"] = "kinetic"
    instant_res["model"] = "instantaneous"
    kinetic_res["model"] = "kinetic"
    comparisons = pd.concat([instant_comp, kinetic_comp], ignore_index=True)
    residuals = pd.concat([instant_res, kinetic_res], ignore_index=True)

    fig = plt.figure(figsize=(7.2, 6.0))
    grid = fig.add_gridspec(2, 3, hspace=0.47, wspace=0.45)
    ax = fig.add_subplot(grid[0, 0])
    ax.set_axis_off()
    add_panel_label(ax, "a", x=-0.03, y=1.0)
    _box(ax, (0.05, 0.58), 0.40, 0.25, "Instantaneous\npairwise\n+ exact K(t)", color="#DCE8F5", fontsize=5.4)
    _box(ax, (0.55, 0.58), 0.40, 0.25, "Kinetic lag-1\n+ exact K(t)", color="#E8DCEC", fontsize=5.4)
    _arrow(ax, (0.25, 0.56), (0.40, 0.26), color=COLORS["instantaneous"])
    _arrow(ax, (0.75, 0.56), (0.60, 0.26), color=COLORS["kinetic"])
    _box(ax, (0.28, 0.10), 0.44, 0.18, "Triplet residual\n+ H1", color="#DDF3DE", fontsize=5.4)

    ax = fig.add_subplot(grid[0, 1])
    add_panel_label(ax, "b")
    qc_data = [
        instant_qc["firing_probability_rmse"],
        kinetic_qc["firing_probability_rmse"],
        instant_qc["simultaneous_pair_moment_rmse"],
        kinetic_qc["simultaneous_pair_moment_rmse_diagnostic"],
        kinetic_qc["lag1_moment_rmse"],
    ]
    positions = np.arange(len(qc_data))
    bp = ax.boxplot(qc_data, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
    for patch_obj, color in zip(bp["boxes"], [COLORS["instantaneous"], COLORS["kinetic"], COLORS["instantaneous"], COLORS["kinetic"], COLORS["kinetic"]]):
        patch_obj.set_facecolor(color)
        patch_obj.set_alpha(0.7)
    ax.set_xticks(positions, ["Firing\ninst.", "Firing\nkin.", "Pair\ninst.", "Pair\nkin.", "Lag-1\nkin."], fontsize=5)
    ax.set_ylabel("Moment RMSE")

    ax = fig.add_subplot(grid[0, 2])
    add_panel_label(ax, "c")
    ax.axhline(0, color="#767676", lw=0.8, ls="--")
    for model, color in [("instantaneous", COLORS["instantaneous"]), ("kinetic", COLORS["kinetic"])]:
        subset = residuals.loc[residuals["model"] == model]
        summary = subset.groupby("DIV")["absolute_residual_z"].agg(median="median", q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75)).reindex(DIVS)
        ax.plot(DIVS, summary["median"], marker="o", ms=3, lw=1, color=color, label=model)
        ax.fill_between(DIVS, summary["q1"], summary["q3"], color=color, alpha=0.15)
    ax.set_xlabel("DIV")
    ax.set_ylabel("Absolute residual z")
    ax.legend(fontsize=5.2)

    for panel_index, (metric, ylabel) in enumerate([
        ("h1_total_persistence_capped", "H1 persistence deviation z"),
        ("h1_infinite_count", "H1 infinite-count deviation z"),
    ]):
        ax = fig.add_subplot(grid[1, panel_index])
        add_panel_label(ax, chr(ord("d") + panel_index))
        ax.axhline(0, color="#767676", lw=0.8, ls="--")
        metric_data = comparisons.loc[comparisons["metric"] == metric]
        summary = _median_iqr(metric_data, "z")
        for model, color in [("instantaneous", COLORS["instantaneous"]), ("kinetic", COLORS["kinetic"])]:
            subset = summary.loc[summary["model"] == model].set_index("DIV").reindex(DIVS)
            ax.plot(DIVS, subset["median"], marker="o", ms=3, lw=1, color=color, label=model)
            ax.fill_between(DIVS, subset["q1"], subset["q3"], color=color, alpha=0.15)
        ax.set_xlabel("DIV")
        ax.set_ylabel(ylabel)
        if panel_index == 0:
            ax.legend(fontsize=5.2)

    ax = fig.add_subplot(grid[1, 2])
    add_panel_label(ax, "f")
    improvement = kinetic_qc["heldout_log_loss_improvement"]
    for culture_name, subset in kinetic_qc.groupby("culture"):
        ax.scatter(subset["DIV"], subset["heldout_log_loss_improvement"], s=10, alpha=0.65, color=COLORS[culture_name], label=culture_name)
    ax.axhline(0, color="#767676", lw=0.8, ls="--")
    ax.set_xlabel("DIV")
    ax.set_ylabel("Held-out log-loss improvement\n(rate-only − kinetic)")
    ax.legend(ncol=3, fontsize=5)
    fig.suptitle("Complementary instantaneous and temporal null models", fontsize=9, fontweight="bold")
    export_publication_figure(fig, figure_dir / "Figure_9_null_model_comparison")
    _save_source(pd.concat([instant_qc.assign(model="instantaneous"), kinetic_qc.assign(model="kinetic")]), source_dir, "Figure_9b_f_null_qc.csv")
    _save_source(residuals, source_dir, "Figure_9c_residuals.csv")
    _save_source(comparisons, source_dir, "Figure_9de_h1_deviations.csv")


def figure_robustness(root: Path, figure_dir: Path, source_dir: Path):
    apply_publication_style(6.1)
    out = root / "tables" / "computational_closure_v2"
    overlap = pd.read_csv(out / "v2_dynamic_fixed_n16_overlap.csv")
    effects = pd.read_csv(out / "v2_node_specification_hierarchical_contrasts.csv")
    bin_agreement = pd.read_csv(out / "v2_bin_width_rank_agreement.csv")
    split = pd.read_csv(out / "v2_split_half_reliability.csv")
    duration = pd.read_csv(out / "v2_duration_convergence.csv")
    qc = pd.read_csv(out / "v2_fixed_node_null_qc_all.csv")

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.4), gridspec_kw={"hspace": 0.47, "wspace": 0.48})
    ax = axes[0, 0]
    add_panel_label(ax, "a")
    for culture_name, subset in overlap.groupby("culture"):
        summary = subset.groupby("DIV")["jaccard"].mean().reindex(DIVS)
        ax.plot(DIVS, summary, marker="o", ms=3, lw=1, color=COLORS[culture_name], label=culture_name)
    ax.set_ylim(0, 1)
    ax.set_xlabel("DIV")
    ax.set_ylabel("Dynamic vs fixed N=16 Jaccard")
    ax.legend(ncol=3, fontsize=4.8)

    ax = axes[0, 1]
    add_panel_label(ax, "b")
    selected = effects.loc[
        (effects["DIV_first"] == 14)
        & (effects["DIV_second"] == 21)
        & (effects["analysis_family"].isin(["residual", "null_deviation_z"]))
        & (effects["outcome"].isin(["log1p_absolute_residual_mass", "absolute_residual_z", "h1_total_persistence_capped"]))
    ].copy()
    selected["label"] = selected["node_specification"] + " | " + selected["outcome"].map({"log1p_absolute_residual_mass": "residual mass", "absolute_residual_z": "residual z", "h1_total_persistence_capped": "H1 z"})
    selected = selected.sort_values(["outcome", "node_specification"])
    y = np.arange(len(selected))
    colors = [COLORS["positive"] if value > 0 else COLORS["negative"] for value in selected["standardized_effect"]]
    ax.barh(y, selected["standardized_effect"], color=colors, alpha=0.75)
    ax.axvline(0, color="#767676", lw=0.8)
    ax.set_yticks(y, selected["label"], fontsize=4.7)
    ax.set_xlabel("Standardized DIV14→21 effect")

    ax = axes[0, 2]
    add_panel_label(ax, "c")
    metrics = ["triplet_repeated5_density", "triplet_support_mass", "h1_total_persistence_capped"]
    matrix = bin_agreement.loc[bin_agreement["metric"].isin(metrics)].pivot(index="metric", columns="comparison_bin_ms", values="spearman_rho").reindex(metrics)
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), [f"{value} ms" for value in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), ["Repeated triplets", "Triplet support", "H1 persistence"])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=5, color="white" if matrix.iloc[i, j] < 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label="Spearman vs 25 ms")

    ax = axes[1, 0]
    add_panel_label(ax, "d")
    split_order = split.sort_values("split_half_spearman_rho")
    ax.barh(np.arange(len(split_order)), split_order["split_half_spearman_rho"], color="#7884B4")
    split_labels = {
        "mean_population_count": "Mean population count",
        "fraction_bins_k_ge_3": "Fraction bins K>=3",
        "triplet_occurrence_density": "Triplet occurrence",
        "triplet_repeated5_density": "Repeated triplets",
        "triplet_support_mass": "Triplet support",
        "h1_total_persistence_capped": "H1 persistence",
        "h1_infinite_count": "H1 infinite count",
    }
    ax.set_yticks(np.arange(len(split_order)), [split_labels.get(item, item) for item in split_order["metric"]], fontsize=4.7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Split-half Spearman ρ")

    ax = axes[1, 1]
    add_panel_label(ax, "e")
    for metric, label, color in [
        ("triplet_repeated5_density", "Repeated triplets", "#3775BA"),
        ("triplet_support_mass", "Triplet support", "#42949E"),
        ("h1_total_persistence_capped", "H1 persistence", "#9A4D8E"),
    ]:
        subset = duration.loc[duration["metric"] == metric].sort_values("duration_s")
        ax.plot(subset["duration_s"], subset["spearman_vs_300s"], marker="o", ms=3, lw=1, label=label, color=color)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Cumulative duration (s)")
    ax.set_ylabel("Spearman vs 300-s estimate")
    ax.legend(fontsize=4.8)

    ax = axes[1, 2]
    add_panel_label(ax, "f")
    summary = qc.groupby("node_specification").agg(
        valid=("inferential_valid", "mean"),
        firing_rmse=("firing_probability_rmse", "max"),
        pair_rmse=("simultaneous_pair_moment_rmse", "max"),
    ).reset_index()
    x = np.arange(len(summary))
    width = 0.35
    ax.bar(x - width / 2, summary["firing_rmse"], width, color="#3775BA", label="Firing RMSE")
    ax.bar(x + width / 2, summary["pair_rmse"], width, color="#9A4D8E", label="Pair RMSE")
    ax.axhline(0.02, color="#E53935", lw=0.8, ls="--")
    ax.set_xticks(x, summary["node_specification"])
    ax.set_ylabel("Maximum RMSE")
    ax.text(0.98, 0.90, "QC threshold", transform=ax.transAxes, ha="right", va="top", fontsize=4.7, color="#A61B1B")
    ax.legend(fontsize=4.7, loc="upper left")
    fig.suptitle("Node, temporal-resolution, and recording-duration robustness", fontsize=9, fontweight="bold")
    export_publication_figure(fig, figure_dir / "Figure_10_robustness")
    _save_source(overlap, source_dir, "Figure_10a_node_overlap.csv")
    _save_source(selected, source_dir, "Figure_10b_node_effects.csv")
    _save_source(bin_agreement, source_dir, "Figure_10c_bin_agreement.csv")
    _save_source(split, source_dir, "Figure_10d_split_half.csv")
    _save_source(duration, source_dir, "Figure_10e_duration.csv")
    _save_source(summary, source_dir, "Figure_10f_fixed_node_qc.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    figure_dir = root / "figures" / "computational_closure_v2"
    source_dir = root / "source_data" / "computational_closure_v2"
    figure_schematic(root, figure_dir, source_dir)
    figure_hierarchical(root, figure_dir, source_dir)
    figure_activity(root, figure_dir, source_dir)
    figure_incremental(root, figure_dir, source_dir)
    figure_nulls(root, figure_dir, source_dir)
    figure_robustness(root, figure_dir, source_dir)
    print(figure_dir)


if __name__ == "__main__":
    main()
