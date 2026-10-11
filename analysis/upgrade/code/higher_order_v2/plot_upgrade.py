from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


BLUE = "#2F6FB0"
ORANGE = "#E68632"
TEAL = "#2A9D8F"
PURPLE = "#7B5AB6"
GRAY = "#6C757D"
RED = "#C44E52"


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    sns.set_style("ticks")


def export_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in ((".svg", 300), (".pdf", 300), (".png", 300), (".tiff", 600)):
        fig.savefig(base.with_suffix(suffix), dpi=dpi, bbox_inches="tight", facecolor="white")


def panel_schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(0.2, 9.4, "Separate representation from added information", weight="bold", fontsize=8.5)
    nodes = np.array([[1.1, 7.2], [2.2, 7.2], [1.65, 8.35]])
    for first, second in ((0, 1), (0, 2), (1, 2)):
        ax.plot(nodes[[first, second], 0], nodes[[first, second], 1], color=BLUE, lw=2)
    ax.scatter(nodes[:, 0], nodes[:, 1], s=85, color=ORANGE, edgecolor="white", zorder=3)
    ax.text(1.65, 6.55, "Triplet value", ha="center", color=GRAY)
    ax.annotate("", xy=(4.0, 7.55), xytext=(2.7, 7.55), arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5))
    ax.text(4.0, 8.3, "Incidence projection", weight="bold", color=BLUE)
    ax.text(4.0, 7.55, "to all 3 edges and 3 nodes", va="center")
    ax.text(4.0, 6.85, "no top-triplet truncation", va="center", color=GRAY)
    ax.plot([0.5, 9.5], [5.55, 5.55], color="#D9D9D9", lw=0.8)
    for y, label, color in ((4.45, "Direct support", BLUE), (3.35, "Pairwise-null residual", RED), (2.25, "O-information", PURPLE)):
        ax.add_patch(mpl.patches.FancyBboxPatch((0.45, y - 0.35), 3.15, 0.7,
                    boxstyle="round,pad=0.05", fc=mpl.colors.to_rgba(color, 0.12), ec=color, lw=1))
        ax.text(2.025, y, label, ha="center", va="center", color=color, weight="bold", fontsize=6.8)
    ax.annotate("", xy=(4.75, 4.0), xytext=(3.65, 4.0), arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5))
    ax.text(4.9, 4.7, "Cross-age retrieval", weight="bold")
    ax.text(4.9, 4.0, "same culture only\nsame age excluded", va="center")
    ax.text(4.9, 3.0, "17 complete wells × 8 ages")
    ax.text(4.9, 2.3, "permutation + whole-well bootstrap", color=GRAY, fontsize=7)
    ax.text(0.6, 0.95, "Does a higher-order profile add information\nbeyond firing and pairwise coactivity?",
            fontsize=7.5, weight="bold", va="center")


def panel_taxonomy(ax: plt.Axes, table_dir: Path) -> None:
    matrix = pd.read_csv(table_dir / "v2_cross_metric_median_correlation_matrix.csv", index_col=0)
    order = [
        "pair_support_probability", "triplet_support_projection", "absolute_residual_projection",
        "oinfo_projection_bits", "oinfo_deviation_projection_bits", "residual_projection",
    ]
    labels = {
        "pair_support_probability": "Pair support",
        "triplet_support_projection": "Triplet support",
        "absolute_residual_projection": "|Residual|",
        "oinfo_projection_bits": "O-information",
        "oinfo_deviation_projection_bits": "O-info deviation",
        "residual_projection": "Signed residual",
    }
    matrix = matrix.loc[order, order]
    sns.heatmap(matrix, ax=ax, cmap="vlag", center=0, vmin=-1, vmax=1, square=True,
                annot=True, fmt=".2f", annot_kws={"fontsize": 6.5}, cbar_kws={"label": "Median Spearman ρ", "shrink": 0.7})
    ax.set_xticklabels([labels[item] for item in order], rotation=45, ha="right")
    ax.set_yticklabels([labels[item] for item in order], rotation=0)
    ax.set_title("Direct coactivity and residual structure separate")


def panel_continuity(ax: plt.Axes, table_dir: Path) -> None:
    data = pd.read_csv(table_dir / "v2_fixed12_temporal_continuity_tests.csv")
    order = [
        "pair_support_probability", "triplet_support_projection", "oinfo_projection_bits",
        "residual_projection", "absolute_residual_projection", "oinfo_deviation_projection_bits",
    ]
    labels = {
        "pair_support_probability": "Pair support",
        "triplet_support_projection": "Triplet support",
        "oinfo_projection_bits": "O-information",
        "residual_projection": "Signed residual",
        "absolute_residual_projection": "|Residual|",
        "oinfo_deviation_projection_bits": "O-info deviation",
    }
    data = data.set_index("metric").loc[order].reset_index()
    y = np.arange(len(data))[::-1]
    ax.hlines(y, data["permuted_ci_low"], data["permuted_ci_high"], color="#A8A8A8", lw=5, label="Age-order permutation 95% interval")
    ax.scatter(data["permuted_mean"], y, color="white", edgecolor=GRAY, s=24, zorder=3)
    ax.scatter(data["observed_mean_adjacent_similarity"], y, color=TEAL, s=30, zorder=4, label="Observed adjacent ages")
    for yi, row in zip(y, data.itertuples()):
        label = "P<0.001" if row.empirical_p_greater < 0.001 else f"P={row.empirical_p_greater:.3f}"
        ax.text(max(row.observed_mean_adjacent_similarity, row.permuted_ci_high) + 0.012, yi, label, va="center", fontsize=6.5)
    ax.set_yticks(y, [labels[item] for item in order])
    ax.set_xlabel("Mean adjacent-age Spearman ρ")
    ax.set_title("Residual profiles retain temporal ordering")
    sns.despine(ax=ax)


def panel_identification(ax: plt.Axes, table_dir: Path) -> None:
    data = pd.read_csv(table_dir / "v2_longitudinal_identification_summary.csv").set_index("metric")
    order = ["firing_node", "pair_edge", "triplet_native", "oinfo_native"]
    labels = ["Firing", "Pair support", "Triplet support", "O-information"]
    colors = [GRAY, BLUE, ORANGE, PURPLE]
    x = np.arange(len(order))
    values = data.loc[order, "top1_accuracy"].to_numpy()
    low = values - data.loc[order, "top1_ci_low"].to_numpy()
    high = data.loc[order, "top1_ci_high"].to_numpy() - values
    for xi, value, lower, upper, color in zip(x, values, low, high, colors):
        ax.errorbar(xi, value, yerr=np.array([[lower], [upper]]), fmt="o", color=color,
                    ecolor=color, elinewidth=1.5, capsize=3, ms=4.5)
    chance = float(data.loc[order[0], "analytic_chance_top1"])
    ax.axhline(chance, color="#999999", ls="--", lw=1, label=f"Chance = {chance:.3f}")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Top-1 cross-age accuracy")
    ax.set_ylim(0, 0.88)
    ax.set_title("Higher-order profiles identify wells,\nbut not better than pair support")
    ax.legend(frameon=False, loc="upper right")
    sns.despine(ax=ax)


def panel_incremental_identity(ax: plt.Axes, table_dir: Path) -> None:
    data = pd.read_csv(table_dir / "v2_identification_incremental_comparisons.csv")
    data = data[data["statistic"] == "top1"].set_index("candidate_metric")
    order = ["triplet_native", "oinfo_native", "pair_plus_triplet_edge", "pair_plus_oinfo_edge"]
    labels = ["Triplet support", "O-information", "Pair + triplet", "Pair + O-info"]
    data = data.loc[order]
    y = np.arange(len(order))[::-1]
    values = data["observed_difference"].to_numpy()
    low = values - data["bootstrap_ci_low"].to_numpy()
    high = data["bootstrap_ci_high"].to_numpy() - values
    ax.errorbar(values, y, xerr=np.vstack([low, high]), fmt="o", color=RED, ecolor=RED, capsize=3, ms=4)
    ax.axvline(0, color="#777777", lw=1, ls="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Top-1 accuracy difference vs pair support")
    ax.set_title("No incremental identification advantage")
    sns.despine(ax=ax)


def panel_oinfo_deviation(ax: plt.Axes, table_dir: Path) -> None:
    summary = pd.read_csv(table_dir / "v2_dynamic16_oinfo_age_summary.csv")
    data = summary[(summary["metric"] == "oinfo_mean") & (summary["quantity"] == "null_deviation")]
    labels = {"instantaneous_count_conditioned": "Instantaneous", "kinetic_count_conditioned": "Kinetic"}
    colors = {"instantaneous_count_conditioned": BLUE, "kinetic_count_conditioned": ORANGE}
    for model, group in data.groupby("model"):
        group = group.sort_values("DIV")
        ax.errorbar(group["DIV"], group["mean"], yerr=group["sem"], marker="o", ms=3.5,
                    lw=1.2, capsize=2, color=colors[model], label=labels[model])
    ax.axhline(0, color="#777777", lw=0.9, ls="--")
    ax.set_xlabel("DIV")
    ax.set_ylabel("Observed − null mean O-information (bits)")
    ax.set_title("Null-calibrated O-information changes\nare small and inconsistent")
    ax.legend(frameon=False)
    sns.despine(ax=ax)


def make_main_figure(output: Path) -> None:
    table_dir = output / "tables/upgrade_v2"
    fig = plt.figure(figsize=(7.2, 8.5))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.05, 1, 1], hspace=0.58, wspace=0.66)
    axes = [fig.add_subplot(grid[i, j]) for i in range(3) for j in range(2)]
    functions = [panel_schematic, panel_taxonomy, panel_continuity, panel_identification,
                 panel_incremental_identity, panel_oinfo_deviation]
    for label, ax, function in zip("ABCDEF", axes, functions):
        function(ax, table_dir) if function is not panel_schematic else function(ax)
        ax.text(-0.12, 1.20, label, transform=ax.transAxes, fontsize=11, weight="bold", va="top")
    fig.suptitle("Higher-order profiles are temporally structured but largely overlap pairwise information",
                 fontsize=10.5, weight="bold", y=0.995)
    export_figure(fig, output / "figures/main/Figure_7_higher_order_profiles")
    for label, ax, function in zip("ABCDEF", axes, functions):
        separate = plt.figure(figsize=(3.5, 3.0))
        panel = separate.add_subplot(111)
        function(panel, table_dir) if function is not panel_schematic else function(panel)
        export_figure(separate, output / f"figures/panels/Figure_7{label}")
        plt.close(separate)
    plt.close(fig)


def make_oinfo_supplement(output: Path) -> None:
    table_dir = output / "tables/upgrade_v2"
    analysis = pd.read_csv(table_dir / "v2_dynamic16_oinfo_analysis_table.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    metrics = ["oinfo_mean", "oinfo_redundant_fraction"]
    titles = ["Mean O-information", "Redundancy-dominated triplet fraction"]
    for ax, metric, title in zip(axes[0], metrics, titles):
        unique = analysis.drop_duplicates(["recording", "metric"])
        sns.boxplot(data=unique[unique["metric"] == metric], x="DIV", y="observed", ax=ax,
                    color="#DDE8F3", width=0.55, showfliers=False, linewidth=0.8)
        sns.stripplot(data=unique[unique["metric"] == metric], x="DIV", y="observed", ax=ax,
                      color=BLUE, alpha=0.55, size=2.3, jitter=0.18)
        ax.set_title(title)
        ax.set_ylabel("Bits" if metric == "oinfo_mean" else "Fraction")
        ax.set_xlabel("DIV")
        sns.despine(ax=ax)
    for ax, model in zip(axes[1], ["instantaneous_count_conditioned", "kinetic_count_conditioned"]):
        subset = analysis[(analysis["metric"] == "oinfo_mean") & (analysis["model"] == model)]
        sns.boxplot(data=subset, x="DIV", y="null_deviation", ax=ax, color="#F2E6D8",
                    width=0.55, showfliers=False, linewidth=0.8)
        sns.stripplot(data=subset, x="DIV", y="null_deviation", hue="culture", ax=ax,
                      palette={"P1": BLUE, "P2": ORANGE, "P3": TEAL}, alpha=0.7, size=2.5, jitter=0.18)
        ax.axhline(0, color="#777777", ls="--", lw=0.8)
        ax.set_title(("Instantaneous" if "instantaneous" in model else "Kinetic") + " null deviation")
        ax.set_ylabel("Observed − null mean (bits)")
        ax.set_xlabel("DIV")
        ax.legend(frameon=False, title="Culture", ncol=3, loc="upper left")
        sns.despine(ax=ax)
    for label, ax in zip("ABCD", axes.flat):
        ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=11, weight="bold", va="top")
    fig.tight_layout()
    export_figure(fig, output / "figures/supplementary/Figure_7_figure_supplement_1_O_information")
    plt.close(fig)


def make_prediction_supplement(output: Path) -> None:
    table_dir = output / "tables/upgrade_v2"
    metrics = pd.read_csv(table_dir / "v2_prediction_metrics.csv")
    boot = pd.read_csv(table_dir / "v2_prediction_bootstrap.csv")
    predictions = pd.read_csv(table_dir / "v2_prediction_predictions.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    order = ["traditional_27", "traditional_27_plus_legacy", "traditional_27_plus_oinfo", "traditional_27_plus_all_higher_order"]
    labels = ["Traditional 27", "+ legacy HO", "+ O-info", "+ all HO"]
    colors = [GRAY, BLUE, PURPLE, ORANGE]
    for ax, analysis_range in zip(axes[0], ["DIV14_63", "DIV21_63"]):
        subset = metrics[(metrics["analysis_range"] == analysis_range) & (metrics["validation"] == "leave_one_culture_out")].set_index("model_set").loc[order]
        ax.bar(np.arange(len(order)), subset["mae"], color=colors, width=0.7)
        ax.set_xticks(np.arange(len(order)), labels, rotation=25, ha="right")
        ax.set_ylabel("LOCO MAE (days)")
        ax.set_title("All ages" if analysis_range == "DIV14_63" else "DIV21–63 only")
        sns.despine(ax=ax)
    subset = boot[(boot["analysis_range"] == "DIV14_63") & (boot["validation"] == "leave_one_culture_out")]
    subset = subset[subset["baseline_name"] == "traditional_27"].set_index("augmented_name").loc[order[1:]]
    values = subset["delta_mae_baseline_minus_augmented"].to_numpy()
    axes[1, 0].errorbar(values, np.arange(3)[::-1],
                        xerr=np.vstack([values - subset["ci95_low"], subset["ci95_high"] - values]),
                        fmt="o", color=BLUE, capsize=3)
    axes[1, 0].axvline(0, color="#777777", ls="--", lw=0.8)
    axes[1, 0].set_yticks(np.arange(3)[::-1], labels[1:])
    axes[1, 0].set_xlabel("MAE improvement vs traditional 27 (days)")
    axes[1, 0].set_title("Whole-well bootstrap, all ages")
    sns.despine(ax=axes[1, 0])
    subset = predictions[(predictions["analysis_range"] == "DIV14_63") &
                         (predictions["validation"] == "leave_one_culture_out") &
                         (predictions["model_set"] == "traditional_27_plus_all_higher_order")]
    for culture, group in subset.groupby("culture"):
        axes[1, 1].scatter(group["observed"], group["prediction"], s=13, alpha=0.65,
                           label=culture, color={"P1": BLUE, "P2": ORANGE, "P3": TEAL}[culture])
    axes[1, 1].plot([14, 63], [14, 63], color="#777777", ls="--", lw=0.8)
    axes[1, 1].set_xlabel("Observed DIV")
    axes[1, 1].set_ylabel("Predicted DIV")
    axes[1, 1].set_title("All-feature LOCO predictions")
    axes[1, 1].legend(frameon=False)
    sns.despine(ax=axes[1, 1])
    for label, ax in zip("ABCD", axes.flat):
        ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=11, weight="bold", va="top")
    fig.tight_layout()
    export_figure(fig, output / "figures/supplementary/Figure_7_figure_supplement_2_prediction")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_style()
    output = args.output.resolve()
    make_main_figure(output)
    make_oinfo_supplement(output)
    make_prediction_supplement(output)


if __name__ == "__main__":
    main()
