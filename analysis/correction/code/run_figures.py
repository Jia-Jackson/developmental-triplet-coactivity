from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from hotopology.plotting import (
    PALETTE,
    add_panel_label,
    apply_publication_style,
    export_publication_figure,
)


DIVS = [14, 21, 28, 35, 42, 49, 56, 63]
AGE_CMAP = LinearSegmentedColormap.from_list(
    "development", [PALETTE["early"], PALETTE["transition"], PALETTE["late"]]
)


def _age_colors() -> dict[int, tuple[float, float, float, float]]:
    return {div: AGE_CMAP(i / (len(DIVS) - 1)) for i, div in enumerate(DIVS)}


def _trajectory_panel(
    ax,
    table: pd.DataFrame,
    metric: str,
    ylabel: str,
    *,
    log_scale: bool = False,
) -> None:
    colors = _age_colors()
    rng = np.random.default_rng(901)
    summary = table.groupby("DIV")[metric].agg(["mean", "sem"])
    for div in DIVS:
        values = table.loc[table["DIV"] == div, metric].dropna().to_numpy(float)
        x = np.full(values.size, div) + rng.uniform(-1.0, 1.0, values.size)
        ax.scatter(x, values, s=8, alpha=0.38, color=colors[div], linewidths=0, zorder=1)
    ax.errorbar(
        DIVS,
        summary.loc[DIVS, "mean"],
        yerr=summary.loc[DIVS, "sem"],
        color=PALETTE["text"],
        marker="o",
        markersize=3.2,
        linewidth=1.1,
        capsize=2,
        zorder=3,
    )
    if log_scale:
        ax.set_yscale("log")
    ax.set_xticks(DIVS)
    ax.set_xlabel("Developmental age (DIV)")
    ax.set_ylabel(ylabel)


def _median_trajectory_panel(ax, table: pd.DataFrame, metric: str, ylabel: str) -> None:
    colors = _age_colors()
    rng = np.random.default_rng(902)
    summary = table.groupby("DIV")[metric].agg(
        median="median",
        q25=lambda values: values.quantile(0.25),
        q75=lambda values: values.quantile(0.75),
    )
    for div in DIVS:
        values = table.loc[table["DIV"] == div, metric].dropna().to_numpy(float)
        x = np.full(values.size, div) + rng.uniform(-1.0, 1.0, values.size)
        ax.scatter(x, values, s=8, alpha=0.38, color=colors[div], linewidths=0, zorder=1)
    medians = summary.loc[DIVS, "median"].to_numpy()
    lower = medians - summary.loc[DIVS, "q25"].to_numpy()
    upper = summary.loc[DIVS, "q75"].to_numpy() - medians
    ax.errorbar(
        DIVS,
        medians,
        yerr=np.vstack([lower, upper]),
        color=PALETTE["text"],
        marker="o",
        markersize=3.2,
        linewidth=1.1,
        capsize=2,
        zorder=3,
    )
    ax.set_xticks(DIVS)
    ax.set_xlabel("Developmental age (DIV)")
    ax.set_ylabel(ylabel)


def figure_observed(root: Path, direct: pd.DataFrame) -> None:
    primary = direct.loc[direct["bin_ms"] == 25].copy()
    source = primary[
        [
            "recording",
            "subject_id",
            "DIV",
            "n_analysis_nodes",
            "mean_population_count",
            "fraction_bins_k_ge_3",
            "triplet_support_mass",
            "h1_total_persistence_capped",
        ]
    ]
    source.to_csv(root / "source_data" / "Figure_1_observed_higher_order.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    panels = [
        ("mean_population_count", "Mean coactive electrodes per bin", False),
        ("fraction_bins_k_ge_3", "Fraction of bins with K ≥ 3", False),
        ("triplet_support_mass", "Support-weighted triplet mass", True),
        ("h1_total_persistence_capped", "Capped H1 total persistence", False),
    ]
    for label, ax, (metric, ylabel, log_scale) in zip("abcd", axes.flat, panels):
        _trajectory_panel(ax, primary, metric, ylabel, log_scale=log_scale)
        add_panel_label(ax, label)
    fig.suptitle("Developmental trajectories of higher-order population organization", fontsize=8)
    export_publication_figure(fig, root / "figures" / "Figure_1_observed_higher_order")


def figure_sensitivity(root: Path, direct: pd.DataFrame) -> None:
    metrics = [
        ("mean_population_count", "Mean coactive electrodes per bin"),
        ("fraction_bins_k_ge_3", "Fraction of bins with K ≥ 3"),
        ("triplet_support_mass", "Support-weighted triplet mass"),
        ("h1_total_persistence_capped", "Capped H1 total persistence"),
    ]
    summary = (
        direct.groupby(["bin_ms", "DIV"])[[m for m, _ in metrics]]
        .agg(["mean", "sem"])
        .reset_index()
    )
    summary.columns = [
        "bin_ms" if col[0] == "bin_ms" else "DIV" if col[0] == "DIV" else f"{col[0]}_{col[1]}"
        for col in summary.columns
    ]
    summary.to_csv(root / "source_data" / "Figure_2_bin_sensitivity.csv", index=False)

    bin_colors = {10: "#4C78A8", 25: "#2E9E44", 50: "#E39C37", 100: "#B54C8A"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    for label, ax, (metric, ylabel) in zip("abcd", axes.flat, metrics):
        for bin_ms in [10, 25, 50, 100]:
            sub = summary.loc[summary["bin_ms"] == bin_ms].set_index("DIV").loc[DIVS]
            ax.errorbar(
                DIVS,
                sub[f"{metric}_mean"],
                yerr=sub[f"{metric}_sem"],
                color=bin_colors[bin_ms],
                marker="o",
                markersize=2.7,
                linewidth=1.0,
                capsize=1.7,
                label=f"{bin_ms} ms",
            )
        ax.set_xticks(DIVS)
        ax.set_xlabel("Developmental age (DIV)")
        ax.set_ylabel(ylabel)
        add_panel_label(ax, label)
    axes[0, 0].legend(ncol=2, loc="best", fontsize=6)
    fig.suptitle("Sensitivity of developmental trajectories to temporal bin width", fontsize=8)
    export_publication_figure(fig, root / "figures" / "Figure_2_bin_size_sensitivity")


def figure_bursts(root: Path, burst: pd.DataFrame) -> None:
    metrics = [
        ("fraction_bins_k_ge_3", "Fraction of bins with K ≥ 3"),
        ("triplet_support_mass", "Support-weighted triplet mass"),
        ("triplet_repeated5_density", "Density of triplets repeated ≥ 5 times"),
        ("h1_total_persistence_capped", "Capped H1 total persistence"),
    ]
    source = (
        burst.groupby(["stratum", "DIV"])[[m for m, _ in metrics]]
        .agg(["count", "mean", "sem", "median"])
        .reset_index()
    )
    source.columns = [
        "stratum" if col[0] == "stratum" else "DIV" if col[0] == "DIV" else f"{col[0]}_{col[1]}"
        for col in source.columns
    ]
    source.to_csv(root / "source_data" / "Figure_3_burst_stratification.csv", index=False)

    colors = {"inside_burst": PALETTE["late"], "outside_burst": PALETTE["pairwise"]}
    labels = {"inside_burst": "Inside network bursts", "outside_burst": "Outside network bursts"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    for panel, ax, (metric, ylabel) in zip("abcd", axes.flat, metrics):
        for stratum in ["outside_burst", "inside_burst"]:
            sub = source.loc[source["stratum"] == stratum].set_index("DIV").loc[DIVS]
            ax.errorbar(
                DIVS,
                sub[f"{metric}_mean"],
                yerr=sub[f"{metric}_sem"],
                color=colors[stratum],
                marker="o",
                markersize=3,
                linewidth=1.1,
                capsize=2,
                label=labels[stratum],
            )
        ax.set_xticks(DIVS)
        ax.set_xlabel("Developmental age (DIV)")
        ax.set_ylabel(ylabel)
        add_panel_label(ax, panel)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Higher-order organization inside and outside detected network bursts", fontsize=8)
    export_publication_figure(fig, root / "figures" / "Figure_3_burst_stratification")


def figure_null_qc(root: Path) -> None:
    tables = root / "tables"
    qc_path = tables / "maxent_model_qc.csv"
    topo_path = tables / "topology_null_comparisons.csv"
    if not qc_path.exists() or not topo_path.exists():
        return
    qc = pd.read_csv(qc_path)
    topo = pd.read_csv(topo_path)
    qc.to_csv(root / "source_data" / "Figure_4_maxent_qc.csv", index=False)
    topo.to_csv(root / "source_data" / "Figure_4_topology_null.csv", index=False)

    metrics = [
        ("pairwise_firing_rmse", "Firing-probability RMSE", 0.02),
        ("pairwise_pair_rmse", "Pairwise-moment RMSE", 0.02),
        ("pairwise_k_rmse", "Population-count RMSE (diagnostic)", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    for panel, ax, (metric, ylabel, threshold) in zip("abc", axes.flat[:3], metrics):
        _trajectory_panel(ax, qc, metric, ylabel)
        if threshold is not None:
            ax.axhline(threshold, color=PALETTE["negative"], linestyle="--", linewidth=0.8)
        add_panel_label(ax, panel)

    ax = axes.flat[3]
    conditioned = topo.loc[
        (topo["model"] == "pairwise_conditioned_k")
        & (topo["metric"] == "h1_total_persistence_capped")
    ].copy()
    _median_trajectory_panel(ax, conditioned, "z", r"H1 persistence versus $P_2(x\mid K(t))$, z")
    ax.axhline(0, color=PALETTE["null"], linewidth=0.8)
    add_panel_label(ax, "d")
    fig.suptitle("Pairwise maximum-entropy fit quality and topological null deviation", fontsize=8)
    export_publication_figure(fig, root / "figures" / "Figure_4_null_model_qc")


def figure_irreducible(root: Path) -> None:
    path = root / "tables" / "irreducible_triplet_summary.csv"
    top_path = root / "tables" / "exploratory_top_triplet_residuals.csv"
    if not path.exists() or not top_path.exists():
        return
    table = pd.read_csv(path)
    top = pd.read_csv(top_path)
    table.to_csv(root / "source_data" / "Figure_5_irreducible_triplets.csv", index=False)
    conditioned = table.loc[table["model"] == "pairwise_conditioned_k"].copy()
    participation_rows: list[dict[str, float | int | str]] = []
    for (recording, div), subset in top.groupby(["recording", "DIV"]):
        values = np.zeros(64, dtype=float)
        for row in subset.itertuples(index=False):
            weight = abs(float(row.residual_probability))
            for node in (row.node_i_zero_based, row.node_j_zero_based, row.node_k_zero_based):
                values[int(node)] += weight
        total = values.sum()
        if total > 0:
            values /= total
        for node, value in enumerate(values):
            participation_rows.append(
                {"recording": recording, "DIV": int(div), "electrode_index": node + 1, "participation": value}
            )
    participation = pd.DataFrame(participation_rows)
    participation_summary = (
        participation.groupby(["DIV", "electrode_index"])["participation"]
        .agg(["count", "mean", "sem"])
        .reset_index()
    )
    participation_summary.to_csv(
        root / "source_data" / "Figure_5_exploratory_spatial_participation.csv", index=False
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    _trajectory_panel(
        axes[0, 0], conditioned, "absolute_residual_mass", "Absolute residual triplet mass"
    )
    add_panel_label(axes[0, 0], "a")
    _median_trajectory_panel(
        axes[0, 1], conditioned, "absolute_residual_z", "Residual redistribution versus null, z"
    )
    axes[0, 1].axhline(0, color=PALETTE["null"], linewidth=0.8)
    add_panel_label(axes[0, 1], "b")

    selected_ages = [21, 63]
    vmax = max(
        participation_summary.loc[participation_summary["DIV"].isin(selected_ages), "mean"].max(),
        1e-6,
    )
    for panel, ax, div in zip("cd", axes[1], selected_ages):
        values = (
            participation_summary.loc[participation_summary["DIV"] == div]
            .set_index("electrode_index")
            .reindex(range(1, 65))["mean"]
            .fillna(0.0)
            .to_numpy()
            .reshape(8, 8)
        )
        image = ax.imshow(values, cmap="magma", vmin=0, vmax=vmax, origin="upper")
        ax.set_title(f"DIV{div}", fontsize=7)
        ax.set_xlabel("Electrode column")
        ax.set_ylabel("Electrode row")
        ax.set_xticks(range(8), range(1, 9))
        ax.set_yticks(range(8), range(1, 9))
        for spine in ax.spines.values():
            spine.set_visible(False)
        add_panel_label(ax, panel)
    colorbar = fig.colorbar(image, ax=list(axes[1]), shrink=0.76, pad=0.02)
    colorbar.set_label("Mean normalized |residual| participation", fontsize=6)
    fig.suptitle("Triplet dependence remaining after pairwise and population-count constraints", fontsize=8)
    export_publication_figure(fig, root / "figures" / "Figure_5_irreducible_triplets")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    root = Path(args.result_dir)
    (root / "figures").mkdir(exist_ok=True)
    (root / "source_data").mkdir(exist_ok=True)
    apply_publication_style(7.0)

    direct = pd.read_csv(root / "tables" / "direct_higher_order_metrics_all_bins.csv")
    burst = pd.read_csv(root / "tables" / "burst_stratified_higher_order_metrics_25ms.csv")
    figure_observed(root, direct)
    figure_sensitivity(root, direct)
    figure_bursts(root, burst)
    figure_null_qc(root)
    figure_irreducible(root)


if __name__ == "__main__":
    main()
