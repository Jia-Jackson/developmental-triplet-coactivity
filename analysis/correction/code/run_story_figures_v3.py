from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from hotopology.plotting import apply_publication_style, export_publication_figure
from hotopology.raincloud import (
    developmental_palette,
    draw_adjacent_stat_band,
    draw_one_sample_symbols,
    draw_raincloud,
)
from hotopology.story_contract import (
    DIVS,
    adjacent_annotations,
    one_sample_annotations,
    p_to_symbol,
    validate_annotation_audit,
)


TABLES = ROOT / "tables"
V2 = TABLES / "computational_closure_v2"
FIGURE_ROOT = ROOT / "figures" / "manuscript_story_v3"
MAIN_DIR = FIGURE_ROOT / "main"
SUPP_DIR = FIGURE_ROOT / "supplementary"
SOURCE_DIR = ROOT / "source_data" / "manuscript_story_v3"

AGE_COLORS = developmental_palette(DIVS)
CULTURE_COLORS = {"P1": "#484878", "P2": "#42949E", "P3": "#D9544D"}
MODEL_COLORS = {
    "pairwise_conditioned_k_100": "#3775BA",
    "kinetic_conditioned_k": "#9A4D8E",
}


def _read(filename: str, *, v2: bool = False) -> pd.DataFrame:
    path = (V2 if v2 else TABLES) / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _panel_label(ax, label: str) -> None:
    ax.text(
        -0.16,
        1.035,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
    )


def _clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8.5, width=0.9, length=3.5)


def _box(ax, xy, width, height, text, color, fontsize=8.0) -> None:
    patch = plt.Rectangle(
        xy,
        width,
        height,
        transform=ax.transAxes,
        facecolor=color,
        edgecolor="#555555",
        linewidth=1.0,
        joinstyle="round",
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
        fontweight="bold",
        color="#272727",
    )


def _arrow(ax, start, end, color="#555555") -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color=color),
    )


def _with_culture(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "culture" not in frame:
        if "plate" in frame:
            frame["culture"] = "P" + frame["plate"].astype(int).astype(str)
        else:
            frame["culture"] = frame["recording"].str.extract(r"_(P[123])_")[0]
    return frame


def _source_csv(number: int, frames: list[tuple[str, pd.DataFrame]]) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    pieces = []
    for dataset, frame in frames:
        part = frame.copy()
        part.insert(0, "dataset", dataset)
        pieces.append(part)
    pd.concat(pieces, ignore_index=True, sort=False).to_csv(
        SOURCE_DIR / f"Figure_{number}_source_data.csv", index=False
    )


def _source_csv_s(number: int, frames: list[tuple[str, pd.DataFrame]]) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    pieces = []
    for dataset, frame in frames:
        part = frame.copy()
        part.insert(0, "dataset", dataset)
        pieces.append(part)
    pd.concat(pieces, ignore_index=True, sort=False).to_csv(
        SOURCE_DIR / f"Figure_S{number}_source_data.csv", index=False
    )


def _adjacent_audit(figure: str, panel: str, annotations: pd.DataFrame, source_file: str):
    return [
        {
            "figure": figure,
            "panel": panel,
            "comparison": row.comparison,
            "adjusted_p": float(row.adjusted_p),
            "symbol": row.symbol,
            "source_file": source_file,
            "test_family": "adjacent_div",
        }
        for row in annotations.itertuples(index=False)
    ]


def _one_sample_audit(
    figure: str,
    panel: str,
    annotations: pd.DataFrame,
    source_file: str,
    test_family: str = "one_sample_vs_zero",
):
    return [
        {
            "figure": figure,
            "panel": panel,
            "comparison": row.comparison,
            "adjusted_p": float(row.adjusted_p),
            "symbol": row.symbol,
            "source_file": source_file,
            "test_family": test_family,
        }
        for row in annotations.itertuples(index=False)
    ]


def _style_rain_axis(ax, ylabel: str, title: str, *, log: bool = False) -> None:
    _clean_axis(ax)
    ax.set_xlabel("Developmental age (DIV)", fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    if log:
        ax.set_yscale("log")


def _plot_developmental_raincloud(
    ax,
    frame: pd.DataFrame,
    *,
    outcome: str,
    annotations: pd.DataFrame,
    ylabel: str,
    title: str,
    log: bool = False,
    one_sample: pd.DataFrame | None = None,
) -> None:
    plot = frame[["recording", "DIV", "culture", outcome]].rename(
        columns={outcome: "value"}
    )
    draw_raincloud(
        ax,
        plot,
        x="DIV",
        y="value",
        culture="culture",
        recording="recording",
        order=DIVS,
        colors=AGE_COLORS,
        point_size=18,
    )
    if one_sample is not None:
        draw_one_sample_symbols(
            ax,
            one_sample,
            order=DIVS,
            y_fraction=1.025,
            label_fontsize=11.5,
        )
        adjacent_y = 1.145
    else:
        adjacent_y = 1.055
    draw_adjacent_stat_band(
        ax,
        annotations,
        order=DIVS,
        y_fraction=adjacent_y,
        label_fontsize=11.5,
    )
    _style_rain_axis(ax, ylabel, title, log=log)


def figure_1() -> list[dict]:
    direct = _with_culture(_read("direct_higher_order_metrics_all_bins.csv"))
    direct = direct.loc[direct["bin_ms"].eq(25)].copy()
    stats = _read("v2_hierarchical_adjacent_contrasts.csv", v2=True)
    specs = [
        ("b", "triplet_support_mass", "Triplet support mass", "Triplet support mass", True),
        (
            "c",
            "triplet_repeated5_density",
            "Repeated-triplet density",
            "Repeated triplets",
            False,
        ),
        (
            "d",
            "h1_total_persistence_capped",
            "Capped H1 persistence",
            "Direct H1 persistence",
            False,
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.35))
    ax = axes[0, 0]
    ax.set_axis_off()
    _panel_label(ax, "a")
    _box(ax, (0.03, 0.62), 0.27, 0.20, "143 longitudinal\nMEA recordings", "#DCE8F5", 7.2)
    _box(ax, (0.38, 0.62), 0.24, 0.20, "Binary population\nactivity", "#DDF0EA", 7.2)
    _box(ax, (0.70, 0.62), 0.27, 0.20, "Triplets and\nsimplicial H1", "#E8DCEC", 7.2)
    _arrow(ax, (0.29, 0.72), (0.38, 0.72))
    _arrow(ax, (0.62, 0.72), (0.71, 0.72))
    _box(ax, (0.13, 0.22), 0.74, 0.18, "Which developmental interval shows\nthe largest higher-order reorganization?", "#F3E4D2", 7.1)
    _arrow(ax, (0.50, 0.62), (0.50, 0.40))
    ax.text(
        0.04,
        0.05,
        "P1-P3: three independent cultures;\n18 longitudinal wells; DIV35 n=17.",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#444444",
    )

    audits: list[dict] = []
    for ax, (panel, outcome, ylabel, title, log) in zip(
        [axes[0, 1], axes[1, 0], axes[1, 1]], specs
    ):
        annotations = adjacent_annotations(
            stats, family="direct_25ms", outcome=outcome
        )
        _plot_developmental_raincloud(
            ax,
            direct,
            outcome=outcome,
            annotations=annotations,
            ylabel=ylabel,
            title=title,
            log=log,
        )
        _panel_label(ax, panel)
        audits.extend(
            _adjacent_audit(
                "Figure_1", panel, annotations, "v2_hierarchical_adjacent_contrasts.csv"
            )
        )

    fig.suptitle(
        "Developmental emergence of higher-order population organization",
        fontsize=12.5,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955], h_pad=3.0, w_pad=2.0)
    export_publication_figure(fig, MAIN_DIR / "Figure_1_developmental_emergence")
    _source_csv(1, [("direct_25ms", direct)])
    return audits


def _burst_differences(metric: str) -> pd.DataFrame:
    burst = _with_culture(_read("burst_stratified_higher_order_metrics_25ms.csv"))
    pivot = burst.pivot_table(
        index=["recording", "DIV", "culture", "subject_id"],
        columns="stratum",
        values=metric,
        aggfunc="first",
    ).reset_index()
    pivot["value"] = pivot["inside_burst"] - pivot["outside_burst"]
    pivot["metric"] = metric
    return pivot


def _burst_annotations(stats: pd.DataFrame, metric: str) -> pd.DataFrame:
    subset = stats.loc[stats["metric"].eq(metric)].copy()
    subset["DIV"] = subset["DIV"].astype(int)
    subset = subset.sort_values("DIV")
    if subset["DIV"].tolist() != list(DIVS):
        raise ValueError(f"Incomplete burst statistics for {metric}")
    subset["symbol"] = subset["adjusted_p"].map(p_to_symbol)
    subset["comparison"] = subset["DIV"].map(
        lambda div: f"DIV{div} inside-outside"
    )
    return subset


def figure_2() -> list[dict]:
    burst_stats = _read("burst_inside_outside_wilcoxon.csv")
    activity = _read("v2_activity_adjusted_adjacent_contrasts.csv", v2=True)
    fig = plt.figure(figsize=(7.25, 6.25))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.58, wspace=0.38)
    ax = fig.add_subplot(grid[0, 0])
    ax.set_axis_off()
    _panel_label(ax, "a")
    _box(ax, (0.03, 0.62), 0.27, 0.20, "All 25-ms bins", "#DCE8F5", 7.0)
    _box(ax, (0.39, 0.62), 0.27, 0.20, "Inside network\nbursts", "#DDF0EA", 7.0)
    _box(ax, (0.71, 0.62), 0.27, 0.20, "Outside network\nbursts", "#F2D8D5", 7.0)
    _arrow(ax, (0.30, 0.72), (0.40, 0.72))
    _arrow(ax, (0.30, 0.68), (0.72, 0.68))
    _box(
        ax,
        (0.18, 0.18),
        0.64,
        0.20,
        "Is higher-order structure concentrated during bursts,\nand does development persist after activity adjustment?",
        "#F3E4D2",
        6.8,
    )

    audits: list[dict] = []
    source_frames: list[tuple[str, pd.DataFrame]] = []
    metrics = [
        ("b", "triplet_support_mass", "Inside - outside support mass", True),
        ("c", "h1_total_persistence_capped", "Inside - outside H1 persistence", False),
    ]
    for grid_slot, (panel, metric, ylabel, log) in zip([grid[0, 1], grid[1, 0]], metrics):
        ax = fig.add_subplot(grid_slot)
        diff = _burst_differences(metric)
        annotations = _burst_annotations(burst_stats, metric)
        draw_raincloud(
            ax,
            diff,
            x="DIV",
            y="value",
            culture="culture",
            recording="recording",
            order=DIVS,
            colors=AGE_COLORS,
            point_size=18,
        )
        draw_one_sample_symbols(
            ax,
            annotations,
            order=DIVS,
            y_fraction=1.055,
            label_fontsize=11.5,
        )
        ax.axhline(0, ls="--", lw=0.9, color="#767676", zorder=0)
        _style_rain_axis(ax, ylabel, metric.replace("_", " "), log=False)
        _panel_label(ax, panel)
        audits.extend(
            _one_sample_audit(
                "Figure_2",
                panel,
                annotations,
                "burst_inside_outside_wilcoxon.csv",
                "paired_inside_outside",
            )
        )
        source_frames.append((metric, diff))

    ax = fig.add_subplot(grid[1, 1])
    tier2 = activity.loc[
        activity["adjustment_tier"].eq("tier2_activity_burst")
        & activity["DIV_first"].eq(14)
        & activity["DIV_second"].eq(21)
    ].copy()
    desired = [
        "log1p_absolute_residual_mass",
        "absolute_residual_z",
        "h1_persistence_deviation_z",
        "h1_infinite_deviation_z",
    ]
    tier2 = tier2.set_index("outcome").loc[desired].reset_index()
    labels = ["Residual mass", "Residual z", "H1 persistence z", "H1 infinite z"]
    y = np.arange(len(tier2))[::-1]
    colors = ["#D9544D" if p < 0.05 else "#A8A8A8" for p in tier2["adjusted_p"]]
    ax.axvline(0, ls="--", lw=0.9, color="#767676")
    ax.scatter(tier2["standardized_effect"], y, s=48, c=colors, zorder=3)
    for yi, row in zip(y, tier2.itertuples(index=False)):
        ax.text(
            row.standardized_effect,
            yi + 0.22,
            p_to_symbol(row.adjusted_p),
            ha="center",
            va="bottom",
            fontsize=11.5,
            fontweight="bold",
        )
        audits.append(
            {
                "figure": "Figure_2",
                "panel": "d",
                "comparison": f"{row.outcome} DIV14-DIV21 adjusted",
                "adjusted_p": float(row.adjusted_p),
                "symbol": p_to_symbol(row.adjusted_p),
                "source_file": "v2_activity_adjusted_adjacent_contrasts.csv",
                "test_family": "activity_adjusted_early_contrast",
            }
        )
    ax.set_yticks(y, labels)
    ax.set_xlim(-1.3, 5.65)
    ax.set_xlabel("Standardized DIV14-DIV21 effect", fontsize=9.5)
    ax.set_title("After activity and burst adjustment", fontsize=10, fontweight="bold")
    _clean_axis(ax)
    _panel_label(ax, "d")
    source_frames.append(("tier2_activity_adjusted", tier2))

    fig.suptitle(
        "Burst context and activity-adjusted higher-order development",
        fontsize=12.5,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.11, right=0.97)
    export_publication_figure(fig, MAIN_DIR / "Figure_2_burst_activity_context")
    _source_csv(2, source_frames)
    return audits


def _null_residual_frame(model: str) -> pd.DataFrame:
    filename = (
        "v2_instantaneous_null100_triplet_residuals.csv"
        if model == "pairwise_conditioned_k_100"
        else "v2_kinetic_triplet_residuals.csv"
    )
    return _with_culture(_read(filename, v2=True))


def figure_3() -> list[dict]:
    extended = _read("v2_extended_null_hierarchical_contrasts.csv", v2=True)
    one_sample = _read("v2_null_deviation_one_sample_statistics.csv", v2=True)
    instant_qc = _read("v2_instantaneous_null100_qc.csv", v2=True)
    kinetic_qc = _read("v2_kinetic_null_qc.csv", v2=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.4))
    ax = axes[0, 0]
    ax.set_axis_off()
    _panel_label(ax, "a")
    _box(ax, (0.04, 0.63), 0.40, 0.20, "Instantaneous pairwise\n+ exact K(t)", "#DCE8F5")
    _box(ax, (0.56, 0.63), 0.40, 0.20, "Lag-1 kinetic\n+ exact K(t)", "#E8DCEC")
    _box(ax, (0.29, 0.20), 0.42, 0.19, "Triplet residual z\nrelative to each null", "#DDF0EA")
    _arrow(ax, (0.24, 0.63), (0.42, 0.39), MODEL_COLORS["pairwise_conditioned_k_100"])
    _arrow(ax, (0.76, 0.63), (0.58, 0.39), MODEL_COLORS["kinetic_conditioned_k"])
    ax.text(
        0.05,
        0.05,
        "Both nulls preserve the observed population-count trajectory exactly.",
        transform=ax.transAxes,
        fontsize=8,
        color="#444444",
    )

    audits: list[dict] = []
    source_frames: list[tuple[str, pd.DataFrame]] = []
    for ax, panel, model, title in [
        (axes[0, 1], "b", "pairwise_conditioned_k_100", "Instantaneous-null residual"),
        (axes[1, 0], "c", "kinetic_conditioned_k", "Kinetic-null residual"),
    ]:
        frame = _null_residual_frame(model)
        adjacent = adjacent_annotations(
            extended, model=model, outcome="absolute_residual_z"
        )
        zero = one_sample_annotations(
            one_sample, model=model, source_metric="absolute_residual_z"
        )
        _plot_developmental_raincloud(
            ax,
            frame,
            outcome="absolute_residual_z",
            annotations=adjacent,
            one_sample=zero,
            ylabel="Absolute triplet residual z",
            title=title,
        )
        ax.text(
            0.98,
            0.98,
            title,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.4,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5),
            zorder=10,
        )
        ax.axhline(0, ls="--", lw=0.9, color="#767676", zorder=0)
        _panel_label(ax, panel)
        audits.extend(
            _one_sample_audit(
                "Figure_3", panel, zero, "v2_null_deviation_one_sample_statistics.csv"
            )
        )
        audits.extend(
            _adjacent_audit(
                "Figure_3",
                panel,
                adjacent,
                "v2_extended_null_hierarchical_contrasts.csv",
            )
        )
        source_frames.append((model, frame))

    ax = axes[1, 1]
    qc_long = pd.DataFrame(
        {
            "label": ["Firing\ninstant.", "Pair\ninstant.", "Firing\nkinetic", "Lag-1\nkinetic"],
            "median": [
                instant_qc["firing_probability_rmse"].median(),
                instant_qc["simultaneous_pair_moment_rmse"].median(),
                kinetic_qc["firing_probability_rmse"].median(),
                kinetic_qc["lag1_moment_rmse"].median(),
            ],
            "maximum": [
                instant_qc["firing_probability_rmse"].max(),
                instant_qc["simultaneous_pair_moment_rmse"].max(),
                kinetic_qc["firing_probability_rmse"].max(),
                kinetic_qc["lag1_moment_rmse"].max(),
            ],
        }
    )
    x = np.arange(len(qc_long))
    ax.bar(x, qc_long["maximum"], color=["#B4C0E4", "#7884B4", "#E4CCD8", "#9A4D8E"], alpha=0.8)
    ax.scatter(x, qc_long["median"], color="#151515", s=32, zorder=3, label="Median")
    ax.axhline(0.02, color="#A61B1B", ls="--", lw=1.1, label="QC threshold")
    ax.set_xticks(x, qc_long["label"])
    ax.set_ylabel("Moment RMSE", fontsize=9.5)
    ax.set_title("Null-model quality control", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    _clean_axis(ax)
    _panel_label(ax, "d")
    source_frames.append(("null_qc_summary", qc_long))

    fig.suptitle(
        "Triplet specificity beyond complementary lower-order constraints",
        fontsize=12.5,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.2, w_pad=2.2)
    export_publication_figure(fig, MAIN_DIR / "Figure_3_triplet_specificity_nulls")
    _source_csv(3, source_frames)
    return audits


def _null_comparison_frame(model: str, metric: str) -> pd.DataFrame:
    filename = (
        "v2_instantaneous_null100_comparisons.csv"
        if model == "pairwise_conditioned_k_100"
        else "v2_kinetic_null_comparisons.csv"
    )
    frame = _with_culture(_read(filename, v2=True))
    return frame.loc[frame["metric"].eq(metric)].copy()


def figure_4() -> list[dict]:
    extended = _read("v2_extended_null_hierarchical_contrasts.csv", v2=True)
    one_sample = _read("v2_null_deviation_one_sample_statistics.csv", v2=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.7))
    specs = [
        ("a", "pairwise_conditioned_k_100", "h1_total_persistence_capped", "Instantaneous H1 persistence z"),
        ("b", "kinetic_conditioned_k", "h1_total_persistence_capped", "Kinetic H1 persistence z"),
        ("c", "pairwise_conditioned_k_100", "h1_infinite_count", "Instantaneous H1 infinite-count z"),
        ("d", "kinetic_conditioned_k", "h1_infinite_count", "Kinetic H1 infinite-count z"),
    ]
    audits: list[dict] = []
    source_frames: list[tuple[str, pd.DataFrame]] = []
    for ax, (panel, model, metric, title) in zip(axes.ravel(), specs):
        frame = _null_comparison_frame(model, metric)
        adjacent = adjacent_annotations(extended, model=model, outcome=metric)
        zero = one_sample_annotations(one_sample, model=model, source_metric=metric)
        _plot_developmental_raincloud(
            ax,
            frame,
            outcome="z",
            annotations=adjacent,
            one_sample=zero,
            ylabel="Observed-null deviation z",
            title=title,
        )
        ax.text(
            0.98,
            0.98,
            title,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.2,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5),
            zorder=10,
        )
        ax.axhline(0, ls="--", lw=0.9, color="#767676", zorder=0)
        _panel_label(ax, panel)
        audits.extend(
            _one_sample_audit(
                "Figure_4", panel, zero, "v2_null_deviation_one_sample_statistics.csv"
            )
        )
        audits.extend(
            _adjacent_audit(
                "Figure_4",
                panel,
                adjacent,
                "v2_extended_null_hierarchical_contrasts.csv",
            )
        )
        source_frames.append((f"{model}_{metric}", frame))
    fig.suptitle(
        "Triplet-specific deviations converge, whereas H1 topology is null-model dependent",
        fontsize=12.2,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.5, w_pad=2.2)
    export_publication_figure(fig, MAIN_DIR / "Figure_4_null_dependent_topology")
    _source_csv(4, source_frames)
    return audits


def _one_row(frame: pd.DataFrame, **criteria) -> pd.Series:
    subset = frame.copy()
    for column, value in criteria.items():
        subset = subset.loc[subset[column].eq(value)]
    if len(subset) != 1:
        raise ValueError(f"Expected one row for {criteria}, found {len(subset)}")
    return subset.iloc[0]


def figure_5() -> list[dict]:
    hierarchical = _read("v2_hierarchical_adjacent_contrasts.csv", v2=True)
    culture = _read("v2_culture_specific_adjacent_changes.csv", v2=True)
    incremental = _read("v2_incremental_metrics.csv", v2=True)
    bootstrap = _read("v2_incremental_bootstrap.csv", v2=True)
    predictions = _read("v2_incremental_predictions.csv", v2=True)
    fig = plt.figure(figsize=(7.25, 6.65))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.08], hspace=0.48, wspace=0.50)
    audits: list[dict] = []
    source_frames: list[tuple[str, pd.DataFrame]] = []

    ax = fig.add_subplot(grid[0, 0])
    changes = culture.loc[
        culture["family"].eq("direct_25ms")
        & culture["outcome"].eq("triplet_support_mass")
        & culture["DIV_first"].eq(14)
        & culture["DIV_second"].eq(21)
    ].sort_values("culture")
    x = np.arange(len(changes))
    ax.axhline(0, ls="--", lw=0.9, color="#767676")
    ax.vlines(x, 0, changes["median_change"], color=[CULTURE_COLORS[c] for c in changes["culture"]], lw=3)
    ax.scatter(x, changes["median_change"], s=56, color=[CULTURE_COLORS[c] for c in changes["culture"]], zorder=3)
    ax.set_xticks(x, changes["culture"])
    ax.set_ylabel("Median DIV14-DIV21 change", fontsize=9.2)
    ax.set_title("Independent-culture direction", fontsize=9.8, fontweight="bold")
    _clean_axis(ax)
    _panel_label(ax, "a")
    source_frames.append(("culture_changes", changes))

    ax = fig.add_subplot(grid[0, 1])
    specs = [
        ("direct_25ms", "triplet_support_mass", "Triplet support"),
        ("direct_25ms", "h1_total_persistence_capped", "Direct H1"),
        ("conditioned_triplet_residual", "log1p_absolute_residual_mass", "Residual mass"),
        ("conditioned_triplet_residual", "absolute_residual_z", "Residual z"),
        ("conditioned_topology_deviation", "z_h1_total_persistence_capped", "H1 deviation z"),
    ]
    rows = []
    for family, outcome, label in specs:
        row = _one_row(
            hierarchical,
            family=family,
            outcome=outcome,
            DIV_first=14,
            DIV_second=21,
        )
        scale = row["estimate"] / row["standardized_effect"] if row["standardized_effect"] != 0 else 1.0
        rows.append(
            {
                "label": label,
                "standardized_effect": row["standardized_effect"],
                "ci_low": row["ci95_low"] / scale,
                "ci_high": row["ci95_high"] / scale,
                "adjusted_p": row["adjusted_p"],
            }
        )
    effects = pd.DataFrame(rows)
    y = np.arange(len(effects))[::-1]
    ax.axvline(0, ls="--", lw=0.9, color="#767676")
    for yi, row in zip(y, effects.itertuples(index=False)):
        ax.plot([row.ci_low, row.ci_high], [yi, yi], color="#D9544D", lw=1.5)
        ax.scatter(row.standardized_effect, yi, color="#D9544D", s=35, zorder=3)
        ax.text(
            row.ci_high,
            yi,
            " " + p_to_symbol(row.adjusted_p),
            ha="left",
            va="center",
            fontsize=10.5,
            fontweight="bold",
        )
        audits.append(
            {
                "figure": "Figure_5",
                "panel": "b",
                "comparison": f"{row.label} DIV14-DIV21",
                "adjusted_p": float(row.adjusted_p),
                "symbol": p_to_symbol(row.adjusted_p),
                "source_file": "v2_hierarchical_adjacent_contrasts.csv",
                "test_family": "early_contrast",
            }
        )
    ax.set_yticks(y, effects["label"])
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("Standardized effect (symlog)", fontsize=9.2)
    ax.set_title("Hierarchical early effects", fontsize=9.8, fontweight="bold")
    _clean_axis(ax)
    _panel_label(ax, "b")
    source_frames.append(("hierarchical_early_effects", effects))

    ax = fig.add_subplot(grid[0, 2])
    mae_rows = incremental.loc[
        incremental["validation"].eq("leave_one_culture_out")
        & incremental["model_set"].isin(["prior_27_baseline", "prior_27_plus_higher_order"])
    ].copy()
    ranges = ["DIV14_63", "DIV21_63"]
    width = 0.34
    for offset, model, color, label in [
        (-width / 2, "prior_27_baseline", "#7884B4", "Prior 27"),
        (width / 2, "prior_27_plus_higher_order", "#D9544D", "+ higher order"),
    ]:
        vals = [
            _one_row(mae_rows, analysis_range=rng, model_set=model)["mae"]
            for rng in ranges
        ]
        ax.bar(np.arange(2) + offset, vals, width=width, color=color, label=label)
    ax.set_xticks(np.arange(2), ["DIV14-63", "DIV21-63"])
    ax.set_ylabel("Held-out MAE (days)", fontsize=9.2)
    ax.set_title("Leave-one-culture-out", fontsize=9.8, fontweight="bold")
    ax.legend(fontsize=7.0, frameon=False, loc="lower center", ncol=1)
    _clean_axis(ax)
    _panel_label(ax, "c")
    source_frames.append(("incremental_metrics", mae_rows))

    ax = fig.add_subplot(grid[1, 0])
    boot = bootstrap.loc[
        bootstrap["validation"].eq("leave_one_culture_out")
        & bootstrap["baseline_name"].eq("prior_27_baseline")
    ].set_index("analysis_range").loc[ranges].reset_index()
    y = np.arange(2)[::-1]
    ax.axvline(0, ls="--", lw=0.9, color="#767676")
    for yi, row in zip(y, boot.itertuples(index=False)):
        ax.plot([row.ci95_low, row.ci95_high], [yi, yi], color="#272727", lw=1.5)
        color = "#2E9E44" if row.delta_mae_baseline_minus_augmented > 0 else "#E53935"
        ax.scatter(row.delta_mae_baseline_minus_augmented, yi, color=color, s=52, zorder=3)
    ax.set_yticks(y, ["DIV14-63", "DIV21-63"])
    ax.set_xlabel("MAE improvement (days)", fontsize=9.2)
    ax.set_title("Cluster-bootstrap delta", fontsize=9.8, fontweight="bold")
    _clean_axis(ax)
    _panel_label(ax, "d")
    source_frames.append(("incremental_bootstrap", boot))

    ax = fig.add_subplot(grid[1, 1:])
    pred = predictions.loc[
        predictions["analysis_range"].eq("DIV14_63")
        & predictions["validation"].eq("leave_one_culture_out")
        & predictions["model_set"].eq("prior_27_plus_higher_order")
    ].copy()
    for culture_name, group in pred.groupby("culture"):
        ax.scatter(
            group["observed"],
            group["prediction"],
            s=24,
            marker={"P1": "o", "P2": "s", "P3": "^"}[culture_name],
            color=CULTURE_COLORS[culture_name],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.35,
            label=culture_name,
        )
    ax.plot([14, 63], [14, 63], ls="--", lw=1.0, color="#767676")
    ax.set_xlabel("Observed DIV", fontsize=9.2)
    ax.set_ylabel("Held-out predicted DIV", fontsize=9.2)
    ax.set_title("Augmented full-range predictions", fontsize=9.8, fontweight="bold")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    _clean_axis(ax)
    _panel_label(ax, "e")
    source_frames.append(("heldout_predictions", pred))

    fig.suptitle(
        "Cross-culture reproducibility and incremental developmental information",
        fontsize=12.2,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.11, right=0.97)
    export_publication_figure(fig, MAIN_DIR / "Figure_5_reproducibility_incremental_value")
    _source_csv(5, source_frames)
    return audits


def build_main_figures() -> None:
    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    audits: list[dict] = []
    for builder in [figure_1, figure_2, figure_3, figure_4, figure_5]:
        audits.extend(builder())
    audit = pd.DataFrame(audits)
    validate_annotation_audit(audit)
    audit.to_csv(SOURCE_DIR / "main_annotation_audit.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-only", action="store_true")
    args = parser.parse_args()
    apply_publication_style(font_size=9.0)
    plt.rcParams.update(
        {
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "figure.dpi": 150,
        }
    )
    build_main_figures()
    if not args.main_only:
        build_supplementary_figures()


def supplementary_1() -> list[dict]:
    nodes = pd.DataFrame(
        [
            ("P1-P3 cultures", "18 longitudinal wells", "biological hierarchy"),
            ("64-electrode MEA", "25-ms binary activity", "encoding"),
            ("Direct triplets", "Burst stratification", "observed structure"),
            ("Instantaneous null", "Kinetic null", "lower-order constraints"),
            ("Hierarchical models", "LOCO prediction", "validation"),
            ("Node/bin/duration", "Claim gates", "robustness"),
        ],
        columns=["left", "right", "stage"],
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.5))
    colors = ["#DCE8F5", "#DDF0EA", "#F3E4D2", "#E8DCEC", "#F2D8D5", "#DDF3DE"]
    for index, (ax, row, color) in enumerate(zip(axes.ravel(), nodes.itertuples(index=False), colors)):
        ax.set_axis_off()
        _panel_label(ax, chr(ord("a") + index))
        _box(ax, (0.08, 0.57), 0.84, 0.22, row.left, color, 7.5)
        _box(ax, (0.08, 0.20), 0.84, 0.22, row.right, color, 7.5)
        _arrow(ax, (0.50, 0.57), (0.50, 0.42))
        ax.text(0.50, 0.92, row.stage, transform=ax.transAxes, ha="center", fontsize=8.5, fontweight="bold")
    fig.suptitle("Complete longitudinal analysis and validation hierarchy", fontsize=12.2, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=1.5, w_pad=1.5)
    export_publication_figure(fig, SUPP_DIR / "Figure_S1_workflow_hierarchy")
    _source_csv_s(1, [("workflow_nodes", nodes)])
    return []


def supplementary_2() -> list[dict]:
    rank = _read("v2_bin_width_rank_agreement.csv", v2=True)
    change = _read("v2_bin_width_div14_21_changes.csv", v2=True)
    selected = ["mean_population_count", "triplet_repeated5_density", "triplet_support_mass", "h1_total_persistence_capped"]
    rank = rank.loc[rank["metric"].isin(selected)].copy()
    change = change.loc[change["metric"].isin(selected)].copy()
    label_map = {
        "mean_population_count": "Mean K",
        "triplet_repeated5_density": "Repeated triplets",
        "triplet_support_mass": "Triplet support",
        "h1_total_persistence_capped": "H1 persistence",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.55))
    for ax, frame, value, title, cmap, limits in [
        (axes[0], rank, "spearman_rho", "Rank agreement with 25 ms", "Blues", (0, 1)),
        (axes[1], change, "fraction_positive", "Positive DIV14-DIV21 changes", "YlGnBu", (0, 1)),
    ]:
        column = "comparison_bin_ms" if "comparison_bin_ms" in frame else "bin_ms"
        matrix = frame.pivot(index="metric", columns=column, values=value).reindex(selected)
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=limits[0], vmax=limits[1])
        ax.set_xticks(np.arange(len(matrix.columns)), [f"{int(v)} ms" for v in matrix.columns])
        ax.set_yticks(np.arange(len(matrix.index)), [label_map[v] for v in matrix.index])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if matrix.iloc[i, j] > 0.65 else "black")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.tick_params(length=0)
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    fig.suptitle("Temporal-bin sensitivity of higher-order measurements", fontsize=12.2, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92], w_pad=2.5)
    export_publication_figure(fig, SUPP_DIR / "Figure_S2_bin_width_sensitivity")
    _source_csv_s(2, [("rank_agreement", rank), ("early_change_direction", change)])
    return []


def supplementary_3() -> list[dict]:
    direct = _with_culture(_read("direct_higher_order_metrics_all_bins.csv"))
    direct = direct.loc[direct["bin_ms"].eq(25)].copy()
    stats = _read("v2_hierarchical_adjacent_contrasts.csv", v2=True)
    specs = [
        ("a", "mean_population_count", "Mean population count", False),
        ("b", "fraction_bins_k_ge_3", "Fraction of bins with K >= 3", False),
        ("c", "pair_support_mass", "Pair support mass", True),
        ("d", "h1_infinite_count", "H1 infinite count", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.3))
    audits: list[dict] = []
    for ax, (panel, outcome, label, log) in zip(axes.ravel(), specs):
        annotations = adjacent_annotations(stats, family="direct_25ms", outcome=outcome)
        _plot_developmental_raincloud(ax, direct, outcome=outcome, annotations=annotations,
                                      ylabel=label, title=label, log=log)
        _panel_label(ax, panel)
        audits.extend(_adjacent_audit("Figure_S3", panel, annotations,
                                      "v2_hierarchical_adjacent_contrasts.csv"))
    fig.suptitle("Additional direct higher-order developmental trajectories", fontsize=12.2, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.2, w_pad=2.2)
    export_publication_figure(fig, SUPP_DIR / "Figure_S3_complete_direct_metrics")
    _source_csv_s(3, [("direct_25ms", direct)])
    return audits


def supplementary_4() -> list[dict]:
    burst_stats = _read("burst_inside_outside_wilcoxon.csv")
    specs = [
        ("a", "fraction_bins_k_ge_3", "Inside - outside fraction K >= 3"),
        ("b", "triplet_repeated5_density", "Inside - outside repeated-triplet density"),
        ("c", "triplet_support_mass", "Inside - outside triplet support mass"),
        ("d", "h1_total_persistence_capped", "Inside - outside H1 persistence"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.3))
    audits: list[dict] = []
    sources: list[tuple[str, pd.DataFrame]] = []
    for ax, (panel, metric, ylabel) in zip(axes.ravel(), specs):
        diff = _burst_differences(metric)
        annotations = _burst_annotations(burst_stats, metric)
        draw_raincloud(ax, diff, x="DIV", y="value", culture="culture", recording="recording",
                       order=DIVS, colors=AGE_COLORS, point_size=18)
        draw_one_sample_symbols(ax, annotations, order=DIVS, y_fraction=1.055, label_fontsize=11.5)
        ax.axhline(0, ls="--", lw=0.9, color="#767676", zorder=0)
        _style_rain_axis(ax, ylabel, metric)
        _panel_label(ax, panel)
        audits.extend(_one_sample_audit("Figure_S4", panel, annotations,
                                        "burst_inside_outside_wilcoxon.csv", "paired_inside_outside"))
        sources.append((metric, diff))
    fig.suptitle("Complete network-burst stratification of higher-order metrics", fontsize=12.2, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.1, w_pad=2.2)
    export_publication_figure(fig, SUPP_DIR / "Figure_S4_complete_burst_metrics")
    _source_csv_s(4, sources)
    return audits


def supplementary_5() -> list[dict]:
    instant = _read("v2_instantaneous_null100_qc.csv", v2=True)
    kinetic = _read("v2_kinetic_null_qc.csv", v2=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.7))
    ax = axes[0, 0]
    data = [instant["firing_probability_rmse"], instant["simultaneous_pair_moment_rmse"],
            kinetic["firing_probability_rmse"], kinetic["lag1_moment_rmse"]]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], ["#B4C0E4", "#7884B4", "#E4CCD8", "#9A4D8E"]):
        patch.set_facecolor(color)
    ax.axhline(0.02, ls="--", color="#A61B1B", lw=1.0)
    ax.set_xticks(range(1, 5), ["Firing\ninst.", "Pair\ninst.", "Firing\nkin.", "Lag-1\nkin."])
    ax.set_ylabel("Moment RMSE")
    ax.set_title("Moment matching", fontweight="bold")
    _clean_axis(ax); _panel_label(ax, "a")

    ax = axes[0, 1]
    valid = pd.concat([
        instant[["recording", "DIV", "n_valid_nulls"]].assign(model="instantaneous"),
        kinetic[["recording", "DIV", "n_valid_nulls"]].assign(model="kinetic"),
    ])
    for model, color in [("instantaneous", "#3775BA"), ("kinetic", "#9A4D8E")]:
        subset = valid.loc[valid["model"].eq(model)]
        ax.scatter(subset["DIV"], subset["n_valid_nulls"], s=10, alpha=0.45, color=color, label=model)
    ax.axhline(90, ls="--", color="#A61B1B", lw=1.0)
    ax.set_ylabel("Valid null realizations")
    ax.set_xlabel("DIV")
    ax.set_title("Simulation completeness", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    _clean_axis(ax); _panel_label(ax, "b")

    ax = axes[1, 0]
    exact = pd.concat([
        instant[["exact_k_mismatch_count"]].assign(model="instantaneous"),
        kinetic[["exact_k_mismatch_count"]].assign(model="kinetic"),
    ])
    counts = exact.groupby("model")["exact_k_mismatch_count"].max().reindex(["instantaneous", "kinetic"])
    ax.bar([0, 1], counts, color=["#3775BA", "#9A4D8E"])
    ax.set_xticks([0, 1], ["Instantaneous", "Kinetic"])
    ax.set_ylabel("Maximum K(t) mismatch")
    ax.set_title("Exact population-count constraint", fontweight="bold")
    _clean_axis(ax); _panel_label(ax, "c")

    ax = axes[1, 1]
    improvement = kinetic["heldout_log_loss_improvement"]
    ax.hist(improvement, bins=16, color="#9A4D8E", alpha=0.75, edgecolor="white")
    ax.axvline(0, ls="--", color="#767676", lw=1.0)
    ax.set_xlabel("Held-out log-loss improvement")
    ax.set_ylabel("Recordings")
    ax.set_title("Kinetic model vs rate-only", fontweight="bold")
    _clean_axis(ax); _panel_label(ax, "d")
    fig.suptitle("Null-model quality control and predictive validation", fontsize=12.2, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=2.2, w_pad=2.0)
    export_publication_figure(fig, SUPP_DIR / "Figure_S5_null_model_qc")
    _source_csv_s(5, [("instantaneous_qc", instant), ("kinetic_qc", kinetic)])
    return []


def supplementary_6() -> list[dict]:
    hierarchical = _read("v2_hierarchical_adjacent_contrasts.csv", v2=True)
    culture = _read("v2_culture_specific_adjacent_changes.csv", v2=True)
    specs = [
        ("direct_25ms", "triplet_support_mass", "Triplet support"),
        ("direct_25ms", "triplet_repeated5_density", "Repeated triplets"),
        ("direct_25ms", "h1_total_persistence_capped", "Direct H1"),
        ("conditioned_triplet_residual", "log1p_absolute_residual_mass", "Residual mass"),
        ("conditioned_triplet_residual", "absolute_residual_z", "Residual z"),
        ("conditioned_topology_deviation", "z_h1_total_persistence_capped", "H1 persistence z"),
        ("conditioned_topology_deviation", "z_h1_infinite_count", "H1 infinite z"),
    ]
    rows = []
    culture_rows = []
    audits: list[dict] = []
    for family, outcome, label in specs:
        row = _one_row(hierarchical, family=family, outcome=outcome, DIV_first=14, DIV_second=21)
        rows.append({"family": family, "outcome": outcome, "label": label,
                     "standardized_effect": row.standardized_effect, "adjusted_p": row.adjusted_p})
        audits.append({"figure": "Figure_S6", "panel": "a", "comparison": f"{label} DIV14-DIV21",
                       "adjusted_p": float(row.adjusted_p), "symbol": p_to_symbol(row.adjusted_p),
                       "source_file": "v2_hierarchical_adjacent_contrasts.csv", "test_family": "early_contrast"})
        subset = culture.loc[(culture["family"].eq(family)) & (culture["outcome"].eq(outcome))
                             & culture["DIV_first"].eq(14) & culture["DIV_second"].eq(21)]
        for item in subset.itertuples(index=False):
            culture_rows.append({"label": label, "culture": item.culture, "median_change": item.median_change})
    effects = pd.DataFrame(rows)
    directions = pd.DataFrame(culture_rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 4.5), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    y = np.arange(len(effects))[::-1]
    ax.axvline(0, ls="--", color="#767676", lw=1)
    for yi, row in zip(y, effects.itertuples(index=False)):
        color = "#D9544D" if row.adjusted_p < 0.05 else "#A8A8A8"
        ax.scatter(row.standardized_effect, yi, color=color, s=42)
        ax.text(row.standardized_effect, yi + 0.22, p_to_symbol(row.adjusted_p), ha="center",
                fontsize=10.5, fontweight="bold")
    ax.set_yticks(y, effects["label"])
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Standardized DIV14-DIV21 effect (symlog)")
    ax.set_title("Hierarchical effects", fontweight="bold")
    _clean_axis(ax); _panel_label(ax, "a")
    ax = axes[1]
    matrix = directions.pivot(index="label", columns="culture", values="median_change").reindex(effects["label"])
    normalized = matrix.div(matrix.abs().max(axis=1).replace(0, 1), axis=0)
    image = ax.imshow(normalized, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2g}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(normalized.iloc[i, j]) > 0.65 else "black")
    ax.set_title("Culture-specific median change", fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03, label="Row-normalized direction")
    _panel_label(ax, "b")
    fig.suptitle("Hierarchical and culture-specific validation of the early transition", fontsize=12.0, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92], w_pad=2.3)
    export_publication_figure(fig, SUPP_DIR / "Figure_S6_hierarchical_culture_effects")
    _source_csv_s(6, [("hierarchical_effects", effects), ("culture_changes", directions)])
    return audits


def supplementary_7() -> list[dict]:
    effects = _read("v2_node_specification_hierarchical_contrasts.csv", v2=True)
    ranks = _read("v2_node_specification_rank_agreement.csv", v2=True)
    effects = effects.loc[effects["DIV_first"].eq(14) & effects["DIV_second"].eq(21)].copy()
    selected_outcomes = ["triplet_support_mass", "absolute_residual_z", "h1_total_persistence_capped"]
    effects_plot = effects.loc[effects["outcome"].isin(selected_outcomes)].copy()
    audits: list[dict] = []
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 4.5), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    specs = ["dynamic_n16", "fixed_n12", "fixed_n16", "fixed_n20"]
    offsets = np.linspace(-0.18, 0.18, len(specs))
    labels = {"triplet_support_mass": "Triplet support", "absolute_residual_z": "Residual z",
              "h1_total_persistence_capped": "H1 persistence"}
    ybase = {outcome: index for index, outcome in enumerate(selected_outcomes[::-1])}
    for offset, spec, color in zip(offsets, specs, ["#484878", "#7884B4", "#42949E", "#D9544D"]):
        subset = effects_plot.loc[effects_plot["node_specification"].eq(spec)]
        for row in subset.itertuples(index=False):
            yi = ybase[row.outcome] + offset
            ax.scatter(row.standardized_effect, yi, color=color, s=34, label=spec if row.outcome == selected_outcomes[0] else None)
            ax.text(row.standardized_effect, yi + 0.08, p_to_symbol(row.adjusted_p), ha="center", fontsize=8.8, fontweight="bold")
            audits.append({"figure": "Figure_S7", "panel": "a",
                           "comparison": f"{spec} {row.analysis_family} {row.outcome} DIV14-DIV21",
                           "adjusted_p": float(row.adjusted_p), "symbol": p_to_symbol(row.adjusted_p),
                           "source_file": "v2_node_specification_hierarchical_contrasts.csv",
                           "test_family": "node_specification_early_contrast"})
    ax.axvline(0, ls="--", color="#767676", lw=1)
    ax.set_yticks(range(len(selected_outcomes)), [labels[o] for o in selected_outcomes[::-1]])
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Standardized effect (symlog)")
    ax.set_title("Node-specification effects", fontweight="bold")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    _clean_axis(ax); _panel_label(ax, "a")
    ax = axes[1]
    rank_metric = ranks.loc[ranks["metric"].eq("absolute_residual_z")].copy()
    names = specs
    matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for row in rank_metric.itertuples(index=False):
        if row.specification_first in names and row.specification_second in names:
            matrix.loc[row.specification_first, row.specification_second] = row.spearman_rho
            matrix.loc[row.specification_second, row.specification_first] = row.spearman_rho
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(names)), [n.replace("_", "\n") for n in names], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(names)), [n.replace("_", " ") for n in names])
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if matrix.iloc[i, j] < 0.55 else "black")
    ax.set_title("Residual-z rank agreement", fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    _panel_label(ax, "b")
    fig.suptitle("Dynamic and longitudinally fixed node specifications", fontsize=12.0, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92], w_pad=2.3)
    export_publication_figure(fig, SUPP_DIR / "Figure_S7_node_specifications")
    _source_csv_s(7, [("node_effects", effects_plot), ("rank_agreement", rank_metric)])
    return audits


def supplementary_8() -> list[dict]:
    split = _read("v2_split_half_reliability.csv", v2=True)
    duration = _read("v2_duration_convergence.csv", v2=True)
    selected = ["mean_population_count", "triplet_repeated5_density", "triplet_support_mass", "h1_total_persistence_capped"]
    split_plot = split.loc[split["metric"].isin(selected)].copy()
    duration_plot = duration.loc[duration["metric"].isin(selected)].copy()
    labels = {"mean_population_count": "Mean K", "triplet_repeated5_density": "Repeated triplets",
              "triplet_support_mass": "Triplet support", "h1_total_persistence_capped": "H1 persistence"}
    fig, axes = plt.subplots(
        1, 3, figsize=(7.25, 4.1),
        gridspec_kw={"width_ratios": [0.90, 1.12, 1.12]},
    )
    ax = axes[0]
    ordered = split_plot.set_index("metric").reindex(selected).reset_index()
    ax.barh(np.arange(len(ordered)), ordered["split_half_spearman_rho"], color="#42949E")
    ax.set_yticks(np.arange(len(ordered)), [labels[v] for v in ordered["metric"]])
    ax.set_xlim(0, 1.05); ax.set_xlabel("Split-half Spearman rho")
    ax.set_title("150-s halves", fontweight="bold"); _clean_axis(ax); _panel_label(ax, "a")
    for ax, value, title, cmap, limits in [
        (axes[1], "spearman_vs_300s", "Rank convergence", "Blues", (0, 1)),
        (axes[2], "median_absolute_relative_error", "Relative error", "Reds", (0, None)),
    ]:
        matrix = duration_plot.pivot(index="metric", columns="duration_s", values=value).reindex(selected)
        vmax = limits[1] if limits[1] is not None else np.nanmax(matrix.to_numpy())
        ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=limits[0], vmax=vmax)
        ax.set_xticks(np.arange(len(matrix.columns)), [f"{int(v)}" for v in matrix.columns])
        ax.set_yticks(np.arange(len(matrix.index)), [labels[v] for v in matrix.index])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.2)
        ax.set_xlabel("Duration (s)"); ax.set_title(title, fontweight="bold")
    _panel_label(axes[1], "b"); _panel_label(axes[2], "c")
    fig.suptitle("Split-half reliability and recording-duration convergence", fontsize=12.0, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90], w_pad=1.9)
    export_publication_figure(fig, SUPP_DIR / "Figure_S8_temporal_stability")
    _source_csv_s(8, [("split_half", split_plot), ("duration", duration_plot)])
    return []


def supplementary_9() -> list[dict]:
    cv = _read("v2_activity_form_cv.csv", v2=True)
    effects = _read("v2_activity_adjusted_adjacent_contrasts.csv", v2=True)
    table = _read("v2_activity_adjustment_analysis_table.csv", v2=True)
    outcomes = ["log1p_absolute_residual_mass", "absolute_residual_z", "h1_persistence_deviation_z", "h1_infinite_deviation_z"]
    cv_plot = cv.loc[cv["outcome"].isin(outcomes)].copy()
    early = effects.loc[effects["outcome"].isin(outcomes) & effects["DIV_first"].eq(14) & effects["DIV_second"].eq(21)].copy()
    audits: list[dict] = []
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 4.1), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    ax = axes[0]
    forms = ["linear", "log1p", "spline"]
    for index, form in enumerate(forms):
        subset = cv_plot.loc[cv_plot["activity_form"].eq(form)].set_index("outcome").reindex(outcomes)
        ax.plot(np.arange(len(outcomes)), subset["cv_mae"], marker="o", lw=1.3,
                color=["#484878", "#42949E", "#D9544D"][index], label=form)
    ax.set_xticks(np.arange(len(outcomes)), ["Mass", "Residual z", "H1 pers.", "H1 inf."], rotation=25, ha="right")
    ax.set_ylabel("Grouped-CV MAE"); ax.set_title("Activity-form selection", fontweight="bold")
    ax.legend(frameon=False, fontsize=7); _clean_axis(ax); _panel_label(ax, "a")
    ax = axes[1]
    tier_colors = {"tier1_mean_k": "#7884B4", "tier2_activity_burst": "#D9544D"}
    positions = np.arange(len(outcomes))
    for offset, tier in [(-0.10, "tier1_mean_k"), (0.10, "tier2_activity_burst")]:
        subset = early.loc[early["adjustment_tier"].eq(tier)].set_index("outcome").reindex(outcomes)
        ax.scatter(subset["standardized_effect"], positions + offset, color=tier_colors[tier], s=36, label=tier)
        for pos, outcome, row in zip(positions + offset, outcomes, subset.itertuples(index=False)):
            ax.text(row.standardized_effect, pos + 0.07, p_to_symbol(row.adjusted_p), ha="center", fontsize=8.5, fontweight="bold")
            audits.append({"figure": "Figure_S9", "panel": "b", "comparison": f"{tier} {outcome} DIV14-DIV21",
                           "adjusted_p": float(row.adjusted_p), "symbol": p_to_symbol(row.adjusted_p),
                           "source_file": "v2_activity_adjusted_adjacent_contrasts.csv", "test_family": "activity_adjusted_early_contrast"})
    ax.axvline(0, ls="--", color="#767676", lw=1)
    ax.set_yticks(positions, ["Mass", "Residual z", "H1 pers.", "H1 inf."])
    ax.set_xlabel("Standardized effect"); ax.set_title("Tier-1 vs Tier-2", fontweight="bold")
    ax.legend(frameon=False, fontsize=6.5); _clean_axis(ax); _panel_label(ax, "b")
    ax = axes[2]
    scatter = ax.scatter(table["mean_k_top16"], table["log1p_absolute_residual_mass"], c=table["DIV"],
                         cmap="viridis_r", s=18, alpha=0.7)
    ax.set_xlabel("Mean K (top 16)"); ax.set_ylabel("log1p residual mass")
    ax.set_title("Activity dependence", fontweight="bold")
    fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.03, label="DIV")
    _clean_axis(ax); _panel_label(ax, "c")
    fig.suptitle("Activity-adjustment model selection and diagnostics", fontsize=12.0, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90], w_pad=1.8)
    export_publication_figure(fig, SUPP_DIR / "Figure_S9_activity_model_selection")
    _source_csv_s(9, [("activity_form_cv", cv_plot), ("adjusted_effects", early), ("analysis_table", table)])
    return audits


def supplementary_10() -> list[dict]:
    predictions = _read("v2_incremental_predictions.csv", v2=True)
    bootstrap = _read("v2_incremental_bootstrap.csv", v2=True)
    spatial = _read("v2_instantaneous_null100_spatial_participation.csv", v2=True)
    late = predictions.loc[predictions["analysis_range"].eq("DIV21_63")
                           & predictions["validation"].eq("leave_one_culture_out")
                           & predictions["model_set"].eq("prior_27_plus_higher_order")].copy()
    boot = bootstrap.loc[bootstrap["analysis_range"].eq("DIV21_63")
                         & bootstrap["validation"].eq("leave_one_culture_out")
                         & bootstrap["baseline_name"].eq("prior_27_baseline")].copy()
    fig = plt.figure(figsize=(7.25, 4.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 0.7], hspace=0.50, wspace=0.35)
    ax = fig.add_subplot(grid[:, 0])
    for culture_name, group in late.groupby("culture"):
        ax.scatter(group["observed"], group["prediction"], s=24, color=CULTURE_COLORS[culture_name],
                   marker={"P1": "o", "P2": "s", "P3": "^"}[culture_name], alpha=0.7, label=culture_name)
    ax.plot([21, 63], [21, 63], ls="--", color="#767676", lw=1)
    ax.set_xlabel("Observed DIV"); ax.set_ylabel("Held-out predicted DIV")
    ax.set_title("Late-range augmented prediction", fontweight="bold")
    ax.legend(frameon=False, ncol=3, fontsize=7); _clean_axis(ax); _panel_label(ax, "a")
    ax = fig.add_subplot(grid[0, 1])
    row = boot.iloc[0]
    ax.axvline(0, ls="--", color="#767676", lw=1)
    ax.plot([row.ci95_low, row.ci95_high], [0, 0], color="#272727", lw=2)
    ax.scatter(row.delta_mae_baseline_minus_augmented, 0, s=58, color="#E53935")
    ax.set_yticks([0], ["DIV21-63"]); ax.set_xlabel("MAE improvement (days)")
    ax.set_title("Late-stage bootstrap", fontweight="bold"); _clean_axis(ax); _panel_label(ax, "b")
    ax = fig.add_subplot(grid[1, 1])
    matrix = spatial.pivot(index="DIV", columns="electrode_index", values="mean").reindex(DIVS)
    image = ax.imshow(matrix, cmap="magma", aspect="auto")
    ax.set_yticks(np.arange(len(DIVS)), DIVS); ax.set_ylabel("DIV")
    ax.set_xticks([0, 15, 31, 47, 63], [1, 16, 32, 48, 64]); ax.set_xlabel("Electrode index")
    ax.set_title("Exploratory triplet participation", fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    _panel_label(ax, "c")
    fig.suptitle("Late-stage prediction boundary and exploratory spatial distribution", fontsize=12.0, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    export_publication_figure(fig, SUPP_DIR / "Figure_S10_late_prediction_spatial")
    _source_csv_s(10, [("late_predictions", late), ("late_bootstrap", boot), ("spatial_participation", spatial)])
    return []


def build_supplementary_figures() -> None:
    SUPP_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    audits: list[dict] = []
    for builder in [supplementary_1, supplementary_2, supplementary_3, supplementary_4,
                    supplementary_5, supplementary_6, supplementary_7, supplementary_8,
                    supplementary_9, supplementary_10]:
        audits.extend(builder())
    audit = pd.DataFrame(audits)
    validate_annotation_audit(audit)
    audit.to_csv(SOURCE_DIR / "supplementary_annotation_audit.csv", index=False)


if __name__ == "__main__":
    main()
