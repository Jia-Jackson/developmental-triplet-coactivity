"""Generate the figure panels affected by the endpoint-z correction.

The module reuses the validated plotting utilities and source tables from the
corrected higher-order project without modifying that project.  Endpoint-count
z scores are excluded because a standardized score is undefined whenever the
reference ensemble has zero variance.
"""

from __future__ import annotations

from release_paths import repository_path

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


SOURCE_PROJECT = repository_path('analysis/correction')
SOURCE_FIGURES = repository_path("analysis/final_revision/figure_templates")
SPATIAL_SOURCE = repository_path('source_data/Figure_4_figure_supplement_2/Figure_4_figure_supplement_2_source_data.csv')

CORRECTED_FIGURE_OUTCOMES: dict[str, tuple[str, ...]] = {
    "Figure_3D": (
        "log1p_absolute_residual_mass",
        "absolute_residual_z",
        "h1_persistence_deviation_z",
    ),
    "Figure_5": ("h1_total_persistence_capped",),
    "Figure_6_figure_supplement_1": (
        "triplet_support_mass",
        "triplet_repeated5_density",
        "h1_total_persistence_capped",
        "log1p_absolute_residual_mass",
        "absolute_residual_z",
        "z_h1_total_persistence_capped",
    ),
    "Figure_3_figure_supplement_2": (
        "log1p_absolute_residual_mass",
        "absolute_residual_z",
        "h1_persistence_deviation_z",
    ),
}


def _load_story_module():
    script = SOURCE_PROJECT / "code" / "run_story_figures_v3.py"
    spec = importlib.util.spec_from_file_location("story_v3_source", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import plotting helpers from {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.apply_publication_style(font_size=8.0)
    return module


def _export_exact_raster(fig, path: Path, *, width_px: int, height_px: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(width_px / 600, height_px / 600)
    fig.savefig(path, dpi=600, facecolor="white", bbox_inches=None, pad_inches=0)


def _write_source(frame: pd.DataFrame, output: Path, name: str) -> None:
    source_dir = output / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(source_dir / f"{name}.csv", index=False)


def _figure_3d(module, output: Path) -> None:
    activity = module._read("v2_activity_adjusted_adjacent_contrasts.csv", v2=True)
    desired = list(CORRECTED_FIGURE_OUTCOMES["Figure_3D"])
    tier2 = activity.loc[
        activity["adjustment_tier"].eq("tier2_activity_burst")
        & activity["DIV_first"].eq(14)
        & activity["DIV_second"].eq(21)
        & activity["outcome"].isin(desired)
    ].set_index("outcome").loc[desired].reset_index()
    if len(tier2) != 3:
        raise RuntimeError(f"Expected three valid activity-adjusted outcomes, found {len(tier2)}")

    fig, ax = plt.subplots(figsize=(3.425, 2.5))
    labels = ["Residual mass", "Residual z", "H1 persistence z"]
    y = np.arange(len(tier2))[::-1]
    colors = ["#D9544D" if p < 0.05 else "#A8A8A8" for p in tier2["adjusted_p"]]
    ax.axvline(0, ls="--", lw=0.9, color="#767676")
    ax.scatter(tier2["standardized_effect"], y, s=48, c=colors, zorder=3)
    for yi, row in zip(y, tier2.itertuples(index=False)):
        ax.text(
            row.standardized_effect,
            yi + 0.18,
            module.p_to_symbol(row.adjusted_p),
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
        )
    ax.set_yticks(y, labels)
    lower = min(-1.35, float(tier2["standardized_effect"].min()) - 0.35)
    upper = max(2.75, float(tier2["standardized_effect"].max()) + 0.50)
    ax.set_xlim(lower, upper)
    ax.set_ylim(-0.65, len(y) - 0.15)
    ax.set_xlabel("Standardized early effect (DIV21 − DIV14)", fontsize=8.3)
    ax.set_title("Activity-adjusted early effects", fontsize=9.2, fontweight="bold")
    module._clean_axis(ax)
    ax.text(-0.34, 1.13, "D", transform=ax.transAxes, fontsize=17, fontweight="bold", va="top")
    fig.subplots_adjust(left=0.38, right=0.97, top=0.78, bottom=0.26)

    panel_base = output / "Figure_3D_activity_adjusted_valid_metrics"
    module.export_publication_figure(fig, panel_base)

    # Rebuild an exact-size raster panel for conservative replacement within
    # the author's polished four-panel composite.
    fig, ax = plt.subplots(figsize=(3.425, 2.5))
    ax.axvline(0, ls="--", lw=0.9, color="#767676")
    ax.scatter(tier2["standardized_effect"], y, s=48, c=colors, zorder=3)
    for yi, row in zip(y, tier2.itertuples(index=False)):
        ax.text(row.standardized_effect, yi + 0.18, module.p_to_symbol(row.adjusted_p),
                ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax.set_yticks(y, labels)
    ax.set_xlim(lower, upper)
    ax.set_ylim(-0.65, len(y) - 0.15)
    ax.set_xlabel("Standardized early effect (DIV21 − DIV14)", fontsize=8.3)
    ax.set_title("Activity-adjusted early effects", fontsize=9.2, fontweight="bold")
    module._clean_axis(ax)
    ax.text(-0.34, 1.13, "D", transform=ax.transAxes, fontsize=17, fontweight="bold", va="top")
    fig.subplots_adjust(left=0.38, right=0.97, top=0.78, bottom=0.26)
    exact_panel = output / "_figure3d_exact.png"
    _export_exact_raster(fig, exact_panel, width_px=2055, height_px=1500)
    plt.close(fig)

    original = Image.open(SOURCE_FIGURES / "Figure.3.tif").convert("RGB")
    if original.size != (4110, 3000):
        raise RuntimeError(f"Unexpected Figure 3 raster size: {original.size}")
    panel = Image.open(exact_panel).convert("RGB")
    original.paste(panel, (2055, 1500))
    original.save(output / "Figure_3_corrected.tiff", dpi=(600, 600), compression="tiff_lzw")
    original.resize((2055, 1500), Image.Resampling.LANCZOS).save(
        output / "Figure_3_corrected.png", dpi=(300, 300)
    )
    exact_panel.unlink(missing_ok=True)
    _write_source(tier2, output, "Figure_3D_source_data")


def _figure_5(module, output: Path) -> None:
    extended = module._read("v2_extended_null_hierarchical_contrasts.csv", v2=True)
    one_sample = module._read("v2_null_deviation_one_sample_statistics.csv", v2=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.55))
    audits: list[dict] = []
    sources: list[pd.DataFrame] = []
    for ax, panel, model, title in [
        (axes[0], "A", "pairwise_conditioned_k_100", "Instantaneous reference"),
        (axes[1], "B", "kinetic_conditioned_k", "Lag-1 kinetic reference"),
    ]:
        metric = "h1_total_persistence_capped"
        frame = module._null_comparison_frame(model, metric)
        adjacent = module.adjacent_annotations(extended, model=model, outcome=metric)
        zero = module.one_sample_annotations(one_sample, model=model, source_metric=metric)
        module._plot_developmental_raincloud(
            ax,
            frame,
            outcome="z",
            annotations=adjacent,
            one_sample=zero,
            ylabel="H1 persistence deviation z",
            title=title,
        )
        ax.axhline(0, ls="--", lw=0.9, color="#767676", zorder=0)
        ax.text(-0.25, 1.17, panel, transform=ax.transAxes, fontsize=17,
                fontweight="bold", va="top")
        ax.text(0.98, 0.98, title, transform=ax.transAxes, ha="right", va="top",
                fontsize=8.4, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5), zorder=10)
        data = frame.copy()
        data["model"] = model
        sources.append(data)
        audits.extend(module._one_sample_audit("Figure_5", panel.lower(), zero,
                                               "v2_null_deviation_one_sample_statistics.csv"))
        audits.extend(module._adjacent_audit("Figure_5", panel.lower(), adjacent,
                                             "v2_extended_null_hierarchical_contrasts.csv"))
    fig.suptitle("H1 persistence deviations depend on the lower-order reference",
                 fontsize=11.5, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.78, wspace=0.40)
    module.export_publication_figure(fig, output / "Figure_5_corrected_H1_persistence_only")
    _write_source(pd.concat(sources, ignore_index=True), output, "Figure_5_source_data")
    _write_source(pd.DataFrame(audits), output, "Figure_5_annotation_audit")


def _figure_6_supplement_1(module, output: Path) -> None:
    hierarchical = module._read("v2_hierarchical_adjacent_contrasts.csv", v2=True)
    culture = module._read("v2_culture_specific_adjacent_changes.csv", v2=True)
    specs = [
        ("direct_25ms", "triplet_support_mass", "Triplet support"),
        ("direct_25ms", "triplet_repeated5_density", "Repeated triplets"),
        ("direct_25ms", "h1_total_persistence_capped", "Direct H1"),
        ("conditioned_triplet_residual", "log1p_absolute_residual_mass", "Residual mass"),
        ("conditioned_triplet_residual", "absolute_residual_z", "Residual z"),
        ("conditioned_topology_deviation", "z_h1_total_persistence_capped", "H1 persistence z"),
    ]
    rows, culture_rows = [], []
    for family, outcome, label in specs:
        row = module._one_row(hierarchical, family=family, outcome=outcome,
                              DIV_first=14, DIV_second=21)
        rows.append({"family": family, "outcome": outcome, "label": label,
                     "standardized_effect": row.standardized_effect,
                     "adjusted_p": row.adjusted_p})
        subset = culture.loc[(culture["family"].eq(family)) & (culture["outcome"].eq(outcome))
                             & culture["DIV_first"].eq(14) & culture["DIV_second"].eq(21)]
        for item in subset.itertuples(index=False):
            culture_rows.append({"label": label, "culture": item.culture,
                                 "median_change": item.median_change})
    effects = pd.DataFrame(rows)
    directions = pd.DataFrame(culture_rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    y = np.arange(len(effects))[::-1]
    ax.axvline(0, ls="--", color="#767676", lw=1)
    for yi, row in zip(y, effects.itertuples(index=False)):
        color = "#D9544D" if row.adjusted_p < 0.05 else "#A8A8A8"
        ax.scatter(row.standardized_effect, yi, color=color, s=42)
        ax.text(row.standardized_effect, yi + 0.20, module.p_to_symbol(row.adjusted_p),
                ha="center", fontsize=10.5, fontweight="bold")
    ax.set_yticks(y, effects["label"])
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Standardized DIV21-minus-DIV14 effect (symlog)")
    ax.set_title("Hierarchical effects", fontweight="bold")
    module._clean_axis(ax)
    ax.text(-0.32, 1.08, "A", transform=ax.transAxes, fontsize=17, fontweight="bold", va="top")

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
    ax.text(-0.40, 1.08, "B", transform=ax.transAxes, fontsize=17, fontweight="bold", va="top")
    fig.suptitle("Hierarchical and culture-specific validation of the early transition",
                 fontsize=11.2, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.18, right=0.93, bottom=0.16, top=0.82, wspace=0.55)
    module.export_publication_figure(fig, output / "Figure_6_figure_supplement_1_corrected")
    _write_source(effects, output, "Figure_6_figure_supplement_1_effects")
    _write_source(directions, output, "Figure_6_figure_supplement_1_culture_changes")


def _figure_3_supplement_2(module, output: Path) -> None:
    cv = module._read("v2_activity_form_cv.csv", v2=True)
    effects = module._read("v2_activity_adjusted_adjacent_contrasts.csv", v2=True)
    table = module._read("v2_activity_adjustment_analysis_table.csv", v2=True)
    outcomes = list(CORRECTED_FIGURE_OUTCOMES["Figure_3_figure_supplement_2"])
    labels = {"log1p_absolute_residual_mass": "Mass", "absolute_residual_z": "Residual z",
              "h1_persistence_deviation_z": "H1 persistence z"}
    cv_plot = cv.loc[cv["outcome"].isin(outcomes)].copy()
    early = effects.loc[effects["outcome"].isin(outcomes)
                        & effects["DIV_first"].eq(14) & effects["DIV_second"].eq(21)].copy()
    fig = plt.figure(figsize=(7.25, 5.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.48, wspace=0.42)
    ax = fig.add_subplot(grid[0, 0])
    for index, form in enumerate(["linear", "log1p", "spline"]):
        subset = cv_plot.loc[cv_plot["activity_form"].eq(form)].set_index("outcome").reindex(outcomes)
        ax.plot(np.arange(len(outcomes)), subset["cv_mae"], marker="o", lw=1.3,
                color=["#484878", "#42949E", "#D9544D"][index], label=form)
    ax.set_xticks(np.arange(len(outcomes)), [labels[o] for o in outcomes], rotation=22, ha="right")
    ax.set_ylabel("Grouped-CV MAE")
    ax.set_title("Activity-form selection", fontweight="bold")
    ax.legend(frameon=False, fontsize=7)
    module._clean_axis(ax)
    ax.text(-0.25, 1.10, "A", transform=ax.transAxes, fontsize=17, fontweight="bold", va="top")

    ax = fig.add_subplot(grid[0, 1])
    positions = np.arange(len(outcomes))
    for offset, tier, color in [(-0.10, "tier1_mean_k", "#7884B4"),
                                (0.10, "tier2_activity_burst", "#D9544D")]:
        subset = early.loc[early["adjustment_tier"].eq(tier)].set_index("outcome").reindex(outcomes)
        ax.scatter(subset["standardized_effect"], positions + offset, color=color, s=36, label=tier)
        for pos, row in zip(positions + offset, subset.itertuples(index=False)):
            ax.text(row.standardized_effect, pos + 0.07, module.p_to_symbol(row.adjusted_p),
                    ha="center", fontsize=8.5, fontweight="bold")
    ax.axvline(0, ls="--", color="#767676", lw=1)
    ax.set_yticks(positions, [labels[o] for o in outcomes])
    ax.set_xlabel("Standardized DIV21-minus-DIV14 effect")
    ax.set_title("Mean-count and activity adjustment", fontweight="bold")
    ax.legend(frameon=False, fontsize=6.5)
    module._clean_axis(ax)
    ax.text(-0.25, 1.10, "B", transform=ax.transAxes, fontsize=17, fontweight="bold", va="top")

    ax = fig.add_subplot(grid[1, :])
    scatter = ax.scatter(table["mean_k_top16"], table["log1p_absolute_residual_mass"],
                         c=table["DIV"], cmap="viridis_r", s=20, alpha=0.75)
    ax.set_xlabel("Mean population count (selected 16 electrodes)")
    ax.set_ylabel("log(1 + 10⁶ absolute residual mass)")
    ax.set_title("Residual magnitude and population activity", fontweight="bold")
    fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.02, label="DIV")
    module._clean_axis(ax)
    ax.text(-0.11, 1.10, "C", transform=ax.transAxes, fontsize=17, fontweight="bold", va="top")
    fig.suptitle("Activity-adjustment specification and diagnostics",
                 fontsize=11.2, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.11, right=0.95, bottom=0.11, top=0.88)
    module.export_publication_figure(fig, output / "Figure_3_figure_supplement_2_corrected")
    _write_source(cv_plot, output, "Figure_3_figure_supplement_2_activity_form_cv")
    _write_source(early, output, "Figure_3_figure_supplement_2_adjusted_effects")
    _write_source(table, output, "Figure_3_figure_supplement_2_analysis_table")


def _figure_4_supplement_2(module, output: Path) -> None:
    source = pd.read_csv(SPATIAL_SOURCE, low_memory=False)
    spatial = source.loc[source["dataset"].eq("spatial_participation")].copy()
    matrix = spatial.pivot(index="DIV", columns="electrode_index", values="mean").reindex(module.DIVS)
    matrix = matrix.sort_index(axis=1)
    if matrix.shape != (8, 64) or matrix.isna().any().any():
        raise RuntimeError(f"Expected a complete 8 x 64 participation matrix, found {matrix.shape}")
    fig, ax = plt.subplots(figsize=(7.25, 3.5))
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="magma", aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(module.DIVS)), module.DIVS)
    ax.set_ylabel("Developmental age (DIV)")
    ax.set_xticks([0, 15, 31, 47, 63], [1, 16, 32, 48, 64])
    ax.set_xlabel("Electrode index")
    ax.set_title("Participation in high-absolute-residual triplets", fontweight="bold")
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mean normalized participation")
    fig.subplots_adjust(left=0.11, right=0.91, bottom=0.20, top=0.84)
    module.export_publication_figure(fig, output / "Figure_4_figure_supplement_2_triplet_participation")
    _write_source(spatial, output, "Figure_4_figure_supplement_2_source_data")


def generate_all(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    module = _load_story_module()
    _figure_3d(module, output)
    _figure_5(module, output)
    _figure_6_supplement_1(module, output)
    _figure_3_supplement_2(module, output)
    _figure_4_supplement_2(module, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate_all(args.output)


if __name__ == "__main__":
    main()
