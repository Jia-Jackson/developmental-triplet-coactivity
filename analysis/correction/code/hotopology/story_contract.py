from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


DIVS: tuple[int, ...] = (14, 21, 28, 35, 42, 49, 56, 63)


@dataclass(frozen=True)
class PanelSpec:
    figure: str
    panel: str
    title: str
    outcome: str
    source_file: str
    statistic_file: str | None = None


def expected_adjacent_pairs() -> list[tuple[int, int]]:
    return list(zip(DIVS[:-1], DIVS[1:]))


def p_to_symbol(p_value: float) -> str:
    p_value = float(p_value)
    if not np.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
        raise ValueError(f"P value must be finite and within [0, 1], got {p_value!r}")
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def adjacent_annotations(
    frame: pd.DataFrame,
    *,
    family: str | None = None,
    outcome: str,
    model: str | None = None,
) -> pd.DataFrame:
    required = {"outcome", "DIV_first", "DIV_second", "adjusted_p"}
    _require_columns(frame, required, "adjacent contrast table")
    subset = frame.loc[frame["outcome"].eq(outcome)].copy()
    if family is not None:
        _require_columns(subset, {"family"}, "adjacent contrast table")
        subset = subset.loc[subset["family"].eq(family)]
    if model is not None:
        _require_columns(subset, {"model"}, "adjacent contrast table")
        subset = subset.loc[subset["model"].eq(model)]
    pairs = list(zip(subset["DIV_first"].astype(int), subset["DIV_second"].astype(int)))
    expected = expected_adjacent_pairs()
    if len(pairs) != len(set(pairs)):
        raise ValueError(f"duplicate adjacent comparisons for {outcome}: {pairs}")
    missing = [pair for pair in expected if pair not in pairs]
    extra = [pair for pair in pairs if pair not in expected]
    if missing or extra:
        raise ValueError(
            f"missing adjacent comparisons for {outcome}: {missing}; unexpected: {extra}"
        )
    subset["pair_order"] = [expected.index(pair) for pair in pairs]
    subset = subset.sort_values("pair_order").reset_index(drop=True)
    subset["comparison"] = subset.apply(
        lambda row: f"DIV{int(row.DIV_first)}-DIV{int(row.DIV_second)}", axis=1
    )
    subset["symbol"] = subset["adjusted_p"].map(p_to_symbol)
    return subset


def one_sample_annotations(
    frame: pd.DataFrame,
    *,
    model: str,
    source_metric: str,
) -> pd.DataFrame:
    required = {"model", "source_metric", "DIV", "adjusted_p"}
    _require_columns(frame, required, "one-sample statistics table")
    subset = frame.loc[
        frame["model"].eq(model) & frame["source_metric"].eq(source_metric)
    ].copy()
    divs = subset["DIV"].astype(int).tolist()
    if sorted(divs) != list(DIVS) or len(divs) != len(set(divs)):
        raise ValueError(
            f"one-sample statistics for {model}/{source_metric} must contain each DIV once; got {divs}"
        )
    subset["DIV"] = subset["DIV"].astype(int)
    subset = subset.sort_values("DIV").reset_index(drop=True)
    subset["comparison"] = subset["DIV"].map(lambda div: f"DIV{div} vs 0")
    subset["symbol"] = subset["adjusted_p"].map(p_to_symbol)
    return subset


def validate_annotation_audit(frame: pd.DataFrame) -> None:
    required = {
        "figure",
        "panel",
        "comparison",
        "adjusted_p",
        "symbol",
        "source_file",
    }
    _require_columns(frame, required, "annotation audit")
    if frame.empty:
        raise ValueError("annotation audit must not be empty")
    duplicated = frame.duplicated(["figure", "panel", "comparison"], keep=False)
    if duplicated.any():
        rows = frame.loc[duplicated, ["figure", "panel", "comparison"]].to_dict("records")
        raise ValueError(f"duplicate annotation audit rows: {rows}")
    expected = frame["adjusted_p"].map(p_to_symbol)
    mismatch = ~frame["symbol"].eq(expected)
    if mismatch.any():
        rows = frame.loc[
            mismatch, ["figure", "panel", "comparison", "adjusted_p", "symbol"]
        ].to_dict("records")
        raise ValueError(f"symbol mismatch in annotation audit: {rows}")


def load_story_tables(root: str | Path) -> Mapping[str, pd.DataFrame]:
    root = Path(root)
    paths = {
        "direct": root / "tables" / "direct_higher_order_metrics_all_bins.csv",
        "hierarchical": root
        / "tables"
        / "computational_closure_v2"
        / "v2_hierarchical_adjacent_contrasts.csv",
        "burst": root / "tables" / "burst_stratified_higher_order_metrics_25ms.csv",
        "burst_stats": root / "tables" / "burst_inside_outside_wilcoxon.csv",
        "one_sample": root
        / "tables"
        / "computational_closure_v2"
        / "v2_null_deviation_one_sample_statistics.csv",
        "extended": root
        / "tables"
        / "computational_closure_v2"
        / "v2_extended_null_hierarchical_contrasts.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing story source tables: {missing}")
    return {name: pd.read_csv(path) for name, path in paths.items()}
