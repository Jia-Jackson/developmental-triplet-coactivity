from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return adjusted
    p = values[finite]
    order = np.argsort(p)
    ranked = p[order]
    raw = ranked * ranked.size / np.arange(1, ranked.size + 1)
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    restored = np.empty_like(monotone)
    restored[order] = np.clip(monotone, 0.0, 1.0)
    adjusted[finite] = restored
    return adjusted


def adjacent_paired_wilcoxon(
    table: pd.DataFrame,
    *,
    metric: str,
    divs: Sequence[int],
    subject_col: str = "subject_id",
    div_col: str = "DIV",
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for first, second in zip(divs[:-1], divs[1:]):
        left = table.loc[table[div_col] == first, [subject_col, metric]].rename(
            columns={metric: "first"}
        )
        right = table.loc[table[div_col] == second, [subject_col, metric]].rename(
            columns={metric: "second"}
        )
        paired = left.merge(right, on=subject_col, how="inner").dropna()
        changes = paired["second"].to_numpy() - paired["first"].to_numpy()
        if changes.size == 0:
            statistic = np.nan
            p_value = np.nan
        elif np.allclose(changes, 0):
            statistic = 0.0
            p_value = 1.0
        else:
            test = stats.wilcoxon(
                paired["first"],
                paired["second"],
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
        rows.append(
            {
                "metric": metric,
                "DIV_first": int(first),
                "DIV_second": int(second),
                "n_pairs": int(changes.size),
                "median_change": float(np.median(changes)) if changes.size else np.nan,
                "wilcoxon_statistic": statistic,
                "raw_p": p_value,
            }
        )
    result = pd.DataFrame(rows)
    result["adjusted_p"] = benjamini_hochberg(result["raw_p"].to_numpy())
    return result


def summarize_by_div(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = table.groupby("DIV", sort=True)[metric]
    summary = grouped.agg(["count", "mean", "std", "median"]).reset_index()
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary["metric"] = metric
    return summary


def repeated_friedman(
    table: pd.DataFrame,
    *,
    metric: str,
    divs: Sequence[int],
    subject_col: str = "subject_id",
    div_col: str = "DIV",
) -> dict[str, float | int | str]:
    """Friedman omnibus test restricted to subjects observed at every requested DIV."""
    data = table.loc[table[div_col].isin(divs), [subject_col, div_col, metric]].copy()
    wide = data.pivot(index=subject_col, columns=div_col, values=metric)
    wide = wide.reindex(columns=list(divs)).dropna()
    arrays = [wide[div].to_numpy(dtype=float) for div in divs]
    if wide.empty or len(divs) < 3:
        statistic = np.nan
        p_value = np.nan
    elif all(np.allclose(array, arrays[0]) for array in arrays[1:]):
        statistic = 0.0
        p_value = 1.0
    else:
        test = stats.friedmanchisquare(*arrays)
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
    return {
        "metric": metric,
        "DIV_range": f"{min(divs)}-{max(divs)}",
        "n_complete_subjects": int(len(wide)),
        "statistic": statistic,
        "df": int(len(divs) - 1),
        "raw_p": p_value,
    }


def one_sample_wilcoxon_by_div(
    table: pd.DataFrame,
    *,
    metric: str,
    divs: Sequence[int],
    null_value: float = 0.0,
    alternative: str = "two-sided",
    div_col: str = "DIV",
) -> pd.DataFrame:
    """One-sample signed-rank tests within age, with BH adjustment across ages."""
    rows: list[dict[str, float | int | str]] = []
    for div in divs:
        values = table.loc[table[div_col] == div, metric].dropna().to_numpy(dtype=float)
        differences = values - null_value
        if values.size == 0:
            statistic = np.nan
            p_value = np.nan
        elif np.allclose(differences, 0):
            statistic = 0.0
            p_value = 1.0
        else:
            test = stats.wilcoxon(
                differences,
                alternative=alternative,
                zero_method="wilcox",
                method="auto",
            )
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
        rows.append(
            {
                "metric": metric,
                "DIV": int(div),
                "n": int(values.size),
                "null_value": float(null_value),
                "alternative": alternative,
                "median": float(np.median(values)) if values.size else np.nan,
                "wilcoxon_statistic": statistic,
                "raw_p": p_value,
            }
        )
    result = pd.DataFrame(rows)
    result["adjusted_p"] = benjamini_hochberg(result["raw_p"].to_numpy())
    return result
