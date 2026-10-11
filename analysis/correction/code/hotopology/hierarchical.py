from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable

from .statistics import benjamini_hochberg


@dataclass
class HierarchicalFit:
    outcome: str
    method: str
    divs: tuple[int, ...]
    reference_div: int
    coefficients: pd.Series
    covariance: pd.DataFrame
    residual_scale: float
    n_observations: int
    n_subjects: int
    n_cultures: int
    converged: bool
    diagnostics: dict[str, object]


def _prepare(
    table: pd.DataFrame,
    *,
    outcome: str,
    divs: Sequence[int],
    div_col: str,
    culture_col: str,
    subject_col: str,
) -> pd.DataFrame:
    required = [outcome, div_col, culture_col, subject_col]
    missing = [column for column in required if column not in table]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    data = table.loc[table[div_col].isin(divs), required].copy()
    data = data.rename(
        columns={
            outcome: "_outcome",
            div_col: "_div",
            culture_col: "_culture",
            subject_col: "_subject",
        }
    )
    data["_outcome"] = pd.to_numeric(data["_outcome"], errors="coerce")
    data["_div"] = pd.to_numeric(data["_div"], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        raise ValueError(f"no finite observations for {outcome}")
    return data


def fit_hierarchical_div_model(
    table: pd.DataFrame,
    *,
    outcome: str,
    divs: Sequence[int],
    div_col: str = "DIV",
    culture_col: str = "culture",
    subject_col: str = "subject_id",
) -> HierarchicalFit:
    """Fit categorical DIV + culture fixed effects with a well random intercept.

    A population-averaged GEE with an exchangeable within-well correlation is used
    only if the mixed model fails to return finite, converged fixed-effect estimates.
    """
    ordered_divs = tuple(int(value) for value in divs)
    reference = ordered_divs[0]
    data = _prepare(
        table,
        outcome=outcome,
        divs=ordered_divs,
        div_col=div_col,
        culture_col=culture_col,
        subject_col=subject_col,
    )
    formula = (
        f"_outcome ~ C(_div, Treatment(reference={reference})) + C(_culture)"
    )
    diagnostics: dict[str, object] = {}
    mixed_result = None
    mixed_errors: list[str] = []
    for method in ("lbfgs", "powell"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                candidate = smf.mixedlm(
                    formula, data=data, groups=data["_subject"], re_formula="1"
                ).fit(reml=False, method=method, maxiter=2000, disp=False)
            finite = np.isfinite(candidate.fe_params.to_numpy(dtype=float)).all()
            if bool(candidate.converged) and finite:
                mixed_result = candidate
                diagnostics["optimizer"] = method
                diagnostics["warnings"] = [str(item.message) for item in caught]
                break
            mixed_errors.append(f"{method}: converged={candidate.converged}, finite={finite}")
        except Exception as exc:  # pragma: no cover - exercised by real-data fallback
            mixed_errors.append(f"{method}: {type(exc).__name__}: {exc}")

    if mixed_result is not None:
        coefficients = mixed_result.fe_params.astype(float)
        covariance = mixed_result.cov_params().loc[
            coefficients.index, coefficients.index
        ].astype(float)
        residual_scale = float(np.sqrt(max(float(mixed_result.scale), 0.0)))
        method_name = "mixedlm"
        converged = bool(mixed_result.converged)
        diagnostics["random_intercept_variance"] = float(
            np.asarray(mixed_result.cov_re)[0, 0]
        )
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gee_result = smf.gee(
                formula,
                groups="_subject",
                data=data,
                cov_struct=Exchangeable(),
            ).fit()
        coefficients = gee_result.params.astype(float)
        covariance = gee_result.cov_params().loc[
            coefficients.index, coefficients.index
        ].astype(float)
        residual_scale = float(np.sqrt(np.mean(np.square(gee_result.resid_response))))
        method_name = "gee"
        converged = True
        diagnostics["mixedlm_errors"] = mixed_errors
        diagnostics["warnings"] = [str(item.message) for item in caught]

    return HierarchicalFit(
        outcome=outcome,
        method=method_name,
        divs=ordered_divs,
        reference_div=reference,
        coefficients=coefficients,
        covariance=covariance,
        residual_scale=residual_scale,
        n_observations=int(len(data)),
        n_subjects=int(data["_subject"].nunique()),
        n_cultures=int(data["_culture"].nunique()),
        converged=converged,
        diagnostics=diagnostics,
    )


def _div_term(fit: HierarchicalFit, div: int) -> str | None:
    if int(div) == fit.reference_div:
        return None
    ending = f"[T.{int(div)}]"
    matches = [name for name in fit.coefficients.index if name.endswith(ending)]
    if len(matches) != 1:
        raise ValueError(f"could not identify coefficient for DIV {div}: {matches}")
    return matches[0]


def adjacent_contrasts_from_fit(fit: HierarchicalFit) -> pd.DataFrame:
    names = list(fit.coefficients.index)
    rows: list[dict[str, object]] = []
    for first, second in zip(fit.divs[:-1], fit.divs[1:]):
        vector = pd.Series(0.0, index=names)
        first_term = _div_term(fit, first)
        second_term = _div_term(fit, second)
        if first_term is not None:
            vector[first_term] -= 1.0
        if second_term is not None:
            vector[second_term] += 1.0
        estimate = float(vector @ fit.coefficients)
        variance = float(vector @ fit.covariance @ vector)
        standard_error = float(np.sqrt(max(variance, 0.0)))
        if standard_error > 0 and np.isfinite(standard_error):
            z_value = estimate / standard_error
            raw_p = float(2.0 * stats.norm.sf(abs(z_value)))
            ci_low = estimate - 1.96 * standard_error
            ci_high = estimate + 1.96 * standard_error
        else:
            z_value = np.nan
            raw_p = np.nan
            ci_low = np.nan
            ci_high = np.nan
        rows.append(
            {
                "outcome": fit.outcome,
                "method": fit.method,
                "DIV_first": first,
                "DIV_second": second,
                "estimate": estimate,
                "standard_error": standard_error,
                "z_value": z_value,
                "raw_p": raw_p,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "standardized_effect": (
                    estimate / fit.residual_scale if fit.residual_scale > 0 else np.nan
                ),
                "n_observations": fit.n_observations,
                "n_subjects": fit.n_subjects,
                "n_cultures": fit.n_cultures,
            }
        )
    result = pd.DataFrame(rows)
    result["adjusted_p"] = benjamini_hochberg(result["raw_p"].to_numpy())
    return result


def coefficient_table(fit: HierarchicalFit) -> pd.DataFrame:
    rows = []
    for name, estimate in fit.coefficients.items():
        se = float(np.sqrt(max(float(fit.covariance.loc[name, name]), 0.0)))
        z_value = float(estimate / se) if se > 0 else np.nan
        rows.append(
            {
                "outcome": fit.outcome,
                "method": fit.method,
                "term": name,
                "estimate": float(estimate),
                "standard_error": se,
                "z_value": z_value,
                "raw_p": float(2 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan,
                "ci95_low": float(estimate - 1.96 * se),
                "ci95_high": float(estimate + 1.96 * se),
                "n_observations": fit.n_observations,
                "n_subjects": fit.n_subjects,
                "n_cultures": fit.n_cultures,
            }
        )
    return pd.DataFrame(rows)


def culture_specific_adjacent_changes(
    table: pd.DataFrame,
    *,
    outcome: str,
    divs: Sequence[int],
    culture_col: str = "culture",
    subject_col: str = "subject_id",
    div_col: str = "DIV",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for culture, subset in table.groupby(culture_col, sort=True):
        for first, second in zip(divs[:-1], divs[1:]):
            left = subset.loc[subset[div_col] == first, [subject_col, outcome]].rename(
                columns={outcome: "first"}
            )
            right = subset.loc[subset[div_col] == second, [subject_col, outcome]].rename(
                columns={outcome: "second"}
            )
            paired = left.merge(right, on=subject_col, how="inner").dropna()
            changes = paired["second"].to_numpy(float) - paired["first"].to_numpy(float)
            rows.append(
                {
                    "outcome": outcome,
                    "culture": culture,
                    "DIV_first": int(first),
                    "DIV_second": int(second),
                    "n_pairs": int(len(changes)),
                    "mean_change": float(np.mean(changes)) if len(changes) else np.nan,
                    "median_change": float(np.median(changes)) if len(changes) else np.nan,
                    "fraction_positive": float(np.mean(changes > 0)) if len(changes) else np.nan,
                }
            )
    return pd.DataFrame(rows)
