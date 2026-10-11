from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable

from .hierarchical import HierarchicalFit


def _activity_transformer(activity_form: str):
    if activity_form == "linear":
        return Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
    if activity_form == "log1p":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]
        )
    if activity_form == "spline":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("spline", SplineTransformer(n_knots=4, degree=2, include_bias=False)),
                ("scale", StandardScaler()),
            ]
        )
    raise ValueError(f"unknown activity form: {activity_form}")


def select_activity_form(
    table: pd.DataFrame,
    *,
    outcome: str,
    activity_col: str,
    n_splits: int = 6,
    subject_col: str = "subject_id",
    div_col: str = "DIV",
    culture_col: str = "culture",
) -> tuple[str, pd.DataFrame]:
    """Select linear/log/spline activity form with group-aware CV."""
    columns = [outcome, activity_col, subject_col, div_col, culture_col]
    data = table[columns].replace([np.inf, -np.inf], np.nan).dropna(
        subset=[outcome, activity_col, subject_col, div_col, culture_col]
    )
    groups = data[subject_col].astype(str).to_numpy()
    n_groups = len(np.unique(groups))
    splitter = GroupKFold(n_splits=min(int(n_splits), n_groups))
    rows: list[dict[str, object]] = []
    for form in ("linear", "log1p", "spline"):
        errors = []
        for train, test in splitter.split(data, data[outcome], groups):
            preprocess = ColumnTransformer(
                [
                    (
                        "categorical",
                        OneHotEncoder(handle_unknown="ignore", drop="first"),
                        [div_col, culture_col],
                    ),
                    ("activity", _activity_transformer(form), [activity_col]),
                ]
            )
            model = Pipeline(
                [("preprocess", preprocess), ("ridge", Ridge(alpha=1.0))]
            )
            model.fit(data.iloc[train], data.iloc[train][outcome])
            prediction = model.predict(data.iloc[test])
            errors.append(mean_absolute_error(data.iloc[test][outcome], prediction))
        rows.append(
            {
                "activity_form": form,
                "cv_mae": float(np.mean(errors)),
                "cv_mae_sd": float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0,
                "n_folds": len(errors),
            }
        )
    scores = pd.DataFrame(rows).sort_values(["cv_mae", "activity_form"]).reset_index(drop=True)
    return str(scores.iloc[0]["activity_form"]), scores


def fit_activity_adjusted_model(
    table: pd.DataFrame,
    *,
    outcome: str,
    divs: Sequence[int],
    activity_col: str,
    activity_form: str,
    additional_covariates: Sequence[str] = (),
    div_col: str = "DIV",
    culture_col: str = "culture",
    subject_col: str = "subject_id",
) -> HierarchicalFit:
    required = [
        outcome,
        div_col,
        culture_col,
        subject_col,
        activity_col,
        *additional_covariates,
    ]
    missing = [column for column in required if column not in table]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    data = table.loc[table[div_col].isin(divs), required].copy()
    rename = {
        outcome: "_outcome",
        div_col: "_div",
        culture_col: "_culture",
        subject_col: "_subject",
        activity_col: "_activity",
    }
    data = data.rename(columns=rename)
    numeric = ["_outcome", "_activity", *additional_covariates]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        raise ValueError("no complete finite observations")

    if activity_form == "linear":
        activity_term = "_activity"
    elif activity_form == "log1p":
        if (data["_activity"] < 0).any():
            raise ValueError("log1p activity form requires non-negative activity")
        data["_activity_log1p"] = np.log1p(data["_activity"])
        activity_term = "_activity_log1p"
    elif activity_form == "spline":
        activity_term = "bs(_activity, df=3, degree=2, include_intercept=False)"
    else:
        raise ValueError(f"unknown activity form: {activity_form}")

    standardized_covariates = []
    for index, column in enumerate(additional_covariates):
        target = f"_covariate_{index}"
        values = data[column].astype(float)
        scale = float(values.std(ddof=0))
        data[target] = (values - float(values.mean())) / scale if scale > 0 else 0.0
        standardized_covariates.append(target)

    reference = int(divs[0])
    extra = " + ".join([activity_term, *standardized_covariates])
    formula = (
        f"_outcome ~ C(_div, Treatment(reference={reference})) + C(_culture)"
        + (f" + {extra}" if extra else "")
    )
    mixed_errors = []
    result = None
    diagnostics: dict[str, object] = {
        "activity_form": activity_form,
        "activity_col": activity_col,
        "additional_covariates": list(additional_covariates),
        "formula": formula,
    }
    for method in ("lbfgs", "powell"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                candidate = smf.mixedlm(
                    formula, data=data, groups=data["_subject"], re_formula="1"
                ).fit(reml=False, method=method, maxiter=2000, disp=False)
            if candidate.converged and np.isfinite(candidate.fe_params).all():
                result = candidate
                diagnostics["optimizer"] = method
                diagnostics["warnings"] = [str(item.message) for item in caught]
                break
            mixed_errors.append(f"{method}: converged={candidate.converged}")
        except Exception as exc:  # pragma: no cover
            mixed_errors.append(f"{method}: {type(exc).__name__}: {exc}")

    if result is not None:
        coefficients = result.fe_params.astype(float)
        covariance = result.cov_params().loc[coefficients.index, coefficients.index].astype(float)
        residual_scale = float(np.sqrt(max(float(result.scale), 0.0)))
        method_name = "mixedlm"
        converged = bool(result.converged)
        diagnostics["random_intercept_variance"] = float(np.asarray(result.cov_re)[0, 0])
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gee = smf.gee(
                formula,
                groups="_subject",
                data=data,
                cov_struct=Exchangeable(),
            ).fit()
        coefficients = gee.params.astype(float)
        covariance = gee.cov_params().loc[coefficients.index, coefficients.index].astype(float)
        residual_scale = float(np.sqrt(np.mean(np.square(gee.resid_response))))
        method_name = "gee"
        converged = True
        diagnostics["mixedlm_errors"] = mixed_errors
        diagnostics["warnings"] = [str(item.message) for item in caught]

    return HierarchicalFit(
        outcome=outcome,
        method=method_name,
        divs=tuple(int(value) for value in divs),
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
