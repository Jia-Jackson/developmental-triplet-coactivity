from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _model(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )


def _select_alpha(
    train: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target: str,
    inner_group: str,
    alphas: Sequence[float],
) -> tuple[float, dict[float, float]]:
    groups = train[inner_group].astype(str).to_numpy()
    unique = np.unique(groups)
    if len(unique) < 2:
        return float(alphas[0]), {float(alphas[0]): np.nan}
    splitter = GroupKFold(n_splits=min(5, len(unique)))
    scores: dict[float, float] = {}
    for alpha in alphas:
        errors = []
        for inner_train, inner_test in splitter.split(train, train[target], groups):
            model = _model(float(alpha))
            model.fit(
                train.iloc[inner_train][list(feature_columns)],
                train.iloc[inner_train][target],
            )
            prediction = model.predict(train.iloc[inner_test][list(feature_columns)])
            errors.append(
                mean_absolute_error(train.iloc[inner_test][target], prediction)
            )
        scores[float(alpha)] = float(np.mean(errors))
    selected = min(scores, key=lambda value: (scores[value], value))
    return float(selected), scores


def nested_group_predictions(
    table: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target: str,
    outer_group: str,
    inner_group: str,
    alphas: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nested group-aware ridge predictions with fold-local preprocessing."""
    required = [*feature_columns, target, outer_group, inner_group]
    missing = [column for column in required if column not in table]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    data = table.copy().reset_index(drop=False).rename(columns={"index": "row_index"})
    data = data.loc[data[target].notna()].copy()
    outer_values = sorted(data[outer_group].astype(str).unique())
    if len(outer_values) < 2:
        raise ValueError("at least two outer groups are required")

    prediction_rows = []
    audit_rows = []
    for fold_index, held_out in enumerate(outer_values):
        test_mask = data[outer_group].astype(str) == held_out
        train = data.loc[~test_mask].copy()
        test = data.loc[test_mask].copy()
        selected_alpha, inner_scores = _select_alpha(
            train,
            feature_columns=feature_columns,
            target=target,
            inner_group=inner_group,
            alphas=alphas,
        )
        model = _model(selected_alpha)
        model.fit(train[list(feature_columns)], train[target])
        prediction = model.predict(test[list(feature_columns)])
        identifiers = list(dict.fromkeys([
            column
            for column in [
                "row_index",
                "recording",
                "DIV",
                "culture",
                "subject_id",
                outer_group,
                inner_group,
            ]
            if column in test.columns
        ]))
        fold_predictions = test[identifiers].copy()
        fold_predictions["observed"] = test[target].to_numpy(dtype=float)
        fold_predictions["prediction"] = prediction.astype(float)
        fold_predictions["outer_fold"] = fold_index
        fold_predictions["held_out_group"] = held_out
        fold_predictions["selected_alpha"] = selected_alpha
        prediction_rows.append(fold_predictions)
        train_groups = sorted(train[outer_group].astype(str).unique())
        test_groups = sorted(test[outer_group].astype(str).unique())
        audit_rows.append(
            {
                "outer_fold": fold_index,
                "held_out_group": held_out,
                "train_outer_groups": train_groups,
                "test_outer_groups": test_groups,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "selected_alpha": selected_alpha,
                "inner_scores": inner_scores,
                "outer_groups_disjoint": set(train_groups).isdisjoint(test_groups),
            }
        )
    predictions = pd.concat(prediction_rows, ignore_index=True).sort_values("row_index")
    if len(predictions) != len(data) or predictions["row_index"].duplicated().any():
        raise RuntimeError("outer predictions do not form a one-to-one partition")
    audit = pd.DataFrame(audit_rows)
    if not audit["outer_groups_disjoint"].all():
        raise RuntimeError("outer-group leakage detected")
    return predictions.reset_index(drop=True), audit


def prediction_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    observed = predictions["observed"].to_numpy(dtype=float)
    predicted = predictions["prediction"].to_numpy(dtype=float)
    rho, p_value = stats.spearmanr(observed, predicted)
    return {
        "n": int(len(predictions)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
        "spearman_rho": float(rho),
        "spearman_p": float(p_value),
    }


def compare_feature_sets(
    table: pd.DataFrame,
    *,
    feature_sets: Mapping[str, Sequence[str]],
    target: str,
    outer_group: str,
    inner_group: str,
    alphas: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_tables = []
    metric_rows = []
    audit_tables = []
    for name, features in feature_sets.items():
        predictions, audit = nested_group_predictions(
            table,
            feature_columns=features,
            target=target,
            outer_group=outer_group,
            inner_group=inner_group,
            alphas=alphas,
        )
        predictions.insert(0, "model_set", name)
        audit.insert(0, "model_set", name)
        metrics = prediction_metrics(predictions)
        metrics["model_set"] = name
        metrics["n_features"] = int(len(features))
        prediction_tables.append(predictions)
        metric_rows.append(metrics)
        audit_tables.append(audit)
    return (
        pd.concat(prediction_tables, ignore_index=True),
        pd.DataFrame(metric_rows),
        pd.concat(audit_tables, ignore_index=True),
    )


def cross_fitted_residuals(
    table: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target: str,
    outer_group: str,
    inner_group: str,
    alphas: Sequence[float],
) -> pd.DataFrame:
    predictions, _ = nested_group_predictions(
        table,
        feature_columns=feature_columns,
        target=target,
        outer_group=outer_group,
        inner_group=inner_group,
        alphas=alphas,
    )
    predictions["residual"] = predictions["observed"] - predictions["prediction"]
    predictions["target_feature"] = target
    return predictions


def clustered_delta_mae_bootstrap(
    predictions: pd.DataFrame,
    *,
    baseline_name: str,
    augmented_name: str,
    subject_col: str = "subject_id",
    culture_col: str = "culture",
    n_bootstrap: int = 2000,
    seed: int = 20260828,
) -> dict[str, float | int]:
    wide = predictions.pivot_table(
        index=[column for column in ["row_index", subject_col, culture_col, "observed"] if column in predictions],
        columns="model_set",
        values="prediction",
        aggfunc="first",
    ).reset_index()
    required = [baseline_name, augmented_name]
    if not set(required).issubset(wide.columns):
        raise ValueError("baseline or augmented predictions are missing")
    rng = np.random.default_rng(seed)
    deltas = []
    cultures = sorted(wide[culture_col].unique()) if culture_col in wide else [None]
    for _ in range(int(n_bootstrap)):
        sampled_frames = []
        for culture in cultures:
            subset = wide if culture is None else wide.loc[wide[culture_col] == culture]
            subjects = subset[subject_col].unique()
            sampled = rng.choice(subjects, size=len(subjects), replace=True)
            for draw_index, subject in enumerate(sampled):
                block = subset.loc[subset[subject_col] == subject].copy()
                block["_bootstrap_subject"] = f"{culture}_{draw_index}"
                sampled_frames.append(block)
        sample = pd.concat(sampled_frames, ignore_index=True)
        baseline_mae = np.mean(np.abs(sample["observed"] - sample[baseline_name]))
        augmented_mae = np.mean(np.abs(sample["observed"] - sample[augmented_name]))
        deltas.append(float(baseline_mae - augmented_mae))
    deltas_array = np.asarray(deltas)
    observed_baseline = np.mean(np.abs(wide["observed"] - wide[baseline_name]))
    observed_augmented = np.mean(np.abs(wide["observed"] - wide[augmented_name]))
    return {
        "n_bootstrap": int(n_bootstrap),
        "delta_mae_baseline_minus_augmented": float(observed_baseline - observed_augmented),
        "ci95_low": float(np.quantile(deltas_array, 0.025)),
        "ci95_high": float(np.quantile(deltas_array, 0.975)),
        "probability_improvement": float(np.mean(deltas_array > 0)),
    }
