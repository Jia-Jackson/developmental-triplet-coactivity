from __future__ import annotations

import numpy as np
import pandas as pd


def _prediction_data(seed: int = 23, incremental: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for culture in ("P1", "P2", "P3"):
        for well in range(5):
            subject = f"{culture}_W{well}"
            for div in (14, 21, 28, 35):
                baseline = rng.normal()
                high = div / 7 + rng.normal(0, 0.15) if incremental else rng.normal()
                rows.append(
                    {
                        "culture": culture,
                        "subject_id": subject,
                        "DIV": div,
                        "baseline": baseline,
                        "high": high,
                    }
                )
    return pd.DataFrame(rows)


def test_grouped_nested_predictions_never_mix_outer_groups():
    from hotopology.incremental import nested_group_predictions

    data = _prediction_data()
    predictions, audit = nested_group_predictions(
        data,
        feature_columns=["baseline", "high"],
        target="DIV",
        outer_group="culture",
        inner_group="subject_id",
        alphas=[0.01, 0.1, 1.0],
    )
    assert len(predictions) == len(data)
    assert predictions["prediction"].notna().all()
    assert len(audit) == 3
    for row in audit.itertuples():
        assert set(row.train_outer_groups).isdisjoint(set(row.test_outer_groups))


def test_augmented_model_improves_when_feature_contains_incremental_signal():
    from hotopology.incremental import compare_feature_sets

    data = _prediction_data(incremental=True)
    _, metrics, _ = compare_feature_sets(
        data,
        feature_sets={"baseline": ["baseline"], "augmented": ["baseline", "high"]},
        target="DIV",
        outer_group="culture",
        inner_group="subject_id",
        alphas=[0.01, 0.1, 1.0],
    )
    metric = metrics.set_index("model_set")
    assert metric.loc["augmented", "mae"] < metric.loc["baseline", "mae"] * 0.4


def test_cross_fitted_residuals_are_returned_for_every_observation():
    from hotopology.incremental import cross_fitted_residuals

    data = _prediction_data(incremental=False)
    result = cross_fitted_residuals(
        data,
        feature_columns=["baseline"],
        target="high",
        outer_group="culture",
        inner_group="subject_id",
        alphas=[0.1, 1.0],
    )
    assert len(result) == len(data)
    assert np.allclose(result["residual"], result["observed"] - result["prediction"])
    assert result["residual"].notna().all()


def test_leave_one_subject_out_does_not_duplicate_identifier_columns():
    from hotopology.incremental import nested_group_predictions

    data = _prediction_data()
    result, audit = nested_group_predictions(
        data,
        feature_columns=["baseline"],
        target="DIV",
        outer_group="subject_id",
        inner_group="subject_id",
        alphas=[0.1, 1.0],
    )
    assert result.columns.is_unique
    assert len(audit) == data["subject_id"].nunique()
