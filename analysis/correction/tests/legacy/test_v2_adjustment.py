from __future__ import annotations

import numpy as np
import pandas as pd


def _confounded_data(seed: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for plate in (1, 2, 3):
        for well in range(6):
            subject = f"P{plate}_W{well}"
            random_intercept = rng.normal(0, 0.2)
            for div, activity in [(14, 0.5), (21, 2.0), (28, 3.0)]:
                measured_activity = activity + rng.normal(0, 0.12)
                rows.append(
                    {
                        "DIV": div,
                        "culture": f"P{plate}",
                        "subject_id": subject,
                        "mean_k": measured_activity,
                        "burst_rate": measured_activity * 0.4 + rng.normal(0, 0.1),
                        "outcome": random_intercept
                        + 1.8 * measured_activity
                        + rng.normal(0, 0.12),
                    }
                )
    return pd.DataFrame(rows)


def test_activity_form_selection_returns_prespecified_candidate():
    from hotopology.activity_adjustment import select_activity_form

    data = _confounded_data()
    selected, scores = select_activity_form(
        data, outcome="outcome", activity_col="mean_k", n_splits=6
    )
    assert selected in {"linear", "log1p", "spline"}
    assert set(scores["activity_form"]) == {"linear", "log1p", "spline"}
    assert scores["cv_mae"].notna().all()


def test_activity_adjustment_attenuates_spurious_div_contrast():
    from hotopology.activity_adjustment import fit_activity_adjusted_model
    from hotopology.hierarchical import (
        adjacent_contrasts_from_fit,
        fit_hierarchical_div_model,
    )

    data = _confounded_data()
    unadjusted = fit_hierarchical_div_model(
        data, outcome="outcome", divs=[14, 21, 28]
    )
    adjusted = fit_activity_adjusted_model(
        data,
        outcome="outcome",
        divs=[14, 21, 28],
        activity_col="mean_k",
        activity_form="linear",
        additional_covariates=["burst_rate"],
    )
    unadjusted_change = adjacent_contrasts_from_fit(unadjusted).iloc[0]["estimate"]
    adjusted_change = adjacent_contrasts_from_fit(adjusted).iloc[0]["estimate"]
    assert abs(adjusted_change) < abs(unadjusted_change) * 0.35
    assert adjusted.diagnostics["activity_form"] == "linear"
