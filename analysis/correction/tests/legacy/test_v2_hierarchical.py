from __future__ import annotations

import numpy as np
import pandas as pd


def _synthetic_longitudinal(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for plate in (1, 2, 3):
        for well_index in range(5):
            subject = f"P{plate}_W{well_index}"
            intercept = rng.normal(0, 0.25)
            for div in (14, 21, 28):
                rows.append(
                    {
                        "DIV": div,
                        "plate": plate,
                        "culture": f"P{plate}",
                        "subject_id": subject,
                        "value": intercept
                        + 0.08 * (div - 14)
                        + 0.15 * (plate - 1)
                        + rng.normal(0, 0.08),
                    }
                )
    return pd.DataFrame(rows)


def test_hierarchical_model_returns_div_and_culture_terms():
    from hotopology.hierarchical import fit_hierarchical_div_model

    fit = fit_hierarchical_div_model(
        _synthetic_longitudinal(), outcome="value", divs=[14, 21, 28]
    )
    assert fit.method in {"mixedlm", "gee"}
    assert set(fit.divs) == {14, 21, 28}
    assert fit.n_subjects == 15
    assert fit.n_cultures == 3
    assert np.isfinite(fit.residual_scale)


def test_adjacent_contrasts_recover_positive_change_and_adjust_p_values():
    from hotopology.hierarchical import (
        adjacent_contrasts_from_fit,
        fit_hierarchical_div_model,
    )

    fit = fit_hierarchical_div_model(
        _synthetic_longitudinal(), outcome="value", divs=[14, 21, 28]
    )
    contrasts = adjacent_contrasts_from_fit(fit)
    assert contrasts[["DIV_first", "DIV_second"]].values.tolist() == [
        [14, 21],
        [21, 28],
    ]
    assert (contrasts["estimate"] > 0).all()
    assert contrasts["adjusted_p"].between(0, 1).all()
    assert np.isfinite(contrasts["standardized_effect"]).all()


def test_culture_direction_summary_uses_within_subject_pairs():
    from hotopology.hierarchical import culture_specific_adjacent_changes

    table = _synthetic_longitudinal()
    result = culture_specific_adjacent_changes(
        table, outcome="value", divs=[14, 21, 28]
    )
    assert len(result) == 6
    assert set(result["culture"]) == {"P1", "P2", "P3"}
    assert (result["n_pairs"] == 5).all()
    assert (result["median_change"] > 0).all()
