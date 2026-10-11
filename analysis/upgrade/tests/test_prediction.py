import pandas as pd

from higher_order_v2.prediction import clustered_delta_mae_bootstrap_fast


def test_fast_cluster_bootstrap_reports_exact_observed_delta():
    rows = []
    for subject, culture, observed, baseline, augmented in [
        ("P1_A1", "P1", 14.0, 20.0, 16.0),
        ("P1_A2", "P1", 21.0, 25.0, 22.0),
        ("P2_A1", "P2", 14.0, 19.0, 15.0),
        ("P2_A2", "P2", 21.0, 24.0, 21.0),
    ]:
        for model, prediction in (("base", baseline), ("aug", augmented)):
            rows.append({
                "row_index": len(rows) // 2,
                "subject_id": subject,
                "culture": culture,
                "observed": observed,
                "model_set": model,
                "prediction": prediction,
            })
    result = clustered_delta_mae_bootstrap_fast(
        pd.DataFrame(rows), baseline_name="base", augmented_name="aug",
        n_bootstrap=100, seed=4,
    )
    # Baseline MAE = 4.5; augmented MAE = 1.0.
    assert result["delta_mae_baseline_minus_augmented"] == 3.5
    assert result["n_bootstrap"] == 100
