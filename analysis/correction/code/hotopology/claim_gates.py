from __future__ import annotations

from typing import Sequence

import numpy as np


def incremental_value_gate(
    delta_mae: float,
    ci_low: float,
    ci_high: float,
    improved_culture_folds: int,
    total_culture_folds: int,
) -> dict[str, object]:
    if not np.all(np.isfinite([delta_mae, ci_low, ci_high])) or total_culture_folds <= 0:
        status = "technically_inconclusive"
    elif delta_mae > 0 and ci_low > 0 and improved_culture_folds == total_culture_folds:
        status = "supported"
    elif delta_mae > 0 and improved_culture_folds >= 2:
        status = "partially_supported"
    else:
        status = "not_supported"
    return {
        "status": status,
        "delta_mae": float(delta_mae),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "improved_culture_folds": int(improved_culture_folds),
        "total_culture_folds": int(total_culture_folds),
    }


def robustness_gate(
    effects: Sequence[float],
    *,
    minimum_agreement: float = 0.75,
) -> dict[str, object]:
    values = np.asarray(effects, dtype=float)
    values = values[np.isfinite(values) & (values != 0)]
    if values.size == 0:
        return {"status": "technically_inconclusive", "direction_agreement": np.nan}
    positive = float(np.mean(values > 0))
    negative = float(np.mean(values < 0))
    agreement = max(positive, negative)
    status = "supported" if agreement >= minimum_agreement else "not_supported"
    return {
        "status": status,
        "direction_agreement": agreement,
        "majority_direction": "positive" if positive >= negative else "negative",
        "n_specifications": int(values.size),
    }


def null_technical_gate(
    *,
    valid_nulls: int,
    minimum_valid: int,
    qc_pass: bool,
) -> dict[str, object]:
    status = (
        "supported"
        if int(valid_nulls) >= int(minimum_valid) and bool(qc_pass)
        else "technically_inconclusive"
    )
    return {
        "status": status,
        "valid_nulls": int(valid_nulls),
        "minimum_valid": int(minimum_valid),
        "qc_pass": bool(qc_pass),
    }
