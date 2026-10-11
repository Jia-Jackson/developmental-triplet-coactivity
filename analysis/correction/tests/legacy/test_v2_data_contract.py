from __future__ import annotations

import pandas as pd
import pytest


def test_normalize_recording_keys_accepts_prior_feature_identifiers():
    from hotopology.v2_contract import normalize_recording_keys

    table = pd.DataFrame(
        {
            "recording": ["WT_DIV14_P1_A1", "WT_DIV21_P3_B2"],
            "DIV": [14, 21],
            "plate": [1, 3],
            "well": ["A1", "B2"],
        }
    )
    result = normalize_recording_keys(table)
    assert result["subject_id"].tolist() == ["P1_A1", "P3_B2"]
    assert result["culture"].tolist() == ["P1", "P3"]
    assert result["recording_key"].tolist() == table["recording"].tolist()


def test_merge_one_to_one_rejects_duplicate_recording_keys():
    from hotopology.v2_contract import audited_one_to_one_merge

    left = pd.DataFrame({"recording": ["WT_DIV14_P1_A1"], "x": [1]})
    right = pd.DataFrame(
        {"recording": ["WT_DIV14_P1_A1", "WT_DIV14_P1_A1"], "y": [2, 3]}
    )
    with pytest.raises(ValueError, match="duplicate"):
        audited_one_to_one_merge(left, right)


def test_expected_longitudinal_grid_has_one_known_missing_recording():
    from hotopology.v2_contract import expected_recording_grid

    grid = expected_recording_grid(
        divs=[14, 21, 28, 35, 42, 49, 56, 63],
        cultures=[1, 2, 3],
        wells=["A1", "A2", "A3", "B1", "B2", "B3"],
        excluded_recording="WT_DIV35_P2_A1",
    )
    assert len(grid) == 143
    assert grid["recording"].is_unique
    assert "WT_DIV35_P2_A1" not in set(grid["recording"])
    assert grid.groupby("DIV").size().to_dict()[35] == 17


def test_audit_recording_table_detects_missing_or_unexpected_keys():
    from hotopology.v2_contract import audit_recording_table, expected_recording_grid

    expected = expected_recording_grid(
        divs=[14, 21], cultures=[1], wells=["A1", "A2"], excluded_recording=None
    )
    observed = expected.iloc[:-1].copy()
    with pytest.raises(ValueError, match="missing"):
        audit_recording_table(observed, expected)
