from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


def test_p_to_symbol_uses_manuscript_thresholds():
    from hotopology.story_contract import p_to_symbol

    assert p_to_symbol(0.20) == "ns"
    assert p_to_symbol(0.05) == "ns"
    assert p_to_symbol(0.049) == "*"
    assert p_to_symbol(0.01) == "*"
    assert p_to_symbol(0.009) == "**"
    assert p_to_symbol(0.001) == "**"
    assert p_to_symbol(0.0009) == "***"
    with pytest.raises(ValueError):
        p_to_symbol(float("nan"))


def test_all_seven_adjacent_div_pairs_are_required():
    from hotopology.story_contract import expected_adjacent_pairs

    assert expected_adjacent_pairs() == [
        (14, 21),
        (21, 28),
        (28, 35),
        (35, 42),
        (42, 49),
        (49, 56),
        (56, 63),
    ]


def test_direct_25ms_source_has_expected_longitudinal_hierarchy():
    frame = pd.read_csv(ROOT / "tables" / "direct_higher_order_metrics_all_bins.csv")
    frame = frame.loc[frame["bin_ms"] == 25]

    assert len(frame) == 143
    assert "WT_DIV35_P2_A1" not in set(frame["recording"])
    assert frame.groupby("DIV").size().to_dict() == {
        14: 18,
        21: 18,
        28: 18,
        35: 17,
        42: 18,
        49: 18,
        56: 18,
        63: 18,
    }


def test_adjacent_annotations_reject_missing_comparison():
    from hotopology.story_contract import adjacent_annotations

    frame = pd.DataFrame(
        {
            "family": ["direct_25ms"] * 6,
            "outcome": ["triplet_support_mass"] * 6,
            "DIV_first": [14, 21, 28, 35, 42, 49],
            "DIV_second": [21, 28, 35, 42, 49, 56],
            "adjusted_p": [0.01] * 6,
        }
    )
    with pytest.raises(ValueError, match="missing adjacent"):
        adjacent_annotations(
            frame,
            family="direct_25ms",
            outcome="triplet_support_mass",
        )


def test_annotation_audit_detects_symbol_mismatch():
    from hotopology.story_contract import validate_annotation_audit

    audit = pd.DataFrame(
        {
            "figure": ["Figure_1"],
            "panel": ["b"],
            "comparison": ["DIV14-DIV21"],
            "adjusted_p": [0.0001],
            "symbol": ["*"],
            "source_file": ["v2_hierarchical_adjacent_contrasts.csv"],
        }
    )
    with pytest.raises(ValueError, match="symbol mismatch"):
        validate_annotation_audit(audit)
