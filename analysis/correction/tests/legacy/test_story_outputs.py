from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = ROOT / "figures" / "manuscript_story_v3"
SOURCE_ROOT = ROOT / "source_data" / "manuscript_story_v3"


MAIN_STEMS = [
    "Figure_1_developmental_emergence",
    "Figure_2_burst_activity_context",
    "Figure_3_triplet_specificity_nulls",
    "Figure_4_null_dependent_topology",
    "Figure_5_reproducibility_incremental_value",
]

SUPPLEMENTARY_STEMS = [
    "Figure_S1_workflow_hierarchy",
    "Figure_S2_bin_width_sensitivity",
    "Figure_S3_complete_direct_metrics",
    "Figure_S4_complete_burst_metrics",
    "Figure_S5_null_model_qc",
    "Figure_S6_hierarchical_culture_effects",
    "Figure_S7_node_specifications",
    "Figure_S8_temporal_stability",
    "Figure_S9_activity_model_selection",
    "Figure_S10_late_prediction_spatial",
]


def test_main_story_figures_have_all_publication_formats():
    for stem in MAIN_STEMS:
        for suffix in [".svg", ".pdf", ".png", ".tiff"]:
            path = FIGURE_ROOT / "main" / f"{stem}{suffix}"
            assert path.exists(), path
            assert path.stat().st_size > 1000, path
        assert "<svg" in (
            FIGURE_ROOT / "main" / f"{stem}.svg"
        ).read_text(encoding="utf-8")[:2000]
        assert (FIGURE_ROOT / "main" / f"{stem}.pdf").read_bytes().startswith(b"%PDF")
        with Image.open(FIGURE_ROOT / "main" / f"{stem}.tiff") as image:
            assert image.width >= 2000
            assert image.height >= 1200


def test_main_annotation_audit_is_complete_and_symbol_consistent():
    from sys import path as sys_path

    sys_path.insert(0, str(ROOT / "code"))
    from hotopology.story_contract import p_to_symbol, validate_annotation_audit

    audit_path = SOURCE_ROOT / "main_annotation_audit.csv"
    assert audit_path.exists(), audit_path
    audit = pd.read_csv(audit_path)
    validate_annotation_audit(audit)
    assert (audit["symbol"] == audit["adjusted_p"].map(p_to_symbol)).all()
    adjacent = audit.loc[audit["test_family"].eq("adjacent_div")]
    counts = adjacent.groupby(["figure", "panel"]).size()
    assert not counts.empty
    assert (counts == 7).all(), counts.to_dict()


def test_main_figure_source_data_are_nonempty():
    for number in range(1, 6):
        path = SOURCE_ROOT / f"Figure_{number}_source_data.csv"
        assert path.exists(), path
        frame = pd.read_csv(path)
        assert not frame.empty, path


def test_supplementary_figures_have_all_publication_formats():
    for stem in SUPPLEMENTARY_STEMS:
        for suffix in [".svg", ".pdf", ".png", ".tiff"]:
            path = FIGURE_ROOT / "supplementary" / f"{stem}{suffix}"
            assert path.exists(), path
            assert path.stat().st_size > 1000, path
        assert "<svg" in (
            FIGURE_ROOT / "supplementary" / f"{stem}.svg"
        ).read_text(encoding="utf-8")[:2000]


def test_supplementary_source_data_and_annotation_audit_exist():
    for number in range(1, 11):
        path = SOURCE_ROOT / f"Figure_S{number}_source_data.csv"
        assert path.exists(), path
        assert not pd.read_csv(path).empty, path
    audit_path = SOURCE_ROOT / "supplementary_annotation_audit.csv"
    assert audit_path.exists(), audit_path
    audit = pd.read_csv(audit_path)
    assert not audit.empty
    from sys import path as sys_path

    sys_path.insert(0, str(ROOT / "code"))
    from hotopology.story_contract import validate_annotation_audit

    validate_annotation_audit(audit)
