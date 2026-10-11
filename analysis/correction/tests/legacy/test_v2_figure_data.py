from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures" / "computational_closure_v2"
SOURCE_DIR = ROOT / "source_data" / "computational_closure_v2"


def test_v2_publication_figures_have_all_required_formats():
    stems = [
        "Figure_S1_computational_closure_schematic",
        "Figure_6_hierarchical_reanalysis",
        "Figure_7_activity_adjusted",
        "Figure_8_incremental_value",
        "Figure_9_null_model_comparison",
        "Figure_10_robustness",
    ]
    for stem in stems:
        for suffix in [".svg", ".pdf", ".png", ".tiff"]:
            path = FIGURE_DIR / f"{stem}{suffix}"
            assert path.exists(), path
            assert path.stat().st_size > 1000, path
        assert "<svg" in (FIGURE_DIR / f"{stem}.svg").read_text(encoding="utf-8")[:1000]
        assert (FIGURE_DIR / f"{stem}.pdf").read_bytes().startswith(b"%PDF")
        with Image.open(FIGURE_DIR / f"{stem}.tiff") as image:
            assert image.width >= 2000
            assert image.height >= 1500


def test_v2_figure_source_data_are_nonempty_and_keyed():
    expected = {
        "Figure_6b_hierarchical_effects.csv": {"outcome", "standardized_effect", "adjusted_p"},
        "Figure_7bd_adjusted_effects.csv": {"outcome", "adjustment_tier", "standardized_effect"},
        "Figure_8c_incremental_bootstrap.csv": {"analysis_range", "validation", "ci95_low", "ci95_high"},
        "Figure_9c_residuals.csv": {"recording", "DIV", "model", "absolute_residual_z"},
        "Figure_10b_node_effects.csv": {"node_specification", "outcome", "standardized_effect"},
    }
    for filename, required in expected.items():
        path = SOURCE_DIR / filename
        assert path.exists(), path
        table = pd.read_csv(path)
        assert not table.empty
        assert required.issubset(table.columns), (filename, required - set(table.columns))
