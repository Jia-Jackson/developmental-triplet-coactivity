from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


def _frame() -> pd.DataFrame:
    rows = []
    for div, base in [(14, 1.0), (21, 2.0)]:
        for index, culture in enumerate(["P1", "P1", "P2", "P2", "P3", "P3"]):
            rows.append(
                {
                    "recording": f"WT_DIV{div}_{culture}_{index}",
                    "DIV": div,
                    "culture": culture,
                    "value": base + index * 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_raincloud_separates_raw_mean_and_density_regions():
    from hotopology.raincloud import draw_raincloud

    fig, ax = plt.subplots()
    result = draw_raincloud(
        ax,
        _frame(),
        x="DIV",
        y="value",
        culture="culture",
        recording="recording",
        order=[14, 21],
    )
    for geometry in result.geometry.values():
        assert geometry.raw_x_max < geometry.mean_x
        assert geometry.mean_x < geometry.violin_x_min
    plt.close(fig)


def test_raincloud_draws_one_summary_and_density_per_group():
    from hotopology.raincloud import draw_raincloud

    fig, ax = plt.subplots()
    result = draw_raincloud(
        ax,
        _frame(),
        x="DIV",
        y="value",
        culture="culture",
        recording="recording",
        order=[14, 21],
    )
    assert set(result.geometry) == {14, 21}
    assert len(result.mean_artists) == 2
    assert len(result.violin_artists) == 2
    plt.close(fig)


def test_stat_band_renders_all_comparisons_including_ns():
    from hotopology.raincloud import draw_adjacent_stat_band

    annotations = pd.DataFrame(
        {
            "DIV_first": [14, 21, 28, 35, 42, 49, 56],
            "DIV_second": [21, 28, 35, 42, 49, 56, 63],
            "symbol": ["***", "ns", "*", "**", "ns", "*", "***"],
        }
    )
    fig, ax = plt.subplots()
    result = draw_adjacent_stat_band(
        ax,
        annotations,
        order=[14, 21, 28, 35, 42, 49, 56, 63],
        row_label="adjacent",
    )
    assert len(result.labels) == 7
    assert [label.get_text() for label in result.labels] == annotations["symbol"].tolist()
    assert all(label.get_fontsize() >= 11 for label in result.labels)
    plt.close(fig)


def test_one_sample_symbols_draw_one_label_per_div():
    from hotopology.raincloud import draw_one_sample_symbols

    annotations = pd.DataFrame({"DIV": [14, 21], "symbol": ["***", "ns"]})
    fig, ax = plt.subplots()
    labels = draw_one_sample_symbols(ax, annotations, order=[14, 21], row_label="vs 0")
    assert [label.get_text() for label in labels] == ["***", "ns"]
    plt.close(fig)
