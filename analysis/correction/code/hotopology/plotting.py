from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Mandatory editable-text and font settings for every project figure.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


PALETTE = {
    "early": "#484878",
    "transition": "#5B8FD6",
    "late": "#D9544D",
    "null": "#A8A8A8",
    "pairwise": "#42949E",
    "kpairwise": "#9A4D8E",
    "positive": "#2E9E44",
    "negative": "#E53935",
    "text": "#272727",
}


def apply_publication_style(font_size: float = 7.0) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": font_size,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def add_panel_label(ax, label: str, *, x: float = -0.15, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def export_publication_figure(fig, base_path: str | Path) -> list[Path]:
    """Export editable vector files plus review and submission rasters."""
    base = Path(base_path).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        base.with_suffix(".svg"),
        base.with_suffix(".pdf"),
        base.with_suffix(".png"),
        base.with_suffix(".tiff"),
    ]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], dpi=300, bbox_inches="tight")
    fig.savefig(outputs[3], dpi=600, bbox_inches="tight")
    plt.close(fig)
    return outputs

