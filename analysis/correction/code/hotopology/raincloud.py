from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Hashable, Sequence

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


@dataclass(frozen=True)
class GroupGeometry:
    raw_x_min: float
    raw_x_max: float
    mean_x: float
    violin_x_min: float
    violin_x_max: float


@dataclass(frozen=True)
class RaincloudResult:
    geometry: dict[Hashable, GroupGeometry]
    mean_artists: list[Artist]
    violin_artists: list[Artist]
    point_artists: list[Artist]


@dataclass(frozen=True)
class StatBandResult:
    labels: list[Artist]
    brackets: list[Line2D]
    row_label: Artist | None


DEFAULT_CULTURE_MARKERS = {"P1": "o", "P2": "s", "P3": "^"}


def developmental_palette(order: Sequence[Hashable]) -> dict[Hashable, tuple[float, ...]]:
    colors = plt.get_cmap("viridis_r")(np.linspace(0.03, 0.92, len(order)))
    return {group: tuple(color) for group, color in zip(order, colors)}


def _stable_jitter(value: str, width: float) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "little") / (2**64 - 1)
    return (unit - 0.5) * 2.0 * width


def _density(values: np.ndarray, n_points: int = 160) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("Raincloud group contains no finite values")
    spread = float(np.ptp(values))
    if len(values) < 2 or spread == 0.0:
        scale = max(abs(float(values[0])) * 0.02, 0.02)
        y = np.linspace(values[0] - scale, values[0] + scale, n_points)
        density = np.exp(-0.5 * ((y - values[0]) / (scale / 2.5)) ** 2)
    else:
        pad = max(spread * 0.18, np.std(values, ddof=1) * 0.15)
        y = np.linspace(values.min() - pad, values.max() + pad, n_points)
        density = gaussian_kde(values)(y)
    maximum = float(np.max(density))
    if maximum <= 0.0 or not np.isfinite(maximum):
        density = np.ones_like(y)
        maximum = 1.0
    return y, density / maximum


def draw_raincloud(
    ax: Axes,
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    culture: str,
    recording: str,
    order: Sequence[Hashable],
    colors: dict[Hashable, tuple[float, ...] | str] | None = None,
    culture_markers: dict[str, str] | None = None,
    point_size: float = 20.0,
    point_alpha: float = 0.72,
    violin_width: float = 0.30,
) -> RaincloudResult:
    required = {x, y, culture, recording}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Raincloud data missing columns: {missing}")
    colors = colors or developmental_palette(order)
    culture_markers = culture_markers or DEFAULT_CULTURE_MARKERS
    geometry: dict[Hashable, GroupGeometry] = {}
    mean_artists: list[Artist] = []
    violin_artists: list[Artist] = []
    point_artists: list[Artist] = []

    for index, group in enumerate(order):
        subset = frame.loc[frame[x].eq(group)].copy()
        values = pd.to_numeric(subset[y], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        subset = subset.loc[finite]
        values = values[finite]
        if not len(values):
            raise ValueError(f"No finite {y} values for {x}={group}")

        raw_center = index - 0.24
        raw_x = np.array(
            [
                raw_center + _stable_jitter(str(identifier), 0.065)
                for identifier in subset[recording]
            ]
        )
        for marker_name, marker in culture_markers.items():
            mask = subset[culture].astype(str).eq(marker_name).to_numpy()
            if mask.any():
                point_artists.append(
                    ax.scatter(
                        raw_x[mask],
                        values[mask],
                        s=point_size,
                        marker=marker,
                        color=colors[group],
                        alpha=point_alpha,
                        edgecolor="white",
                        linewidth=0.35,
                        zorder=3,
                    )
                )

        mean = float(np.mean(values))
        sem = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        summary = ax.errorbar(
            index,
            mean,
            yerr=sem,
            fmt="o",
            ms=5.2,
            color="#151515",
            ecolor="#151515",
            elinewidth=1.25,
            capsize=2.5,
            capthick=1.1,
            zorder=5,
        )
        mean_artists.append(summary)

        density_y, density = _density(values)
        violin_left = index + 0.10
        violin_right = violin_left + density * violin_width
        violin = ax.fill_betweenx(
            density_y,
            violin_left,
            violin_right,
            color=colors[group],
            alpha=0.72,
            linewidth=0.75,
            edgecolor=colors[group],
            zorder=2,
        )
        violin_artists.append(violin)
        geometry[group] = GroupGeometry(
            raw_x_min=float(raw_x.min()),
            raw_x_max=float(raw_x.max()),
            mean_x=float(index),
            violin_x_min=float(violin_left),
            violin_x_max=float(violin_left + violin_width),
        )

    ax.set_xticks(range(len(order)), [str(group) for group in order])
    ax.set_xlim(-0.55, len(order) - 0.45)
    return RaincloudResult(
        geometry=geometry,
        mean_artists=mean_artists,
        violin_artists=violin_artists,
        point_artists=point_artists,
    )


def draw_adjacent_stat_band(
    ax: Axes,
    annotations: pd.DataFrame,
    *,
    order: Sequence[Hashable],
    row_label: str | None = None,
    y_fraction: float = 1.055,
    label_fontsize: float = 11.0,
) -> StatBandResult:
    required = {"DIV_first", "DIV_second", "symbol"}
    missing = sorted(required - set(annotations.columns))
    if missing:
        raise ValueError(f"Stat-band annotations missing columns: {missing}")
    positions = {int(group): index for index, group in enumerate(order)}
    labels: list[Artist] = []
    brackets: list[Line2D] = []
    transform = ax.get_xaxis_transform()
    for row in annotations.itertuples(index=False):
        first = positions[int(row.DIV_first)]
        second = positions[int(row.DIV_second)]
        line = ax.plot(
            [first, first, second, second],
            [y_fraction - 0.018, y_fraction, y_fraction, y_fraction - 0.018],
            color="#333333",
            lw=0.9,
            transform=transform,
            clip_on=False,
        )[0]
        label = ax.text(
            (first + second) / 2,
            y_fraction + 0.012,
            str(row.symbol),
            transform=transform,
            ha="center",
            va="bottom",
            fontsize=label_fontsize,
            fontweight="bold",
            color="#171717",
            clip_on=False,
        )
        brackets.append(line)
        labels.append(label)
    row_artist = None
    if row_label:
        row_artist = ax.text(
            -0.52,
            y_fraction,
            row_label,
            transform=transform,
            ha="right",
            va="center",
            fontsize=max(label_fontsize - 1.0, 8.0),
            color="#333333",
            clip_on=False,
        )
    return StatBandResult(labels=labels, brackets=brackets, row_label=row_artist)


def draw_one_sample_symbols(
    ax: Axes,
    annotations: pd.DataFrame,
    *,
    order: Sequence[Hashable],
    row_label: str | None = None,
    y_fraction: float = 1.055,
    label_fontsize: float = 11.0,
) -> list[Artist]:
    required = {"DIV", "symbol"}
    missing = sorted(required - set(annotations.columns))
    if missing:
        raise ValueError(f"One-sample annotations missing columns: {missing}")
    positions = {int(group): index for index, group in enumerate(order)}
    transform = ax.get_xaxis_transform()
    labels: list[Artist] = []
    for row in annotations.itertuples(index=False):
        labels.append(
            ax.text(
                positions[int(row.DIV)],
                y_fraction,
                str(row.symbol),
                transform=transform,
                ha="center",
                va="bottom",
                fontsize=label_fontsize,
                fontweight="bold",
                color="#171717",
                clip_on=False,
            )
        )
    if row_label:
        ax.text(
            -0.52,
            y_fraction,
            row_label,
            transform=transform,
            ha="right",
            va="bottom",
            fontsize=max(label_fontsize - 1.0, 8.0),
            color="#333333",
            clip_on=False,
        )
    return labels
