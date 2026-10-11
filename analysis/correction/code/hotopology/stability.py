from __future__ import annotations

import numpy as np


def rebin_binary(
    x: np.ndarray,
    *,
    source_bin_ms: int,
    target_bin_ms: int,
) -> np.ndarray:
    data = (np.asarray(x) > 0).astype(np.uint8)
    if data.ndim != 2:
        raise ValueError("x must be node-by-bin")
    if target_bin_ms < source_bin_ms or target_bin_ms % source_bin_ms != 0:
        raise ValueError("target bin must be an integer multiple of source bin")
    factor = int(target_bin_ms // source_bin_ms)
    usable = data.shape[1] - data.shape[1] % factor
    if usable == 0:
        raise ValueError("no complete target bins")
    reshaped = data[:, :usable].reshape(data.shape[0], usable // factor, factor)
    return np.any(reshaped, axis=2).astype(np.uint8)


def temporal_windows(
    *,
    n_bins: int,
    bin_ms: int,
    block_duration_s: int = 50,
    cumulative_duration_s: tuple[int, ...] = (60, 120, 180, 240, 300),
) -> dict[str, slice]:
    duration_s = n_bins * bin_ms / 1000.0
    windows: dict[str, slice] = {}
    midpoint = n_bins // 2
    windows["half_first"] = slice(0, midpoint)
    windows["half_second"] = slice(midpoint, n_bins)
    block_bins = int(round(block_duration_s * 1000 / bin_ms))
    n_blocks = n_bins // block_bins
    for index in range(n_blocks):
        windows[f"block{block_duration_s}s_{index + 1}"] = slice(
            index * block_bins, (index + 1) * block_bins
        )
    for seconds in cumulative_duration_s:
        if seconds <= duration_s + 1e-9:
            stop = min(n_bins, int(round(seconds * 1000 / bin_ms)))
            windows[f"cumulative_{seconds}s"] = slice(0, stop)
    return windows
