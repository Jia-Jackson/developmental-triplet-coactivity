from __future__ import annotations

import numpy as np


def bin_spikes(
    spike_times_s: np.ndarray,
    channel_indices: np.ndarray,
    *,
    n_channels: int,
    duration_s: float,
    bin_s: float,
) -> np.ndarray:
    """Convert spike events to a channel-by-time binary occupancy matrix.

    Bins are half-open intervals ``[start, end)``. Multiple spikes from the
    same channel in the same bin remain a single active observation.
    """
    times = np.asarray(spike_times_s, dtype=float).reshape(-1)
    channels = np.asarray(channel_indices, dtype=int).reshape(-1)
    if times.size != channels.size:
        raise ValueError("spike_times_s and channel_indices must have equal length")
    if n_channels <= 0 or duration_s <= 0 or bin_s <= 0:
        raise ValueError("n_channels, duration_s, and bin_s must be positive")

    n_bins = int(np.ceil(duration_s / bin_s))
    result = np.zeros((n_channels, n_bins), dtype=np.uint8)
    valid = (
        np.isfinite(times)
        & (times >= 0)
        & (times < duration_s)
        & (channels >= 0)
        & (channels < n_channels)
    )
    if not np.any(valid):
        return result
    bins = np.floor(times[valid] / bin_s).astype(np.int64)
    result[channels[valid], bins] = 1
    return result

