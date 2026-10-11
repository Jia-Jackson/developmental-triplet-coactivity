from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class BurstResult:
    intervals: np.ndarray
    threshold_s: float
    spike_mask: np.ndarray
    n_channels: np.ndarray

    def bin_mask(self, *, bin_s: float, n_bins: int) -> np.ndarray:
        mask = np.zeros(n_bins, dtype=bool)
        for start, end in self.intervals:
            first = max(0, int(np.floor(start / bin_s)))
            last = min(n_bins - 1, int(np.floor(end / bin_s)))
            if last >= first:
                mask[first : last + 1] = True
        return mask


def automatic_bakkum_threshold(unique_times: np.ndarray, n_spikes: int) -> float:
    unique_times = np.asarray(unique_times, dtype=float)
    if unique_times.size < n_spikes * 2:
        return 0.1
    isi_n = unique_times[n_spikes - 1 :] - unique_times[: -(n_spikes - 1)]
    isi_n = isi_n[np.isfinite(isi_n) & (isi_n > 0)]
    if isi_n.size == 0 or np.unique(np.diff(unique_times)).size <= 10:
        return 0.1
    if np.max(np.diff(unique_times)) < 0.1:
        return 0.0
    steps_ms = np.power(10.0, np.arange(-5.0, 1.5001, 0.05))
    counts, _ = np.histogram(isi_n * 1000.0, bins=steps_ms)
    if counts.sum() == 0:
        return 0.1
    width = max(3, min(11, int(round(len(counts) * 0.04))))
    if width % 2 == 0:
        width += 1
    curve = np.convolve(counts.astype(float), np.ones(width) / width, mode="same")
    if curve.sum() > 0:
        curve /= curve.sum()
    peaks, _ = signal.find_peaks(curve, distance=2)
    if peaks.size < 2:
        return 0.1
    first, second = int(peaks[0]), int(peaks[1])
    valley = first + int(np.argmin(curve[first : second + 1]))
    threshold_s = float(steps_ms[min(valley, len(steps_ms) - 1)] / 1000.0)
    return min(max(threshold_s, 0.0), 0.1)


def detect_network_bursts(
    spike_times_s: np.ndarray,
    channel_indices: np.ndarray,
    *,
    duration_s: float,
    fs: float,
    min_spikes: int = 10,
    min_channels: int = 3,
) -> BurstResult:
    """Detect Bakkum-style network bursts using the prior project definition."""
    times = np.asarray(spike_times_s, dtype=float).reshape(-1)
    channels = np.asarray(channel_indices, dtype=int).reshape(-1)
    if times.size != channels.size:
        raise ValueError("spike times and channels must have equal length")
    order = np.argsort(times)
    times = times[order]
    channels = channels[order]
    empty = BurstResult(
        intervals=np.empty((0, 2), dtype=float),
        threshold_s=np.nan,
        spike_mask=np.zeros(times.size, dtype=bool),
        n_channels=np.empty(0, dtype=int),
    )
    if times.size < min_spikes:
        return empty
    sample_times = np.round(times * fs) / fs
    unique_times = np.unique(sample_times)
    if unique_times.size < min_spikes:
        return empty
    threshold = automatic_bakkum_threshold(unique_times, min_spikes)
    if threshold <= 0:
        return BurstResult(empty.intervals, threshold, empty.spike_mask, empty.n_channels)
    ok = (
        unique_times[min_spikes - 1 :] - unique_times[: -(min_spikes - 1)]
    ) <= threshold
    in_burst = np.zeros(unique_times.size, dtype=bool)
    for start in np.flatnonzero(ok):
        in_burst[start : start + min_spikes] = True

    intervals: list[tuple[float, float]] = []
    channel_counts: list[int] = []
    idx = 0
    while idx < in_burst.size:
        if not in_burst[idx]:
            idx += 1
            continue
        start_idx = idx
        while idx + 1 < in_burst.size and in_burst[idx + 1]:
            idx += 1
        end_idx = idx
        start_t = float(unique_times[start_idx])
        end_t = float(unique_times[end_idx])
        tolerance = 0.5 / fs + np.finfo(float).eps
        event = (times >= start_t - tolerance) & (times <= end_t + tolerance)
        n_event_channels = int(np.unique(channels[event]).size)
        if int(event.sum()) >= min_spikes and n_event_channels >= min_channels:
            intervals.append((float(times[event].min()), float(times[event].max())))
            channel_counts.append(n_event_channels)
        idx += 1

    if not intervals:
        return BurstResult(empty.intervals, threshold, empty.spike_mask, empty.n_channels)
    interval_array = np.asarray(intervals, dtype=float)
    spike_mask = np.zeros(times.size, dtype=bool)
    tolerance = 0.5 / fs + np.finfo(float).eps
    for start_t, end_t in interval_array:
        spike_mask |= (times >= start_t - tolerance) & (times <= end_t + tolerance)
    return BurstResult(
        intervals=interval_array,
        threshold_s=threshold,
        spike_mask=spike_mask,
        n_channels=np.asarray(channel_counts, dtype=int),
    )
