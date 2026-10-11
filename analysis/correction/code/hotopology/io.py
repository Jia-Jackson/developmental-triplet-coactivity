from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class Recording:
    recording: str
    div: int
    plate: int
    well: str
    spike_times_s: np.ndarray
    channel_indices: np.ndarray
    duration_s: float
    fs: float
    source_path: Path


def _scalar(value: object) -> object:
    current = value
    while isinstance(current, np.ndarray) and current.size == 1:
        current = current.reshape(-1)[0]
    return current


def _text(value: object) -> str:
    current = _scalar(value)
    if isinstance(current, bytes):
        return current.decode("utf-8")
    if isinstance(current, np.ndarray):
        return "".join(str(x) for x in current.reshape(-1))
    return str(current)


def load_recording_mat(path: str | Path) -> Recording:
    """Load one MATLAB v7 well file and convert channel indices to zero-based."""
    source = Path(path)
    mat = loadmat(source, squeeze_me=True, struct_as_record=False)
    times = np.asarray(mat["spike_time_s"], dtype=float).reshape(-1)
    channels = np.asarray(mat["spike_channel_index"], dtype=int).reshape(-1) - 1
    if times.size != channels.size:
        raise ValueError(f"mismatched spike arrays in {source}")
    valid = np.isfinite(times) & (channels >= 0) & (channels < 64)
    times = times[valid]
    channels = channels[valid]
    order = np.argsort(times, kind="stable")
    match = re.fullmatch(
        r"(?P<recording>WT_DIV(?P<div>\d+)_P(?P<plate>\d+)_(?P<well>[AB]\d))_arrays",
        source.stem,
    )
    if match:
        recording = match.group("recording")
        div = int(match.group("div"))
        plate = int(match.group("plate"))
        well = match.group("well")
    else:
        recording = _text(mat["recording"])
        div = int(_scalar(mat["divValue"]))
        plate = int(_scalar(mat["plateIndex"]))
        well = _text(mat["wellLabel"])
    return Recording(
        recording=recording,
        div=div,
        plate=plate,
        well=well,
        spike_times_s=times[order],
        channel_indices=channels[order],
        duration_s=float(_scalar(mat["duration_s"])),
        fs=float(_scalar(mat["fs"])),
        source_path=source,
    )
