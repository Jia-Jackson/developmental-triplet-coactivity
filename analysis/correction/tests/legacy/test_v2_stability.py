from __future__ import annotations

import numpy as np


def test_temporal_windows_cover_halves_blocks_and_cumulative_duration():
    from hotopology.stability import temporal_windows

    windows = temporal_windows(n_bins=12000, bin_ms=25)
    assert windows["half_first"].start == 0 and windows["half_first"].stop == 6000
    assert windows["half_second"].start == 6000 and windows["half_second"].stop == 12000
    block_names = [name for name in windows if name.startswith("block50s_")]
    assert len(block_names) == 6
    assert windows["cumulative_60s"].stop == 2400
    assert windows["cumulative_300s"].stop == 12000


def test_rebin_binary_preserves_spikes_when_bins_are_merged():
    from hotopology.stability import rebin_binary

    x = np.array([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=np.uint8)
    merged = rebin_binary(x, source_bin_ms=25, target_bin_ms=50)
    assert np.array_equal(merged, np.array([[1, 1], [1, 0]], dtype=np.uint8))


def test_rebin_binary_rejects_noninteger_ratio():
    import pytest

    from hotopology.stability import rebin_binary

    with pytest.raises(ValueError, match="integer multiple"):
        rebin_binary(np.zeros((2, 10), dtype=np.uint8), source_bin_ms=25, target_bin_ms=40)
