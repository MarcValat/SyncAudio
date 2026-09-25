from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import median_filter

from syncaudio.features import _parallel_median_filter


@pytest.mark.parametrize("size", [(1, 17), (17, 1)])
@pytest.mark.parametrize("frames", [100, 5003])
def test_parallel_median_filter_is_bit_identical(size: tuple[int, int], frames: int) -> None:
    mag = np.random.default_rng(0).random((513, frames), dtype=np.float32)
    assert np.array_equal(_parallel_median_filter(mag, size), median_filter(mag, size=size))
