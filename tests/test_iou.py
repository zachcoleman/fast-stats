import numpy as np
import pytest

import fast_stats


@pytest.mark.parametrize(
    "arr1,arr2,zero_division,expected",
    [
        (
            np.zeros(4, dtype=np.uint64),
            np.ones(4, dtype=np.uint64),
            "zero",
            0.0,
        ),
        (
            np.ones(4, dtype=np.uint64),
            np.ones(4, dtype=np.uint64),
            "zero",
            1.0,
        ),
        (
            np.zeros(4, dtype=np.uint64),
            np.zeros(4, dtype=np.uint64),
            "zero",
            0.0,
        ),
        (
            np.zeros(4, dtype=np.uint64),
            np.zeros(4, dtype=np.uint64),
            "none",
            None,
        ),
        (
            np.ones(4, dtype=np.uint64),
            np.array([1, 0, 0, 0], dtype=np.uint64),
            "none",
            0.25,
        ),
        (
            np.array([1, 1, 0, 0], dtype=np.uint64),
            np.array([1, 0, 1, 0], dtype=np.uint64),
            "none",
            1.0 / 3.0,
        ),
    ],
)
def test_iou(arr1, arr2, zero_division, expected):
    assert fast_stats.iou(arr1, arr2, zero_division) == expected
