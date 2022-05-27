import numpy as np
import pytest

import fast_stats


@pytest.mark.parametrize(
    "y_true,y_pred,zero_division,expected",
    [
        (
            np.zeros(4, dtype=np.uint64),
            np.ones(4, dtype=np.uint64),
            "zero",
            0,
        ),  # all FP
        (
            np.ones(4, dtype=np.uint64),
            np.ones(4, dtype=np.uint64),
            "zero",
            1.0,
        ),  # all TP
        (
            np.zeros(4, dtype=np.uint64),
            np.zeros(4, dtype=np.uint64),
            "zero",
            0.0,
        ),  # No TP & No FP
        (
            np.zeros(4, dtype=np.uint64),
            np.zeros(4, dtype=np.uint64),
            "none",
            None,
        ),  # No TP & No FP
        (
            np.ones(4, dtype=np.uint64),
            np.array([1, 0, 0, 0], dtype=np.uint64),
            "none",
            1.0,
        ),  # 1 TP & 0 FP
        (
            np.array([1, 0, 0, 0], dtype=np.uint64),
            np.ones(4, dtype=np.uint64),
            "none",
            0.25,
        ),  # 1 TP & 3 FP
        (
            np.zeros(4, dtype=np.uint64),
            np.array([1, 0, 0, 0], dtype=np.uint64),
            "none",
            0.0,
        ),  # 0 TP & 1 TP
    ],
)
def test_precision(y_true, y_pred, zero_division, expected):
    assert fast_stats.binary_precision(y_true, y_pred, zero_division) == expected


@pytest.mark.parametrize(
    "y_true,y_pred,zero_division,expected",
    [
        (
            np.ones(4, dtype=np.uint64),
            np.zeros(4, dtype=np.uint64),
            "zero",
            0,
        ),  # all FN
        (
            np.ones(4, dtype=np.uint64),
            np.ones(4, dtype=np.uint64),
            "zero",
            1.0,
        ),  # all TP
        (
            np.zeros(4, dtype=np.uint64),
            np.zeros(4, dtype=np.uint64),
            "zero",
            0.0,
        ),  # No TP & No FN
        (
            np.zeros(4, dtype=np.uint64),
            np.zeros(4, dtype=np.uint64),
            "none",
            None,
        ),  # No TP & No FN
        (
            np.ones(4, dtype=np.uint64),
            np.array([1, 0, 0, 0], dtype=np.uint64),
            "none",
            0.25,
        ),  # 1 TP & 3 FN
        (
            np.array([1, 0, 0, 0], dtype=np.uint64),
            np.ones(4, dtype=np.uint64),
            "none",
            1.0,
        ),  # 1 TP & 0 FN
        (
            np.array([1, 0, 0, 0], dtype=np.uint64),
            np.zeros(4, dtype=np.uint64),
            "none",
            0.0,
        ),  # 0 TP & 1 FN
    ],
)
def test_recall(y_true, y_pred, zero_division, expected):
    assert fast_stats.binary_recall(y_true, y_pred, zero_division) == expected


@pytest.mark.parametrize(
    "y_true,y_pred,zero_division,expected",
    [
        (
            np.ones(4, dtype=np.uint64),
            np.zeros(4, dtype=np.uint64),
            "zero",
            0.0,
        ),  # all FN
        (
            np.ones(4, dtype=np.uint64),
            np.ones(4, dtype=np.uint64),
            "zero",
            1.0,
        ),  # all TP
        (
            np.zeros(4, dtype=np.uint64),
            np.zeros(4, dtype=np.uint64),
            "none",
            None,
        ),  # No TP & No FP, & No FN
        (
            np.ones(4, dtype=np.uint64),
            np.array([1, 0, 0, 0], dtype=np.uint64),
            "none",
            2 * (1 / 4 * 1.0) / (1 / 4 + 1.0),
        ),  # 1 TP & 3 FN
        (
            np.array([1, 1, 0, 0], dtype=np.uint64),
            np.array([0, 1, 1, 0], dtype=np.uint64),
            "none",
            0.5,
        ),
    ],
)
def test_f1(y_true, y_pred, zero_division, expected):
    if expected is None:
        assert fast_stats.binary_f1_score(y_true, y_pred, zero_division) is None
    else:
        assert np.allclose(
            fast_stats.binary_f1_score(y_true, y_pred, zero_division), expected
        )
