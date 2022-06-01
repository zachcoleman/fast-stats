import numpy as np
import pytest
from dictdiffer import diff

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


@pytest.mark.parametrize(
    "y_true,y_pred,expected",
    [
        (
            np.ones(4, dtype=np.uint64),
            np.zeros(4, dtype=np.uint64),
            (0, 0, 4),
        ),  # all FN
        (np.ones(4, dtype=np.uint64), np.ones(4, dtype=np.uint64), (4, 0, 0)),  # all TP
        (
            np.zeros(4, dtype=np.uint64),
            np.zeros(4, dtype=np.uint64),
            (0, 0, 0),
        ),  # No TP & No FP, & No FN
        (
            np.ones(4, dtype=np.uint64),
            np.array([1, 0, 0, 0], dtype=np.uint64),
            (1, 0, 3),
        ),  # 1 TP & 3 FN
        (
            np.array([1, 1, 0, 0], dtype=np.uint64),
            np.array([0, 1, 1, 0], dtype=np.uint64),
            (1, 1, 1),
        ),  # 1 TP, 1 FP, & 1 FN
    ],
)
def test_tpfpfn(y_true, y_pred, expected):
    assert np.allclose(fast_stats.binary_tp_fp_fn(y_true, y_pred), expected)


@pytest.mark.parametrize(
    "y_true,y_pred,zero_division,expected",
    [
        (
            np.ones(4, dtype=np.uint64),
            np.zeros(4, dtype=np.uint64),
            "zero",
            {"precision": 0.0, "recall": 0.0, "f1-score": 0.0},
        ),  # all FN
        (
            np.ones(4, dtype=np.uint64),
            np.ones(4, dtype=np.uint64),
            "zero",
            {"precision": 1.0, "recall": 1.0, "f1-score": 1.0},
        ),  # all TP
        (
            np.zeros(4, dtype=np.uint64),
            np.zeros(4, dtype=np.uint64),
            "none",
            {"precision": None, "recall": None, "f1-score": None},
        ),  # No TP & No FP, & No FN
        (
            np.ones(4, dtype=np.uint64),
            np.array([1, 0, 0, 0], dtype=np.uint64),
            "none",
            {
                "precision": 1.0,
                "recall": 0.25,
                "f1-score": 2 * (1 / 4 * 1.0) / (1 / 4 + 1.0),
            },
        ),  # 1 TP & 3 FN
        (
            np.array([1, 1, 0, 0], dtype=np.uint64),
            np.array([0, 1, 1, 0], dtype=np.uint64),
            "none",
            {"precision": 0.5, "recall": 0.5, "f1-score": 0.5},
        ),
    ],
)
def test_stats(y_true, y_pred, zero_division, expected):
    assert (
        len(
            list(diff(fast_stats.binary_stats(y_true, y_pred, zero_division), expected))
        )
        == 0
    )
