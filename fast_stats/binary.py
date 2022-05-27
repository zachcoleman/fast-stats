from typing import Union

import numpy as np

from ._fast_stats_ext import (
    _binary_f1_score_reqs,
    _binary_precision_reqs,
    _binary_recall_reqs,
)

Result = Union[None, float]


def _precision(tp: int, tp_fp: int, zero_division: str = "none") -> Result:
    if tp_fp == 0:
        if zero_division == "none":
            return None
        elif zero_division == "zero":
            return 0.0
    return tp / tp_fp


def _recall(tp: int, tp_fn: int, zero_division: str = "none") -> Result:
    if tp_fn == 0:
        if zero_division == "none":
            return None
        elif zero_division == "zero":
            return 0.0
    return tp / tp_fn


def binary_precision(
    y_true: np.ndarray, y_pred: np.ndarray, zero_division: str = "none"
) -> Result:
    assert y_true.shape == y_pred.shape, "y_true and y_pred must be same shape"
    assert all(
        [
            isinstance(y_pred, np.ndarray),
            isinstance(y_true, np.ndarray),
        ]
    ), "y_true and y_pred must be numpy arrays"

    tp, tp_fp, _ = _binary_precision_reqs(y_true, y_pred)
    return _precision(tp, tp_fp, zero_division)


def binary_recall(
    y_true: np.ndarray, y_pred: np.ndarray, zero_division: str = "none"
) -> Result:
    assert y_true.shape == y_pred.shape, "y_true and y_pred must be same shape"
    assert all(
        [
            isinstance(y_pred, np.ndarray),
            isinstance(y_true, np.ndarray),
        ]
    ), "y_true and y_pred must be numpy arrays"

    tp, tp_fn, _ = _binary_recall_reqs(y_true, y_pred)
    return _recall(tp, tp_fn, zero_division)


def binary_f1_score(
    y_true: np.ndarray, y_pred: np.ndarray, zero_division: str = "none"
) -> Result:
    assert y_true.shape == y_pred.shape, "y_true and y_pred must be same shape"
    assert all(
        [
            isinstance(y_pred, np.ndarray),
            isinstance(y_true, np.ndarray),
        ]
    ), "y_true and y_pred must be numpy arrays"

    tp, tp_fp, tp_fn = _binary_f1_score_reqs(y_true, y_pred)
    p, r = _precision(tp, tp_fp, "zero"), _recall(tp, tp_fn, "zero")

    if p + r == 0:
        if zero_division == "none":
            return None
        elif zero_division == "zero":
            return 0.0

    return 2 * p * r / (p + r)
