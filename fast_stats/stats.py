from typing import Union

import numpy as np

from .fast_stats import _tp_fp_fn_tn

Result = Union[None, float]


def _precision(tp: int, fp: int, zero_division: str = "none") -> Result:
    if tp + fp == 0:
        if zero_division == "none":
            return None
        elif zero_division == "zero":
            return 0.0
    return tp / (tp + fp)


def _recall(tp: int, fn: int, zero_division: str = "none") -> Result:
    if tp + fn == 0:
        if zero_division == "none":
            return None
        elif zero_division == "zero":
            return 0.0
    return tp / (tp + fn)


def precision(
    y_true: np.ndarray, y_pred: np.ndarray, zero_division: str = "none"
) -> Result:
    assert y_true.shape == y_pred.shape, "y_true and y_pred must be same shape"
    assert isinstance(y_pred, np.ndarray) and isinstance(
        y_true, np.ndarray
    ), "y_true and y_pred must be numpy arrays"

    tp, fp, _, _ = _tp_fp_fn_tn(y_true, y_pred)
    return _precision(tp, fp, zero_division)


def recall(
    y_true: np.ndarray, y_pred: np.ndarray, zero_division: str = "none"
) -> Result:
    assert y_true.shape == y_pred.shape, "y_true and y_pred must be same shape"
    assert isinstance(y_pred, np.ndarray) and isinstance(
        y_true, np.ndarray
    ), "y_true and y_pred must be numpy arrays"

    tp, _, fn, _ = _tp_fp_fn_tn(y_true, y_pred)
    return _recall(tp, fn, zero_division)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, zero_division: str = "none"):
    assert y_true.shape == y_pred.shape, "y_true and y_pred must be same shape"
    assert isinstance(y_pred, np.ndarray) and isinstance(
        y_true, np.ndarray
    ), "y_true and y_pred must be numpy arrays"

    tp, fp, fn, _ = _tp_fp_fn_tn(y_true, y_pred)
    p, r = _precision(tp, fp, "0"), _recall(tp, fn, "0")

    if p + r == 0:
        if zero_division == "none":
            return None
        elif zero_division == "zero":
            return 0.0

    return 2 * p * r / (p + r)
