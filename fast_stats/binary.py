from enum import Enum
from typing import Dict, Tuple, Union

import numpy as np

from ._fast_stats_ext import (
    _binary_f1_score_reqs,
    _binary_precision_reqs,
    _binary_recall_reqs,
)
from .exceptions import ShapeError

Result = Union[None, float]


class ZeroDivision(Enum):
    ZERO = "zero"
    NONE = "none"


def _precision(
    tp: int, tp_fp: int, zero_division: ZeroDivision = ZeroDivision.NONE
) -> Result:
    if tp_fp == 0:
        if zero_division == ZeroDivision.NONE:
            return None
        elif zero_division == ZeroDivision.ZERO:
            return 0.0
    return tp / tp_fp


def _recall(
    tp: int, tp_fn: int, zero_division: ZeroDivision = ZeroDivision.NONE
) -> Result:
    if tp_fn == 0:
        if zero_division == ZeroDivision.NONE:
            return None
        elif zero_division == ZeroDivision.ZERO:
            return 0.0
    return tp / tp_fn


def binary_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_division: ZeroDivision = ZeroDivision.NONE,
) -> Result:
    """Binary calculation for precision

    Args:
        y_true (np.ndarray): array of true values (must be bool or int types)
        y_pred (np.ndarray): array of pred values (must be bool or int types)
        zero_division (str): determines how to handle division by zero
    Returns:
        Result: None or float depending on values and zero division
    """
    if not all(
        [
            isinstance(y_pred, np.ndarray),
            isinstance(y_true, np.ndarray),
        ]
    ):
        raise TypeError("y_true and y_pred must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ShapeError("y_true and y_pred must be same shape")
    zero_division = ZeroDivision(zero_division)

    tp, tp_fp, _ = _binary_precision_reqs(y_true, y_pred)
    return _precision(tp, tp_fp, zero_division)


def binary_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_division: ZeroDivision = ZeroDivision.NONE,
) -> Result:
    """Binary calculation for recall

    Args:
        y_true (np.ndarray): array of true values (must be bool or int types)
        y_pred (np.ndarray): array of pred values (must be bool or int types)
        zero_division (str): determines how to handle division by zero
    Returns:
        Result: None or float depending on values and zero division
    """
    if not all(
        [
            isinstance(y_pred, np.ndarray),
            isinstance(y_true, np.ndarray),
        ]
    ):
        raise TypeError("y_true and y_pred must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ShapeError("y_true and y_pred must be same shape")
    zero_division = ZeroDivision(zero_division)

    tp, tp_fn, _ = _binary_recall_reqs(y_true, y_pred)
    return _recall(tp, tp_fn, zero_division)


def binary_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_division: ZeroDivision = ZeroDivision.NONE,
) -> Result:
    """Binary calculation for f1 score

    Args:
        y_true (np.ndarray): array of true values (must be bool or int types)
        y_pred (np.ndarray): array of pred values (must be bool or int types)
        zero_division (str): determines how to handle division by zero
    Returns:
        Result: None or float depending on values and zero division
    """
    if not all(
        [
            isinstance(y_pred, np.ndarray),
            isinstance(y_true, np.ndarray),
        ]
    ):
        raise TypeError("y_true and y_pred must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ShapeError("y_true and y_pred must be same shape")
    zero_division = ZeroDivision(zero_division)

    tp, tp_fp, tp_fn = _binary_f1_score_reqs(y_true, y_pred)
    p, r = _precision(tp, tp_fp, ZeroDivision.ZERO), _recall(
        tp, tp_fn, ZeroDivision.ZERO
    )

    if p + r == 0:  # type: ignore
        if zero_division == ZeroDivision.NONE:
            return None
        elif zero_division == ZeroDivision.ZERO:
            return 0.0
    return 2 * p * r / (p + r)  # type: ignore


def binary_tp_fp_fn(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[int, int, int]:
    """Binary calculations for TP, FP, and FN

    Args:
        y_true (np.ndarray): array of true values (must be bool or int types)
        y_pred (np.ndarray): array of pred values (must be bool or int types)
    Returns:
        Tuple[int]: counts for TP, FP, and FN
    """
    if not all(
        [
            isinstance(y_pred, np.ndarray),
            isinstance(y_true, np.ndarray),
        ]
    ):
        raise TypeError("y_true and y_pred must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ShapeError("y_true and y_pred must be same shape")

    tp, tp_fp, tp_fn = _binary_f1_score_reqs(y_true, y_pred)
    fp, fn = tp_fp - tp, tp_fn - tp
    return tp, fp, fn


def binary_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_division: ZeroDivision = ZeroDivision.NONE,
) -> Dict[str, Result]:
    """Binary calculations for precision, recall and f1-score

    Args:
        y_true (np.ndarray): array of true values (must be bool or int types)
        y_pred (np.ndarray): array of pred values (must be bool or int types)
    Returns:
        Dict[str, Result]: stats for precision, recall and f1-score
    """
    if not all(
        [
            isinstance(y_pred, np.ndarray),
            isinstance(y_true, np.ndarray),
        ]
    ):
        raise TypeError("y_true and y_pred must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ShapeError("y_true and y_pred must be same shape")
    zero_division = ZeroDivision(zero_division)

    tp, tp_fp, tp_fn = _binary_f1_score_reqs(y_true, y_pred)
    p, r = _precision(tp, tp_fp, zero_division), _recall(tp, tp_fn, zero_division)
    stats = dict({"precision": p, "recall": r})

    # convert p and/or r to 0 if None
    if p is None:
        p = 0.0
    if r is None:
        r = 0.0

    # handle 0 cases
    if p + r == 0:
        if zero_division == ZeroDivision.NONE:
            f1 = None
        elif zero_division == ZeroDivision.ZERO:
            f1 = 0.0
    else:
        f1 = 2 * p * r / (p + r)

    stats.update({"f1-score": f1})

    return stats
