from enum import Enum
from typing import Union

import numpy as np

from ._fast_stats_ext import _binary_f1_score_reqs

Result = Union[None, float]


class ZeroDivision(Enum):
    ZERO = "zero"
    NONE = "none"


def _iou(
    tp: int, fp: int, fn: int, zero_division: ZeroDivision = ZeroDivision.NONE
) -> Result:
    if tp + fp + fn == 0:
        if zero_division == ZeroDivision.NONE:
            return None
        elif zero_division == ZeroDivision.ZERO:
            return 0.0
    return tp / (tp + fp + fn)


def iou(
    array1: np.ndarray,
    array2: np.ndarray,
    zero_division: ZeroDivision = ZeroDivision.NONE,
) -> Result:
    """Calculation for IoU

    Args:
        array1 (np.ndarray): array of 0/1 values (must be bool or int types)
        array2 (np.ndarray): array of 0/1 values (must be bool or int types)
        zero_division (str): determines how to handle division by zero
    Returns:
        Result: None or float depending on values and zero division
    """
    assert array1.shape == array2.shape, "y_true and y_pred must be same shape"
    assert all(
        [
            isinstance(array1, np.ndarray),
            isinstance(array2, np.ndarray),
        ]
    ), "y_true and y_pred must be numpy arrays"
    zero_division = ZeroDivision(zero_division)

    tp, tp_fp, tp_fn = _binary_f1_score_reqs(array1, array2)
    fp, fn = tp_fp - tp, tp_fn - tp
    return _iou(tp, fp, fn, zero_division)
