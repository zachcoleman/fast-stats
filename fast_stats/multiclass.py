from enum import Enum
from functools import partial
from typing import Dict, List, Union

import numpy as np

from ._fast_stats_ext import _f1_score, _precision, _recall, _unique

Result = Union[None, float, np.ndarray]


class ZeroDivision(Enum):
    ZERO = "zero"
    NONE = "none"


class AverageType(Enum):
    NONE = "none"
    MICRO = "micro"
    MACRO = "macro"


def precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Union[List, np.ndarray] = None,
    zero_division: ZeroDivision = ZeroDivision.NONE,
    average: AverageType = AverageType.NONE,
) -> Result:
    """Multi-class calculation of precision

    Args:
        y_true (np.ndarray): array of true values (must be bool or int types)
        y_pred (np.ndarray): array of pred values (must be bool or int types)
        labels (optional | list or np.ndarray):
            labels to calculate confusion matrix for (must be bool or int types)
        zero_division (optional | str): strategy to handle division by 0
        average (optional | str): strategy for averaging across classes
    Returns:
        precision (np.ndarray): 1D array or scalar values depending on averaging
    """
    assert y_true.shape == y_pred.shape, "y_true and y_pred must be same shape"
    assert all(
        [
            isinstance(y_pred, np.ndarray),
            isinstance(y_true, np.ndarray),
        ]
    ), "y_true and y_pred must be numpy arrays"
    zero_division = ZeroDivision(zero_division)
    average = AverageType(average)

    if labels is None:
        labels = np.array(
            sorted(list(_unique(np.concatenate([y_true, y_pred])))), dtype=y_true.dtype
        )
    elif isinstance(labels, list):
        labels = np.array(labels, dtype=y_true.dtype)

    x = _precision(y_true, y_pred, labels)

    if zero_division == ZeroDivision.NONE:
        zero_handle = partial(
            np.nan_to_num, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan
        )
    elif zero_division == zero_division.ZERO:
        zero_handle = partial(
            np.nan_to_num, copy=False, nan=0.0, posinf=0.0, neginf=0.0
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        if average == AverageType.NONE:
            return zero_handle(x[:, 0] / x[:, 1])
        elif average == AverageType.MICRO:
            return zero_handle(x[:, 0].sum() / x[:, 1].sum())
        elif average == AverageType.MACRO:
            return np.nanmean(zero_handle(x[:, 0] / x[:, 1]))


def recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Union[List, np.ndarray] = None,
    zero_division: ZeroDivision = ZeroDivision.NONE,
    average: AverageType = AverageType.NONE,
) -> Result:
    """Multi-class calculation of recall

    Args:
        y_true (np.ndarray): array of true values (must be bool or int types)
        y_pred (np.ndarray): array of pred values (must be bool or int types)
        labels (optional | list or np.ndarray):
            labels to calculate confusion matrix for (must be bool or int types)
        zero_division (optional | str): strategy to handle division by 0
        average (optional | str): strategy for averaging across classes
    Returns:
        recall (np.ndarray): 1D array or scalar values depending on averaging
    """
    assert y_true.shape == y_pred.shape, "y_true and y_pred must be same shape"
    assert all(
        [
            isinstance(y_pred, np.ndarray),
            isinstance(y_true, np.ndarray),
        ]
    ), "y_true and y_pred must be numpy arrays"
    zero_division = ZeroDivision(zero_division)
    average = AverageType(average)

    if labels is None:
        labels = np.array(
            sorted(list(_unique(np.concatenate([y_true, y_pred])))), dtype=y_true.dtype
        )
    elif isinstance(labels, list):
        labels = np.array(labels, dtype=y_true.dtype)

    x = _recall(y_true, y_pred, labels)

    if zero_division == ZeroDivision.NONE:
        zero_handle = partial(
            np.nan_to_num, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan
        )
    elif zero_division == zero_division.ZERO:
        zero_handle = partial(
            np.nan_to_num, copy=False, nan=0.0, posinf=0.0, neginf=0.0
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        if average == AverageType.NONE:
            return zero_handle(x[:, 0] / x[:, 1])
        elif average == AverageType.MICRO:
            return zero_handle(x[:, 0].sum() / x[:, 1].sum())
        elif average == AverageType.MACRO:
            return np.nanmean(zero_handle(x[:, 0] / x[:, 1]))


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Union[List, np.ndarray] = None,
    zero_division: ZeroDivision = ZeroDivision.NONE,
    average: AverageType = AverageType.NONE,
) -> Result:
    """Multi-class calculation of f1 score

    Args:
        y_true (np.ndarray): array of true values (must be bool or int types)
        y_pred (np.ndarray): array of pred values (must be bool or int types)
        labels (optional | list or np.ndarray):
            labels to calculate confusion matrix for (must be bool or int types)
        zero_division (optional | str): strategy to handle division by 0
        average (optional | str): strategy for averaging across classes
    Returns:
        f1 score (np.ndarray): 1D array or scalar values depending on averaging
    """
    assert y_true.shape == y_pred.shape, "y_true and y_pred must be same shape"
    assert all(
        [
            isinstance(y_pred, np.ndarray),
            isinstance(y_true, np.ndarray),
        ]
    ), "y_true and y_pred must be numpy arrays"
    zero_division = ZeroDivision(zero_division)
    average = AverageType(average)

    if labels is None:
        labels = np.array(
            sorted(list(_unique(np.concatenate([y_true, y_pred])))), dtype=y_true.dtype
        )
    elif isinstance(labels, list):
        labels = np.array(labels, dtype=y_true.dtype)

    x = _f1_score(y_true, y_pred, labels)

    if zero_division == ZeroDivision.NONE:
        zero_handle = partial(
            np.nan_to_num, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan
        )
    elif zero_division == zero_division.ZERO:
        zero_handle = partial(
            np.nan_to_num, copy=False, nan=0.0, posinf=0.0, neginf=0.0
        )

    def f1_from_ext(x, y, z):
        p, r = x / y, x / z
        return 2 * p * r / (p + r)

    with np.errstate(divide="ignore", invalid="ignore"):
        if average == AverageType.NONE:
            return zero_handle(f1_from_ext(x[:, 0], x[:, 1], x[:, 2]))
        elif average == AverageType.MICRO:
            return zero_handle(f1_from_ext(x[:, 0].sum(), x[:, 1].sum(), x[:, 2].sum()))
        elif average == AverageType.MACRO:
            return np.nanmean(f1_from_ext(x[:, 0], x[:, 1], x[:, 2]))


def stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Union[List, np.ndarray] = None,
    zero_division: ZeroDivision = ZeroDivision.NONE,
    average: AverageType = AverageType.NONE,
) -> Dict[str, Result]:
    """Multi-class calculation of f1 score

    Args:
        y_true (np.ndarray): array of true values (must be bool or int types)
        y_pred (np.ndarray): array of pred values (must be bool or int types)
        labels (optional | list or np.ndarray):
            labels to calculate confusion matrix for (must be bool or int types)
        zero_division (optional | str): strategy to handle division by 0
        average (optional | str): strategy for averaging across classes
    Returns:
        Dict[str, Result]: dictionary of strings to 1D array or scalar values
            depending on averaging
    """
    assert y_true.shape == y_pred.shape, "y_true and y_pred must be same shape"
    assert all(
        [
            isinstance(y_pred, np.ndarray),
            isinstance(y_true, np.ndarray),
        ]
    ), "y_true and y_pred must be numpy arrays"
    zero_division = ZeroDivision(zero_division)
    average = AverageType(average)

    if labels is None:
        labels = np.array(
            sorted(list(_unique(np.concatenate([y_true, y_pred])))), dtype=y_true.dtype
        )
    elif isinstance(labels, list):
        labels = np.array(labels, dtype=y_true.dtype)

    x = _f1_score(y_true, y_pred, labels)

    if zero_division == ZeroDivision.NONE:
        zero_handle = partial(
            np.nan_to_num, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan
        )
    elif zero_division == zero_division.ZERO:
        zero_handle = partial(
            np.nan_to_num, copy=False, nan=0.0, posinf=0.0, neginf=0.0
        )

    def f1_from_ext(x, y, z):
        p, r = x / y, x / z
        return 2 * p * r / (p + r)

    stats = dict()

    # precision
    with np.errstate(divide="ignore", invalid="ignore"):
        if average == AverageType.NONE:
            stats.update({"precision": zero_handle(x[:, 0] / x[:, 1])})
        elif average == AverageType.MICRO:
            stats.update({"precision": zero_handle(x[:, 0].sum() / x[:, 1].sum())})
        elif average == AverageType.MACRO:
            stats.update({"precision": np.nanmean(zero_handle(x[:, 0] / x[:, 1]))})

    # recall
    with np.errstate(divide="ignore", invalid="ignore"):
        if average == AverageType.NONE:
            stats.update({"recall": zero_handle(x[:, 0] / x[:, 2])})
        elif average == AverageType.MICRO:
            stats.update({"recall": zero_handle(x[:, 0].sum() / x[:, 2].sum())})
        elif average == AverageType.MACRO:
            stats.update({"recall": np.nanmean(zero_handle(x[:, 0] / x[:, 2]))})

    # f1-score
    with np.errstate(divide="ignore", invalid="ignore"):
        if average == AverageType.NONE:
            stats.update(
                {"f1-score": zero_handle(f1_from_ext(x[:, 0], x[:, 1], x[:, 2]))}
            )
        elif average == AverageType.MICRO:
            stats.update(
                {
                    "f1-score": zero_handle(
                        f1_from_ext(x[:, 0].sum(), x[:, 1].sum(), x[:, 2].sum())
                    )
                }
            )
        elif average == AverageType.MACRO:
            stats.update(
                {"f1-score": np.nanmean(f1_from_ext(x[:, 0], x[:, 1], x[:, 2]))}
            )

    return stats
