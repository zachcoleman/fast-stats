from enum import Enum
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from ._fast_stats_ext import _f1_score, _precision, _recall, _unique
from .exceptions import ShapeError

Result = Union[None, float, np.floating, np.ndarray]


class ZeroDivision(Enum):
    ZERO = "zero"
    NONE = "none"


class AverageType(Enum):
    NONE = "none"
    MICRO = "micro"
    MACRO = "macro"


def _get_zero_handler(
    zero_division: ZeroDivision,
) -> Callable[[Union[float, np.ndarray]], np.ndarray]:
    if zero_division == ZeroDivision.NONE:

        def zero_handle(x: Union[float, np.ndarray]) -> np.ndarray:
            return np.where(np.isfinite(x), x, np.nan)

    elif zero_division == zero_division.ZERO:

        def zero_handle(x: Union[float, np.ndarray]) -> np.ndarray:
            return np.where(np.isfinite(x), x, 0.0)

    return zero_handle


def precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[Union[List, np.ndarray]] = None,
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
    average = AverageType(average)

    if labels is None:
        labels = np.array(
            sorted(list(_unique(y_true).union(_unique(y_pred)))), dtype=y_true.dtype
        )
    elif isinstance(labels, list):
        labels = np.array(labels, dtype=y_true.dtype)

    x = _precision(y_true, y_pred, labels)
    zero_handle = _get_zero_handler(zero_division)
    with np.errstate(divide="ignore", invalid="ignore"):
        if average == AverageType.NONE:
            return zero_handle(x[:, 0] / x[:, 1])
        elif average == AverageType.MICRO:
            return zero_handle(x[:, 0].sum() / x[:, 1].sum()).item()
        elif average == AverageType.MACRO:
            return np.nanmean(zero_handle(x[:, 0] / x[:, 1])).item()
        return None  # pragma: no cover


def recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[Union[List, np.ndarray]] = None,
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
    average = AverageType(average)

    if labels is None:
        labels = np.array(
            sorted(list(_unique(y_true).union(_unique(y_pred)))), dtype=y_true.dtype
        )
    elif isinstance(labels, list):
        labels = np.array(labels, dtype=y_true.dtype)

    x = _recall(y_true, y_pred, labels)
    zero_handle = _get_zero_handler(zero_division)
    with np.errstate(divide="ignore", invalid="ignore"):
        if average == AverageType.NONE:
            return zero_handle(x[:, 0] / x[:, 1])
        elif average == AverageType.MICRO:
            return zero_handle(x[:, 0].sum() / x[:, 1].sum()).item()
        elif average == AverageType.MACRO:
            return np.nanmean(zero_handle(x[:, 0] / x[:, 1])).item()
        return None  # pragma: no cover


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[Union[List, np.ndarray]] = None,
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
    average = AverageType(average)

    if labels is None:
        labels = np.array(
            sorted(list(_unique(y_true).union(_unique(y_pred)))), dtype=y_true.dtype
        )
    elif isinstance(labels, list):
        labels = np.array(labels, dtype=y_true.dtype)

    x = _f1_score(y_true, y_pred, labels)
    zero_handle = _get_zero_handler(zero_division)

    def f1_from_ext(x, y, z):
        p, r = x / y, x / z
        return 2 * p * r / (p + r)

    with np.errstate(divide="ignore", invalid="ignore"):
        if average == AverageType.NONE:
            return zero_handle(f1_from_ext(x[:, 0], x[:, 1], x[:, 2]))
        elif average == AverageType.MICRO:
            return zero_handle(
                f1_from_ext(x[:, 0].sum(), x[:, 1].sum(), x[:, 2].sum())
            ).item()
        elif average == AverageType.MACRO:
            return np.nanmean(f1_from_ext(x[:, 0], x[:, 1], x[:, 2])).item()
        return None  # pragma: no cover


def stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[Union[List, np.ndarray]] = None,
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
    average = AverageType(average)

    if labels is None:
        labels = np.array(
            sorted(list(_unique(y_true).union(_unique(y_pred)))), dtype=y_true.dtype
        )
    elif isinstance(labels, list):
        labels = np.array(labels, dtype=y_true.dtype)

    x = _f1_score(y_true, y_pred, labels)
    zero_handle = _get_zero_handler(zero_division)

    def f1_from_ext(x, y, z):
        p, r = x / y, x / z
        return 2 * p * r / (p + r)

    stats: Dict[str, Result] = dict()

    # precision
    with np.errstate(divide="ignore", invalid="ignore"):
        if average == AverageType.NONE:
            stats.update({"precision": zero_handle(x[:, 0] / x[:, 1])})
        elif average == AverageType.MICRO:
            stats.update(
                {"precision": zero_handle(x[:, 0].sum() / x[:, 1].sum()).item()}
            )
        elif average == AverageType.MACRO:
            stats.update(
                {"precision": np.nanmean(zero_handle(x[:, 0] / x[:, 1])).item()}
            )

    # recall
    with np.errstate(divide="ignore", invalid="ignore"):
        if average == AverageType.NONE:
            stats.update({"recall": zero_handle(x[:, 0] / x[:, 2])})
        elif average == AverageType.MICRO:
            stats.update({"recall": zero_handle(x[:, 0].sum() / x[:, 2].sum()).item()})
        elif average == AverageType.MACRO:
            stats.update({"recall": np.nanmean(zero_handle(x[:, 0] / x[:, 2])).item()})

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
                    ).item()
                }
            )
        elif average == AverageType.MACRO:
            stats.update(
                {"f1-score": np.nanmean(f1_from_ext(x[:, 0], x[:, 1], x[:, 2])).item()}
            )

    # for none average add labels and support
    if average == AverageType.NONE:
        stats.update({"labels": labels})
        stats.update({"support": x[:, 2]})  # support total y_true (TP + FN)

    return stats
