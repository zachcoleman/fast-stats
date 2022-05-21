import numpy as np

from .fast_stats import _tp_fp_fn_tn


def precision(y_true: np.ndarray, y_pred: np.ndarray, zero_division: str = "nan"):
    assert y_true.shape == y_pred.shape, "y_true and y_pred must be same shape"
    assert isinstance(y_pred, np.ndarray) and isinstance(
        y_true, np.ndarray
    ), "y_true and y_pred must be numpy arrays"

    tp, fp, _, _ = _tp_fp_fn_tn(y_true, y_pred)

    if tp + fp == 0:
        if zero_division == "nan":
            return np.nan
        elif zero_division == "zero":
            return 0.0

    return tp / (tp + fp)


def recall(y_true: np.ndarray, y_pred: np.ndarray, zero_division: str = "nan"):
    assert y_true.shape == y_pred.shape, "y_true and y_pred must be same shape"
    assert isinstance(y_pred, np.ndarray) and isinstance(
        y_true, np.ndarray
    ), "y_true and y_pred must be numpy arrays"

    tp, _, fn, _ = _tp_fp_fn_tn(y_true, y_pred)

    if tp + fn == 0:
        if zero_division == "nan":
            return np.nan
        elif zero_division == "zero":
            return 0.0

    return tp / (tp + fn)
