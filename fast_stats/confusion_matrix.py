from typing import List

import numpy as np

from ._fast_stats_ext import _confusion_matrix, _unique


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List = None
) -> np.ndarray:
    assert y_true.shape == y_pred.shape, "y_true and y_pred must be same shape"
    assert all(
        [
            isinstance(y_pred, np.ndarray),
            isinstance(y_true, np.ndarray),
        ]
    ), "y_true and y_pred must be numpy arrays"

    if labels is None:
        labels = sorted(list(_unique(np.concatenate([y_true, y_pred]))))

    return _confusion_matrix(y_true, y_pred, labels)
