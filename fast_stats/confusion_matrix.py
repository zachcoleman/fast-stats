from typing import List, Union

import numpy as np

from ._fast_stats_ext import _confusion_matrix, _unique


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Union[List, np.ndarray] = None
) -> np.ndarray:
    """Calculation of confusion matrix

    Args:
        y_true (np.ndarray): array of true values (must be bool or int types)
        y_pred (np.ndarray): array of pred values (must be bool or int types)
        labels (optional | list or np.ndarray):
            labels to calculate confusion matrix for (must be bool or int types)
    Returns:
        confusion matrix (np.ndarray): 2D np.ndarray confusion matrix
    """
    assert y_true.shape == y_pred.shape, "y_true and y_pred must be same shape"
    assert all(
        [
            isinstance(y_pred, np.ndarray),
            isinstance(y_true, np.ndarray),
        ]
    ), "y_true and y_pred must be numpy arrays"

    if labels is None:
        labels = np.array(
            sorted(list(_unique(np.concatenate([y_true, y_pred])))), dtype=y_true.dtype
        )
    elif isinstance(labels, list):
        labels = np.array(labels, dtype=y_true.dtype)

    return _confusion_matrix(y_true, y_pred, labels)
