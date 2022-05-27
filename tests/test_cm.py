import numpy as np
import pytest

from fast_stats import confusion_matrix


@pytest.mark.parametrize(
    "y_true,y_pred,expected",
    [
        (
            np.ones(4, dtype=np.uint64),
            np.ones(4, dtype=np.uint64),
            np.array([[4]]),
        ),  # all one value
        (
            np.array(
                [
                    [1, 2],
                    [1, 2],
                ]
            ),
            np.array(
                [
                    [1, 1],
                    [2, 2],
                ]
            ),
            np.ones((2, 2)),
        ),  # 2x2
        (
            np.array(
                [
                    [1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3],
                ]
            ),
            np.array(
                [
                    [1, 2, 1],
                    [1, 1, 1],
                    [1, 1, 3],
                ]
            ),
            np.array(
                [
                    [3, 0, 0],
                    [2, 1, 0],
                    [2, 0, 1],
                ]
            ),
        ),  # 3x3
    ],
)
def test_precision(y_true, y_pred, expected):
    assert np.allclose(confusion_matrix(y_true, y_pred), expected)
