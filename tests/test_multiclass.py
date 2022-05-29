import numpy as np
import pytest

import fast_stats


@pytest.mark.parametrize(
    "y_true,y_pred,kwargs,expected",
    [
        (
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    1.0,
                    2.0,
                    3.0,
                ],
                dtype=np.uint64,
            ),
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    1.0,
                    2.0,
                    3.0,
                ],
                dtype=np.uint64,
            ),
            {},
            np.array([1.0, 1.0, 1.0]),
        ),  # perfect
        (
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    1.0,
                    2.0,
                    3.0,
                ],
                dtype=np.uint64,
            ),
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    2.0,
                    3.0,
                    1.0,
                ],
                dtype=np.uint64,
            ),
            {"labels": [1, 2, 3]},
            np.array([0.5, 0.5, 0.5]),
        ),  # 50%
        (
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    1.0,
                    2.0,
                    3.0,
                ],
                dtype=np.uint64,
            ),
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    2.0,
                    1.0,
                    3.0,
                ],
                dtype=np.uint64,
            ),
            {"labels": [1, 2]},
            np.array([0.5, 0.5]),
        ),  # 50% subset
        (
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    1.0,
                    2.0,
                    3.0,
                ],
                dtype=np.uint64,
            ),
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    2.0,
                    3.0,
                    1.0,
                ],
                dtype=np.uint64,
            ),
            {"average": "micro", "zero_division": "none"},
            np.array([0.5]),
        ),  # 50% micro
        (
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    1.0,
                    2.0,
                    3.0,
                ],
                dtype=np.uint64,
            ),
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    2.0,
                    3.0,
                    1.0,
                ],
                dtype=np.uint64,
            ),
            {"average": "macro", "zero_division": "zero"},
            np.array([0.5]),
        ),  # 50% macro
    ],
)
def test_precision(y_true, y_pred, kwargs, expected):
    assert np.allclose(fast_stats.precision(y_true, y_pred, **kwargs), expected)


@pytest.mark.parametrize(
    "y_true,y_pred,kwargs,expected",
    [
        (
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    1.0,
                    2.0,
                    3.0,
                ],
                dtype=np.uint64,
            ),
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    1.0,
                    2.0,
                    3.0,
                ],
                dtype=np.uint64,
            ),
            {},
            np.array([1.0, 1.0, 1.0]),
        ),  # perfect
        (
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    1.0,
                    2.0,
                    3.0,
                ],
                dtype=np.uint64,
            ),
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    2.0,
                    3.0,
                    1.0,
                ],
                dtype=np.uint64,
            ),
            {"labels": [1, 2, 3]},
            np.array([0.5, 0.5, 0.5]),
        ),  # 50%
        (
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    1.0,
                    2.0,
                    3.0,
                ],
                dtype=np.uint64,
            ),
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    2.0,
                    1.0,
                    3.0,
                ],
                dtype=np.uint64,
            ),
            {"labels": [1, 2]},
            np.array([0.5, 0.5]),
        ),  # 50% subset
        (
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    1.0,
                    2.0,
                    3.0,
                ],
                dtype=np.uint64,
            ),
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    2.0,
                    3.0,
                    1.0,
                ],
                dtype=np.uint64,
            ),
            {"average": "micro", "zero_division": "none"},
            np.array([0.5]),
        ),  # 50% micro
        (
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    1.0,
                    2.0,
                    3.0,
                ],
                dtype=np.uint64,
            ),
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    2.0,
                    3.0,
                    1.0,
                ],
                dtype=np.uint64,
            ),
            {"average": "macro", "zero_division": "zero"},
            np.array([0.5]),
        ),  # 50% macro
    ],
)
def test_recall(y_true, y_pred, kwargs, expected):
    assert np.allclose(fast_stats.recall(y_true, y_pred, **kwargs), expected)


@pytest.mark.parametrize(
    "y_true,y_pred,kwargs,expected",
    [
        (
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    1.0,
                    2.0,
                    3.0,
                ],
                dtype=np.uint64,
            ),
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    1.0,
                    2.0,
                    3.0,
                ],
                dtype=np.uint64,
            ),
            {},
            np.array([1.0, 1.0, 1.0]),
        ),  # perfect
        (
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    1.0,
                    2.0,
                    3.0,
                ],
                dtype=np.uint64,
            ),
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    2.0,
                    3.0,
                    1.0,
                ],
                dtype=np.uint64,
            ),
            {"labels": [1, 2, 3]},
            np.array([0.5, 0.5, 0.5]),
        ),  # 50%
        (
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    1.0,
                    2.0,
                    3.0,
                ],
                dtype=np.uint64,
            ),
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    2.0,
                    1.0,
                    3.0,
                ],
                dtype=np.uint64,
            ),
            {"labels": [1, 2]},
            np.array([0.5, 0.5]),
        ),  # 50% subset
        (
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    1.0,
                    2.0,
                    3.0,
                ],
                dtype=np.uint64,
            ),
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    2.0,
                    3.0,
                    1.0,
                ],
                dtype=np.uint64,
            ),
            {"average": "micro", "zero_division": "none"},
            np.array([0.5]),
        ),  # 50% micro
        (
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    1.0,
                    2.0,
                    3.0,
                ],
                dtype=np.uint64,
            ),
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    2.0,
                    3.0,
                    1.0,
                ],
                dtype=np.uint64,
            ),
            {"average": "macro", "zero_division": "zero"},
            np.array([0.5]),
        ),  # 50% macro
    ],
)
def test_f1_score(y_true, y_pred, kwargs, expected):
    assert np.allclose(fast_stats.f1_score(y_true, y_pred, **kwargs), expected)
