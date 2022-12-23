"""
`fast-stats` is a fast and simple library for calculating basic statistics such as:
precision, recall, and f1-score. The library also supports the calculation of confusion
matrices. For examples, please look at the `examples/` folder.

The project was developed using the `maturin` framework.

The main functions are:
- fast_stats.binary.binary_stats
- fast_stats.multiclass.stats
"""

from .binary import (
    binary_f1_score,
    binary_precision,
    binary_recall,
    binary_stats,
    binary_tp_fp_fn,
)
from .confusion_matrix import confusion_matrix
from .iou import iou
from .multiclass import f1_score, precision, recall, stats
