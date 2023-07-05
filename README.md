![GitHub Workflow Status (branch)](https://img.shields.io/github/actions/workflow/status/zachcoleman/fast-stats/tests.yml?branch=main)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fast-stats)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/fast-stats)
[![License](https://img.shields.io/badge/license-Apache2.0-green)](./LICENSE)

# fast-stats
`fast-stats` is a fast and simple library for calculating basic statistics such as: precision, recall, and f1-score. The library also supports the calculation of confusion matrices. For examples, please look at the `examples/` folder.

`fast-stats` is designed and tested against the `scikit-learn` library and provides formatted results to be a drop-in replacement in most cases. There are both generic functions like `fast_stats.precision` that provide significant speedup and specialized binary functions such as `fast_stats.binary_precision` that provide even faster optimal performance for binary data. `fast-stats` was specifically created to provide a limited alternative to `scikit-learn.metrics` for calculating statistics quickly on large multi-dimensional arrays or tensors such as those produced by computer vision models.

[Benchmarks](`examples/benchmarks.ipynb`) show approximately: 
- 100x improvement in binary calculations
- 10x improvement in multiclass calculations
- 2x improvement in computing confusion matrices
- 15% speed-up over an equivalent `numpy` binary calculation

The project was developed using the [maturin](https://maturin.rs) framework. 

See docs here: https://zachcoleman.github.io/fast-stats/

## Installation
From PyPI:
```shell
pip install fast-stats
```

Build from source:
```
maturin build -r -i=path/to/python
pip install .../fast-stats/target/wheels/<whl file name>.whl
```

## Running Tests
Tests are run with `pytest`.
