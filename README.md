![GitHub Workflow Status (branch)](https://img.shields.io/github/workflow/status/zachcoleman/fast-stats/tests/main)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fast-stats)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/fast-stats)
[![License](https://img.shields.io/badge/license-Apache2.0-green)](./LICENSE)

# fast-stats
`fast-stats` is a fast and simple library for calculating basic statistics such as: precision, recall, and f1-score. `fast-stats` supports the calculation of confusion matrices.

The project was developed using the [maturin](https://maturin.rs) framework. 

This project is still in development.

## Installation
From PyPi:
```shell
pip install fast-stats
```
Build from source
```
maturin build -r -i=path/to/python
pip install .../fast-stats/target/wheels/<whl file name>.whl
```

## Limitations
- The Rust code does not enable releasing the GIL

## Running Tests
Tests are run with `pytest`.
