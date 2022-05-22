![GitHub Workflow Status (branch)](https://img.shields.io/github/workflow/status/zachcoleman/fast-stats/tests/main)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fast-stats)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/fast-stats)
[![License](https://img.shields.io/badge/license-Apache2.0-green)](./LICENSE)
![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)

# fast-stats
`fast-stats` is a fast and simple library for calculating simple statistics like precision, recall, and f1-score. The library uses Python to wrap performant 

The project was developed using the [maturin](https://maturin.rs) framework. 

This project is still in development.

### Limitations
- The Rust code does not enable releasing the GIL
- Only binary operations are supported at this time

### Running Tests
Tests are run with `pytest`:
```shell
pytest --cov=fast_stats tests
```
