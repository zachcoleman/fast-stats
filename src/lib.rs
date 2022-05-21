use ndarray::*;
use numpy::*;
use pyo3::prelude::*;
use std::iter::zip;

/// TP and FP
#[pyfunction]
fn _tp_fp_fn_tn(_py: Python<'_>, actual: &PyArrayDyn<isize>, pred: &PyArrayDyn<isize>) -> (usize, usize, usize, usize) {
    let mut _tp = 0;
    let mut _fp: usize = 0;
    let mut _fn: usize = 0;
    let mut _tn: usize = 0;

    for (y_pred, y_actual) in zip(pred.readonly().as_array().iter(), actual.readonly().as_array().iter()){
        if *y_pred == 1 && *y_actual == 1{
            _tp = _tp + 1;
        } else if *y_pred == 1 && *y_actual == 0{
            _fp = _fp + 1;
        } else if *y_pred == 0 && *y_actual == 1{
            _fn = _fn + 1;
        } else{
            _tn = _tn + 1;
        }
    }
    (_tp, _fp, _fn, _tn)
}

/// A Python module implemented in Rust.
#[pymodule]
fn fast_stats(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_tp_fp_fn_tn, m)?)?;
    Ok(())
}