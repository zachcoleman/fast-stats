use numpy::*;
use pyo3::{exceptions, prelude::*};
use std::iter::zip;
mod binary;

/// Get tp, fp, fn, tn counts by looping
#[pyfunction]
#[pyo3(name = "_tp_fp_fn_tn")]
#[pyo3(text_signature = "(actual: np.ndarray, pred: np.ndarray, /)")]
pub fn tp_fp_fn_tn(
    _py: Python<'_>,
    actual: &PyArrayDyn<usize>,
    pred: &PyArrayDyn<usize>,
) -> (usize, usize, usize, usize) {
    let mut _tp = 0;
    let mut _fp: usize = 0;
    let mut _fn: usize = 0;
    let mut _tn: usize = 0;

    for (y_pred, y_actual) in zip(
        pred.readonly().as_array().iter(),
        actual.readonly().as_array().iter(),
    ) {
        if *y_pred == 1 && *y_actual == 1 {
            _tp = _tp + 1;
        } else if *y_pred == 1 && *y_actual == 0 {
            _fp = _fp + 1;
        } else if *y_pred == 0 && *y_actual == 1 {
            _fn = _fn + 1;
        } else {
            _tn = _tn + 1;
        }
    }
    (_tp, _fp, _fn, _tn)
}

/// A Python module implemented in Rust.
#[pymodule]
fn fast_stats(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tp_fp_fn_tn, m)?)?;
    m.add_function(wrap_pyfunction!(binary::py_binary_precision_reqs, m)?)?;
    m.add_function(wrap_pyfunction!(binary::py_binary_recall_reqs, m)?)?;
    m.add_function(wrap_pyfunction!(binary::py_binary_f1_score_reqs, m)?)?;
    Ok(())
}
