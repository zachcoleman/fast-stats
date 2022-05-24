use numpy::*;
use pyo3::{exceptions, prelude::*, ffi::Py_TPFLAGS_LIST_SUBCLASS};
use std::{iter::zip, collections::HashMap};
mod binary;

/// Confusion Matrix
#[pyfunction]
#[pyo3(name = "_confusion_matrix")]
#[pyo3(text_signature = "(actual: np.ndarray, pred: np.ndarray, labels: List[int], /)")]
fn confusion_matrix<'a>(
    py: Python<'a>,
    actual: &PyArrayDyn<i64>,
    pred: &PyArrayDyn<i64>,
    labels: Vec<i64>,
) -> &'a PyArray2<i64>  {
    let mut cm = ndarray::Array2::<i64>::from_elem((labels.len(), labels.len()), 0);
    let idx_map: HashMap<i64, usize> = HashMap::from_iter(
        labels
        .iter()
        .enumerate()
        .map(|(x, y)| (*y, x))
    );
    
    for (y_pred, y_actual) in zip(
        pred.to_owned_array().iter(),
        actual.to_owned_array().iter(),
    ) {
        let ix1 = *idx_map.get(y_pred).unwrap();
        let ix2 = *idx_map.get(y_actual).unwrap();
        *cm.get_mut((ix1, ix2)).unwrap() = *cm.get_mut((ix1, ix2)).unwrap() + 1;
    }

    return &PyArray2::from_array(py, &cm);
}

/// A Python module implemented in Rust.
#[pymodule]
fn fast_stats(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(confusion_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(binary::py_binary_precision_reqs, m)?)?;
    m.add_function(wrap_pyfunction!(binary::py_binary_recall_reqs, m)?)?;
    m.add_function(wrap_pyfunction!(binary::py_binary_f1_score_reqs, m)?)?;
    Ok(())
}
