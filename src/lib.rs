use pyo3::prelude::*;
use numpy;

mod binary;
mod cm;
mod dispatch;
mod multiclass;
mod utils;


#[pyfunction]
#[pyo3(name = "_to_owned_iter")]
#[pyo3(text_signature = "(arr: np.ndarray, /)")]
pub fn py_to_owned_iter<'a>(
    _py: Python<'a>,
    arr: numpy::PyReadonlyArrayDyn<u8>,
) -> PyResult<u64> {
    let mut sum = 0;
    let owned_arr = arr.to_owned_array();
    for v in owned_arr.iter(){
        sum = sum + (*v as u64);
    }
    Ok(sum)
}

#[pyfunction]
#[pyo3(name = "_not_owned_iter")]
#[pyo3(text_signature = "(arr: np.ndarray, /)")]
pub fn py_not_owned_iter<'a>(
    _py: Python<'a>,
    arr: numpy::PyReadonlyArrayDyn<u8>,
) -> PyResult<u64> {
    let mut sum = 0;
    for v in arr.as_array().iter(){
        sum = sum + (*v as u64);
    }
    Ok(sum)
}

/// A Python module implemented in Rust.
#[pymodule]
fn _fast_stats_ext(_py: Python, m: &PyModule) -> PyResult<()> {
    // test
    m.add_function(wrap_pyfunction!(py_to_owned_iter, m)?)?;
    m.add_function(wrap_pyfunction!(py_not_owned_iter, m)?)?;

    // cm
    m.add_function(wrap_pyfunction!(cm::py_confusion_matrix, m)?)?;

    // utils
    m.add_function(wrap_pyfunction!(utils::py_unique, m)?)?;

    // binary calcs
    m.add_function(wrap_pyfunction!(binary::py_binary_precision_reqs, m)?)?;
    // m.add_function(wrap_pyfunction!(binary::py_binary_recall_reqs, m)?)?;
    // m.add_function(wrap_pyfunction!(binary::py_binary_f1_score_reqs, m)?)?;

    // multiclass calcs
    m.add_function(wrap_pyfunction!(multiclass::py_precision, m)?)?;
    m.add_function(wrap_pyfunction!(multiclass::py_recall, m)?)?;
    m.add_function(wrap_pyfunction!(multiclass::py_f1_score, m)?)?;

    Ok(())
}
