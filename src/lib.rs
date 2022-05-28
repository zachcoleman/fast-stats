use pyo3::prelude::*;

mod binary;
mod cm;
mod dispatch;
mod utils;

/// A Python module implemented in Rust.
#[pymodule]
fn _fast_stats_ext(_py: Python, m: &PyModule) -> PyResult<()> {
    // cm
    m.add_function(wrap_pyfunction!(cm::py_confusion_matrix, m)?)?;

    // utils
    m.add_function(wrap_pyfunction!(utils::py_unique, m)?)?;

    // binary calcs
    m.add_function(wrap_pyfunction!(binary::py_binary_precision_reqs, m)?)?;
    m.add_function(wrap_pyfunction!(binary::py_binary_recall_reqs, m)?)?;
    m.add_function(wrap_pyfunction!(binary::py_binary_f1_score_reqs, m)?)?;

    Ok(())
}
