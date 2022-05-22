use numpy::*;
use pyo3::{exceptions, prelude::*};
use std::iter::zip;

fn sum<T>(arr: ndarray::ArrayD<T>) -> i128
where
    T: Clone + std::ops::Add<Output = T> + num_traits::Num + Into<i128>,
{
    let mut sum = 0;
    for row in arr.rows() {
        sum = sum + row.iter().fold(0, |acc, elt| acc + elt.clone().into());
    }
    sum
}

/// Get tp, fp, fn, tn counts by looping
#[pyfunction]
#[pyo3(name = "_tp_fp_fn_tn")]
#[pyo3(text_signature = "(actual: np.ndarray, pred: np.ndarray, /)")]
fn tp_fp_fn_tn(
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

/// Array-based binary precision req calculating
#[pyfunction]
#[pyo3(name = "_binary_precision_reqs")]
#[pyo3(text_signature = "(actual: np.ndarray, pred: np.ndarray, /)")]
fn py_binary_precision_reqs(
    _py: Python<'_>,
    actual: &PyAny,
    pred: &PyAny,
) -> PyResult<(i128, i128)> {
    // TODO macro this out
    // bool
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<bool>>(),
        pred.extract::<PyReadonlyArrayDyn<bool>>(),
    ) {
        return Ok(binary_precision_reqs::<u8>(
            i.to_owned_array().mapv(|e| e as u8),
            j.to_owned_array().mapv(|e| e as u8),
        ));
    }
    // i8
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<i8>>(),
        pred.extract::<PyReadonlyArrayDyn<i8>>(),
    ) {
        return Ok(binary_precision_reqs::<i8>(
            i.to_owned_array(),
            j.to_owned_array(),
        ));
    }
    // i16
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<i16>>(),
        pred.extract::<PyReadonlyArrayDyn<i16>>(),
    ) {
        return Ok(binary_precision_reqs::<i16>(
            i.to_owned_array(),
            j.to_owned_array(),
        ));
    }
    // i32
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<i32>>(),
        pred.extract::<PyReadonlyArrayDyn<i32>>(),
    ) {
        return Ok(binary_precision_reqs::<i32>(
            i.to_owned_array(),
            j.to_owned_array(),
        ));
    }
    // i64
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<i64>>(),
        pred.extract::<PyReadonlyArrayDyn<i64>>(),
    ) {
        return Ok(binary_precision_reqs::<i64>(
            i.to_owned_array(),
            j.to_owned_array(),
        ));
    }
    // u8
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<u8>>(),
        pred.extract::<PyReadonlyArrayDyn<u8>>(),
    ) {
        return Ok(binary_precision_reqs::<u8>(
            i.to_owned_array(),
            j.to_owned_array(),
        ));
    }
    // u16
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<u16>>(),
        pred.extract::<PyReadonlyArrayDyn<u16>>(),
    ) {
        return Ok(binary_precision_reqs::<u16>(
            i.to_owned_array(),
            j.to_owned_array(),
        ));
    }
    // u32
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<u32>>(),
        pred.extract::<PyReadonlyArrayDyn<u32>>(),
    ) {
        return Ok(binary_precision_reqs::<u32>(
            i.to_owned_array(),
            j.to_owned_array(),
        ));
    }
    // u64
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<u64>>(),
        pred.extract::<PyReadonlyArrayDyn<u64>>(),
    ) {
        return Ok(binary_precision_reqs::<u64>(
            i.to_owned_array(),
            j.to_owned_array(),
        ));
    }

    Err(PyErr::new::<exceptions::PyTypeError, _>(
        "Unsupport numpy dtype",
    ))
}

fn binary_precision_reqs<T>(actual: ndarray::ArrayD<T>, pred: ndarray::ArrayD<T>) -> (i128, i128)
where
    T: Clone + std::ops::Add<Output = T> + num_traits::Num + Into<i128>,
{
    // TP, TP + FP
    (sum(actual * &pred), sum(pred))
}

/// Array-based binary recall req calculating
#[pyfunction]
#[pyo3(name = "_binary_recall_reqs")]
#[pyo3(text_signature = "(actual: np.ndarray, pred: np.ndarray, /)")]
fn py_binary_recall_reqs(_py: Python<'_>, actual: &PyAny, pred: &PyAny) -> PyResult<(i128, i128)> {
    // TODO macro this out
    // bool
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<bool>>(),
        pred.extract::<PyReadonlyArrayDyn<bool>>(),
    ) {
        return Ok(binary_recall_reqs::<u8>(
            i.to_owned_array().mapv(|e| e as u8),
            j.to_owned_array().mapv(|e| e as u8),
        ));
    }
    // i8
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<i8>>(),
        pred.extract::<PyReadonlyArrayDyn<i8>>(),
    ) {
        return Ok(binary_recall_reqs::<i8>(
            i.to_owned_array(),
            j.to_owned_array(),
        ));
    }
    // i16
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<i16>>(),
        pred.extract::<PyReadonlyArrayDyn<i16>>(),
    ) {
        return Ok(binary_recall_reqs::<i16>(
            i.to_owned_array(),
            j.to_owned_array(),
        ));
    }
    // i32
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<i32>>(),
        pred.extract::<PyReadonlyArrayDyn<i32>>(),
    ) {
        return Ok(binary_recall_reqs::<i32>(
            i.to_owned_array(),
            j.to_owned_array(),
        ));
    }
    // i64
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<i64>>(),
        pred.extract::<PyReadonlyArrayDyn<i64>>(),
    ) {
        return Ok(binary_recall_reqs::<i64>(
            i.to_owned_array(),
            j.to_owned_array(),
        ));
    }
    // u8
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<u8>>(),
        pred.extract::<PyReadonlyArrayDyn<u8>>(),
    ) {
        return Ok(binary_recall_reqs::<u8>(
            i.to_owned_array(),
            j.to_owned_array(),
        ));
    }
    // u16
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<u16>>(),
        pred.extract::<PyReadonlyArrayDyn<u16>>(),
    ) {
        return Ok(binary_recall_reqs::<u16>(
            i.to_owned_array(),
            j.to_owned_array(),
        ));
    }
    // u32
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<u32>>(),
        pred.extract::<PyReadonlyArrayDyn<u32>>(),
    ) {
        return Ok(binary_recall_reqs::<u32>(
            i.to_owned_array(),
            j.to_owned_array(),
        ));
    }
    // u64
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<u64>>(),
        pred.extract::<PyReadonlyArrayDyn<u64>>(),
    ) {
        return Ok(binary_recall_reqs::<u64>(
            i.to_owned_array(),
            j.to_owned_array(),
        ));
    }

    Err(PyErr::new::<exceptions::PyTypeError, _>(
        "Unsupport numpy dtype",
    ))
}

fn binary_recall_reqs<T>(actual: ndarray::ArrayD<T>, pred: ndarray::ArrayD<T>) -> (i128, i128)
where
    T: Clone + std::ops::Add<Output = T> + num_traits::Num + Into<i128>,
{
    // TP, TP + FN
    (sum(&actual * pred), sum(actual))
}

/// Array-based binary recall req calculating
#[pyfunction]
#[pyo3(name = "_binary_f1_score_reqs")]
#[pyo3(text_signature = "(actual: np.ndarray, pred: np.ndarray, /)")]
fn py_binary_f1_score_reqs(
    _py: Python<'_>,
    actual: &PyAny,
    pred: &PyAny,
) -> PyResult<(i128, i128, i128)> {
    // TODO macro this out

    // bool
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<bool>>(),
        pred.extract::<PyReadonlyArrayDyn<bool>>(),
    ) {
        return Ok(binary_f1_score_reqs::<u8>(
            i.to_owned_array().mapv(|e| e as u8),
            j.to_owned_array().mapv(|e| e as u8),
        ));
    }

    // i8
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<i8>>(),
        pred.extract::<PyReadonlyArrayDyn<i8>>(),
    ) {
        return Ok(binary_f1_score_reqs::<i8>(
            i.to_owned_array(),
            j.to_owned_array(),
        ));
    }
    // i16
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<i16>>(),
        pred.extract::<PyReadonlyArrayDyn<i16>>(),
    ) {
        return Ok(binary_f1_score_reqs::<i16>(
            i.to_owned_array(),
            j.to_owned_array(),
        ));
    }
    // i32
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<i32>>(),
        pred.extract::<PyReadonlyArrayDyn<i32>>(),
    ) {
        return Ok(binary_f1_score_reqs::<i32>(
            i.to_owned_array(),
            j.to_owned_array(),
        ));
    }
    // i64
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<i64>>(),
        pred.extract::<PyReadonlyArrayDyn<i64>>(),
    ) {
        return Ok(binary_f1_score_reqs::<i64>(
            i.to_owned_array(),
            j.to_owned_array(),
        ));
    }
    // u8
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<u8>>(),
        pred.extract::<PyReadonlyArrayDyn<u8>>(),
    ) {
        return Ok(binary_f1_score_reqs::<u8>(
            i.to_owned_array(),
            j.to_owned_array(),
        ));
    }
    // u16
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<u16>>(),
        pred.extract::<PyReadonlyArrayDyn<u16>>(),
    ) {
        return Ok(binary_f1_score_reqs::<u16>(
            i.to_owned_array(),
            j.to_owned_array(),
        ));
    }
    // u32
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<u32>>(),
        pred.extract::<PyReadonlyArrayDyn<u32>>(),
    ) {
        return Ok(binary_f1_score_reqs::<u32>(
            i.to_owned_array(),
            j.to_owned_array(),
        ));
    }
    // u64
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<u64>>(),
        pred.extract::<PyReadonlyArrayDyn<u64>>(),
    ) {
        return Ok(binary_f1_score_reqs::<u64>(
            i.to_owned_array(),
            j.to_owned_array(),
        ));
    }

    Err(PyErr::new::<exceptions::PyTypeError, _>(
        "Unsupport numpy dtype",
    ))
}

fn binary_f1_score_reqs<T>(
    actual: ndarray::ArrayD<T>,
    pred: ndarray::ArrayD<T>,
) -> (i128, i128, i128)
where
    T: Clone + std::ops::Add<Output = T> + num_traits::Num + Into<i128>,
{
    // TP, TP + FP, TP + FN
    (sum(&actual * &pred), sum(pred), sum(actual))
}

/// A Python module implemented in Rust.
#[pymodule]
fn fast_stats(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tp_fp_fn_tn, m)?)?;
    m.add_function(wrap_pyfunction!(py_binary_precision_reqs, m)?)?;
    m.add_function(wrap_pyfunction!(py_binary_recall_reqs, m)?)?;
    m.add_function(wrap_pyfunction!(py_binary_f1_score_reqs, m)?)?;
    Ok(())
}
