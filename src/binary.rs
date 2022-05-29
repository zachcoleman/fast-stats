use numpy::*;
use pyo3::prelude::*;

use crate::numpy_dispatch_no_bool;

/// Binary precision computational requirements
#[pyfunction]
#[pyo3(name = "_binary_precision_reqs")]
#[pyo3(text_signature = "(actual: np.ndarray, pred: np.ndarray, /)")]
pub fn py_binary_precision_reqs<'a>(
    py: Python<'a>,
    actual: &'a PyAny,
    pred: &'a PyAny,
) -> PyResult<(i128, i128, i128)> {
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<bool>>(),
        pred.extract::<PyReadonlyArrayDyn<bool>>(),
    ) {
        return binary_precision_reqs_owned::<u8>(
            py,
            i.to_owned_array().mapv(|e| e as u8),
            j.to_owned_array().mapv(|e| e as u8),
        );
    }

    numpy_dispatch_no_bool!(
        py,
        binary_precision_reqs,
        PyResult<(i128, i128, i128)>,
        actual,
        pred
    )
}

/// Binary recall computational requirements
#[pyfunction]
#[pyo3(name = "_binary_recall_reqs")]
#[pyo3(text_signature = "(actual: np.ndarray, pred: np.ndarray, /)")]
pub fn py_binary_recall_reqs<'a>(
    py: Python<'a>,
    actual: &'a PyAny,
    pred: &'a PyAny,
) -> PyResult<(i128, i128, i128)> {
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<bool>>(),
        pred.extract::<PyReadonlyArrayDyn<bool>>(),
    ) {
        return binary_recall_reqs_owned::<u8>(
            py,
            i.to_owned_array().mapv(|e| e as u8),
            j.to_owned_array().mapv(|e| e as u8),
        );
    }

    numpy_dispatch_no_bool!(
        py,
        binary_recall_reqs,
        PyResult<(i128, i128, i128)>,
        actual,
        pred
    )
}

/// Binary f1 computational requirements
#[pyfunction]
#[pyo3(name = "_binary_f1_score_reqs")]
#[pyo3(text_signature = "(actual: np.ndarray, pred: np.ndarray, /)")]
pub fn py_binary_f1_score_reqs<'a>(
    py: Python<'a>,
    actual: &'a PyAny,
    pred: &'a PyAny,
) -> PyResult<(i128, i128, i128)> {
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<bool>>(),
        pred.extract::<PyReadonlyArrayDyn<bool>>(),
    ) {
        return binary_f1_score_reqs_owned::<u8>(
            py,
            i.to_owned_array().mapv(|e| e as u8),
            j.to_owned_array().mapv(|e| e as u8),
        );
    }

    numpy_dispatch_no_bool!(
        py,
        binary_f1_score_reqs,
        PyResult<(i128, i128, i128)>,
        actual,
        pred
    )
}

// move into utils?
fn custom_sum<T>(arr: ndarray::ArrayD<T>) -> i128
where
    T: Clone + std::ops::Add<Output = T> + num_traits::Num + Into<i128>,
{
    let mut sum = 0;
    for row in arr.rows() {
        sum = sum + row.iter().fold(0, |acc, elt| acc + elt.clone().into());
    }
    sum
}

fn binary_precision_reqs<'a, T>(
    py: Python<'a>,
    actual: numpy::PyReadonlyArrayDyn<T>,
    pred: numpy::PyReadonlyArrayDyn<T>,
) -> PyResult<(i128, i128, i128)>
where
    T: Clone
        + std::marker::Send
        + numpy::Element
        + std::ops::Add<Output = T>
        + num_traits::Num
        + Into<i128>,
{
    let actual = actual.to_owned_array();
    let pred = pred.to_owned_array();

    // TP, TP + FP, 0
    Ok(py.allow_threads(move || {
        return (custom_sum(actual * &pred), custom_sum(pred), 0);
    }))
}

fn binary_precision_reqs_owned<'a, T>(
    py: Python<'a>,
    actual: ndarray::ArrayD<T>,
    pred: ndarray::ArrayD<T>,
) -> PyResult<(i128, i128, i128)>
where
    T: Clone
        + std::marker::Send
        + numpy::Element
        + std::ops::Add<Output = T>
        + num_traits::Num
        + Into<i128>,
{
    // TP, TP + FP, 0
    Ok(py.allow_threads(move || {
        return (custom_sum(actual * &pred), custom_sum(pred), 0);
    }))
}

fn binary_recall_reqs<'a, T>(
    py: Python<'a>,
    actual: numpy::PyReadonlyArrayDyn<T>,
    pred: numpy::PyReadonlyArrayDyn<T>,
) -> PyResult<(i128, i128, i128)>
where
    T: Clone
        + std::marker::Send
        + numpy::Element
        + std::ops::Add<Output = T>
        + num_traits::Num
        + Into<i128>,
{
    let actual = actual.to_owned_array();
    let pred = pred.to_owned_array();

    // TP, TP + FN, 0
    Ok(py.allow_threads(move || {
        return (custom_sum(&actual * pred), custom_sum(actual), 0);
    }))
}

fn binary_recall_reqs_owned<'a, T>(
    py: Python<'a>,
    actual: ndarray::ArrayD<T>,
    pred: ndarray::ArrayD<T>,
) -> PyResult<(i128, i128, i128)>
where
    T: Clone
        + std::marker::Send
        + numpy::Element
        + std::ops::Add<Output = T>
        + num_traits::Num
        + Into<i128>,
{
    // TP, TP + FN, 0
    Ok(py.allow_threads(move || {
        return (custom_sum(&actual * pred), custom_sum(actual), 0);
    }))
}

fn binary_f1_score_reqs<'a, T>(
    py: Python<'a>,
    actual: numpy::PyReadonlyArrayDyn<T>,
    pred: numpy::PyReadonlyArrayDyn<T>,
) -> PyResult<(i128, i128, i128)>
where
    T: Clone
        + std::marker::Send
        + numpy::Element
        + std::ops::Add<Output = T>
        + num_traits::Num
        + Into<i128>,
{
    let actual = actual.to_owned_array();
    let pred = pred.to_owned_array();

    // TP, TP + FP, TP + FN
    Ok(py.allow_threads(move || {
        return (
            custom_sum(&actual * &pred),
            custom_sum(pred),
            custom_sum(actual),
        );
    }))
}

fn binary_f1_score_reqs_owned<'a, T>(
    py: Python<'a>,
    actual: ndarray::ArrayD<T>,
    pred: ndarray::ArrayD<T>,
) -> PyResult<(i128, i128, i128)>
where
    T: Clone
        + std::marker::Send
        + numpy::Element
        + std::ops::Add<Output = T>
        + num_traits::Num
        + Into<i128>,
{
    // TP, TP + FP, TP + FN
    Ok(py.allow_threads(move || {
        return (
            custom_sum(&actual * &pred),
            custom_sum(pred),
            custom_sum(actual),
        );
    }))
}
