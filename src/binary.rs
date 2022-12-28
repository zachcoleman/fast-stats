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
        binary_precision_reqs_bool(py, i, j)
    } else {
        numpy_dispatch_no_bool!(
            py,
            binary_precision_reqs,
            PyResult<(i128, i128, i128)>,
            actual,
            pred
        )
    }
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
        binary_recall_reqs_bool(py, i, j)
    } else {
        numpy_dispatch_no_bool!(
            py,
            binary_recall_reqs,
            PyResult<(i128, i128, i128)>,
            actual,
            pred
        )
    }
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
        binary_f1_score_reqs_bool(py, i, j)
    } else {
        numpy_dispatch_no_bool!(
            py,
            binary_f1_score_reqs,
            PyResult<(i128, i128, i128)>,
            actual,
            pred
        )
    }
}

fn binary_precision_reqs<'a, T>(
    _py: Python<'a>,
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
    let mut reqs = (0, 0, 0);
    unsafe {
        for (r1, r2) in std::iter::zip(pred.as_array_mut().rows(), actual.as_array_mut().rows()) {
            let row_reqs = std::iter::zip(r1, r2).fold((0, 0), |acc, elt| {
                (
                    acc.0 + (elt.0.clone() * elt.1.clone()).into(),
                    acc.1 + elt.0.clone().into(),
                )
            });
            reqs.0 = reqs.0 + row_reqs.0;
            reqs.1 = reqs.1 + row_reqs.1;
        }
    }
    Ok(reqs)
}

fn binary_precision_reqs_bool<'a>(
    _py: Python<'a>,
    actual: numpy::PyReadonlyArrayDyn<bool>,
    pred: numpy::PyReadonlyArrayDyn<bool>,
) -> PyResult<(i128, i128, i128)> {
    let mut reqs = (0, 0, 0);
    unsafe {
        for (r1, r2) in std::iter::zip(pred.as_array_mut().rows(), actual.as_array_mut().rows()) {
            let row_reqs = std::iter::zip(r1, r2).fold((0, 0), |acc, elt| {
                (
                    acc.0 + (elt.0.clone() & elt.1.clone()) as i128,
                    acc.1 + (elt.0.clone()) as i128,
                )
            });
            reqs.0 = reqs.0 + row_reqs.0;
            reqs.1 = reqs.1 + row_reqs.1;
        }
    }
    Ok(reqs)
}

fn binary_recall_reqs<'a, T>(
    _py: Python<'a>,
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
    let mut reqs = (0, 0, 0);
    unsafe {
        for (r1, r2) in std::iter::zip(pred.as_array_mut().rows(), actual.as_array_mut().rows()) {
            let row_reqs = std::iter::zip(r1, r2).fold((0, 0), |acc, elt| {
                (
                    acc.0 + (elt.0.clone() * elt.1.clone()).into(),
                    acc.1 + elt.1.clone().into(),
                )
            });
            reqs.0 = reqs.0 + row_reqs.0;
            reqs.1 = reqs.1 + row_reqs.1;
        }
    }
    Ok(reqs)
}

fn binary_recall_reqs_bool<'a>(
    _py: Python<'a>,
    actual: numpy::PyReadonlyArrayDyn<bool>,
    pred: numpy::PyReadonlyArrayDyn<bool>,
) -> PyResult<(i128, i128, i128)> {
    let mut reqs = (0, 0, 0);
    unsafe {
        for (r1, r2) in std::iter::zip(pred.as_array_mut().rows(), actual.as_array_mut().rows()) {
            let row_reqs = std::iter::zip(r1, r2).fold((0, 0), |acc, elt| {
                (
                    acc.0 + (elt.0.clone() & elt.1.clone()) as i128,
                    acc.1 + (elt.1.clone()) as i128,
                )
            });
            reqs.0 = reqs.0 + row_reqs.0;
            reqs.1 = reqs.1 + row_reqs.1;
        }
    }
    Ok(reqs)
}

fn binary_f1_score_reqs<'a, T>(
    _py: Python<'a>,
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
    let mut reqs = (0, 0, 0);
    unsafe {
        for (r1, r2) in std::iter::zip(pred.as_array_mut().rows(), actual.as_array_mut().rows()) {
            let row_reqs = std::iter::zip(r1, r2).fold((0, 0, 0), |acc, elt| {
                (
                    acc.0 + (elt.0.clone() * elt.1.clone()).into(),
                    acc.1 + elt.0.clone().into(),
                    acc.2 + elt.1.clone().into(),
                )
            });
            reqs.0 = reqs.0 + row_reqs.0;
            reqs.1 = reqs.1 + row_reqs.1;
            reqs.2 = reqs.2 + row_reqs.2;
        }
    }
    Ok(reqs)
}

fn binary_f1_score_reqs_bool<'a>(
    _py: Python<'a>,
    actual: numpy::PyReadonlyArrayDyn<bool>,
    pred: numpy::PyReadonlyArrayDyn<bool>,
) -> PyResult<(i128, i128, i128)> {
    let mut reqs = (0, 0, 0);
    unsafe {
        for (r1, r2) in std::iter::zip(pred.as_array_mut().rows(), actual.as_array_mut().rows()) {
            let row_reqs = std::iter::zip(r1, r2).fold((0, 0, 0), |acc, elt| {
                (
                    acc.0 + (elt.0.clone() & elt.1.clone()) as i128,
                    acc.1 + (elt.0.clone()) as i128,
                    acc.2 + (elt.1.clone()) as i128,
                )
            });
            reqs.0 = reqs.0 + row_reqs.0;
            reqs.1 = reqs.1 + row_reqs.1;
            reqs.2 = reqs.2 + row_reqs.2;
        }
    }
    Ok(reqs)
}
