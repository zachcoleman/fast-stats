use numpy::*;
use pyo3::prelude::*;
use std::{collections::HashMap, iter::zip};

/// Confusion Matrix
#[pyfunction]
#[pyo3(name = "_confusion_matrix")]
#[pyo3(text_signature = "(actual: np.ndarray, pred: np.ndarray, labels: List[int], /)")]
pub fn py_confusion_matrix<'a>(
    py: Python<'a>,
    actual: &PyAny,
    pred: &PyAny,
    labels: &PyAny,
) -> PyResult<&'a PyArray2<usize>>{
    dispatch(py, actual, pred, labels)
}

pub fn confusion_matrix<'a, T>(
    py: Python<'a>,
    actual: &PyArrayDyn<T>,
    pred: &PyArrayDyn<T>,
    labels: Vec<T>,
) -> &'a PyArray2<usize>
where
    T: Copy + Clone + numpy::Element + std::hash::Hash + std::cmp::Eq,
{
    let mut cm = ndarray::Array2::<usize>::from_elem((labels.len(), labels.len()), 0);
    let idx_map: HashMap<T, usize> =
        HashMap::from_iter(labels.iter().enumerate().map(|(x, y)| (*y, x)));

    for (y_pred, y_actual) in zip(pred.to_owned_array().iter(), actual.to_owned_array().iter()) {
        if let (Some(ix1), Some(ix2)) = (idx_map.get(y_pred), idx_map.get(y_actual)){
            *cm.get_mut((*ix1, *ix2)).unwrap() = *cm.get_mut((*ix1, *ix2)).unwrap() + 1;
        }
    }

    return PyArray2::from_array(py, &cm);
}

/// dispatching
fn dispatch<'a>(
    py: Python<'a>,
    actual: &PyAny,
    pred: &PyAny,
    labels: &PyAny,
) -> PyResult<&'a PyArray2<usize>> {
    // bool
    if let (Ok(i), Ok(j), Ok(l)) = (
        actual.extract::<PyReadonlyArrayDyn<bool>>(),
        pred.extract::<PyReadonlyArrayDyn<bool>>(),
        labels.extract::<Vec<bool>>(),
    ) {
        return Ok(confusion_matrix::<bool>(py, &i, &j, l));
    }

    // i8
    if let (Ok(i), Ok(j), Ok(l)) = (
        actual.extract::<PyReadonlyArrayDyn<i8>>(),
        pred.extract::<PyReadonlyArrayDyn<i8>>(),
        labels.extract::<Vec<i8>>(),
    ) {
        return Ok(confusion_matrix::<i8>(py, &i, &j, l));
    }

    // i16
    if let (Ok(i), Ok(j), Ok(l)) = (
        actual.extract::<PyReadonlyArrayDyn<i16>>(),
        pred.extract::<PyReadonlyArrayDyn<i16>>(),
        labels.extract::<Vec<i16>>(),
    ) {
        return Ok(confusion_matrix::<i16>(py, &i, &j, l));
    }

    // i32
    if let (Ok(i), Ok(j), Ok(l)) = (
        actual.extract::<PyReadonlyArrayDyn<i32>>(),
        pred.extract::<PyReadonlyArrayDyn<i32>>(),
        labels.extract::<Vec<i32>>(),
    ) {
        return Ok(confusion_matrix::<i32>(py, &i, &j, l));
    }

    // i64
    if let (Ok(i), Ok(j), Ok(l)) = (
        actual.extract::<PyReadonlyArrayDyn<i64>>(),
        pred.extract::<PyReadonlyArrayDyn<i64>>(),
        labels.extract::<Vec<i64>>(),
    ) {
        return Ok(confusion_matrix::<i64>(py, &i, &j, l));
    }

    // u8
    if let (Ok(i), Ok(j), Ok(l)) = (
        actual.extract::<PyReadonlyArrayDyn<u8>>(),
        pred.extract::<PyReadonlyArrayDyn<u8>>(),
        labels.extract::<Vec<u8>>(),
    ) {
        return Ok(confusion_matrix::<u8>(py, &i, &j, l));
    }

    // u16
    if let (Ok(i), Ok(j), Ok(l)) = (
        actual.extract::<PyReadonlyArrayDyn<u16>>(),
        pred.extract::<PyReadonlyArrayDyn<u16>>(),
        labels.extract::<Vec<u16>>(),
    ) {
        return Ok(confusion_matrix::<u16>(py, &i, &j, l));
    }

    // u32
    if let (Ok(i), Ok(j), Ok(l)) = (
        actual.extract::<PyReadonlyArrayDyn<u32>>(),
        pred.extract::<PyReadonlyArrayDyn<u32>>(),
        labels.extract::<Vec<u32>>(),
    ) {
        return Ok(confusion_matrix::<u32>(py, &i, &j, l));
    }

    // u64
    if let (Ok(i), Ok(j), Ok(l)) = (
        actual.extract::<PyReadonlyArrayDyn<u64>>(),
        pred.extract::<PyReadonlyArrayDyn<u64>>(),
        labels.extract::<Vec<u64>>(),
    ) {
        return Ok(confusion_matrix::<u64>(py, &i, &j, l));
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Unsupported numpy dtype",
    ))
}
