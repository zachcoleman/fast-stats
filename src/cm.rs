use numpy::*;
use pyo3::prelude::*;
use std::{collections::HashMap, iter::zip};

use crate::numpy_dispatch_bool;

/// Confusion Matrix
#[pyfunction]
#[pyo3(name = "_confusion_matrix")]
#[pyo3(
    text_signature = "(actual: np.ndarray, pred: np.ndarray, labels: Union[List, np.ndarray], /)"
)]
pub fn py_confusion_matrix<'a>(
    py: Python<'a>,
    actual: &'a PyAny,
    pred: &'a PyAny,
    labels: &'a PyAny,
) -> PyResult<&'a PyArray2<i64>> {
    numpy_dispatch_bool!(
        py,
        confusion_matrix,
        PyResult<&'a PyArray2<i64>>,
        actual,
        pred,
        labels
    )
}

pub fn _confusion_matrix<'a, T>(
    _py: Python<'a>,
    actual: PyReadonlyArrayDyn<T>,
    pred: PyReadonlyArrayDyn<T>,
    labels: PyReadonlyArrayDyn<T>,
) -> ndarray::Array2<i64>
where
    T: Clone + numpy::Element + std::hash::Hash + std::cmp::Eq,
{
    let labels = labels.to_vec().unwrap();
    let mut cm = ndarray::Array2::<i64>::from_elem((labels.len(), labels.len()), 0);
    let idx_map: HashMap<T, usize> = HashMap::from_iter(
        labels
            .iter()
            .enumerate()
            .map(|(x, y)| (y.clone(), x.clone())),
    );
    for (y_pred, y_actual) in zip(pred.as_array().iter(), actual.as_array().iter()) {
        if let (Some(ix1), Some(ix2)) = (idx_map.get(y_actual), idx_map.get(y_pred)) {
            *cm.get_mut((*ix1, *ix2)).unwrap() = *cm.get_mut((*ix1, *ix2)).unwrap() + 1;
        }
    }
    cm
}

pub fn confusion_matrix<'a, T>(
    py: Python<'a>,
    actual: PyReadonlyArrayDyn<T>,
    pred: PyReadonlyArrayDyn<T>,
    labels: PyReadonlyArrayDyn<T>,
) -> PyResult<&'a PyArray2<i64>>
where
    T: Clone + numpy::Element + std::hash::Hash + std::cmp::Eq,
{
    Ok(PyArray2::from_array(
        py,
        &_confusion_matrix(py, actual, pred, labels),
    ))
}
