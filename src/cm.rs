use ndarray::ArrayD;
use numpy::*;
use pyo3::prelude::*;
use std::{collections::HashMap, iter::zip};

use crate::numpy_dispatch_bool;

/// Confusion Matrix
#[pyfunction]
#[pyo3(name = "_confusion_matrix")]
#[pyo3(text_signature = "(actual: np.ndarray, pred: np.ndarray, labels: List[int], /)")]
pub fn py_confusion_matrix<'a>(
    py: Python<'a>,
    actual: &'a PyAny,
    pred: &'a PyAny,
    labels: &'a PyAny,
) -> PyResult<&'a PyArray2<usize>> {
    numpy_dispatch_bool!(
        py,
        confusion_matrix,
        PyResult<&'a PyArray2<usize>>,
        actual,
        pred,
        labels
    )
}

fn confusion_matrix<'a, T>(
    py: Python<'a>,
    actual: PyReadonlyArrayDyn<T>,
    pred: PyReadonlyArrayDyn<T>,
    labels: PyReadonlyArrayDyn<T>,
) -> PyResult<&'a PyArray2<usize>>
where
    T: Copy + Clone + std::marker::Send + numpy::Element + std::hash::Hash + std::cmp::Eq,
{
    let actual = actual.to_owned_array();
    let pred = pred.to_owned_array();
    let labels = labels.to_vec().unwrap();

    let threadable = |actual: ArrayD<T>, pred: ArrayD<T>| -> ndarray::Array2<usize> {
        py.allow_threads(move || {
            let mut cm = ndarray::Array2::<usize>::from_elem((labels.len(), labels.len()), 0);
            let idx_map: HashMap<T, usize> =
                HashMap::from_iter(labels.iter().enumerate().map(|(x, y)| (*y, x)));

            for (y_pred, y_actual) in zip(pred.iter(), actual.iter()) {
                if let (Some(ix1), Some(ix2)) = (idx_map.get(y_actual), idx_map.get(y_pred)) {
                    *cm.get_mut((*ix1, *ix2)).unwrap() = *cm.get_mut((*ix1, *ix2)).unwrap() + 1;
                }
            }
            return cm;
        })
    };

    Ok(PyArray2::from_array(py, &threadable(actual, pred)))
}
