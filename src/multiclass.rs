use numpy::*;
use pyo3::prelude::*;

use crate::cm;
use crate::numpy_dispatch_bool;

/// Precision computational requirements
#[pyfunction]
#[pyo3(name = "_precision")]
#[pyo3(
    text_signature = "(actual: np.ndarray, pred: np.ndarray, labels: Union[List, np.ndarray], /)"
)]
pub fn py_precision<'a>(
    py: Python<'a>,
    actual: &'a PyAny,
    pred: &'a PyAny,
    labels: &'a PyAny,
) -> PyResult<&'a PyArray2<usize>> {
    numpy_dispatch_bool!(
        py,
        precision,
        PyResult<&'a PyArray2<usize>>,
        actual,
        pred,
        labels
    )
}

/// Recall computational requirements
#[pyfunction]
#[pyo3(name = "_recall")]
#[pyo3(
    text_signature = "(actual: np.ndarray, pred: np.ndarray, labels: Union[List, np.ndarray], /)"
)]
pub fn py_recall<'a>(
    py: Python<'a>,
    actual: &'a PyAny,
    pred: &'a PyAny,
    labels: &'a PyAny,
) -> PyResult<&'a PyArray2<usize>> {
    numpy_dispatch_bool!(
        py,
        recall,
        PyResult<&'a PyArray2<usize>>,
        actual,
        pred,
        labels
    )
}

/// f1 score computational requirements
#[pyfunction]
#[pyo3(name = "_f1_score")]
#[pyo3(
    text_signature = "(actual: np.ndarray, pred: np.ndarray, labels: Union[List, np.ndarray], /)"
)]
pub fn py_f1_score<'a>(
    py: Python<'a>,
    actual: &'a PyAny,
    pred: &'a PyAny,
    labels: &'a PyAny,
) -> PyResult<&'a PyArray2<usize>> {
    numpy_dispatch_bool!(
        py,
        f1_score,
        PyResult<&'a PyArray2<usize>>,
        actual,
        pred,
        labels
    )
}

fn precision<'a, T>(
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

    let threadable =
        |actual: ndarray::ArrayD<T>, pred: ndarray::ArrayD<T>| -> ndarray::Array2<usize> {
            py.allow_threads(move || {
                let cm = cm::confusion_matrix_owned(actual, pred, labels);
                let mut ret = ndarray::Array2::<usize>::from_elem((cm.shape()[0], 2), 0);
                for (idx, col) in cm.columns().into_iter().enumerate() {
                    // get TP
                    *ret.get_mut((idx, 0)).unwrap() = *col.get(idx).unwrap();
                    // get TP + FP
                    *ret.get_mut((idx, 1)).unwrap() = col.sum();
                }
                return ret;
            })
        };
    Ok(PyArray2::from_array(py, &threadable(actual, pred)))
}

fn recall<'a, T>(
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

    let threadable =
        |actual: ndarray::ArrayD<T>, pred: ndarray::ArrayD<T>| -> ndarray::Array2<usize> {
            py.allow_threads(move || {
                let cm = cm::confusion_matrix_owned(actual, pred, labels);
                let mut ret = ndarray::Array2::<usize>::from_elem((cm.shape()[0], 2), 0);
                for (idx, row) in cm.rows().into_iter().enumerate() {
                    // get TP
                    *ret.get_mut((idx, 0)).unwrap() = *row.get(idx).unwrap();
                    // get TP + FN
                    *ret.get_mut((idx, 1)).unwrap() = row.sum();
                }
                return ret;
            })
        };
    Ok(PyArray2::from_array(py, &threadable(actual, pred)))
}

fn f1_score<'a, T>(
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

    let threadable =
        |actual: ndarray::ArrayD<T>, pred: ndarray::ArrayD<T>| -> ndarray::Array2<usize> {
            py.allow_threads(move || {
                let cm = cm::confusion_matrix_owned(actual, pred, labels);
                let mut ret = ndarray::Array2::<usize>::from_elem((cm.shape()[0], 3), 0);
                for (idx, col) in cm.columns().into_iter().enumerate() {
                    // get TP
                    *ret.get_mut((idx, 0)).unwrap() = *col.get(idx).unwrap();
                    // get TP + FP
                    *ret.get_mut((idx, 1)).unwrap() = col.sum();
                }
                for (idx, row) in cm.rows().into_iter().enumerate() {
                    // get TP + FN
                    *ret.get_mut((idx, 2)).unwrap() = row.sum();
                }
                return ret;
            })
        };
    Ok(PyArray2::from_array(py, &threadable(actual, pred)))
}
