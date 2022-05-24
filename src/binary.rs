use numpy::*;
use pyo3::{exceptions, prelude::*};

/// Binary precision computational requirements
#[pyfunction]
#[pyo3(name = "_binary_precision_reqs")]
#[pyo3(text_signature = "(actual: np.ndarray, pred: np.ndarray, /)")]
pub fn py_binary_precision_reqs(
    _py: Python<'_>,
    actual: &PyAny,
    pred: &PyAny,
) -> PyResult<(i128, i128, i128)> {
    dispatch(_py, "precision", actual, pred)
}

/// Binary recall computational requirements
#[pyfunction]
#[pyo3(name = "_binary_recall_reqs")]
#[pyo3(text_signature = "(actual: np.ndarray, pred: np.ndarray, /)")]
pub fn py_binary_recall_reqs(
    _py: Python<'_>,
    actual: &PyAny,
    pred: &PyAny,
) -> PyResult<(i128, i128, i128)> {
    dispatch(_py, "recall", actual, pred)
}

/// Binary f1 computational requirements
#[pyfunction]
#[pyo3(name = "_binary_f1_score_reqs")]
#[pyo3(text_signature = "(actual: np.ndarray, pred: np.ndarray, /)")]
pub fn py_binary_f1_score_reqs(
    _py: Python<'_>,
    actual: &PyAny,
    pred: &PyAny,
) -> PyResult<(i128, i128, i128)> {
    dispatch(_py, "f1", actual, pred)
}

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

fn binary_precision_reqs<T>(
    actual: ndarray::ArrayD<T>,
    pred: ndarray::ArrayD<T>,
) -> (i128, i128, i128)
where
    T: Clone + std::ops::Add<Output = T> + num_traits::Num + Into<i128>,
{
    // TP, TP + FP
    (custom_sum(actual * &pred), custom_sum(pred), 0)
}

fn binary_recall_reqs<T>(actual: ndarray::ArrayD<T>, pred: ndarray::ArrayD<T>) -> (i128, i128, i128)
where
    T: Clone + std::ops::Add<Output = T> + num_traits::Num + Into<i128>,
{
    // TP, TP + FN
    (custom_sum(&actual * pred), custom_sum(actual), 0)
}

fn binary_f1_score_reqs<T>(
    actual: ndarray::ArrayD<T>,
    pred: ndarray::ArrayD<T>,
) -> (i128, i128, i128)
where
    T: Clone + std::ops::Add<Output = T> + num_traits::Num + Into<i128>,
{
    // TP, TP + FP, TP + FN
    (
        custom_sum(&actual * &pred),
        custom_sum(pred),
        custom_sum(actual),
    )
}

/// Dispatching!
fn dispatch(
    _py: Python<'_>,
    stat: &str,
    actual: &PyAny,
    pred: &PyAny,
) -> PyResult<(i128, i128, i128)> {
    // bool
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<bool>>(),
        pred.extract::<PyReadonlyArrayDyn<bool>>(),
    ) {
        match stat {
            "recall" => {
                return Ok(binary_recall_reqs::<u8>(
                    i.to_owned_array().mapv(|e| e as u8),
                    j.to_owned_array().mapv(|e| e as u8),
                ));
            }
            "precision" => {
                return Ok(binary_precision_reqs::<u8>(
                    i.to_owned_array().mapv(|e| e as u8),
                    j.to_owned_array().mapv(|e| e as u8),
                ));
            }
            "f1" => {
                return Ok(binary_f1_score_reqs::<u8>(
                    i.to_owned_array().mapv(|e| e as u8),
                    j.to_owned_array().mapv(|e| e as u8),
                ));
            }
            _ => {
                return Err(PyErr::new::<exceptions::PyTypeError, _>(
                    "Internal Error: not implemented stat type",
                ));
            }
        }
    }

    // i8
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<i8>>(),
        pred.extract::<PyReadonlyArrayDyn<i8>>(),
    ) {
        match stat {
            "recall" => {
                return Ok(binary_recall_reqs::<i8>(
                    i.to_owned_array(),
                    j.to_owned_array(),
                ));
            }
            "precision" => {
                return Ok(binary_precision_reqs::<i8>(
                    i.to_owned_array(),
                    j.to_owned_array(),
                ));
            }
            "f1" => {
                return Ok(binary_f1_score_reqs::<i8>(
                    i.to_owned_array(),
                    j.to_owned_array(),
                ));
            }
            _ => {
                return Err(PyErr::new::<exceptions::PyTypeError, _>(
                    "Internal Error: not implemented stat type",
                ));
            }
        }
    }

    // i16
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<i16>>(),
        pred.extract::<PyReadonlyArrayDyn<i16>>(),
    ) {
        match stat {
            "recall" => {
                return Ok(binary_recall_reqs::<i16>(
                    i.to_owned_array(),
                    j.to_owned_array(),
                ));
            }
            "precision" => {
                return Ok(binary_precision_reqs::<i16>(
                    i.to_owned_array(),
                    j.to_owned_array(),
                ));
            }
            "f1" => {
                return Ok(binary_f1_score_reqs::<i16>(
                    i.to_owned_array(),
                    j.to_owned_array(),
                ));
            }
            _ => {
                return Err(PyErr::new::<exceptions::PyTypeError, _>(
                    "Internal Error: not implemented stat type",
                ));
            }
        }
    }

    // i32
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<i32>>(),
        pred.extract::<PyReadonlyArrayDyn<i32>>(),
    ) {
        match stat {
            "recall" => {
                return Ok(binary_recall_reqs::<i32>(
                    i.to_owned_array(),
                    j.to_owned_array(),
                ));
            }
            "precision" => {
                return Ok(binary_precision_reqs::<i32>(
                    i.to_owned_array(),
                    j.to_owned_array(),
                ));
            }
            "f1" => {
                return Ok(binary_f1_score_reqs::<i32>(
                    i.to_owned_array(),
                    j.to_owned_array(),
                ));
            }
            _ => {
                return Err(PyErr::new::<exceptions::PyTypeError, _>(
                    "Internal Error: not implemented stat type",
                ));
            }
        }
    }

    // i64
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<i64>>(),
        pred.extract::<PyReadonlyArrayDyn<i64>>(),
    ) {
        match stat {
            "recall" => {
                return Ok(binary_recall_reqs::<i64>(
                    i.to_owned_array(),
                    j.to_owned_array(),
                ));
            }
            "precision" => {
                return Ok(binary_precision_reqs::<i64>(
                    i.to_owned_array(),
                    j.to_owned_array(),
                ));
            }
            "f1" => {
                return Ok(binary_f1_score_reqs::<i64>(
                    i.to_owned_array(),
                    j.to_owned_array(),
                ));
            }
            _ => {
                return Err(PyErr::new::<exceptions::PyTypeError, _>(
                    "Internal Error: not implemented stat type",
                ));
            }
        }
    }

    // u8
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<u8>>(),
        pred.extract::<PyReadonlyArrayDyn<u8>>(),
    ) {
        match stat {
            "recall" => {
                return Ok(binary_recall_reqs::<u8>(
                    i.to_owned_array(),
                    j.to_owned_array(),
                ));
            }
            "precision" => {
                return Ok(binary_precision_reqs::<u8>(
                    i.to_owned_array(),
                    j.to_owned_array(),
                ));
            }
            "f1" => {
                return Ok(binary_f1_score_reqs::<u8>(
                    i.to_owned_array(),
                    j.to_owned_array(),
                ));
            }
            _ => {
                return Err(PyErr::new::<exceptions::PyTypeError, _>(
                    "Internal Error: not implemented stat type",
                ));
            }
        }
    }

    // u16
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<u16>>(),
        pred.extract::<PyReadonlyArrayDyn<u16>>(),
    ) {
        match stat {
            "recall" => {
                return Ok(binary_recall_reqs::<u16>(
                    i.to_owned_array(),
                    j.to_owned_array(),
                ));
            }
            "precision" => {
                return Ok(binary_precision_reqs::<u16>(
                    i.to_owned_array(),
                    j.to_owned_array(),
                ));
            }
            "f1" => {
                return Ok(binary_f1_score_reqs::<u16>(
                    i.to_owned_array(),
                    j.to_owned_array(),
                ));
            }
            _ => {
                return Err(PyErr::new::<exceptions::PyTypeError, _>(
                    "Internal Error: not implemented stat type",
                ));
            }
        }
    }

    // u32
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<u32>>(),
        pred.extract::<PyReadonlyArrayDyn<u32>>(),
    ) {
        match stat {
            "recall" => {
                return Ok(binary_recall_reqs::<u32>(
                    i.to_owned_array(),
                    j.to_owned_array(),
                ));
            }
            "precision" => {
                return Ok(binary_precision_reqs::<u32>(
                    i.to_owned_array(),
                    j.to_owned_array(),
                ));
            }
            "f1" => {
                return Ok(binary_f1_score_reqs::<u32>(
                    i.to_owned_array(),
                    j.to_owned_array(),
                ));
            }
            _ => {
                return Err(PyErr::new::<exceptions::PyTypeError, _>(
                    "Internal Error: not implemented stat type",
                ));
            }
        }
    }

    // u64
    if let (Ok(i), Ok(j)) = (
        actual.extract::<PyReadonlyArrayDyn<u64>>(),
        pred.extract::<PyReadonlyArrayDyn<u64>>(),
    ) {
        match stat {
            "recall" => {
                return Ok(binary_recall_reqs::<u64>(
                    i.to_owned_array(),
                    j.to_owned_array(),
                ));
            }
            "precision" => {
                return Ok(binary_precision_reqs::<u64>(
                    i.to_owned_array(),
                    j.to_owned_array(),
                ));
            }
            "f1" => {
                return Ok(binary_f1_score_reqs::<u64>(
                    i.to_owned_array(),
                    j.to_owned_array(),
                ));
            }
            _ => {
                return Err(PyErr::new::<exceptions::PyTypeError, _>(
                    "Internal Error: not implemented stat type",
                ));
            }
        }
    }

    Err(PyErr::new::<exceptions::PyTypeError, _>(
        "Unsupport numpy dtype",
    ))
}
