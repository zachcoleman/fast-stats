use ndarray::*;
use numpy::*;
use pyo3::prelude::*;

use crate::numpy_dispatch_quant;
use crate::utils::custom_float_sum;

/// Confusion Matrix
#[pyfunction]
#[pyo3(name = "_mean")]
#[pyo3(text_signature = "(arr: np.ndarray, /)")]
pub fn py_mean<'a>(py: Python<'a>, arr: &'a PyAny) -> PyResult<f64> {
    numpy_dispatch_quant!(py, mean, PyResult<f64>, arr)
}

fn mean<T>(py: Python<'_>, arr: PyReadonlyArrayDyn<T>) -> PyResult<f64>
where
    T: Copy
        + Clone
        + std::marker::Send
        + numpy::Element
        + std::ops::Add<Output = T>
        + num_traits::Num
        + Into<f64>,
{
    let arr = arr.to_owned_array();
    let threadable = |arr: ArrayD<T>| -> f64 {
        py.allow_threads(move || {
            let n_elements = usize::from(arr.len());
            custom_float_sum(arr) / (n_elements as f64)
        })
    };
    Ok(threadable(arr))
}

fn arr_mean<T>(arr: ArrayD<T>) -> f64
where
    T: Clone + std::ops::Add<Output = T> + num_traits::Num + Into<f64>,
{
    let mut ret: f64 = 0.;
    let mut n: f64 = 0.;
    for val in arr.iter() {
        ret = ret * (n / (n + 1.)) + val.clone().into() / (n + 1.);
        n = n + 1.;
    }
    ret
}
