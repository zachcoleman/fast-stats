use numpy::*;
use pyo3::{prelude::*, types::PySet, PyAny};
use std::collections::HashSet;

use crate::{numpy_dispatch_bool, numpy_dispatch_no_bool, numpy_dispatch_quant};

/// unique
#[pyfunction]
#[pyo3(name = "_unique")]
#[pyo3(text_signature = "(arr: np.ndarray, /)")]
pub fn py_unique<'a>(py: Python<'a>, arr: &'a PyAny) -> PyResult<&'a PySet> {
    numpy_dispatch_bool!(py, unique, PyResult<&'a PySet>, arr)
}

#[pyfunction]
#[pyo3(name = "_int_sum")]
#[pyo3(text_signature = "(arr: np.ndarray, /)")]
pub fn py_int_sum<'a>(py: Python<'a>, arr: &'a PyAny) -> PyResult<i128> {
    numpy_dispatch_no_bool!(py, int_sum, PyResult<i128>, arr)
}

#[pyfunction]
#[pyo3(name = "_float_sum")]
#[pyo3(text_signature = "(arr: np.ndarray, /)")]
pub fn py_float_sum<'a>(py: Python<'a>, arr: &'a PyAny) -> PyResult<f64> {
    numpy_dispatch_quant!(py, float_sum, PyResult<f64>, arr)
}

/// ndarray unique
fn unique<'a, T>(py: Python<'a>, arr: numpy::PyReadonlyArrayDyn<T>) -> PyResult<&'a PySet>
where
    T: Clone
        + std::marker::Send
        + numpy::Element
        + std::hash::Hash
        + std::cmp::Eq
        + pyo3::ToPyObject,
{
    let arr = arr.to_owned_array();

    let threadable = |arr: ndarray::ArrayD<T>| -> Vec<T> {
        py.allow_threads(move || {
            let mut track = HashSet::<T>::new();
            let mut ret: Vec<T> = vec![];
            for val in arr.iter() {
                if !track.contains(val) {
                    track.insert(val.clone());
                    ret.push(val.clone());
                }
            }
            return ret;
        })
    };
    PySet::new(py, threadable(arr).as_slice())
}

fn int_sum<'a, T>(py: Python<'a>, arr: numpy::PyReadonlyArrayDyn<T>) -> PyResult<i128>
where
    T: Clone + std::ops::Add<Output = T> + num_traits::Num + Into<i128> + numpy::Element,
{
    let arr = arr.to_owned_array();
    Ok(custom_int_sum::<T>(arr))
}

fn float_sum<'a, T>(py: Python<'a>, arr: numpy::PyReadonlyArrayDyn<T>) -> PyResult<f64>
where
    T: Clone + std::ops::Add<Output = T> + num_traits::Num + Into<f64> + numpy::Element,
{
    let arr = arr.to_owned_array();
    Ok(custom_float_sum::<T>(arr))
}

#[pyfunction]
#[pyo3(name = "_sum")]
#[pyo3(text_signature = "(arr: np.ndarray, /)")]
pub fn py_sum<'a>(py: Python<'a>, arr: &'a PyAny) -> PyResult<f64> {
    numpy_dispatch_quant!(py, sum, PyResult<f64>, arr)
}

fn sum<'a, T>(py: Python<'a>, arr: numpy::PyReadonlyArrayDyn<T>) -> PyResult<f64>
where
    T: Clone + std::ops::Add<Output = T> + num_traits::Num + numpy::Element + Into<f64>,
{
    let arr = arr.to_owned_array();
    let res = arr.sum();
    Ok(res.into())
}

pub fn custom_int_sum<T>(arr: ndarray::ArrayD<T>) -> i128
where
    T: Clone + std::ops::Add<Output = T> + num_traits::Num + Into<i128>,
{
    let mut sum = 0;
    for row in arr.rows() {
        sum = sum + row.iter().fold(0, |acc, elt| acc + elt.clone().into());
    }
    sum
}

pub fn custom_float_sum<T>(arr: ndarray::ArrayD<T>) -> f64
where
    T: Clone + std::ops::Add<Output = T> + num_traits::Num + Into<f64>,
{
    let mut sum: f64 = 0.;
    for row in arr.rows() {
        sum = sum + row.iter().fold(0., |acc, elt| acc + elt.clone().into());
    }
    sum
}
