use numpy::*;
use pyo3::{prelude::*, types::PySet};
use std::collections::HashSet;

use crate::numpy_dispatch_bool;

/// unique
#[pyfunction]
#[pyo3(name = "_unique")]
#[pyo3(text_signature = "(arr: np.ndarray, /)")]
pub fn py_unique<'a>(py: Python<'a>, arr: &'a PyAny) -> PyResult<&'a PySet> {
    numpy_dispatch_bool!(py, unique, PyResult<&'a PySet>, arr)
}

/// ndarray unique
fn unique<'a, T>(py: Python<'a>, arr: numpy::PyReadonlyArrayDyn<T>) -> PyResult<&'a PySet>
where
    T: Clone + numpy::Element + std::hash::Hash + std::cmp::Eq + pyo3::ToPyObject,
{
    let mut track = HashSet::<T>::new();
    let mut ret: Vec<T> = vec![];
    unsafe {
        for val in arr.as_array_mut().iter() {
            if !track.contains(val) {
                track.insert(val.clone());
                ret.push(val.clone());
            }
        }
    }
    PySet::new(py, ret.as_slice())
}
