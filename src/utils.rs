use numpy::*;
use pyo3::{prelude::*, types::PySet};
use std::{collections::HashSet};

/// unique
#[pyfunction]
#[pyo3(name = "_unique")]
#[pyo3(text_signature = "(arr: np.ndarray, /)")]
pub fn py_unique<'a>(_py: Python<'a>, arr: &PyAny) -> PyResult<&'a PySet> {
    dispatch(_py, arr)
}

// &PyArrayDyn? 
fn unique<'a, T>(py: Python<'a>, arr: numpy::PyReadonlyArrayDyn<T>) -> PyResult<&'a PySet>
where
    T: Clone + numpy::Element + std::hash::Hash + std::cmp::Eq + pyo3::ToPyObject, 
{
    let mut track = HashSet::<T>::new();
    let mut ret: Vec<T> = vec![];
    for val in arr.readonly().as_array().iter(){
        if !track.contains(val){
            track.insert(val.clone());
            ret.push(val.clone());
        }
    }

    PySet::new(py, ret.as_slice())
}

/// dispatching
fn dispatch<'a>(
    py: Python<'a>,
    actual: &PyAny,
) -> PyResult<&'a PySet> {
    // bool
    if let Ok(i) = actual.extract::<PyReadonlyArrayDyn<bool>>() {
        return unique::<bool>(py, i);
    }

    // i8
    if let Ok(i) = actual.extract::<PyReadonlyArrayDyn<i8>>() {
        return unique::<i8>(py, i);
    }

    // i16
    if let Ok(i) = actual.extract::<PyReadonlyArrayDyn<i16>>() {
        return unique::<i16>(py, i);
        
    }

    // i32
    if let Ok(i) = actual.extract::<PyReadonlyArrayDyn<i32>>() {
        return unique::<i32>(py, i);
    }

    // i64
    if let Ok(i) = actual.extract::<PyReadonlyArrayDyn<i64>>() {
        return unique::<i64>(py, i);
    }

    // u8
    if let Ok(i) = actual.extract::<PyReadonlyArrayDyn<u8>>() {
        return unique::<u8>(py, i);
    }

    // u16
    if let Ok(i) = actual.extract::<PyReadonlyArrayDyn<u16>>() {
        return unique::<u16>(py, i);
    }

    // u32
    if let Ok(i) = actual.extract::<PyReadonlyArrayDyn<u32>>() {
        return unique::<u32>(py, i);
    }

    // u64
    if let Ok(i) = actual.extract::<PyReadonlyArrayDyn<u64>>() {
        return unique::<u64>(py, i);
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Unsupported numpy dtype",
    ))
}