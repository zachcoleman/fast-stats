use numpy::*;
use pyo3::prelude::*;
use std::{iter::zip, collections::HashMap};

/// Confusion Matrix
#[pyfunction]
#[pyo3(name = "_confusion_matrix")]
#[pyo3(text_signature = "(actual: np.ndarray, pred: np.ndarray, labels: List[int], /)")]
pub fn confusion_matrix<'a>(
    py: Python<'a>,
    actual: &PyArrayDyn<i64>,
    pred: &PyArrayDyn<i64>,
    labels: Vec<i64>,
) -> &'a PyArray2<i64>  {
    let mut cm = ndarray::Array2::<i64>::from_elem((labels.len(), labels.len()), 0);
    let idx_map: HashMap<i64, usize> = HashMap::from_iter(
        labels
        .iter()
        .enumerate()
        .map(|(x, y)| (*y, x))
    );
    
    for (y_pred, y_actual) in zip(
        pred.to_owned_array().iter(),
        actual.to_owned_array().iter(),
    ) {
        let ix1 = *idx_map.get(y_pred).unwrap();
        let ix2 = *idx_map.get(y_actual).unwrap();
        *cm.get_mut((ix1, ix2)).unwrap() = *cm.get_mut((ix1, ix2)).unwrap() + 1;
    }

    return &PyArray2::from_array(py, &cm);
}


// dispatchin
// fn dispatch(
//     _py: Python<'_>,
//     actual: &PyAny,
//     pred: &PyAny,
// ) -> PyResult<(i128, i128, i128)> {
//     // bool
//     if let (Ok(i), Ok(j)) = (
//         actual.extract::<PyReadonlyArrayDyn<bool>>(),
//         pred.extract::<PyReadonlyArrayDyn<bool>>(),
//     ) {
    
//     }

//     // i8
//     if let (Ok(i), Ok(j)) = (
//         actual.extract::<PyReadonlyArrayDyn<i8>>(),
//         pred.extract::<PyReadonlyArrayDyn<i8>>(),
//     ) {
    
//     }

//     // i16
//     if let (Ok(i), Ok(j)) = (
//         actual.extract::<PyReadonlyArrayDyn<i16>>(),
//         pred.extract::<PyReadonlyArrayDyn<i16>>(),
//     ) {

//     }

//     // i32
//     if let (Ok(i), Ok(j)) = (
//         actual.extract::<PyReadonlyArrayDyn<i32>>(),
//         pred.extract::<PyReadonlyArrayDyn<i32>>(),
//     ) {

//     }

//     // i64
//     if let (Ok(i), Ok(j)) = (
//         actual.extract::<PyReadonlyArrayDyn<i64>>(),
//         pred.extract::<PyReadonlyArrayDyn<i64>>(),
//     ) {
    
//     }

//     // u8
//     if let (Ok(i), Ok(j)) = (
//         actual.extract::<PyReadonlyArrayDyn<u8>>(),
//         pred.extract::<PyReadonlyArrayDyn<u8>>(),
//     ) {
    
//     }

//     // u16
//     if let (Ok(i), Ok(j)) = (
//         actual.extract::<PyReadonlyArrayDyn<u16>>(),
//         pred.extract::<PyReadonlyArrayDyn<u16>>(),
//     ) {
    
//     }

//     // u32
//     if let (Ok(i), Ok(j)) = (
//         actual.extract::<PyReadonlyArrayDyn<u32>>(),
//         pred.extract::<PyReadonlyArrayDyn<u32>>(),
//     ) {
     
//     }

//     // u64
//     if let (Ok(i), Ok(j)) = (
//         actual.extract::<PyReadonlyArrayDyn<u64>>(),
//         pred.extract::<PyReadonlyArrayDyn<u64>>(),
//     ) {
        
//     }

//     Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
//         "Unsupported numpy dtype",
//     ))
// }