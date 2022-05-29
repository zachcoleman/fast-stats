/// Dispatching always calls functions w/ PyReadonlyArrayDyn<T>
#[macro_export]
macro_rules! numpy_dispatch_no_bool {
    // single arg function
    ($py:ident, $f:ident, $ret_type:ty, $arr:ident) => {
        |x: &'a PyAny| -> $ret_type {
            if let Ok(i) = x.extract::<numpy::PyReadonlyArrayDyn<i8>>() {
                return $f::<i8>($py, i);
            }
            if let Ok(i) = x.extract::<numpy::PyReadonlyArrayDyn<i16>>() {
                return $f::<i16>($py, i);
            }
            if let Ok(i) = x.extract::<numpy::PyReadonlyArrayDyn<i32>>() {
                return $f::<i32>($py, i);
            }
            if let Ok(i) = x.extract::<numpy::PyReadonlyArrayDyn<i64>>() {
                return $f::<i64>($py, i);
            }
            if let Ok(i) = x.extract::<numpy::PyReadonlyArrayDyn<u8>>() {
                return $f::<u8>($py, i);
            }
            if let Ok(i) = x.extract::<numpy::PyReadonlyArrayDyn<u16>>() {
                return $f::<u16>($py, i);
            }
            if let Ok(i) = x.extract::<numpy::PyReadonlyArrayDyn<u32>>() {
                return $f::<u32>($py, i);
            }
            if let Ok(i) = x.extract::<numpy::PyReadonlyArrayDyn<u64>>() {
                return $f::<u64>($py, i);
            }
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported numpy dtype",
            ))
        }($arr)
    };

    // two arg function
    ($py:ident, $f:ident, $ret_type:ty, $arr1:ident, $arr2:ident) => {
        |x: &'a PyAny, y: &'a PyAny| -> $ret_type {
            if let (Ok(i), Ok(j)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<bool>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<bool>>(),
            ) {
                let i =
                    PyArrayDyn::from_array($py, &i.to_owned_array().mapv(|e| e as u8)).readonly();
                let j =
                    PyArrayDyn::from_array($py, &j.to_owned_array().mapv(|e| e as u8)).readonly();
                return $f::<u8>($py, i, j);
            }
            if let (Ok(i), Ok(j)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<i8>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<i8>>(),
            ) {
                return $f::<i8>($py, i, j);
            }
            if let (Ok(i), Ok(j)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<i16>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<i16>>(),
            ) {
                return $f::<i16>($py, i, j);
            }
            if let (Ok(i), Ok(j)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<i32>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<i32>>(),
            ) {
                return $f::<i32>($py, i, j);
            }
            if let (Ok(i), Ok(j)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<i64>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<i64>>(),
            ) {
                return $f::<i64>($py, i, j);
            }
            if let (Ok(i), Ok(j)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<u8>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<u8>>(),
            ) {
                return $f::<u8>($py, i, j);
            }
            if let (Ok(i), Ok(j)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<u16>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<u16>>(),
            ) {
                return $f::<u16>($py, i, j);
            }
            if let (Ok(i), Ok(j)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<u32>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<u32>>(),
            ) {
                return $f::<u32>($py, i, j);
            }
            if let (Ok(i), Ok(j)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<u64>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<u64>>(),
            ) {
                return $f::<u64>($py, i, j);
            }
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported numpy dtype",
            ))
        }($arr1, $arr2)
    };

    // three arg function
    ($py:ident, $f:ident, $ret_type:ty, $arr1:ident, $arr2:ident, $arr3:ident) => {
        |x: &'a PyAny, y: &'a PyAny, z: &'a PyAny| -> $ret_type {
            if let (Ok(i), Ok(j), Ok(k)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<bool>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<bool>>(),
                z.extract::<numpy::PyReadonlyArrayDyn<bool>>(),
            ) {
                return $f::<bool>($py, i, j, k);
            }
            if let (Ok(i), Ok(j), Ok(k)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<i8>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<i8>>(),
                z.extract::<numpy::PyReadonlyArrayDyn<i8>>(),
            ) {
                return $f::<i8>($py, i, j, k);
            }
            if let (Ok(i), Ok(j), Ok(k)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<i16>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<i16>>(),
                z.extract::<numpy::PyReadonlyArrayDyn<i16>>(),
            ) {
                return $f::<i16>($py, i, j, k);
            }
            if let (Ok(i), Ok(j), Ok(k)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<i32>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<i32>>(),
                z.extract::<numpy::PyReadonlyArrayDyn<i32>>(),
            ) {
                return $f::<i32>($py, i, j, k);
            }
            if let (Ok(i), Ok(j), Ok(k)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<i64>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<i64>>(),
                z.extract::<numpy::PyReadonlyArrayDyn<i64>>(),
            ) {
                return $f::<i64>($py, i, j, k);
            }
            if let (Ok(i), Ok(j), Ok(k)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<u8>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<u8>>(),
                z.extract::<numpy::PyReadonlyArrayDyn<u8>>(),
            ) {
                return $f::<u8>($py, i, j, k);
            }
            if let (Ok(i), Ok(j), Ok(k)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<u16>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<u16>>(),
                z.extract::<numpy::PyReadonlyArrayDyn<u16>>(),
            ) {
                return $f::<u16>($py, i, j, k);
            }
            if let (Ok(i), Ok(j), Ok(k)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<u32>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<u32>>(),
                z.extract::<numpy::PyReadonlyArrayDyn<u32>>(),
            ) {
                return $f::<u32>($py, i, j, k);
            }
            if let (Ok(i), Ok(j), Ok(k)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<u64>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<u64>>(),
                z.extract::<numpy::PyReadonlyArrayDyn<u64>>(),
            ) {
                return $f::<u64>($py, i, j, k);
            }
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported numpy dtype",
            ))
        }($arr1, $arr2, $arr3)
    };
}

#[macro_export]
macro_rules! numpy_dispatch_bool {
    // single arg function
    ($py:ident, $f:ident, $ret_type:ty, $arr:ident) => {
        |x: &'a PyAny| -> $ret_type {
            if let Ok(i) = x.extract::<numpy::PyReadonlyArrayDyn<bool>>() {
                return $f::<bool>($py, i);
            }
            if let Ok(i) = x.extract::<numpy::PyReadonlyArrayDyn<i8>>() {
                return $f::<i8>($py, i);
            }
            if let Ok(i) = x.extract::<numpy::PyReadonlyArrayDyn<i16>>() {
                return $f::<i16>($py, i);
            }
            if let Ok(i) = x.extract::<numpy::PyReadonlyArrayDyn<i32>>() {
                return $f::<i32>($py, i);
            }
            if let Ok(i) = x.extract::<numpy::PyReadonlyArrayDyn<i64>>() {
                return $f::<i64>($py, i);
            }
            if let Ok(i) = x.extract::<numpy::PyReadonlyArrayDyn<u8>>() {
                return $f::<u8>($py, i);
            }
            if let Ok(i) = x.extract::<numpy::PyReadonlyArrayDyn<u16>>() {
                return $f::<u16>($py, i);
            }
            if let Ok(i) = x.extract::<numpy::PyReadonlyArrayDyn<u32>>() {
                return $f::<u32>($py, i);
            }
            if let Ok(i) = x.extract::<numpy::PyReadonlyArrayDyn<u64>>() {
                return $f::<u64>($py, i);
            }
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported numpy dtype",
            ))
        }($arr)
    };

    // two arg function
    ($py:ident, $f:ident, $ret_type:ty, $arr1:ident, $arr2:ident) => {
        |x: &'a PyAny, y: &'a PyAny| -> $ret_type {
            if let (Ok(i), Ok(j)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<bool>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<bool>>(),
            ) {
                return $f::<bool>($py, i, j);
            }
            if let (Ok(i), Ok(j)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<i8>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<i8>>(),
            ) {
                return $f::<i8>($py, i, j);
            }
            if let (Ok(i), Ok(j)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<i16>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<i16>>(),
            ) {
                return $f::<i16>($py, i, j);
            }
            if let (Ok(i), Ok(j)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<i32>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<i32>>(),
            ) {
                return $f::<i32>($py, i, j);
            }
            if let (Ok(i), Ok(j)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<i64>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<i64>>(),
            ) {
                return $f::<i64>($py, i, j);
            }
            if let (Ok(i), Ok(j)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<u8>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<u8>>(),
            ) {
                return $f::<u8>($py, i, j);
            }
            if let (Ok(i), Ok(j)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<u16>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<u16>>(),
            ) {
                return $f::<u16>($py, i, j);
            }
            if let (Ok(i), Ok(j)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<u32>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<u32>>(),
            ) {
                return $f::<u32>($py, i, j);
            }
            if let (Ok(i), Ok(j)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<u64>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<u64>>(),
            ) {
                return $f::<u64>($py, i, j);
            }
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported numpy dtype",
            ))
        }($arr1, $arr2)
    };

    // three arg function
    ($py:ident, $f:ident, $ret_type:ty, $arr1:ident, $arr2:ident, $arr3:ident) => {
        |x: &'a PyAny, y: &'a PyAny, z: &'a PyAny| -> $ret_type {
            if let (Ok(i), Ok(j), Ok(k)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<bool>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<bool>>(),
                z.extract::<numpy::PyReadonlyArrayDyn<bool>>(),
            ) {
                return $f::<bool>($py, i, j, k);
            }
            if let (Ok(i), Ok(j), Ok(k)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<i8>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<i8>>(),
                z.extract::<numpy::PyReadonlyArrayDyn<i8>>(),
            ) {
                return $f::<i8>($py, i, j, k);
            }
            if let (Ok(i), Ok(j), Ok(k)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<i16>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<i16>>(),
                z.extract::<numpy::PyReadonlyArrayDyn<i16>>(),
            ) {
                return $f::<i16>($py, i, j, k);
            }
            if let (Ok(i), Ok(j), Ok(k)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<i32>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<i32>>(),
                z.extract::<numpy::PyReadonlyArrayDyn<i32>>(),
            ) {
                return $f::<i32>($py, i, j, k);
            }
            if let (Ok(i), Ok(j), Ok(k)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<i64>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<i64>>(),
                z.extract::<numpy::PyReadonlyArrayDyn<i64>>(),
            ) {
                return $f::<i64>($py, i, j, k);
            }
            if let (Ok(i), Ok(j), Ok(k)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<u8>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<u8>>(),
                z.extract::<numpy::PyReadonlyArrayDyn<u8>>(),
            ) {
                return $f::<u8>($py, i, j, k);
            }
            if let (Ok(i), Ok(j), Ok(k)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<u16>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<u16>>(),
                z.extract::<numpy::PyReadonlyArrayDyn<u16>>(),
            ) {
                return $f::<u16>($py, i, j, k);
            }
            if let (Ok(i), Ok(j), Ok(k)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<u32>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<u32>>(),
                z.extract::<numpy::PyReadonlyArrayDyn<u32>>(),
            ) {
                return $f::<u32>($py, i, j, k);
            }
            if let (Ok(i), Ok(j), Ok(k)) = (
                x.extract::<numpy::PyReadonlyArrayDyn<u64>>(),
                y.extract::<numpy::PyReadonlyArrayDyn<u64>>(),
                z.extract::<numpy::PyReadonlyArrayDyn<u64>>(),
            ) {
                return $f::<u64>($py, i, j, k);
            }
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported numpy dtype",
            ))
        }($arr1, $arr2, $arr3)
    };
}
