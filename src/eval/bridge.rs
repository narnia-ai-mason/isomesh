use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Result of a batch evaluation from the Python callable.
pub struct EvalResult {
    /// Function values at each query point, shape (N,).
    pub values: Vec<f64>,
    /// Gradients at each query point, shape (N, 3). None if the function didn't provide them.
    pub gradients: Option<Vec<[f64; 3]>>,
}

/// Calls the Python implicit function with a batch of query points.
///
/// The Python function signature is:
///   func(positions: ndarray(N, 3)) -> (values: ndarray(N,), gradients: ndarray(N, 3) | None)
///
/// This function acquires the GIL, calls the Python function, extracts the results,
/// and returns owned Rust data. The GIL is released when this function returns.
///
/// SAFETY: Must NOT be called from within a Rayon parallel section that was entered
/// while holding the GIL. The caller must have released the GIL before entering Rayon,
/// and this function will acquire it independently.
pub fn batch_evaluate(py: Python<'_>, func: &PyObject, points: &[[f64; 3]]) -> PyResult<EvalResult> {
    let n = points.len();
    if n == 0 {
        return Ok(EvalResult {
            values: Vec::new(),
            gradients: None,
        });
    }

    // Create (N, 3) numpy array from points — this copies the data into a Python-owned array
    let positions = PyArray2::from_vec2(py, &points.iter().map(|p| p.to_vec()).collect::<Vec<_>>())?;

    // Call the Python function
    let result = func.call1(py, (positions,))?;

    // Unpack the (values, gradients) tuple
    let tuple = result.bind(py);
    let values_obj = tuple.get_item(0)?;
    let gradients_obj = tuple.get_item(1)?;

    // Extract values as (N,) f64 array
    let values_array: PyReadonlyArray1<f64> = values_obj.extract()?;
    let values: Vec<f64> = values_array.as_slice()?.to_vec();

    // Extract gradients as (N, 3) f64 array or None
    let gradients = if gradients_obj.is_none() {
        None
    } else {
        let grad_array: PyReadonlyArray2<f64> = gradients_obj.extract()?;
        let grad_view = grad_array.as_array();
        let mut grads = Vec::with_capacity(n);
        for row in grad_view.rows() {
            grads.push([row[0], row[1], row[2]]);
        }
        Some(grads)
    };

    Ok(EvalResult { values, gradients })
}

/// Calls the Python implicit function, releasing the GIL around Rust-side work.
///
/// This is the primary entry point for batch evaluation from the algorithm core.
/// It acquires the GIL only for the Python call, keeping it released otherwise.
pub fn batch_evaluate_with_gil(func: &PyObject, points: &[[f64; 3]]) -> PyResult<EvalResult> {
    Python::with_gil(|py| batch_evaluate(py, func, points))
}

/// Estimates gradients via centered finite differences.
///
/// For each point, evaluates f(x ± h·e_i) for i=0,1,2 and computes:
///   grad_i = (f(x + h·e_i) - f(x - h·e_i)) / (2h)
///
/// This batches all 6N evaluation points into a single Python call.
pub fn estimate_gradients_fd(
    py: Python<'_>,
    func: &PyObject,
    points: &[[f64; 3]],
    step: f64,
) -> PyResult<Vec<[f64; 3]>> {
    let n = points.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Build 6N points: for each original point, 2 offsets per axis
    let mut fd_points = Vec::with_capacity(n * 6);
    for p in points {
        for axis in 0..3usize {
            let mut p_plus = *p;
            p_plus[axis] += step;
            fd_points.push(p_plus);

            let mut p_minus = *p;
            p_minus[axis] -= step;
            fd_points.push(p_minus);
        }
    }

    // Single batch evaluation for all FD points
    let result = batch_evaluate(py, func, &fd_points)?;
    let vals = &result.values;

    // Reconstruct gradients
    let inv_2h = 0.5 / step;
    let mut gradients = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * 6;
        gradients.push([
            (vals[base] - vals[base + 1]) * inv_2h,
            (vals[base + 2] - vals[base + 3]) * inv_2h,
            (vals[base + 4] - vals[base + 5]) * inv_2h,
        ]);
    }

    Ok(gradients)
}

#[cfg(test)]
mod tests {
    // Rust-side unit tests require Python interpreter, tested via pytest instead.
}
