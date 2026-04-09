use pyo3::prelude::*;
use crate::eval::bridge;

/// Estimates gradients via centered finite differences using a batch evaluation.
///
/// For N points, creates 6N query points (+/- h along each axis),
/// calls the Python function once, and reconstructs the gradients.
pub fn estimate_gradients_batch(
    py: Python<'_>,
    func: &PyObject,
    points: &[[f64; 3]],
    step: f64,
) -> PyResult<Vec<[f64; 3]>> {
    bridge::estimate_gradients_fd(py, func, points, step)
}
