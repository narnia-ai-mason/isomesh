mod bbox;
mod dc;
mod eval;
mod mesh;
mod octree;
mod qef;
mod util;

use bbox::BoundingBox;
use dc::extract;
use eval::bridge;
use numpy::IntoPyArray;
use numpy::PyArray2;
use octree::build;
use pyo3::prelude::*;

/// Isosurface extraction entry point.
///
/// Builds an adaptive octree, runs Dual Contouring, and returns a triangle mesh.
#[pyfunction]
#[pyo3(signature = (
    eval_fn,
    bbox_min,
    bbox_max,
    min_depth = 3,
    max_depth = 7,
    angle_threshold_deg = 30.0,
    iso_value = 0.0,
))]
fn extract_mesh(
    py: Python<'_>,
    eval_fn: PyObject,
    bbox_min: [f64; 3],
    bbox_max: [f64; 3],
    min_depth: u32,
    max_depth: u32,
    angle_threshold_deg: f64,
    iso_value: f64,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<i64>>)> {
    let _ = angle_threshold_deg;

    let bb = BoundingBox::new(bbox_min, bbox_max);
    let (octree, _total_evals) = build::build_adaptive(
        py, &eval_fn, bb,
        min_depth as u8, max_depth as u8,
        iso_value,
    )?;

    // Run Dual Contouring extraction
    let mesh = extract::extract_dc(py, &eval_fn, &octree)?;

    let nv = mesh.vertices.len();
    let nf = mesh.faces.len();

    let verts_flat: Vec<f64> = mesh.vertices.iter().flat_map(|v| v.iter().copied()).collect();
    let faces_flat: Vec<i64> = mesh.faces.iter().flat_map(|f| f.iter().copied()).collect();

    let verts_array = numpy::ndarray::Array2::from_shape_vec((nv, 3), verts_flat)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        .into_pyarray(py).into();
    let faces_array = numpy::ndarray::Array2::from_shape_vec((nf, 3), faces_flat)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        .into_pyarray(py).into();

    Ok((verts_array, faces_array))
}

/// Build a uniform octree and return statistics.
#[pyfunction]
#[pyo3(signature = (eval_fn, bbox_min, bbox_max, depth, iso_value = 0.0))]
fn build_octree_uniform(
    py: Python<'_>,
    eval_fn: PyObject,
    bbox_min: [f64; 3],
    bbox_max: [f64; 3],
    depth: u32,
    iso_value: f64,
) -> PyResult<(u64, u64, u64, u64, u8, u64)> {
    let bb = BoundingBox::new(bbox_min, bbox_max);
    let (octree, total_evals) = build::build_uniform(py, &eval_fn, bb, depth as u8, iso_value)?;
    let stats = octree.stats();
    Ok((
        stats.leaf_count, stats.branch_count,
        stats.empty_count, stats.full_count,
        stats.max_actual_depth, total_evals,
    ))
}

/// Build an adaptive octree and return statistics.
#[pyfunction]
#[pyo3(signature = (eval_fn, bbox_min, bbox_max, min_depth, max_depth, iso_value = 0.0))]
fn build_octree_adaptive(
    py: Python<'_>,
    eval_fn: PyObject,
    bbox_min: [f64; 3],
    bbox_max: [f64; 3],
    min_depth: u32,
    max_depth: u32,
    iso_value: f64,
) -> PyResult<(u64, u64, u64, u64, u8, u64)> {
    let bb = BoundingBox::new(bbox_min, bbox_max);
    let (octree, total_evals) = build::build_adaptive(
        py, &eval_fn, bb,
        min_depth as u8, max_depth as u8,
        iso_value,
    )?;
    let stats = octree.stats();
    Ok((
        stats.leaf_count, stats.branch_count,
        stats.empty_count, stats.full_count,
        stats.max_actual_depth, total_evals,
    ))
}

/// Test function to verify the batch evaluation bridge works.
#[pyfunction]
fn test_batch_eval(
    py: Python<'_>,
    eval_fn: PyObject,
    points: Vec<[f64; 3]>,
) -> PyResult<(Vec<f64>, bool, usize)> {
    let result = bridge::batch_evaluate(py, &eval_fn, &points)?;
    let has_grads = result.gradients.is_some();
    let n = result.values.len();
    Ok((result.values, has_grads, n))
}

/// Test function for finite-difference gradient estimation.
#[pyfunction]
#[pyo3(signature = (eval_fn, points, step = 1e-5))]
fn test_fd_gradients(
    py: Python<'_>,
    eval_fn: PyObject,
    points: Vec<[f64; 3]>,
    step: f64,
) -> PyResult<Vec<[f64; 3]>> {
    bridge::estimate_gradients_fd(py, &eval_fn, &points, step)
}

#[pymodule]
fn _isomesh_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(build_octree_uniform, m)?)?;
    m.add_function(wrap_pyfunction!(build_octree_adaptive, m)?)?;
    m.add_function(wrap_pyfunction!(test_batch_eval, m)?)?;
    m.add_function(wrap_pyfunction!(test_fd_gradients, m)?)?;
    Ok(())
}
