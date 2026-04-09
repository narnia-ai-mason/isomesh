use pyo3::prelude::*;
use crate::bbox::CubicBounds;
use crate::eval::bridge;
use crate::eval::cache::{EvalCache, CachedEval};
use crate::octree::Octree;
use crate::octree::cell::{Cell, LeafData};
use crate::octree::types::GridPos;

/// Find all edge crossings in the octree and compute Hermite data
/// (intersection points + normals) via bisection.
///
/// Returns the number of Python batch calls made.
pub fn compute_hermite_data(
    py: Python<'_>,
    func: &PyObject,
    octree: &mut Octree,
    cache: &EvalCache,
) -> PyResult<u64> {
    let iso_value = octree.iso_value;
    let max_depth = octree.max_depth;
    let cell_size_at_max = octree.bounds.size / (1u32 << max_depth) as f64;
    let mut batch_calls = 0u64;

    // Collect all sign-change edges from leaf cells
    struct EdgeCrossing {
        p0: [f64; 3],
        p1: [f64; 3],
        v0: f64,
        v1: f64,
    }

    let mut crossings = Vec::new();
    let mut crossing_cell_info: Vec<(usize, u8)> = Vec::new(); // (leaf_index, edge_index)

    // Gather all leaves and their edges
    let mut leaf_info: Vec<(LeafData, CubicBounds)> = Vec::new();
    octree.for_each_leaf(|data, bounds, _depth| {
        leaf_info.push((data.clone(), *bounds));
    });

    for (leaf_idx, (data, bounds)) in leaf_info.iter().enumerate() {
        for edge in 0..12u8 {
            let (c0, c1) = CubicBounds::edge_corners(edge);
            let s0 = (data.corner_values[c0 as usize] - iso_value) < 0.0;
            let s1 = (data.corner_values[c1 as usize] - iso_value) < 0.0;

            if s0 != s1 {
                // Sign change on this edge
                let p0 = bounds.corner(c0);
                let p1 = bounds.corner(c1);
                let v0 = data.corner_values[c0 as usize] - iso_value;
                let v1 = data.corner_values[c1 as usize] - iso_value;

                crossings.push(EdgeCrossing { p0, p1, v0, v1 });
                crossing_cell_info.push((leaf_idx, edge));
            }
        }
    }

    if crossings.is_empty() {
        return Ok(0);
    }

    // Bisection: find zero crossing on each edge
    // Use linear interpolation as initial guess, then refine with bisection
    let num_crossings = crossings.len();
    let mut t_values: Vec<f64> = crossings.iter().map(|c| {
        // Linear interpolation: t = -v0 / (v1 - v0)
        let dv = c.v1 - c.v0;
        if dv.abs() < 1e-15 { 0.5 } else { (-c.v0 / dv).clamp(0.0, 1.0) }
    }).collect();

    let mut lo: Vec<f64> = vec![0.0; num_crossings];
    let mut hi: Vec<f64> = vec![1.0; num_crossings];
    let mut lo_val: Vec<f64> = crossings.iter().map(|c| c.v0).collect();
    let mut hi_val: Vec<f64> = crossings.iter().map(|c| c.v1).collect();

    // Bisection iterations (batched)
    let bisection_iters = 10; // ~1/1024 cell accuracy
    for _ in 0..bisection_iters {
        // Compute midpoints
        let mid_points: Vec<[f64; 3]> = (0..num_crossings).map(|i| {
            let t = (lo[i] + hi[i]) * 0.5;
            let c = &crossings[i];
            [
                c.p0[0] + t * (c.p1[0] - c.p0[0]),
                c.p0[1] + t * (c.p1[1] - c.p0[1]),
                c.p0[2] + t * (c.p1[2] - c.p0[2]),
            ]
        }).collect();

        // Batch evaluate
        let result = bridge::batch_evaluate(py, func, &mid_points)?;
        batch_calls += 1;

        // Update intervals
        for i in 0..num_crossings {
            let mid_val = result.values[i] - iso_value;
            let mid_t = (lo[i] + hi[i]) * 0.5;

            if (mid_val < 0.0) == (lo_val[i] < 0.0) {
                lo[i] = mid_t;
                lo_val[i] = mid_val;
            } else {
                hi[i] = mid_t;
                hi_val[i] = mid_val;
            }
        }
    }

    // Final intersection points
    let final_points: Vec<[f64; 3]> = (0..num_crossings).map(|i| {
        let t = (lo[i] + hi[i]) * 0.5;
        let c = &crossings[i];
        [
            c.p0[0] + t * (c.p1[0] - c.p0[0]),
            c.p0[1] + t * (c.p1[1] - c.p0[1]),
            c.p0[2] + t * (c.p1[2] - c.p0[2]),
        ]
    }).collect();

    // Get gradients at crossing points (for normals)
    let result = bridge::batch_evaluate(py, func, &final_points)?;
    batch_calls += 1;

    let normals: Vec<[f64; 3]> = if let Some(ref grads) = result.gradients {
        grads.iter().map(|g| {
            let len = (g[0]*g[0] + g[1]*g[1] + g[2]*g[2]).sqrt();
            if len > 1e-12 { [g[0]/len, g[1]/len, g[2]/len] } else { [0.0, 0.0, 1.0] }
        }).collect()
    } else {
        // Estimate gradients via FD
        let fd_grads = bridge::estimate_gradients_fd(py, func, &final_points, cell_size_at_max * 0.01)?;
        batch_calls += 1;
        fd_grads.iter().map(|g| {
            let len = (g[0]*g[0] + g[1]*g[1] + g[2]*g[2]).sqrt();
            if len > 1e-12 { [g[0]/len, g[1]/len, g[2]/len] } else { [0.0, 0.0, 1.0] }
        }).collect()
    };

    // Store Hermite data in the octree
    for i in 0..num_crossings {
        let t = (lo[i] + hi[i]) * 0.5;
        let hermite_idx = octree.hermite_data.len() as u32;
        octree.hermite_data.push(crate::octree::HermitePoint {
            pos: final_points[i],
            normal: normals[i],
            t,
        });

        // Store in leaf's edge_intersections
        let (leaf_idx, edge_idx) = crossing_cell_info[i];
        // We need to update the leaf in the octree
        // Since we collected leaf_info by order, we track with a counter
        // Store mapping for later use in extract
    }

    // Store crossings indexed by leaf for DC extraction
    // We store them in a separate structure since we can't easily update leaves in-place
    // The extract module will re-collect them

    Ok(batch_calls)
}
