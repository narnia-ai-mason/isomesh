use pyo3::prelude::*;
use std::collections::HashMap;
use crate::bbox::CubicBounds;
use crate::eval::bridge;
use crate::octree::Octree;
use crate::octree::cell::{Cell, LeafData};
use crate::qef::quadric::QefData;

pub struct ExtractedMesh {
    pub vertices: Vec<[f64; 3]>,
    pub faces: Vec<[i64; 3]>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct CellKey(u32, u32, u32, u8);

struct LeafInfo {
    data: LeafData,
    bounds: CubicBounds,
    depth: u8,
    key: CellKey,
}

/// Extract a triangle mesh using Dual Contouring.
///
/// Edge crossings are found via linear interpolation from cached corner values
/// (zero extra Python calls). Normals are obtained from the user function's
/// gradient output, or estimated via finite differences if not provided
/// (one extra Python call).
pub fn extract_dc(
    py: Python<'_>,
    func: &PyObject,
    octree: &Octree,
) -> PyResult<ExtractedMesh> {
    let iso_value = octree.iso_value;

    // Step 1: Collect all leaf cells with sign changes
    let mut leaves: Vec<LeafInfo> = Vec::new();
    collect_leaves_recursive(octree, &octree.root, &octree.bounds, 0, 0, 0, 0, &mut leaves);

    if leaves.is_empty() {
        return Ok(ExtractedMesh { vertices: Vec::new(), faces: Vec::new() });
    }

    let mut cell_map: HashMap<CellKey, usize> = HashMap::with_capacity(leaves.len());
    for (i, leaf) in leaves.iter().enumerate() {
        cell_map.insert(leaf.key, i);
    }

    // Step 2: Find edge crossings via linear interpolation (no extra Python calls)
    // and collect crossing points for normal estimation.
    let mut crossing_points: Vec<[f64; 3]> = Vec::new();
    let mut edge_leaf_map: Vec<(usize, u8)> = Vec::new();

    for (idx, leaf) in leaves.iter().enumerate() {
        for edge in 0..12u8 {
            let (c0, c1) = CubicBounds::edge_corners(edge);
            let v0 = leaf.data.corner_values[c0 as usize] - iso_value;
            let v1 = leaf.data.corner_values[c1 as usize] - iso_value;
            if (v0 < 0.0) != (v1 < 0.0) {
                // Linear interpolation: t = -v0 / (v1 - v0)
                let dv = v1 - v0;
                let t = if dv.abs() > 1e-15 { (-v0 / dv).clamp(0.001, 0.999) } else { 0.5 };
                let p0 = leaf.bounds.corner(c0);
                let p1 = leaf.bounds.corner(c1);
                crossing_points.push(lerp3(p0, p1, t));
                edge_leaf_map.push((idx, edge));
            }
        }
    }

    // Step 3: Get normals at crossing points (single batch Python call)
    let normals = if !crossing_points.is_empty() {
        let result = bridge::batch_evaluate(py, func, &crossing_points)?;
        if let Some(ref grads) = result.gradients {
            grads.iter().map(|g| normalize3(*g)).collect::<Vec<_>>()
        } else {
            let cell_size = octree.bounds.size / (1u32 << octree.max_depth) as f64;
            let fd = bridge::estimate_gradients_fd(py, func, &crossing_points, cell_size * 0.01)?;
            fd.iter().map(|g| normalize3(*g)).collect()
        }
    } else {
        Vec::new()
    };

    // Step 4: QEF vertex placement per leaf cell
    let mut qefs = vec![QefData::new(); leaves.len()];
    for (i, &(leaf_idx, _)) in edge_leaf_map.iter().enumerate() {
        qefs[leaf_idx].add(crossing_points[i], normals[i]);
    }

    let mut vertices = Vec::with_capacity(leaves.len());
    let mut leaf_vertex: Vec<Option<usize>> = vec![None; leaves.len()];
    for (i, leaf) in leaves.iter().enumerate() {
        if qefs[i].count == 0 { continue; }
        let (pos, _) = qefs[i].solve(leaf.bounds.corner(0), leaf.bounds.corner(7));
        leaf_vertex[i] = Some(vertices.len());
        vertices.push(pos);
    }

    // Step 5: Generate quads from canonical edges (0, 4, 8)
    let mut faces = Vec::new();

    const CANONICAL_EDGES: [u8; 3] = [0, 4, 8];
    const NON_AXIS_DIMS: [(usize, usize); 3] = [(1, 2), (0, 2), (0, 1)];
    const CCW_ORDER: [[usize; 4]; 3] = [
        [0, 1, 3, 2], // X-axis
        [0, 2, 3, 1], // Y-axis
        [0, 1, 3, 2], // Z-axis
    ];

    for (leaf_idx, leaf) in leaves.iter().enumerate() {
        if leaf_vertex[leaf_idx].is_none() { continue; }

        let d = leaf.depth;
        let cell_coords = [leaf.key.0, leaf.key.1, leaf.key.2];

        for axis in 0..3usize {
            let edge_idx = CANONICAL_EDGES[axis];
            let (a1, a2) = NON_AXIS_DIMS[axis];

            let (c0, c1) = CubicBounds::edge_corners(edge_idx);
            let val0 = leaf.data.corner_values[c0 as usize] - iso_value;
            let val1 = leaf.data.corner_values[c1 as usize] - iso_value;
            if (val0 < 0.0) == (val1 < 0.0) { continue; }

            if cell_coords[a1] == 0 || cell_coords[a2] == 0 { continue; }

            let mut n_a1 = cell_coords; n_a1[a1] -= 1;
            let mut n_a2 = cell_coords; n_a2[a2] -= 1;
            let mut n_both = cell_coords; n_both[a1] -= 1; n_both[a2] -= 1;

            let Some(&ni_a1) = cell_map.get(&CellKey(n_a1[0], n_a1[1], n_a1[2], d)) else { continue; };
            let Some(&ni_a2) = cell_map.get(&CellKey(n_a2[0], n_a2[1], n_a2[2], d)) else { continue; };
            let Some(&ni_both) = cell_map.get(&CellKey(n_both[0], n_both[1], n_both[2], d)) else { continue; };

            let Some(v_self) = leaf_vertex[leaf_idx] else { continue; };
            let Some(v_a1) = leaf_vertex[ni_a1] else { continue; };
            let Some(v_a2) = leaf_vertex[ni_a2] else { continue; };
            let Some(v_both) = leaf_vertex[ni_both] else { continue; };

            let verts = [v_self, v_a1, v_a2, v_both];
            let ccw = CCW_ORDER[axis];

            let quad = if val0 < 0.0 {
                [verts[ccw[0]], verts[ccw[1]], verts[ccw[2]], verts[ccw[3]]]
            } else {
                [verts[ccw[3]], verts[ccw[2]], verts[ccw[1]], verts[ccw[0]]]
            };

            // Split quad along the shorter diagonal for better triangle quality.
            // Diagonal a-c vs b-d: shorter diagonal produces more equilateral
            // triangles and better surface approximation on curved regions.
            let [a, b, c, d] = quad;
            let diag_ac = dist_sq(&vertices[a], &vertices[c]);
            let diag_bd = dist_sq(&vertices[b], &vertices[d]);
            if diag_ac <= diag_bd {
                faces.push([a as i64, b as i64, c as i64]);
                faces.push([a as i64, c as i64, d as i64]);
            } else {
                faces.push([a as i64, b as i64, d as i64]);
                faces.push([b as i64, c as i64, d as i64]);
            }
        }
    }

    Ok(ExtractedMesh { vertices, faces })
}

fn collect_leaves_recursive(
    octree: &Octree, cell: &Cell, bounds: &CubicBounds,
    cx: u32, cy: u32, cz: u32, depth: u8,
    result: &mut Vec<LeafInfo>,
) {
    match cell {
        Cell::Leaf(data) if data.has_sign_change() => {
            result.push(LeafInfo {
                data: data.clone(), bounds: *bounds, depth,
                key: CellKey(cx, cy, cz, depth),
            });
        }
        Cell::Branch { children_index } => {
            let children = &octree.children[*children_index as usize];
            for octant in 0..8u8 {
                collect_leaves_recursive(
                    octree, &children[octant as usize], &bounds.child(octant),
                    cx * 2 + if octant & 1 != 0 { 1 } else { 0 },
                    cy * 2 + if octant & 2 != 0 { 1 } else { 0 },
                    cz * 2 + if octant & 4 != 0 { 1 } else { 0 },
                    depth + 1, result,
                );
            }
        }
        _ => {}
    }
}

fn dist_sq(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

fn lerp3(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])]
}

fn normalize3(v: [f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-12 { [v[0] / len, v[1] / len, v[2] / len] } else { [0.0, 0.0, 1.0] }
}
