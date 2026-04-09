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
/// For each leaf cell, processes the 3 canonical edges (edges 0, 4, 8 — at the
/// cell's min corner in both non-axis dimensions). For each edge with a sign
/// change, finds the 3 neighboring cells sharing that edge and emits a quad
/// (split into 2 triangles) connecting the 4 QEF vertices.
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

    // Build cell lookup by (x, y, z, depth)
    let mut cell_map: HashMap<CellKey, usize> = HashMap::new();
    for (i, leaf) in leaves.iter().enumerate() {
        cell_map.insert(leaf.key, i);
    }

    // Step 2: Find edge crossings via bisection (batched)
    let mut all_edge_data: Vec<([f64; 3], [f64; 3], f64, f64)> = Vec::new();
    let mut edge_leaf_map: Vec<(usize, u8)> = Vec::new();

    for (idx, leaf) in leaves.iter().enumerate() {
        for edge in 0..12u8 {
            let (c0, c1) = CubicBounds::edge_corners(edge);
            let v0 = leaf.data.corner_values[c0 as usize] - iso_value;
            let v1 = leaf.data.corner_values[c1 as usize] - iso_value;
            if (v0 < 0.0) != (v1 < 0.0) {
                all_edge_data.push((leaf.bounds.corner(c0), leaf.bounds.corner(c1), v0, v1));
                edge_leaf_map.push((idx, edge));
            }
        }
    }

    let num_crossings = all_edge_data.len();

    // Bisection: 10 iterations (~1/1024 cell accuracy)
    let mut lo = vec![0.0f64; num_crossings];
    let mut hi = vec![1.0f64; num_crossings];
    let mut lo_val: Vec<f64> = all_edge_data.iter().map(|e| e.2).collect();

    if num_crossings > 0 {
        for _ in 0..10 {
            let mid_points: Vec<[f64; 3]> = (0..num_crossings).map(|i| {
                lerp3(all_edge_data[i].0, all_edge_data[i].1, (lo[i] + hi[i]) * 0.5)
            }).collect();
            let result = bridge::batch_evaluate(py, func, &mid_points)?;
            for i in 0..num_crossings {
                let mid_val = result.values[i] - iso_value;
                let mid_t = (lo[i] + hi[i]) * 0.5;
                if (mid_val < 0.0) == (lo_val[i] < 0.0) {
                    lo[i] = mid_t;
                    lo_val[i] = mid_val;
                } else {
                    hi[i] = mid_t;
                }
            }
        }
    }

    // Final crossing points + normals
    let crossing_points: Vec<[f64; 3]> = (0..num_crossings).map(|i| {
        lerp3(all_edge_data[i].0, all_edge_data[i].1, (lo[i] + hi[i]) * 0.5)
    }).collect();

    let normals = if num_crossings > 0 {
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

    // Step 3: QEF vertex placement per leaf cell
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

    // Step 4: Generate quads from shared edges
    //
    // For each leaf cell, process only its 3 "canonical" edges (edges 0, 4, 8):
    //   edge 0: X-axis, at cell's (Y=min, Z=min) corner
    //   edge 4: Y-axis, at cell's (X=min, Z=min) corner
    //   edge 8: Z-axis, at cell's (X=min, Y=min) corner
    //
    // Each canonical edge is shared by exactly 4 cells:
    //   self, neighbor at -1 in a1, neighbor at -1 in a2, neighbor at -1 in both
    //
    // This ensures each grid edge is processed exactly once.

    let mut faces = Vec::new();

    // Non-axis dimension pairs and CCW vertex ordering per axis.
    // CCW order gives outward normal in +axis direction.
    //
    // For X-axis (a1=Y, a2=Z): 4 cells at offsets (0,0), (-1,0), (0,-1), (-1,-1) in (Y,Z)
    //   CCW from +X: [self, n_a1, n_both, n_a2] = [0, 1, 3, 2]
    // For Y-axis (a1=X, a2=Z): offsets in (X,Z)
    //   CCW from +Y: [self, n_a2, n_both, n_a1] = [0, 2, 3, 1]
    // For Z-axis (a1=X, a2=Y): offsets in (X,Y)
    //   CCW from +Z: [self, n_a1, n_both, n_a2] = [0, 1, 3, 2]
    const CANONICAL_EDGES: [u8; 3] = [0, 4, 8];
    const NON_AXIS_DIMS: [(usize, usize); 3] = [(1, 2), (0, 2), (0, 1)]; // (a1, a2) per axis
    const CCW_ORDER: [[usize; 4]; 3] = [
        [0, 1, 3, 2], // X-axis
        [0, 2, 3, 1], // Y-axis
        [0, 1, 3, 2], // Z-axis
    ];

    for (leaf_idx, leaf) in leaves.iter().enumerate() {
        if leaf_vertex[leaf_idx].is_none() { continue; }

        let d = leaf.depth;
        let cx = leaf.key.0;
        let cy = leaf.key.1;
        let cz = leaf.key.2;
        let cell_coords = [cx, cy, cz];

        for axis in 0..3usize {
            let edge_idx = CANONICAL_EDGES[axis];
            let (a1, a2) = NON_AXIS_DIMS[axis];

            // Canonical edge: corners 0 and (1 << axis)
            let (c0, c1) = CubicBounds::edge_corners(edge_idx);
            let val0 = leaf.data.corner_values[c0 as usize] - iso_value;
            let val1 = leaf.data.corner_values[c1 as usize] - iso_value;
            if (val0 < 0.0) == (val1 < 0.0) { continue; }

            // Check boundary: neighbors at -1 in a1 and a2 must exist
            if cell_coords[a1] == 0 || cell_coords[a2] == 0 { continue; }

            // Find 3 neighbor cells
            let mut n_a1_coords = cell_coords;
            n_a1_coords[a1] -= 1;
            let mut n_a2_coords = cell_coords;
            n_a2_coords[a2] -= 1;
            let mut n_both_coords = cell_coords;
            n_both_coords[a1] -= 1;
            n_both_coords[a2] -= 1;

            let Some(&ni_a1) = cell_map.get(&CellKey(n_a1_coords[0], n_a1_coords[1], n_a1_coords[2], d)) else { continue; };
            let Some(&ni_a2) = cell_map.get(&CellKey(n_a2_coords[0], n_a2_coords[1], n_a2_coords[2], d)) else { continue; };
            let Some(&ni_both) = cell_map.get(&CellKey(n_both_coords[0], n_both_coords[1], n_both_coords[2], d)) else { continue; };

            // All 4 cells must have QEF vertices
            let Some(v_self) = leaf_vertex[leaf_idx] else { continue; };
            let Some(v_a1) = leaf_vertex[ni_a1] else { continue; };
            let Some(v_a2) = leaf_vertex[ni_a2] else { continue; };
            let Some(v_both) = leaf_vertex[ni_both] else { continue; };

            // Vertex array: [self=0, n_a1=1, n_a2=2, n_both=3]
            let verts = [v_self, v_a1, v_a2, v_both];
            let ccw = CCW_ORDER[axis];

            // If val0 < 0 (inside) and val1 >= 0 (outside):
            //   gradient points in +axis direction → quad normal should be +axis → use CCW
            // Otherwise: use CW (reversed CCW)
            let quad = if val0 < 0.0 {
                [verts[ccw[0]], verts[ccw[1]], verts[ccw[2]], verts[ccw[3]]]
            } else {
                [verts[ccw[3]], verts[ccw[2]], verts[ccw[1]], verts[ccw[0]]]
            };

            // Split quad into 2 triangles
            faces.push([quad[0] as i64, quad[1] as i64, quad[2] as i64]);
            faces.push([quad[0] as i64, quad[2] as i64, quad[3] as i64]);
        }
    }

    Ok(ExtractedMesh { vertices, faces })
}

fn collect_leaves_recursive(
    octree: &Octree,
    cell: &Cell,
    bounds: &CubicBounds,
    cx: u32, cy: u32, cz: u32,
    depth: u8,
    result: &mut Vec<LeafInfo>,
) {
    match cell {
        Cell::Leaf(data) if data.has_sign_change() => {
            result.push(LeafInfo {
                data: data.clone(),
                bounds: *bounds,
                depth,
                key: CellKey(cx, cy, cz, depth),
            });
        }
        Cell::Branch { children_index } => {
            let children = &octree.children[*children_index as usize];
            for octant in 0..8u8 {
                collect_leaves_recursive(
                    octree,
                    &children[octant as usize],
                    &bounds.child(octant),
                    cx * 2 + if octant & 1 != 0 { 1 } else { 0 },
                    cy * 2 + if octant & 2 != 0 { 1 } else { 0 },
                    cz * 2 + if octant & 4 != 0 { 1 } else { 0 },
                    depth + 1,
                    result,
                );
            }
        }
        _ => {}
    }
}

fn lerp3(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])]
}

fn normalize3(v: [f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-12 { [v[0] / len, v[1] / len, v[2] / len] } else { [0.0, 0.0, 1.0] }
}
