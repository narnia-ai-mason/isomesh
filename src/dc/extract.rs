use pyo3::prelude::*;
use std::collections::HashMap;
use crate::bbox::CubicBounds;
use crate::eval::bridge;
use crate::octree::Octree;
use crate::octree::cell::{Cell, LeafData};
use crate::qef::quadric::QefData;

/// Extracted mesh: vertices and triangle faces.
pub struct ExtractedMesh {
    pub vertices: Vec<[f64; 3]>,
    pub faces: Vec<[i64; 3]>,
}

/// Key for a leaf cell: grid coordinates at its depth.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct CellKey(u32, u32, u32, u8);

struct LeafInfo {
    data: LeafData,
    bounds: CubicBounds,
    depth: u8,
    key: CellKey,
}

/// Extract a triangle mesh using Dual Contouring.
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

    // Determine the effective depth for uniform-like processing
    // For now, all leaves should be at the same depth (uniform or adaptively refined to same level)
    let leaf_depth = leaves[0].depth;

    // Build cell lookup by grid coordinates
    let mut cell_map: HashMap<CellKey, usize> = HashMap::new();
    for (i, leaf) in leaves.iter().enumerate() {
        cell_map.insert(leaf.key, i);
    }

    // Step 2: Find edge crossings via bisection
    let mut all_edge_data: Vec<([f64; 3], [f64; 3], f64, f64)> = Vec::new();
    let mut edge_leaf_map: Vec<(usize, u8)> = Vec::new();

    for (idx, leaf) in leaves.iter().enumerate() {
        for edge in 0..12u8 {
            let (c0, c1) = CubicBounds::edge_corners(edge);
            let v0 = leaf.data.corner_values[c0 as usize] - iso_value;
            let v1 = leaf.data.corner_values[c1 as usize] - iso_value;
            if (v0 < 0.0) != (v1 < 0.0) {
                all_edge_data.push((
                    leaf.bounds.corner(c0),
                    leaf.bounds.corner(c1),
                    v0, v1,
                ));
                edge_leaf_map.push((idx, edge));
            }
        }
    }

    let num_crossings = all_edge_data.len();

    // Bisection (10 iterations)
    let mut lo = vec![0.0f64; num_crossings];
    let mut hi = vec![1.0f64; num_crossings];
    let mut lo_val: Vec<f64> = all_edge_data.iter().map(|e| e.2).collect();

    if num_crossings > 0 {
        for _ in 0..10 {
            let mid_points: Vec<[f64; 3]> = (0..num_crossings).map(|i| {
                let t = (lo[i] + hi[i]) * 0.5;
                lerp3(all_edge_data[i].0, all_edge_data[i].1, t)
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

    // Final crossing points
    let crossing_points: Vec<[f64; 3]> = (0..num_crossings).map(|i| {
        lerp3(all_edge_data[i].0, all_edge_data[i].1, (lo[i] + hi[i]) * 0.5)
    }).collect();

    // Get normals
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

    // Step 3: QEF vertex placement
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

    // Step 4: Generate quads from grid edges
    // In DC, for each grid edge with a sign change, we create a quad
    // connecting the 4 cells sharing that edge.
    //
    // Grid edge approach: iterate over all internal edges of the grid.
    // An internal edge parallel to axis A at position (e0, e1) in the
    // two non-axis dimensions is shared by 4 cells at:
    //   (e0-1, e1-1), (e0, e1-1), (e0-1, e1), (e0, e1)
    // in the non-axis dimensions (using the min-corner convention).

    let mut faces = Vec::new();
    let cells_per_axis = 1u32 << leaf_depth;

    // For each axis (X=0, Y=1, Z=2), iterate over all internal edges
    for axis in 0..3u32 {
        let (a1, a2) = match axis {
            0 => (1u32, 2u32),
            1 => (0u32, 2u32),
            _ => (0u32, 1u32),
        };

        // Edge at position (ea, e1, e2) where ea ranges over [0, cells_per_axis)
        // and e1, e2 range over [1, cells_per_axis) (internal edges only)
        for ea in 0..cells_per_axis {
            for e1 in 1..cells_per_axis {
                for e2 in 1..cells_per_axis {
                    // The 4 cells sharing this edge
                    let mut coords = [[0u32; 3]; 4];
                    for (idx, &(d1, d2)) in [(0i32, 0i32), (-1, 0), (0, -1), (-1, -1)].iter().enumerate() {
                        coords[idx][axis as usize] = ea;
                        coords[idx][a1 as usize] = (e1 as i32 + d1) as u32;
                        coords[idx][a2 as usize] = (e2 as i32 + d2) as u32;
                    }

                    // Look up all 4 cells
                    let cell_indices: Vec<usize> = coords.iter().filter_map(|c| {
                        cell_map.get(&CellKey(c[0], c[1], c[2], leaf_depth)).copied()
                    }).collect();

                    if cell_indices.len() != 4 {
                        continue; // not all 4 cells are leaves with sign changes
                    }

                    // Check for sign change along this edge
                    // The edge connects two corners that differ only in the axis bit
                    // Use the first cell's corner values
                    let first_leaf = &leaves[cell_indices[0]];
                    // The relevant edge for this cell: parallel to `axis`,
                    // at the max corner in both non-axis dimensions
                    // corner c0: axis bit = 0, a1 bit = 1, a2 bit = 1
                    // corner c1: axis bit = 1, a1 bit = 1, a2 bit = 1
                    // (since we're at the meeting point of 4 cells, the edge is at the max of a1 and a2 for cell [0])
                    let c0_bits = (1u8 << a1) | (1u8 << a2);
                    let c1_bits = c0_bits | (1u8 << axis);
                    let v0 = first_leaf.data.corner_values[c0_bits as usize] - iso_value;
                    let v1 = first_leaf.data.corner_values[c1_bits as usize] - iso_value;

                    if (v0 < 0.0) == (v1 < 0.0) {
                        continue; // no sign change
                    }

                    // Get vertex indices
                    let verts: Vec<usize> = cell_indices.iter().filter_map(|&ci| leaf_vertex[ci]).collect();
                    if verts.len() != 4 { continue; }

                    // Winding order based on sign of v0
                    // If v0 < 0 (inside), gradient points outward; winding should match
                    let flip = v0 >= 0.0;

                    // Order the quad vertices consistently
                    // The 4 cells are at: (0,0), (-1,0), (0,-1), (-1,-1) offsets
                    // We want them in CCW order around the edge
                    let ordered = if flip {
                        [verts[0], verts[2], verts[3], verts[1]]
                    } else {
                        [verts[0], verts[1], verts[3], verts[2]]
                    };

                    faces.push([ordered[0] as i64, ordered[1] as i64, ordered[2] as i64]);
                    faces.push([ordered[0] as i64, ordered[2] as i64, ordered[3] as i64]);
                }
            }
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
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}

fn normalize3(v: [f64; 3]) -> [f64; 3] {
    let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    if len > 1e-12 { [v[0]/len, v[1]/len, v[2]/len] } else { [0.0, 0.0, 1.0] }
}
