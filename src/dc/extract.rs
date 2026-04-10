use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use crate::bbox::CubicBounds;
use crate::eval::bridge;
use crate::octree::Octree;
use crate::octree::cell::{Cell, LeafData};
use crate::qef::quadric::QefData;
use crate::dc::component::{CellComponents, COMPONENT_TABLE, edge_component};

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

/// Extract a triangle mesh using Manifold Dual Contouring.
///
/// Each leaf cell may produce multiple vertices (one per connected component
/// of inside corners), preventing non-manifold "bowtie" vertices that occur
/// in basic DC. Edge crossings are found via linear interpolation from cached
/// corner values (zero extra Python calls). Normals are obtained from the
/// user function's gradient output, or estimated via finite differences if
/// not provided (one extra Python call).
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

    // Depth-aware cell map: supports cross-depth neighbor lookup for true adaptive.
    // With 2:1 balance, depth difference is at most 1, so we try:
    // same depth → coarser (depth-1) → finer (depth+1).
    let mut cell_map: HashMap<(u32, u32, u32, u8), usize> = HashMap::with_capacity(leaves.len());
    for (i, leaf) in leaves.iter().enumerate() {
        cell_map.insert((leaf.key.0, leaf.key.1, leaf.key.2, leaf.key.3), i);
    }

    // Step 0 (MDC): Connected component analysis per leaf cell.
    // Most cells have 1 component; multi-component cells get multiple vertices.
    let leaf_components: Vec<CellComponents> = leaves.iter()
        .map(|leaf| COMPONENT_TABLE[leaf.data.corner_mask as usize])
        .collect();

    // Compute component offsets for flat QEF array indexing
    let mut component_offsets: Vec<usize> = Vec::with_capacity(leaves.len());
    let mut total_components = 0usize;
    for lc in &leaf_components {
        component_offsets.push(total_components);
        total_components += lc.num_components as usize;
    }

    // Step 2: Find edge crossings via linear interpolation (no extra Python calls)
    // and collect crossing points for normal estimation.
    // MDC: each crossing is tagged with its component ID.
    let edge_results: Vec<Vec<([f64; 3], (usize, u8, u8))>> = leaves
        .par_iter()
        .enumerate()
        .map(|(idx, leaf)| {
            let mut local = Vec::new();
            let components = &leaf_components[idx];
            for edge in 0..12u8 {
                let (c0, c1) = CubicBounds::edge_corners(edge);
                let v0 = leaf.data.corner_values[c0 as usize] - iso_value;
                let v1 = leaf.data.corner_values[c1 as usize] - iso_value;
                if (v0 < 0.0) != (v1 < 0.0) {
                    let dv = v1 - v0;
                    let t = if dv.abs() > 1e-15 { (-v0 / dv).clamp(0.001, 0.999) } else { 0.5 };
                    let p0 = leaf.bounds.corner(c0);
                    let p1 = leaf.bounds.corner(c1);
                    let comp_id = edge_component(components, leaf.data.corner_mask, edge);
                    local.push((lerp3(p0, p1, t), (idx, edge, comp_id)));
                }
            }
            local
        })
        .collect();

    let mut crossing_points: Vec<[f64; 3]> = Vec::new();
    let mut edge_leaf_map: Vec<(usize, u8, u8)> = Vec::new(); // (leaf_idx, edge, component_id)
    for chunk in &edge_results {
        for &(pt, mapping) in chunk {
            crossing_points.push(pt);
            edge_leaf_map.push(mapping);
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

    // Step 4: QEF vertex placement per component (probabilistic quadrics)
    // Per-leaf sigma_p: cell_size varies by depth in adaptive mode.
    let sigma_n: f64 = 0.01;

    // MDC: one QEF per (leaf, component), stored in flat array
    let mut qefs = vec![QefData::new(); total_components];
    for (i, &(leaf_idx, _edge, comp_id)) in edge_leaf_map.iter().enumerate() {
        let leaf_cell_size = octree.bounds.size / (1u32 << leaves[leaf_idx].depth) as f64;
        let sigma_p = 0.01 * leaf_cell_size;
        let qef_idx = component_offsets[leaf_idx] + comp_id as usize;
        qefs[qef_idx].add(crossing_points[i], normals[i], sigma_n, sigma_p);
    }

    // Pass 1 (parallel): solve QEF per component independently
    // Each component within a cell shares the same cell bounds.
    struct ComponentSolveInput {
        qef_idx: usize,
        cell_min: [f64; 3],
        cell_max: [f64; 3],
    }

    let mut solve_inputs: Vec<ComponentSolveInput> = Vec::with_capacity(total_components);
    for (i, leaf) in leaves.iter().enumerate() {
        let n = leaf_components[i].num_components as usize;
        let cell_min = leaf.bounds.corner(0);
        let cell_max = leaf.bounds.corner(7);
        for c in 0..n {
            solve_inputs.push(ComponentSolveInput {
                qef_idx: component_offsets[i] + c,
                cell_min,
                cell_max,
            });
        }
    }

    let qef_solutions: Vec<Option<([f64; 3], [f64; 3], [f64; 3])>> = solve_inputs
        .par_iter()
        .map(|input| {
            if qefs[input.qef_idx].count == 0 {
                None
            } else {
                let (pos, _) = qefs[input.qef_idx].solve(input.cell_min, input.cell_max);
                Some((pos, input.cell_min, input.cell_max))
            }
        })
        .collect();

    // Pass 2 (sequential): assign contiguous vertex indices per leaf
    let mut vertices = Vec::with_capacity(total_components);
    let mut leaf_vertex_base: Vec<Option<usize>> = vec![None; leaves.len()];
    let mut vertex_cell_bounds: Vec<([f64; 3], [f64; 3])> = Vec::new();

    let mut sol_idx = 0;
    for (i, _leaf) in leaves.iter().enumerate() {
        let n = leaf_components[i].num_components as usize;
        let has_any = (0..n).any(|c| qef_solutions[sol_idx + c].is_some());
        if has_any {
            leaf_vertex_base[i] = Some(vertices.len());
            for c in 0..n {
                if let Some((pos, cell_min, cell_max)) = qef_solutions[sol_idx + c] {
                    vertices.push(pos);
                    vertex_cell_bounds.push((cell_min, cell_max));
                } else {
                    // Fallback: cell center for empty-QEF component
                    let leaf = &leaves[i];
                    let center = leaf.bounds.center();
                    let cell_min = leaf.bounds.corner(0);
                    let cell_max = leaf.bounds.corner(7);
                    vertices.push(center);
                    vertex_cell_bounds.push((cell_min, cell_max));
                }
            }
        }
        sol_idx += n;
    }

    // Step 4b: Project vertices onto isosurface via Newton step.
    if !vertices.is_empty() {
        let proj_result = bridge::batch_evaluate(py, func, &vertices)?;
        if let Some(ref grads) = proj_result.gradients {
            vertices.par_iter_mut().enumerate().for_each(|(i, vert)| {
                let f_val = proj_result.values[i] - iso_value;
                let g = grads[i];
                let g_sq = g[0]*g[0] + g[1]*g[1] + g[2]*g[2];
                if g_sq > 1e-20 {
                    let step = f_val / g_sq;
                    vert[0] -= step * g[0];
                    vert[1] -= step * g[1];
                    vert[2] -= step * g[2];
                    let (cmin, cmax) = vertex_cell_bounds[i];
                    for d in 0..3 {
                        vert[d] = vert[d].clamp(cmin[d], cmax[d]);
                    }
                }
            });
        } else {
            // FD gradient for projection (use finest cell size for step)
            let cell_size = octree.bounds.size / (1u32 << octree.max_depth) as f64;
            let proj_grads = bridge::estimate_gradients_fd(
                py, func, &vertices, cell_size * 0.01
            )?;
            vertices.par_iter_mut().enumerate().for_each(|(i, vert)| {
                let f_val = proj_result.values[i] - iso_value;
                let g = proj_grads[i];
                let g_sq = g[0]*g[0] + g[1]*g[1] + g[2]*g[2];
                if g_sq > 1e-20 {
                    let step = f_val / g_sq;
                    vert[0] -= step * g[0];
                    vert[1] -= step * g[1];
                    vert[2] -= step * g[2];
                    let (cmin, cmax) = vertex_cell_bounds[i];
                    for d in 0..3 {
                        vert[d] = vert[d].clamp(cmin[d], cmax[d]);
                    }
                }
            });
        }
    }

    // Step 5: Generate quads from canonical edges (0, 4, 8)
    // MDC: each cell selects the vertex for the component adjacent to the shared edge.
    const CANONICAL_EDGES: [u8; 3] = [0, 4, 8];
    const NON_AXIS_DIMS: [(usize, usize); 3] = [(1, 2), (0, 2), (0, 1)];
    const CCW_ORDER: [[usize; 4]; 3] = [
        [0, 1, 3, 2], // X-axis
        [0, 2, 3, 1], // Y-axis
        [0, 1, 3, 2], // Z-axis
    ];

    // For a canonical edge along axis A at cell (x,y,z), the 4 sharing cells
    // see it as different local edges:
    //   cell[0] (current):     local_edge = axis*4 + 0
    //   cell[1] (shifted a1):  local_edge = axis*4 + 1
    //   cell[2] (shifted a2):  local_edge = axis*4 + 2
    //   cell[3] (shifted both):local_edge = axis*4 + 3

    let faces: Vec<[i64; 3]> = leaves
        .par_iter()
        .enumerate()
        .map(|(leaf_idx, leaf)| {
            let mut local_faces = Vec::new();
            if leaf_vertex_base[leaf_idx].is_none() { return local_faces; }

            let cell_coords = [leaf.key.0, leaf.key.1, leaf.key.2];
            let cell_depth = leaf.key.3;

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

                // Cross-depth neighbor lookup: same depth → coarser → finer
                let Some(ni_a1) = find_neighbor(&cell_map, n_a1[0], n_a1[1], n_a1[2], cell_depth) else { continue; };
                let Some(ni_a2) = find_neighbor(&cell_map, n_a2[0], n_a2[1], n_a2[2], cell_depth) else { continue; };
                let Some(ni_both) = find_neighbor(&cell_map, n_both[0], n_both[1], n_both[2], cell_depth) else { continue; };

                // MDC: select the correct component vertex for each of the 4 cells.
                let v0 = match vertex_for_edge(
                    leaf_idx, (axis * 4) as u8,
                    leaf.data.corner_mask, &leaf_components, &leaf_vertex_base,
                ) {
                    Some(v) => v,
                    None => continue,
                };
                let v1 = match vertex_for_edge(
                    ni_a1, (axis * 4 + 1) as u8,
                    leaves[ni_a1].data.corner_mask, &leaf_components, &leaf_vertex_base,
                ) {
                    Some(v) => v,
                    None => continue,
                };
                let v2 = match vertex_for_edge(
                    ni_a2, (axis * 4 + 2) as u8,
                    leaves[ni_a2].data.corner_mask, &leaf_components, &leaf_vertex_base,
                ) {
                    Some(v) => v,
                    None => continue,
                };
                let v3 = match vertex_for_edge(
                    ni_both, (axis * 4 + 3) as u8,
                    leaves[ni_both].data.corner_mask, &leaf_components, &leaf_vertex_base,
                ) {
                    Some(v) => v,
                    None => continue,
                };

                let verts = [v0, v1, v2, v3];
                let ccw = CCW_ORDER[axis];

                let quad = if val0 < 0.0 {
                    [verts[ccw[0]], verts[ccw[1]], verts[ccw[2]], verts[ccw[3]]]
                } else {
                    [verts[ccw[3]], verts[ccw[2]], verts[ccw[1]], verts[ccw[0]]]
                };

                // Flatness-based diagonal: choose flatter split
                let [a, b, c, d] = quad;
                let pa = vertices[a]; let pb = vertices[b];
                let pc = vertices[c]; let pd = vertices[d];

                // Split into 2 triangles, skipping degenerate ones
                let n1_ac = tri_normal(pa, pb, pc);
                let n2_ac = tri_normal(pa, pc, pd);
                let dot_ac = n1_ac[0]*n2_ac[0] + n1_ac[1]*n2_ac[1] + n1_ac[2]*n2_ac[2];
                let n1_bd = tri_normal(pa, pb, pd);
                let n2_bd = tri_normal(pb, pc, pd);
                let dot_bd = n1_bd[0]*n2_bd[0] + n1_bd[1]*n2_bd[1] + n1_bd[2]*n2_bd[2];

                if dot_ac >= dot_bd {
                    if a != b && b != c && a != c {
                        local_faces.push([a as i64, b as i64, c as i64]);
                    }
                    if a != c && c != d && a != d {
                        local_faces.push([a as i64, c as i64, d as i64]);
                    }
                } else {
                    if a != b && b != d && a != d {
                        local_faces.push([a as i64, b as i64, d as i64]);
                    }
                    if b != c && c != d && b != d {
                        local_faces.push([b as i64, c as i64, d as i64]);
                    }
                }
            }
            local_faces
        })
        .collect::<Vec<Vec<[i64; 3]>>>()
        .into_iter()
        .flatten()
        .collect();

    Ok(ExtractedMesh { vertices, faces })
}

/// Same-depth neighbor lookup for adaptive octrees.
///
/// Edge completion guarantees that all 4 cells sharing a sign-change edge
/// exist at the same depth, so we only need same-depth lookup.
#[inline]
fn find_neighbor(
    cell_map: &HashMap<(u32, u32, u32, u8), usize>,
    cx: u32, cy: u32, cz: u32,
    depth: u8,
) -> Option<usize> {
    cell_map.get(&(cx, cy, cz, depth)).copied()
}

/// Select the vertex for a given cell and local edge.
///
/// For MDC, each cell may have multiple vertices (one per connected component).
/// This function determines which component is adjacent to `local_edge` by
/// finding which corner of that edge is inside, then returning the vertex
/// index for that corner's component.
#[inline]
fn vertex_for_edge(
    leaf_idx: usize,
    local_edge: u8,
    corner_mask: u8,
    leaf_components: &[CellComponents],
    leaf_vertex_base: &[Option<usize>],
) -> Option<usize> {
    let base = leaf_vertex_base[leaf_idx]?;
    let components = &leaf_components[leaf_idx];

    // Fast path: single component (vast majority of cells)
    if components.num_components == 1 {
        return Some(base);
    }

    let (c0, c1) = CubicBounds::edge_corners(local_edge);
    let inside_corner = if (corner_mask >> c0) & 1 == 1 { c0 } else { c1 };
    let comp = components.corner_component[inside_corner as usize];
    debug_assert!(comp != 0xFF);
    Some(base + comp as usize)
}

fn collect_leaves_recursive(
    octree: &Octree, cell: &Cell, bounds: &CubicBounds,
    cx: u32, cy: u32, cz: u32, depth: u8,
    result: &mut Vec<LeafInfo>,
) {
    match cell {
        Cell::Leaf(data) if data.has_sign_change() && depth >= octree.min_depth => {
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

/// Unnormalized triangle normal via cross product.
fn tri_normal(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    let ab = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
    let ac = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
    let n = [
        ab[1]*ac[2] - ab[2]*ac[1],
        ab[2]*ac[0] - ab[0]*ac[2],
        ab[0]*ac[1] - ab[1]*ac[0],
    ];
    let len = (n[0]*n[0] + n[1]*n[1] + n[2]*n[2]).sqrt();
    if len > 1e-15 { [n[0]/len, n[1]/len, n[2]/len] } else { [0.0, 0.0, 0.0] }
}

fn lerp3(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])]
}

fn normalize3(v: [f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-12 { [v[0] / len, v[1] / len, v[2] / len] } else { [0.0, 0.0, 1.0] }
}
