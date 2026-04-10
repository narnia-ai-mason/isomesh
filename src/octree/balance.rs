use std::collections::{HashSet, VecDeque};
use pyo3::prelude::*;
use crate::eval::bridge;
use crate::eval::cache::{self, CachedEval, EvalCache};
use crate::octree::cell::{Cell, LeafData};
use crate::octree::types::GridPos;
use crate::octree::Octree;

/// Enforce 2:1 balance on the octree: no two face-adjacent leaf cells
/// may differ by more than 1 depth level.
///
/// Uses BFS ripple propagation from all surface leaves. When a cell's
/// neighbor is more than 1 level coarser, the coarser cell is subdivided.
/// New corner values are batch-evaluated via the Python function.
///
/// Returns the number of new SDF evaluations performed.
pub fn enforce_balance_2to1(
    py: Python<'_>,
    func: &PyObject,
    octree: &mut Octree,
    cache: &mut EvalCache,
) -> PyResult<u64> {
    let min_depth = octree.min_depth;
    let max_depth = octree.max_depth;
    let cell_size_at_max = octree.bounds.size / (1u32 << max_depth) as f64;
    let mut total_evals = 0u64;

    loop {
        // Collect all leaf cells with their coordinates and depths
        let mut all_leaves: Vec<(u32, u32, u32, u8)> = Vec::new();
        collect_all_leaves_recursive(
            octree, &octree.root, 0, 0, 0, 0, &mut all_leaves,
        );

        // Find cells that need subdivision for two reasons:
        // 1. 2:1 balance violation: face-neighbor depth difference > 1
        // 2. Surface propagation: surface cell's face-neighbor is non-surface
        //    (Empty/Full) at a coarser depth, which would create gaps in the mesh
        let mut cells_to_subdivide: Vec<(u32, u32, u32, u8)> = Vec::new();
        let mut seen_subdivide: HashSet<(u32, u32, u32, u8)> = HashSet::new();

        for &(cx, cy, cz, depth) in &all_leaves {
            if depth == 0 { continue; }

            let neighbors = super::face_neighbors(cx, cy, cz, depth);
            let is_surface = matches!(octree.cell_at(cx, cy, cz, depth), (Cell::Leaf(d), _) if d.has_sign_change());

            for neighbor in neighbors.iter().flatten() {
                let (nx, ny, nz) = *neighbor;
                let (neighbor_cell, actual_depth) = octree.cell_at(nx, ny, nz, depth);
                // Rule 1: 2:1 balance — neighbor is more than 1 level coarser
                let needs_balance = actual_depth < depth.saturating_sub(1);
                // Rule 2: Surface propagation — surface cell borders a non-surface
                // cell at a coarser depth. Subdividing may reveal sign changes.
                let needs_surface_prop = is_surface
                    && actual_depth < depth
                    && !matches!(neighbor_cell, Cell::Leaf(d) if d.has_sign_change());

                if (needs_balance || needs_surface_prop) && actual_depth >= min_depth {
                    // Only subdivide cells at or above min_depth.
                    // Cells below min_depth are collapsed tree nodes; leave them.
                    let scale = 1u32 << (depth - actual_depth);
                    let coarse_cx = nx / scale;
                    let coarse_cy = ny / scale;
                    let coarse_cz = nz / scale;
                    let key = (coarse_cx, coarse_cy, coarse_cz, actual_depth);
                    if seen_subdivide.insert(key) {
                        cells_to_subdivide.push(key);
                    }
                }
            }
        }

        if cells_to_subdivide.is_empty() {
            break;
        }

        // Collect new corner positions needed for subdivision
        let mut new_points: Vec<[f64; 3]> = Vec::new();
        let mut new_grid: Vec<GridPos> = Vec::new();
        let mut seen_grid: HashSet<GridPos> = HashSet::new();

        for &(cx, cy, cz, depth) in &cells_to_subdivide {
            let cs = 1u32 << (max_depth - depth - 1); // child scale in max_depth grid
            let gx = cx * (1u32 << (max_depth - depth));
            let gy = cy * (1u32 << (max_depth - depth));
            let gz = cz * (1u32 << (max_depth - depth));

            for octant in 0..8u8 {
                let ox = gx + if octant & 1 != 0 { cs } else { 0 };
                let oy = gy + if octant & 2 != 0 { cs } else { 0 };
                let oz = gz + if octant & 4 != 0 { cs } else { 0 };

                for corner in 0..8u8 {
                    let px = ox + if corner & 1 != 0 { cs } else { 0 };
                    let py = oy + if corner & 2 != 0 { cs } else { 0 };
                    let pz = oz + if corner & 4 != 0 { cs } else { 0 };
                    let gp = GridPos::new(px, py, pz);
                    if !cache.contains(&gp) && seen_grid.insert(gp) {
                        new_grid.push(gp);
                        new_points.push(cache::grid_to_world(
                            &gp, &octree.bounds.origin, cell_size_at_max,
                        ));
                    }
                }
            }
        }

        // Batch evaluate new corners
        if !new_points.is_empty() {
            let result = bridge::batch_evaluate(py, func, &new_points)?;
            total_evals += new_points.len() as u64;
            let fd_grads = if result.gradients.is_none() {
                let g = bridge::estimate_gradients_fd(
                    py, func, &new_points, cell_size_at_max * 0.01,
                )?;
                total_evals += new_points.len() as u64 * 6;
                Some(g)
            } else {
                None
            };
            for (i, gp) in new_grid.iter().enumerate() {
                let gradient = result.gradients.as_ref()
                    .map(|g| g[i])
                    .or_else(|| fd_grads.as_ref().map(|g| g[i]));
                cache.insert(*gp, CachedEval {
                    value: result.values[i],
                    gradient,
                });
            }
        }

        // Subdivide identified cells
        for &(cx, cy, cz, depth) in &cells_to_subdivide {
            let gx = cx * (1u32 << (max_depth - depth));
            let gy = cy * (1u32 << (max_depth - depth));
            let gz = cz * (1u32 << (max_depth - depth));
            subdivide_cell_for_balance(octree, cache, gx, gy, gz, depth, max_depth);
        }
    }

    Ok(total_evals)
}

/// Subdivide any cell (Leaf, Empty, or Full) for 2:1 balance enforcement.
///
/// Unlike `build::subdivide_leaf`, this handles Empty/Full cells by creating
/// child cells from the cached corner values.
fn subdivide_cell_for_balance(
    octree: &mut Octree,
    cache: &EvalCache,
    grid_x: u32, grid_y: u32, grid_z: u32,
    depth: u8, max_depth: u8,
) {
    let child_scale = 1u32 << (max_depth - depth - 1);
    let iso_value = octree.iso_value;
    let mut child_cells: [Cell; 8] = std::array::from_fn(|_| Cell::Empty);

    for octant in 0..8u8 {
        let cx = grid_x + if octant & 1 != 0 { child_scale } else { 0 };
        let cy = grid_y + if octant & 2 != 0 { child_scale } else { 0 };
        let cz = grid_z + if octant & 4 != 0 { child_scale } else { 0 };

        let mut corner_values = [0.0f64; 8];
        let mut corner_grads = [[0.0f64; 3]; 8];
        let mut has_grads = true;

        for corner in 0..8u8 {
            let gx = cx + if corner & 1 != 0 { child_scale } else { 0 };
            let gy = cy + if corner & 2 != 0 { child_scale } else { 0 };
            let gz = cz + if corner & 4 != 0 { child_scale } else { 0 };
            let gp = GridPos::new(gx, gy, gz);
            let eval = cache.get(&gp).expect("corner not in cache during balance subdivision");
            corner_values[corner as usize] = eval.value;
            match eval.gradient {
                Some(g) => corner_grads[corner as usize] = g,
                None => has_grads = false,
            }
        }

        let mut leaf = LeafData::new(corner_values, iso_value);
        if has_grads { leaf.corner_gradients = Some(corner_grads); }

        child_cells[octant as usize] = if leaf.has_sign_change() {
            Cell::Leaf(leaf)
        } else if leaf.corner_mask == 0 {
            Cell::Empty
        } else {
            Cell::Full
        };
    }

    let idx = octree.children.len() as u32;
    octree.children.push(child_cells);

    // Navigate to the cell and replace it with a Branch
    let path = compute_path_for_balance(grid_x, grid_y, grid_z, depth, max_depth);
    let cell_ref = navigate_to_cell_mut_balance(&mut octree.root, &mut octree.children, &path);
    *cell_ref = Cell::Branch { children_index: idx };
}

fn compute_path_for_balance(grid_x: u32, grid_y: u32, grid_z: u32, depth: u8, max_depth: u8) -> Vec<u8> {
    let mut path = Vec::with_capacity(depth as usize);
    let mut scale = 1u32 << (max_depth - 1);
    let mut rx = grid_x;
    let mut ry = grid_y;
    let mut rz = grid_z;
    for _ in 0..depth {
        let mut octant = 0u8;
        if rx >= scale { octant |= 1; rx -= scale; }
        if ry >= scale { octant |= 2; ry -= scale; }
        if rz >= scale { octant |= 4; rz -= scale; }
        path.push(octant);
        scale >>= 1;
    }
    path
}

fn navigate_to_cell_mut_balance<'a>(
    root: &'a mut Cell,
    children: &'a mut Vec<[Cell; 8]>,
    path: &[u8],
) -> &'a mut Cell {
    if path.is_empty() { return root; }
    let mut current = root as *mut Cell;
    for &octant in path.iter() {
        unsafe {
            match &*current {
                Cell::Branch { children_index } => {
                    let idx = *children_index as usize;
                    current = &mut children[idx][octant as usize] as *mut Cell;
                }
                _ => { return &mut *current; }
            }
        }
    }
    unsafe { &mut *current }
}

fn collect_all_leaves_recursive(
    octree: &Octree, cell: &Cell,
    cx: u32, cy: u32, cz: u32, depth: u8,
    result: &mut Vec<(u32, u32, u32, u8)>,
) {
    match cell {
        Cell::Empty | Cell::Full | Cell::Leaf(_) => {
            result.push((cx, cy, cz, depth));
        }
        Cell::Branch { children_index } => {
            let children = &octree.children[*children_index as usize];
            for octant in 0..8u8 {
                collect_all_leaves_recursive(
                    octree, &children[octant as usize],
                    cx * 2 + if octant & 1 != 0 { 1 } else { 0 },
                    cy * 2 + if octant & 2 != 0 { 1 } else { 0 },
                    cz * 2 + if octant & 4 != 0 { 1 } else { 0 },
                    depth + 1, result,
                );
            }
        }
    }
}
