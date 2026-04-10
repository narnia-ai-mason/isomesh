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

                if needs_balance || needs_surface_prop {
                    let scale = 1u32 << (depth - actual_depth);
                    let coarse_cx = nx / scale;
                    let coarse_cy = ny / scale;
                    let coarse_cz = nz / scale;
                    let key = (coarse_cx, coarse_cy, coarse_cz, actual_depth);
                    if seen_subdivide.insert(key) {
                        cells_to_subdivide.push(key);
                    }
                }

                // Rule 3: Surface depth propagation — if THIS surface cell's
                // face-neighbor is a Branch containing surface descendants,
                // THIS cell must also be subdivided to prevent cross-depth
                // gaps in mesh extraction (same-depth lookup).
                if is_surface
                    && actual_depth == depth
                    && matches!(neighbor_cell, Cell::Branch { .. })
                    && has_surface_descendant(octree, neighbor_cell)
                    && depth < max_depth
                {
                    let key = (cx, cy, cz, depth);
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

    // Phase 2: Edge completion — ensure every sign-change edge has all 4 sharing cells.
    total_evals += ensure_edge_completeness(py, func, octree, cache)?;

    Ok(total_evals)
}

/// Ensure every canonical edge with a sign change has all 4 sharing cells present
/// as surface leaves. This eliminates gaps at depth boundaries.
fn ensure_edge_completeness(
    py: Python<'_>,
    func: &PyObject,
    octree: &mut Octree,
    cache: &mut EvalCache,
) -> PyResult<u64> {
    use crate::bbox::CubicBounds;

    let max_depth = octree.max_depth;
    let min_depth = octree.min_depth;
    let cell_size_at_max = octree.bounds.size / (1u32 << max_depth) as f64;
    let iso_value = octree.iso_value;
    let mut total_evals = 0u64;

    const CANONICAL_EDGES: [u8; 3] = [0, 4, 8];
    const NON_AXIS_DIMS: [(usize, usize); 3] = [(1, 2), (0, 2), (0, 1)];

    // Also check non-canonical edges (1-3, 5-7, 9-11) because surface cells
    // that DON'T own a canonical edge still participate in quads generated
    // by their neighbors. We must ensure ALL edges with sign changes have
    // complete 4-cell neighborhoods.
    const ALL_EDGES_BY_AXIS: [[u8; 4]; 3] = [
        [0, 1, 2, 3],   // X-axis edges
        [4, 5, 6, 7],   // Y-axis edges
        [8, 9, 10, 11],  // Z-axis edges
    ];

    loop {
        // Collect surface leaves
        let mut surface_leaves: Vec<(u32, u32, u32, u8)> = Vec::new();
        collect_surface_leaves_recursive(
            octree, &octree.root, 0, 0, 0, 0, min_depth, &mut surface_leaves,
        );

        // Build a lookup set for fast existence checks
        let leaf_set: HashSet<(u32, u32, u32, u8)> = surface_leaves.iter().copied().collect();

        // For each surface leaf, check canonical edges
        let mut cells_to_subdivide: Vec<(u32, u32, u32, u8)> = Vec::new();
        let mut seen: HashSet<(u32, u32, u32, u8)> = HashSet::new();
        let mut target_depths: std::collections::HashMap<(u32,u32,u32,u8), u8> = std::collections::HashMap::new();

        for &(cx, cy, cz, depth) in &surface_leaves {
            // Get corner values to check sign changes on ALL 12 edges
            let scale = 1u32 << (max_depth - depth);
            let gx = cx * scale;
            let gy = cy * scale;
            let gz = cz * scale;

            let mut corner_values = [0.0f64; 8];
            for c in 0..8u8 {
                let px = gx + if c & 1 != 0 { scale } else { 0 };
                let py_c = gy + if c & 2 != 0 { scale } else { 0 };
                let pz = gz + if c & 4 != 0 { scale } else { 0 };
                if let Some(eval) = cache.get(&GridPos::new(px, py_c, pz)) {
                    corner_values[c as usize] = eval.value;
                }
            }

            let cell_coords = [cx, cy, cz];
            let max_coord = 1u32 << depth;

            // Check ALL 12 edges, not just canonical 3.
            // Each edge with a sign change needs 4 sharing cells at the same depth.
            // The 4 cells sharing edge E (parallel to axis A, at specific a1, a2 offsets)
            // are determined by the edge's offset bits in the two perpendicular axes.
            for axis in 0..3usize {
                let (a1, a2) = NON_AXIS_DIMS[axis];
                for edge_in_group in 0..4u8 {
                    let edge_idx = (axis as u8) * 4 + edge_in_group;
                    let (c0, c1) = CubicBounds::edge_corners(edge_idx);
                    let val0 = corner_values[c0 as usize] - iso_value;
                    let val1 = corner_values[c1 as usize] - iso_value;
                    if (val0 < 0.0) == (val1 < 0.0) { continue; }

                    // Edge offset bits: bit 0 = a1 offset, bit 1 = a2 offset
                    // For the 4 cells sharing this edge, the "owner" cell has
                    // the edge at its minimum coordinates. We compute the owner
                    // and the 3 other sharing cells.
                    let a1_off = (edge_in_group & 1) as u32;
                    let a2_off = ((edge_in_group >> 1) & 1) as u32;

                    // The 4 sharing cells in coordinate space
                    let mut sharing = Vec::new();
                    for da1 in 0..2u32 {
                        for da2 in 0..2u32 {
                            let mut nc = cell_coords;
                            // The cell at (nc[a1] + a1_off - da1, nc[a2] + a2_off - da2)
                            let na1 = cell_coords[a1] + a1_off;
                            let na2 = cell_coords[a2] + a2_off;
                            if na1 < da1 || na2 < da2 { continue; }
                            let ca1 = na1 - da1;
                            let ca2 = na2 - da2;
                            if ca1 >= max_coord || ca2 >= max_coord { continue; }
                            nc[a1] = ca1;
                            nc[a2] = ca2;
                            if nc[0] != cx || nc[1] != cy || nc[2] != cz {
                                sharing.push((nc[0], nc[1], nc[2]));
                            }
                        }
                    }

                    for (nx, ny, nz) in sharing {
                        if !find_surface_neighbor_same_depth(&leaf_set, nx, ny, nz, depth) {
                            let (cell, actual_depth) = octree.cell_at(nx, ny, nz, depth);
                            if actual_depth < depth
                                && !matches!(cell, Cell::Branch { .. })
                            {
                                // Neighbor is coarser → subdivide neighbor
                                // to match current cell's depth.
                                let s = 1u32 << (depth - actual_depth);
                                let key = (nx / s, ny / s, nz / s, actual_depth);
                                if seen.insert(key) {
                                    cells_to_subdivide.push((key.0, key.1, key.2, key.3));
                                    target_depths.insert(key, depth);
                                }
                            }
                            // Note: when neighbor is a Branch at the same depth
                            // (already refined to finer depth), we do NOT cascade
                            // subdivision. Instead, extract_dc handles cross-depth
                            // quad generation via find_neighbor's fallback lookup.
                        }
                    }
                }
            }
        }

        if cells_to_subdivide.is_empty() {
            break;
        }

        // Collect ALL corners needed for deep subdivision (from current depth to target)
        let mut new_points: Vec<[f64; 3]> = Vec::new();
        let mut new_grid: Vec<GridPos> = Vec::new();
        let mut seen_grid: HashSet<GridPos> = HashSet::new();

        for &(cx, cy, cz, depth) in &cells_to_subdivide {
            let target = target_depths.get(&(cx, cy, cz, depth)).copied().unwrap_or(depth + 1);
            let gx = cx * (1u32 << (max_depth - depth));
            let gy = cy * (1u32 << (max_depth - depth));
            let gz = cz * (1u32 << (max_depth - depth));
            // Collect corners for ALL levels from depth+1 to target
            collect_deep_corners(
                gx, gy, gz, depth, target, max_depth,
                cache, &octree.bounds.origin, cell_size_at_max,
                &mut new_grid, &mut new_points, &mut seen_grid,
            );
        }

        if !new_points.is_empty() {
            let result = bridge::batch_evaluate(py, func, &new_points)?;
            total_evals += new_points.len() as u64;
            let fd_grads = if result.gradients.is_none() {
                let g = bridge::estimate_gradients_fd(py, func, &new_points, cell_size_at_max * 0.01)?;
                total_evals += new_points.len() as u64 * 6;
                Some(g)
            } else { None };
            for (i, gp) in new_grid.iter().enumerate() {
                let gradient = result.gradients.as_ref().map(|g| g[i])
                    .or_else(|| fd_grads.as_ref().map(|g| g[i]));
                cache.insert(*gp, CachedEval { value: result.values[i], gradient });
            }
        }

        for &(cx, cy, cz, depth) in &cells_to_subdivide {
            let gx = cx * (1u32 << (max_depth - depth));
            let gy = cy * (1u32 << (max_depth - depth));
            let gz = cz * (1u32 << (max_depth - depth));
            let target = target_depths.get(&(cx, cy, cz, depth)).copied().unwrap_or(depth + 1);
            subdivide_deep(octree, cache, gx, gy, gz, depth, target, max_depth);
        }
    }

    Ok(total_evals)
}

/// Collect all corner grid positions needed to subdivide from current_depth to target_depth.
fn collect_deep_corners(
    gx: u32, gy: u32, gz: u32,
    current_depth: u8, target_depth: u8, max_depth: u8,
    cache: &EvalCache, origin: &[f64; 3], cell_size_at_max: f64,
    new_grid: &mut Vec<GridPos>, new_points: &mut Vec<[f64; 3]>,
    seen: &mut HashSet<GridPos>,
) {
    if current_depth >= target_depth || current_depth >= max_depth {
        return;
    }
    let cs = 1u32 << (max_depth - current_depth - 1);
    for octant in 0..8u8 {
        let ox = gx + if octant & 1 != 0 { cs } else { 0 };
        let oy = gy + if octant & 2 != 0 { cs } else { 0 };
        let oz = gz + if octant & 4 != 0 { cs } else { 0 };
        for corner in 0..8u8 {
            let px = ox + if corner & 1 != 0 { cs } else { 0 };
            let py = oy + if corner & 2 != 0 { cs } else { 0 };
            let pz = oz + if corner & 4 != 0 { cs } else { 0 };
            let gp = GridPos::new(px, py, pz);
            if !cache.contains(&gp) && seen.insert(gp) {
                new_grid.push(gp);
                new_points.push(cache::grid_to_world(&gp, origin, cell_size_at_max));
            }
        }
        // Recurse for deeper levels
        if current_depth + 1 < target_depth {
            collect_deep_corners(
                ox, oy, oz, current_depth + 1, target_depth, max_depth,
                cache, origin, cell_size_at_max, new_grid, new_points, seen,
            );
        }
    }
}

/// Recursively subdivide a cell from `current_depth` down to `target_depth`.
/// All intermediate cells become Branches; only the leaves at target_depth are kept.
fn subdivide_deep(
    octree: &mut Octree,
    cache: &EvalCache,
    grid_x: u32, grid_y: u32, grid_z: u32,
    current_depth: u8, target_depth: u8, max_depth: u8,
) {
    if current_depth >= target_depth || current_depth >= max_depth {
        return;
    }

    // First subdivide one level
    subdivide_cell_for_balance(octree, cache, grid_x, grid_y, grid_z, current_depth, max_depth);

    // If we need to go deeper, subdivide each child that needs it
    if current_depth + 1 < target_depth {
        let child_scale = 1u32 << (max_depth - current_depth - 1);
        for octant in 0..8u8 {
            let cx = grid_x + if octant & 1 != 0 { child_scale } else { 0 };
            let cy = grid_y + if octant & 2 != 0 { child_scale } else { 0 };
            let cz = grid_z + if octant & 4 != 0 { child_scale } else { 0 };

            // Check if child is a leaf (not already a Branch)
            let child_depth = current_depth + 1;
            let (cell, actual) = octree.cell_at(
                cx / (1u32 << (max_depth - child_depth)),
                cy / (1u32 << (max_depth - child_depth)),
                cz / (1u32 << (max_depth - child_depth)),
                child_depth,
            );
            if actual == child_depth && !matches!(cell, Cell::Branch { .. }) {
                subdivide_deep(octree, cache, cx, cy, cz, child_depth, target_depth, max_depth);
            }
        }
    }
}

/// Check if a surface leaf exists at (nx, ny, nz) at EXACTLY the given depth.
/// Cross-depth vertex sharing is incorrect because the coarser cell may not
/// have a sign change on the specific edge, causing wrong vertex selection.
fn find_surface_neighbor_same_depth(
    leaf_set: &HashSet<(u32, u32, u32, u8)>,
    nx: u32, ny: u32, nz: u32,
    depth: u8,
) -> bool {
    leaf_set.contains(&(nx, ny, nz, depth))
}

fn collect_surface_leaves_recursive(
    octree: &Octree, cell: &Cell,
    cx: u32, cy: u32, cz: u32, depth: u8, _min_depth: u8,
    result: &mut Vec<(u32, u32, u32, u8)>,
) {
    match cell {
        Cell::Leaf(data) if data.has_sign_change() => {
            result.push((cx, cy, cz, depth));
        }
        Cell::Branch { children_index } => {
            let children = &octree.children[*children_index as usize];
            for octant in 0..8u8 {
                collect_surface_leaves_recursive(
                    octree, &children[octant as usize],
                    cx * 2 + if octant & 1 != 0 { 1 } else { 0 },
                    cy * 2 + if octant & 2 != 0 { 1 } else { 0 },
                    cz * 2 + if octant & 4 != 0 { 1 } else { 0 },
                    depth + 1, _min_depth, result,
                );
            }
        }
        _ => {}
    }
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

    // Navigate to the cell and check if it's already a Branch.
    // Overwriting a Branch would orphan its children (including deeper
    // refinement from adaptive passes), so we must skip.
    let path = compute_path_for_balance(grid_x, grid_y, grid_z, depth, max_depth);
    // Use raw pointer to avoid borrow conflicts with octree.children
    let cell_ptr = {
        let cell_ref = navigate_to_cell_mut_balance(&mut octree.root, &mut octree.children, &path);
        cell_ref as *mut Cell
    };
    unsafe {
        if matches!(&*cell_ptr, Cell::Branch { .. }) {
            return; // Already subdivided; don't overwrite
        }
        let idx = octree.children.len() as u32;
        octree.children.push(child_cells);
        *cell_ptr = Cell::Branch { children_index: idx };
    }
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

/// Check if a cell's subtree contains any surface leaf (sign change).
fn has_surface_descendant(octree: &Octree, cell: &Cell) -> bool {
    match cell {
        Cell::Leaf(d) => d.has_sign_change(),
        Cell::Branch { children_index } => {
            let children = &octree.children[*children_index as usize];
            children.iter().any(|c| has_surface_descendant(octree, c))
        }
        _ => false,
    }
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
