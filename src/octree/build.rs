use pyo3::prelude::*;
use crate::bbox::BoundingBox;
use crate::eval::bridge;
use crate::eval::cache::{self, CachedEval, EvalCache};
use crate::octree::cell::{Cell, LeafData};
use crate::octree::types::GridPos;
use crate::octree::Octree;

/// Build a uniform octree to the specified depth (no adaptive refinement).
pub fn build_uniform(
    py: Python<'_>,
    func: &PyObject,
    bbox: BoundingBox,
    depth: u8,
    iso_value: f64,
) -> PyResult<(Octree, u64)> {
    build_adaptive(py, func, bbox, depth, depth, iso_value)
}

/// Build an adaptive octree with refinement from min_depth to max_depth.
///
/// Strategy:
/// 1. Build uniform to min_depth, evaluate all corners in one batch.
/// 2. For each depth from min_depth to max_depth-1:
///    a. Identify leaf cells that need subdivision (4 criteria).
///    b. Collect new corner + edge midpoint positions.
///    c. Batch-evaluate all new points in one Python call per depth.
///    d. Subdivide and populate children.
/// 3. Apply 2:1 balancing.
pub fn build_adaptive(
    py: Python<'_>,
    func: &PyObject,
    bbox: BoundingBox,
    min_depth: u8,
    max_depth: u8,
    iso_value: f64,
) -> PyResult<(Octree, u64)> {
    let bounds = bbox.to_cubic();
    let mut total_evals = 0u64;

    // Phase 1: Evaluate all corners at min_depth
    let cells_per_axis = 1u32 << min_depth;
    let corners_per_axis = cells_per_axis + 1;
    let cell_size_at_min = bounds.size / cells_per_axis as f64;

    let mut cache = EvalCache::with_capacity((corners_per_axis as usize).pow(3));
    let mut points = Vec::new();
    let mut grid_positions = Vec::new();

    for z in 0..corners_per_axis {
        for y in 0..corners_per_axis {
            for x in 0..corners_per_axis {
                let gp = GridPos::new(x, y, z);
                let world = cache::grid_to_world(&gp, &bounds.origin, cell_size_at_min);
                grid_positions.push(gp);
                points.push(world);
            }
        }
    }

    let result = bridge::batch_evaluate(py, func, &points)?;
    total_evals += points.len() as u64;
    populate_cache(&mut cache, &grid_positions, &result);

    // Build uniform tree to min_depth
    let mut children_storage: Vec<[Cell; 8]> = Vec::new();
    let root = build_cell_recursive(
        &cache, &mut children_storage,
        0, 0, 0, min_depth, 0, iso_value,
    );

    let mut octree = Octree {
        bounds,
        root,
        children: children_storage,
        vertices: Vec::new(),
        hermite_data: Vec::new(),
        min_depth,
        max_depth,
        iso_value,
    };

    // Phase 2: Adaptive refinement from min_depth to max_depth
    for current_depth in min_depth..max_depth {
        let cell_size = bounds.size / (1u32 << current_depth) as f64;
        let child_cell_size = cell_size * 0.5;

        // Collect leaves at current_depth that need subdivision
        let mut leaves_to_refine = Vec::new();
        collect_refineable_leaves(
            &octree, &octree.root, &cache,
            0, 0, 0, 0, current_depth,
            cell_size, iso_value, &mut leaves_to_refine,
        );

        if leaves_to_refine.is_empty() {
            break;
        }

        // Collect all new points needed for subdivision
        let child_depth = current_depth + 1;
        let mut new_points = Vec::new();
        let mut new_grid_positions = Vec::new();

        for &(cx, cy, cz) in &leaves_to_refine {
            // Each leaf subdivides into 8 children, each needing 8 corners
            // Many corners are shared; use cache to deduplicate
            let child_scale = 1u32 << (max_depth - child_depth);
            let base_x = cx * 2;
            let base_y = cy * 2;
            let base_z = cz * 2;

            for octant in 0..8u8 {
                let ocx = base_x + if octant & 1 != 0 { 1 } else { 0 };
                let ocy = base_y + if octant & 2 != 0 { 1 } else { 0 };
                let ocz = base_z + if octant & 4 != 0 { 1 } else { 0 };

                for corner in 0..8u8 {
                    let gx = ocx * child_scale + if corner & 1 != 0 { child_scale } else { 0 };
                    let gy = ocy * child_scale + if corner & 2 != 0 { child_scale } else { 0 };
                    let gz = ocz * child_scale + if corner & 4 != 0 { child_scale } else { 0 };
                    let gp = GridPos::new(gx, gy, gz);

                    if !cache.contains(&gp) {
                        let cell_size_at_max = bounds.size / (1u32 << max_depth) as f64;
                        let world = cache::grid_to_world(&gp, &bounds.origin, cell_size_at_max);
                        new_grid_positions.push(gp);
                        new_points.push(world);
                    }
                }
            }
        }

        // Deduplicate new_points (some may appear multiple times from different cells)
        let mut dedup_points = Vec::new();
        let mut dedup_grid = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for (gp, pt) in new_grid_positions.into_iter().zip(new_points.into_iter()) {
            if seen.insert(gp) {
                dedup_grid.push(gp);
                dedup_points.push(pt);
            }
        }

        // Batch evaluate new points
        if !dedup_points.is_empty() {
            let result = bridge::batch_evaluate(py, func, &dedup_points)?;
            total_evals += dedup_points.len() as u64;
            populate_cache(&mut cache, &dedup_grid, &result);
        }

        // Now subdivide the identified leaves
        for &(cx, cy, cz) in &leaves_to_refine {
            let scale = 1u32 << (max_depth - current_depth);
            let gx = cx * scale;
            let gy = cy * scale;
            let gz = cz * scale;
            subdivide_leaf(
                &mut octree, &cache,
                gx, gy, gz,
                current_depth, max_depth, iso_value,
            );
        }
    }

    Ok((octree, total_evals))
}

/// Populate cache from batch evaluation results.
fn populate_cache(cache: &mut EvalCache, positions: &[GridPos], result: &bridge::EvalResult) {
    for (i, gp) in positions.iter().enumerate() {
        let gradient = result.gradients.as_ref().map(|g| g[i]);
        cache.insert(*gp, CachedEval {
            value: result.values[i],
            gradient,
        });
    }
}

/// Collect leaf cells at `target_depth` that should be subdivided.
///
/// Refinement criteria (applied in order):
/// 1. Sign change at corners — always subdivide
/// 2. Lipschitz guard — surface might be inside even without sign change
/// 3. Gradient magnitude variation — suspicious rapidly-changing function
fn collect_refineable_leaves(
    octree: &Octree,
    cell: &Cell,
    cache: &EvalCache,
    cell_x: u32, cell_y: u32, cell_z: u32,
    current_depth: u8,
    target_depth: u8,
    cell_size: f64,
    iso_value: f64,
    result: &mut Vec<(u32, u32, u32)>,
) {
    match cell {
        Cell::Leaf(data) => {
            if current_depth != target_depth {
                return;
            }
            if should_subdivide(data, cache, cell_x, cell_y, cell_z, target_depth, cell_size, iso_value) {
                result.push((cell_x, cell_y, cell_z));
            }
        }
        Cell::Branch { children_index } => {
            if current_depth >= target_depth {
                return;
            }
            let children = &octree.children[*children_index as usize];
            for octant in 0..8u8 {
                let cx = cell_x * 2 + if octant & 1 != 0 { 1 } else { 0 };
                let cy = cell_y * 2 + if octant & 2 != 0 { 1 } else { 0 };
                let cz = cell_z * 2 + if octant & 4 != 0 { 1 } else { 0 };
                collect_refineable_leaves(
                    octree, &children[octant as usize], cache,
                    cx, cy, cz,
                    current_depth + 1, target_depth,
                    cell_size * 0.5, iso_value,
                    result,
                );
            }
        }
        Cell::Empty | Cell::Full => {
            if current_depth != target_depth {
                return;
            }
            // For Empty/Full cells, check Lipschitz guard
            // (surface might pass through without touching corners)
            if should_subdivide_empty_full(cache, cell, cell_x, cell_y, cell_z, target_depth, cell_size, iso_value) {
                result.push((cell_x, cell_y, cell_z));
            }
        }
    }
}

/// Decide if a Leaf cell should be subdivided.
fn should_subdivide(
    data: &LeafData,
    _cache: &EvalCache,
    _cell_x: u32, _cell_y: u32, _cell_z: u32,
    _depth: u8,
    cell_size: f64,
    iso_value: f64,
) -> bool {
    // Criterion 1: Sign change at corners
    if data.has_sign_change() {
        return true;
    }

    // Criterion 2: Lipschitz guard
    // If the minimum absolute value at corners is small relative to
    // the estimated Lipschitz constant * cell diagonal, surface might be inside
    let shifted: Vec<f64> = data.corner_values.iter().map(|v| v - iso_value).collect();
    let min_abs = shifted.iter().map(|v| v.abs()).fold(f64::MAX, f64::min);
    let cell_diag = cell_size * (3.0f64).sqrt();

    if let Some(ref grads) = data.corner_gradients {
        let grad_mags: Vec<f64> = grads.iter().map(|g| {
            (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt()
        }).collect();
        let max_grad = grad_mags.iter().cloned().fold(0.0f64, f64::max);
        let min_grad = grad_mags.iter().cloned().fold(f64::MAX, f64::min);

        let l_est = max_grad * 1.5; // safety factor
        if min_abs < l_est * cell_diag {
            return true;
        }

        // Criterion 3: Gradient magnitude variation
        if min_grad > 1e-12 && max_grad > 4.0 * min_grad {
            return true;
        }
    }

    false
}

/// Decide if an Empty/Full cell should be subdivided (Lipschitz guard only).
fn should_subdivide_empty_full(
    cache: &EvalCache,
    cell: &Cell,
    cell_x: u32, cell_y: u32, cell_z: u32,
    depth: u8,
    cell_size: f64,
    iso_value: f64,
) -> bool {
    let scale = 1u32; // cells at leaf level
    let _ = depth;

    // Read corner values from cache
    let mut min_abs = f64::MAX;
    let mut max_grad = 0.0f64;

    for corner in 0..8u8 {
        let gx = cell_x + if corner & 1 != 0 { scale } else { 0 };
        let gy = cell_y + if corner & 2 != 0 { scale } else { 0 };
        let gz = cell_z + if corner & 4 != 0 { scale } else { 0 };
        let gp = GridPos::new(gx, gy, gz);

        if let Some(eval) = cache.get(&gp) {
            let shifted = (eval.value - iso_value).abs();
            min_abs = min_abs.min(shifted);
            if let Some(g) = eval.gradient {
                let mag = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
                max_grad = max_grad.max(mag);
            }
        }
    }

    let cell_diag = cell_size * (3.0f64).sqrt();
    let l_est = max_grad * 1.5;

    // Lipschitz guard: if surface might pass through
    min_abs < l_est * cell_diag
}

/// Subdivide a leaf cell in the octree, replacing it with a Branch + 8 children.
fn subdivide_leaf(
    octree: &mut Octree,
    cache: &EvalCache,
    grid_x: u32, grid_y: u32, grid_z: u32,
    depth: u8,
    max_depth: u8,
    iso_value: f64,
) {
    // Build new children first (before mutating the tree)
    let child_scale = 1u32 << (max_depth - depth - 1);
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

            let eval = cache.get(&gp).expect("corner not in cache during subdivision");
            corner_values[corner as usize] = eval.value;
            match eval.gradient {
                Some(g) => corner_grads[corner as usize] = g,
                None => has_grads = false,
            }
        }

        let mut leaf = LeafData::new(corner_values, iso_value);
        if has_grads {
            leaf.corner_gradients = Some(corner_grads);
        }

        child_cells[octant as usize] = if leaf.has_sign_change() {
            Cell::Leaf(leaf)
        } else if leaf.corner_mask == 0 {
            Cell::Empty
        } else {
            Cell::Full
        };
    }

    // Push children and get the index
    let idx = octree.children.len() as u32;
    octree.children.push(child_cells);

    // Now navigate to the cell and replace it with a Branch
    let path = compute_path(grid_x, grid_y, grid_z, depth, max_depth);
    let cell_ref = navigate_to_cell_mut(&mut octree.root, &mut octree.children, &path);
    *cell_ref = Cell::Branch { children_index: idx };
}

/// Compute the path of octant indices from root to a cell at given depth.
fn compute_path(grid_x: u32, grid_y: u32, grid_z: u32, depth: u8, max_depth: u8) -> Vec<u8> {
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

/// Navigate the octree to find a mutable reference to the cell at the given path.
fn navigate_to_cell_mut<'a>(
    root: &'a mut Cell,
    children: &'a mut Vec<[Cell; 8]>,
    path: &[u8],
) -> &'a mut Cell {
    if path.is_empty() {
        return root;
    }

    let mut current = root as *mut Cell;

    for &octant in path.iter() {
        unsafe {
            match &*current {
                Cell::Branch { children_index } => {
                    let idx = *children_index as usize;
                    current = &mut children[idx][octant as usize] as *mut Cell;
                }
                _ => {
                    // Cell is a leaf/empty/full — can't navigate further
                    // Return the current cell (it will be replaced)
                    return &mut *current;
                }
            }
        }
    }

    unsafe { &mut *current }
}

/// Recursively build a cell in the octree (used for initial uniform construction).
fn build_cell_recursive(
    cache: &EvalCache,
    children_storage: &mut Vec<[Cell; 8]>,
    cell_x: u32, cell_y: u32, cell_z: u32,
    target_depth: u8,
    current_depth: u8,
    iso_value: f64,
) -> Cell {
    if current_depth == target_depth {
        let mut corner_values = [0.0f64; 8];
        let mut corner_grads = [[0.0f64; 3]; 8];
        let mut has_grads = true;

        let scale = 1u32 << (target_depth - current_depth);
        for c in 0..8u8 {
            let gp = cache::corner_grid_pos(
                cell_x * scale, cell_y * scale, cell_z * scale, c,
            );
            let eval = cache.get(&gp).expect("corner not in cache");
            corner_values[c as usize] = eval.value;
            match eval.gradient {
                Some(g) => corner_grads[c as usize] = g,
                None => has_grads = false,
            }
        }

        let mut leaf = LeafData::new(corner_values, iso_value);
        if has_grads {
            leaf.corner_gradients = Some(corner_grads);
        }

        if !leaf.has_sign_change() {
            return if leaf.corner_mask == 0 { Cell::Empty } else { Cell::Full };
        }
        Cell::Leaf(leaf)
    } else {
        let mut child_cells: [Cell; 8] = std::array::from_fn(|_| Cell::Empty);

        for octant in 0..8u8 {
            let cx = cell_x * 2 + if octant & 1 != 0 { 1 } else { 0 };
            let cy = cell_y * 2 + if octant & 2 != 0 { 1 } else { 0 };
            let cz = cell_z * 2 + if octant & 4 != 0 { 1 } else { 0 };
            child_cells[octant as usize] = build_cell_recursive(
                cache, children_storage,
                cx, cy, cz,
                target_depth, current_depth + 1, iso_value,
            );
        }

        let all_empty = child_cells.iter().all(|c| matches!(c, Cell::Empty));
        let all_full = child_cells.iter().all(|c| matches!(c, Cell::Full));

        if all_empty {
            Cell::Empty
        } else if all_full {
            Cell::Full
        } else {
            let idx = children_storage.len() as u32;
            children_storage.push(child_cells);
            Cell::Branch { children_index: idx }
        }
    }
}
