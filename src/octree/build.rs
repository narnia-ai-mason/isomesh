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
    build_adaptive(py, func, bbox, depth, depth, iso_value, 30.0)
}

/// Build an adaptive octree with refinement from min_depth to max_depth.
///
/// Refinement strategy:
/// - All cells subdivide uniformly to min_depth.
/// - From min_depth to max_depth, cells subdivide only if they contain
///   a feature (normal angle spread > angle_threshold) or if the Lipschitz
///   guard detects a potential thin feature.
/// - Smooth surface cells are NOT subdivided beyond min_depth.
pub fn build_adaptive(
    py: Python<'_>,
    func: &PyObject,
    bbox: BoundingBox,
    min_depth: u8,
    max_depth: u8,
    iso_value: f64,
    angle_threshold_deg: f64,
) -> PyResult<(Octree, u64)> {
    build_adaptive_inner(py, func, bbox, min_depth, max_depth, iso_value, angle_threshold_deg, false)
}

pub fn build_adaptive_true(
    py: Python<'_>,
    func: &PyObject,
    bbox: BoundingBox,
    min_depth: u8,
    max_depth: u8,
    iso_value: f64,
    angle_threshold_deg: f64,
) -> PyResult<(Octree, u64)> {
    build_adaptive_inner(py, func, bbox, min_depth, max_depth, iso_value, angle_threshold_deg, true)
}

fn build_adaptive_inner(
    py: Python<'_>,
    func: &PyObject,
    bbox: BoundingBox,
    min_depth: u8,
    max_depth: u8,
    iso_value: f64,
    angle_threshold_deg: f64,
    true_adaptive: bool,
) -> PyResult<(Octree, u64)> {
    let bounds = bbox.to_cubic();
    let mut total_evals = 0u64;
    let angle_threshold_cos = (angle_threshold_deg * std::f64::consts::PI / 180.0).cos();

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

    // If function doesn't provide gradients, estimate via FD
    let has_gradients = result.gradients.is_some();
    populate_cache(&mut cache, &grid_positions, &result);

    if !has_gradients {
        let fd_grads = bridge::estimate_gradients_fd(py, func, &points, cell_size_at_min * 0.01)?;
        total_evals += points.len() as u64 * 6; // 6 FD evaluations per point
        for (i, gp) in grid_positions.iter().enumerate() {
            if let Some(entry) = cache.get_mut(gp) {
                entry.gradient = Some(fd_grads[i]);
            }
        }
    }

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

        // Collect leaves that need subdivision (feature-based criteria)
        let mut leaves_to_refine = Vec::new();
        collect_refineable_leaves(
            &octree, &octree.root, &cache,
            0, 0, 0, 0, current_depth,
            cell_size, iso_value, angle_threshold_cos,
            &mut leaves_to_refine,
        );

        if leaves_to_refine.is_empty() {
            break;
        }

        // Collect new corner points needed for subdivision
        let mut new_points = Vec::new();
        let mut new_grid_positions = Vec::new();
        let child_scale = 1u32 << (max_depth - current_depth - 1);
        let cell_size_at_max = bounds.size / (1u32 << max_depth) as f64;

        for &(cx, cy, cz) in &leaves_to_refine {
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
                        let world = cache::grid_to_world(&gp, &bounds.origin, cell_size_at_max);
                        new_grid_positions.push(gp);
                        new_points.push(world);
                    }
                }
            }
        }

        // Deduplicate
        let mut dedup_points = Vec::new();
        let mut dedup_grid = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for (gp, pt) in new_grid_positions.into_iter().zip(new_points.into_iter()) {
            if seen.insert(gp) {
                dedup_grid.push(gp);
                dedup_points.push(pt);
            }
        }

        if !dedup_points.is_empty() {
            let result = bridge::batch_evaluate(py, func, &dedup_points)?;
            total_evals += dedup_points.len() as u64;

            let fd_grads = if result.gradients.is_none() {
                let g = bridge::estimate_gradients_fd(py, func, &dedup_points, cell_size_at_max * 0.01)?;
                total_evals += dedup_points.len() as u64 * 6;
                Some(g)
            } else {
                None
            };

            for (i, gp) in dedup_grid.iter().enumerate() {
                let gradient = result.gradients.as_ref()
                    .map(|g| g[i])
                    .or_else(|| fd_grads.as_ref().map(|g| g[i]));
                cache.insert(*gp, CachedEval {
                    value: result.values[i],
                    gradient,
                });
            }
        }

        // Subdivide identified leaves
        for &(cx, cy, cz) in &leaves_to_refine {
            let scale = 1u32 << (max_depth - current_depth);
            let gx = cx * scale;
            let gy = cy * scale;
            let gz = cz * scale;
            subdivide_leaf(&mut octree, &cache, gx, gy, gz, current_depth, max_depth, iso_value);
        }
    }

    // Phase 3: Either force uniform depth (legacy) or 2:1 balance (true adaptive).
    if true_adaptive && min_depth < max_depth {
        let balance_evals = crate::octree::balance::enforce_balance_2to1(
            py, func, &mut octree, &mut cache,
        )?;
        total_evals += balance_evals;
    } else if min_depth < max_depth {
    // Legacy Phase 3: Force all remaining surface leaves to max_depth.
    // This ensures all leaves in DC extraction are at the same depth,
    // avoiding T-junction holes at depth boundaries.
        loop {
            let mut coarse_leaves = Vec::new();
            collect_coarse_surface_leaves(
                &octree, &octree.root, 0, 0, 0, 0, max_depth,
                &mut coarse_leaves,
            );
            if coarse_leaves.is_empty() { break; }

            // Collect new corner points
            let child_scale_base = max_depth; // for computing child_scale
            let cell_size_at_max = bounds.size / (1u32 << max_depth) as f64;
            let mut new_points = Vec::new();
            let mut new_grid = Vec::new();
            let mut seen = std::collections::HashSet::new();

            for &(cx, cy, cz, depth) in &coarse_leaves {
                let cs = 1u32 << (max_depth - depth - 1);
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
                        if !cache.contains(&gp) && seen.insert(gp) {
                            new_grid.push(gp);
                            new_points.push(cache::grid_to_world(&gp, &bounds.origin, cell_size_at_max));
                        }
                    }
                }
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

            // Subdivide
            for &(cx, cy, cz, depth) in &coarse_leaves {
                let gx = cx * (1u32 << (max_depth - depth));
                let gy = cy * (1u32 << (max_depth - depth));
                let gz = cz * (1u32 << (max_depth - depth));
                subdivide_leaf(&mut octree, &cache, gx, gy, gz, depth, max_depth, iso_value);
            }
        }
    }

    Ok((octree, total_evals))
}

/// Collect surface-crossing leaf cells that are coarser than max_depth.
fn collect_coarse_surface_leaves(
    octree: &Octree, cell: &Cell,
    cx: u32, cy: u32, cz: u32, depth: u8, max_depth: u8,
    result: &mut Vec<(u32, u32, u32, u8)>,
) {
    match cell {
        Cell::Leaf(data) if data.has_sign_change() && depth < max_depth => {
            result.push((cx, cy, cz, depth));
        }
        Cell::Branch { children_index } => {
            let children = &octree.children[*children_index as usize];
            for octant in 0..8u8 {
                collect_coarse_surface_leaves(
                    octree, &children[octant as usize],
                    cx * 2 + if octant & 1 != 0 { 1 } else { 0 },
                    cy * 2 + if octant & 2 != 0 { 1 } else { 0 },
                    cz * 2 + if octant & 4 != 0 { 1 } else { 0 },
                    depth + 1, max_depth, result,
                );
            }
        }
        _ => {}
    }
}

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
/// Feature-based criteria (only cells with sign changes AND feature indicators):
/// 1. Normal angle spread > threshold (feature boundary detected)
/// 2. Lipschitz guard (potential thin feature in non-sign-change cells)
/// 3. Gradient magnitude variation > 4x
fn collect_refineable_leaves(
    octree: &Octree,
    cell: &Cell,
    cache: &EvalCache,
    cell_x: u32, cell_y: u32, cell_z: u32,
    current_depth: u8,
    target_depth: u8,
    cell_size: f64,
    iso_value: f64,
    angle_threshold_cos: f64,
    result: &mut Vec<(u32, u32, u32)>,
) {
    match cell {
        Cell::Leaf(data) => {
            if current_depth != target_depth { return; }
            if should_subdivide_leaf(data, cell_size, iso_value, angle_threshold_cos) {
                result.push((cell_x, cell_y, cell_z));
            }
        }
        Cell::Branch { children_index } => {
            if current_depth >= target_depth { return; }
            let children = &octree.children[*children_index as usize];
            for octant in 0..8u8 {
                let cx = cell_x * 2 + if octant & 1 != 0 { 1 } else { 0 };
                let cy = cell_y * 2 + if octant & 2 != 0 { 1 } else { 0 };
                let cz = cell_z * 2 + if octant & 4 != 0 { 1 } else { 0 };
                collect_refineable_leaves(
                    octree, &children[octant as usize], cache,
                    cx, cy, cz,
                    current_depth + 1, target_depth,
                    cell_size * 0.5, iso_value, angle_threshold_cos,
                    result,
                );
            }
        }
        Cell::Empty | Cell::Full => {
            if current_depth != target_depth { return; }
            if should_subdivide_empty_full(cache, cell, cell_x, cell_y, cell_z, cell_size, iso_value) {
                result.push((cell_x, cell_y, cell_z));
            }
        }
    }
}

/// Decide if a Leaf cell should be subdivided beyond min_depth.
///
/// Only subdivides if the cell contains a detectable feature:
/// - Normal angle spread exceeds threshold (sharp edge/corner)
/// - Gradient magnitude varies significantly (rapidly changing function)
fn should_subdivide_leaf(
    data: &LeafData,
    cell_size: f64,
    iso_value: f64,
    angle_threshold_cos: f64,
) -> bool {
    // Must have sign change to be worth subdividing
    if !data.has_sign_change() {
        // For non-sign-change cells: check Lipschitz guard
        if let Some(ref grads) = data.corner_gradients {
            let shifted: Vec<f64> = data.corner_values.iter().map(|v| v - iso_value).collect();
            let min_abs = shifted.iter().map(|v| v.abs()).fold(f64::MAX, f64::min);
            let cell_diag = cell_size * (3.0f64).sqrt();
            let max_grad = grads.iter()
                .map(|g| (g[0]*g[0] + g[1]*g[1] + g[2]*g[2]).sqrt())
                .fold(0.0f64, f64::max);
            if min_abs < max_grad * 1.5 * cell_diag {
                return true;
            }
        }
        return false;
    }

    // Has sign change — check if this cell straddles a feature
    if let Some(ref grads) = data.corner_gradients {
        // Criterion 1: Normal angle spread
        // Check min cos(angle) between any pair of normalized corner gradients
        let mut normals = Vec::new();
        for g in grads.iter() {
            let len = (g[0]*g[0] + g[1]*g[1] + g[2]*g[2]).sqrt();
            if len > 1e-12 {
                normals.push([g[0]/len, g[1]/len, g[2]/len]);
            }
        }

        if normals.len() >= 2 {
            let mut min_dot = 1.0f64;
            for i in 0..normals.len() {
                for j in (i+1)..normals.len() {
                    let d = normals[i][0]*normals[j][0]
                          + normals[i][1]*normals[j][1]
                          + normals[i][2]*normals[j][2];
                    min_dot = min_dot.min(d);
                }
            }
            // If min dot product < cos(threshold), the angle exceeds threshold
            if min_dot < angle_threshold_cos {
                return true; // Feature detected: subdivide
            }
        }

        // Criterion 2: Gradient magnitude variation
        let grad_mags: Vec<f64> = grads.iter()
            .map(|g| (g[0]*g[0] + g[1]*g[1] + g[2]*g[2]).sqrt())
            .collect();
        let max_grad = grad_mags.iter().cloned().fold(0.0f64, f64::max);
        let min_grad = grad_mags.iter().cloned().fold(f64::MAX, f64::min);
        if min_grad > 1e-12 && max_grad > 4.0 * min_grad {
            return true;
        }
    }

    false // Smooth surface cell: don't subdivide further
}

fn should_subdivide_empty_full(
    cache: &EvalCache,
    _cell: &Cell,
    cell_x: u32, cell_y: u32, cell_z: u32,
    cell_size: f64,
    iso_value: f64,
) -> bool {
    let scale = 1u32;
    let mut min_abs = f64::MAX;
    let mut max_grad = 0.0f64;

    for corner in 0..8u8 {
        let gx = cell_x + if corner & 1 != 0 { scale } else { 0 };
        let gy = cell_y + if corner & 2 != 0 { scale } else { 0 };
        let gz = cell_z + if corner & 4 != 0 { scale } else { 0 };
        let gp = GridPos::new(gx, gy, gz);
        if let Some(eval) = cache.get(&gp) {
            min_abs = min_abs.min((eval.value - iso_value).abs());
            if let Some(g) = eval.gradient {
                max_grad = max_grad.max((g[0]*g[0] + g[1]*g[1] + g[2]*g[2]).sqrt());
            }
        }
    }

    let cell_diag = cell_size * (3.0f64).sqrt();
    min_abs < max_grad * 1.5 * cell_diag
}

fn subdivide_leaf(
    octree: &mut Octree, cache: &EvalCache,
    grid_x: u32, grid_y: u32, grid_z: u32,
    depth: u8, max_depth: u8, iso_value: f64,
) {
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

    let path = compute_path(grid_x, grid_y, grid_z, depth, max_depth);
    let cell_ref = navigate_to_cell_mut(&mut octree.root, &mut octree.children, &path);
    *cell_ref = Cell::Branch { children_index: idx };
}

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

fn navigate_to_cell_mut<'a>(
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

fn build_cell_recursive(
    cache: &EvalCache,
    children_storage: &mut Vec<[Cell; 8]>,
    cell_x: u32, cell_y: u32, cell_z: u32,
    target_depth: u8, current_depth: u8,
    iso_value: f64,
) -> Cell {
    if current_depth == target_depth {
        let mut corner_values = [0.0f64; 8];
        let mut corner_grads = [[0.0f64; 3]; 8];
        let mut has_grads = true;
        let scale = 1u32 << (target_depth - current_depth);
        for c in 0..8u8 {
            let gp = cache::corner_grid_pos(cell_x * scale, cell_y * scale, cell_z * scale, c);
            let eval = cache.get(&gp).expect("corner not in cache");
            corner_values[c as usize] = eval.value;
            match eval.gradient {
                Some(g) => corner_grads[c as usize] = g,
                None => has_grads = false,
            }
        }
        let mut leaf = LeafData::new(corner_values, iso_value);
        if has_grads { leaf.corner_gradients = Some(corner_grads); }
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
                cache, children_storage, cx, cy, cz,
                target_depth, current_depth + 1, iso_value,
            );
        }
        let all_empty = child_cells.iter().all(|c| matches!(c, Cell::Empty));
        let all_full = child_cells.iter().all(|c| matches!(c, Cell::Full));
        if all_empty { Cell::Empty }
        else if all_full { Cell::Full }
        else {
            let idx = children_storage.len() as u32;
            children_storage.push(child_cells);
            Cell::Branch { children_index: idx }
        }
    }
}
