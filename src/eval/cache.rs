use std::collections::HashMap;
use crate::octree::types::GridPos;

/// Cached evaluation result for a single point.
#[derive(Clone, Copy, Debug)]
pub struct CachedEval {
    pub value: f64,
    pub gradient: Option<[f64; 3]>,
}

/// Evaluation cache using quantized grid positions as keys.
///
/// Corners shared between adjacent cells map to the same GridPos,
/// ensuring each unique point is evaluated only once.
pub struct EvalCache {
    entries: HashMap<GridPos, CachedEval>,
}

impl EvalCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(cap),
        }
    }

    pub fn get(&self, pos: &GridPos) -> Option<&CachedEval> {
        self.entries.get(pos)
    }

    pub fn insert(&mut self, pos: GridPos, eval: CachedEval) {
        self.entries.insert(pos, eval);
    }

    pub fn contains(&self, pos: &GridPos) -> bool {
        self.entries.contains_key(pos)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Convert a cell's corner position to a quantized grid position.
///
/// At depth `d`, the grid has `2^d` cells per axis, meaning
/// corner coordinates are integers in `[0, 2^d]`.
///
/// `cell_x, cell_y, cell_z` are the cell's integer coordinates at depth `d`.
/// `corner_idx` is 0-7, with bits encoding the X, Y, Z offset.
pub fn corner_grid_pos(cell_x: u32, cell_y: u32, cell_z: u32, corner_idx: u8) -> GridPos {
    GridPos::new(
        cell_x + (corner_idx & 1) as u32,
        cell_y + ((corner_idx >> 1) & 1) as u32,
        cell_z + ((corner_idx >> 2) & 1) as u32,
    )
}

/// Convert a quantized grid position to a world-space coordinate.
///
/// `grid_pos` is an integer grid position where each axis ranges from 0 to 2^depth.
/// `origin` is the world-space origin of the octree.
/// `cell_size` is the size of a single cell at the given depth.
pub fn grid_to_world(pos: &GridPos, origin: &[f64; 3], cell_size: f64) -> [f64; 3] {
    [
        origin[0] + pos.x as f64 * cell_size,
        origin[1] + pos.y as f64 * cell_size,
        origin[2] + pos.z as f64 * cell_size,
    ]
}
