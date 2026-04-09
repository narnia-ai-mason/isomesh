pub mod cell;
pub mod types;
pub mod morton;
pub mod build;

use crate::bbox::CubicBounds;
use cell::{Cell, LeafData};

/// Hermite data: intersection point and normal on an edge.
#[derive(Clone, Copy, Debug)]
pub struct HermitePoint {
    pub pos: [f64; 3],
    pub normal: [f64; 3],
    pub t: f64, // parameter along edge [0, 1]
}

/// Vertex computed by QEF for a cell.
#[derive(Clone, Copy, Debug)]
pub struct CellVertex {
    pub pos: [f64; 3],
    pub qef_error: f64,
}

/// The adaptive octree data structure.
///
/// Uses flat-children array storage: each Branch stores an index into `children`,
/// where `children[i]` holds the 8 child cells.
pub struct Octree {
    /// Root cell bounds (always cubic).
    pub bounds: CubicBounds,
    /// The root cell.
    pub root: Cell,
    /// Flat storage: children[i] = [Cell; 8] for Branch { children_index: i }.
    pub children: Vec<[Cell; 8]>,
    /// Vertex positions computed by QEF (Phase 5-6).
    pub vertices: Vec<CellVertex>,
    /// Hermite data pool (Phase 4).
    pub hermite_data: Vec<HermitePoint>,
    /// Configuration.
    pub min_depth: u8,
    pub max_depth: u8,
    pub iso_value: f64,
}

/// Statistics about the octree.
#[derive(Debug, Clone)]
pub struct OctreeStats {
    pub leaf_count: u64,
    pub branch_count: u64,
    pub empty_count: u64,
    pub full_count: u64,
    pub max_actual_depth: u8,
    pub total_evaluations: u64,
}

impl Octree {
    /// Count leaves, branches, etc.
    pub fn stats(&self) -> OctreeStats {
        let mut stats = OctreeStats {
            leaf_count: 0,
            branch_count: 0,
            empty_count: 0,
            full_count: 0,
            max_actual_depth: 0,
            total_evaluations: 0,
        };
        self.count_recursive(&self.root, 0, &mut stats);
        stats
    }

    fn count_recursive(&self, cell: &Cell, depth: u8, stats: &mut OctreeStats) {
        match cell {
            Cell::Empty => stats.empty_count += 1,
            Cell::Full => stats.full_count += 1,
            Cell::Leaf(_) => {
                stats.leaf_count += 1;
                stats.max_actual_depth = stats.max_actual_depth.max(depth);
            }
            Cell::Branch { children_index } => {
                stats.branch_count += 1;
                let children = &self.children[*children_index as usize];
                for child in children {
                    self.count_recursive(child, depth + 1, stats);
                }
            }
        }
    }

    /// Iterate over all leaf cells with their bounds and depth.
    pub fn for_each_leaf<F>(&self, mut f: F)
    where
        F: FnMut(&LeafData, &CubicBounds, u8),
    {
        self.for_each_leaf_recursive(&self.root, &self.bounds, 0, &mut f);
    }

    fn for_each_leaf_recursive<F>(
        &self,
        cell: &Cell,
        bounds: &CubicBounds,
        depth: u8,
        f: &mut F,
    ) where
        F: FnMut(&LeafData, &CubicBounds, u8),
    {
        match cell {
            Cell::Empty | Cell::Full => {}
            Cell::Leaf(data) => f(data, bounds, depth),
            Cell::Branch { children_index } => {
                let children = &self.children[*children_index as usize];
                for (i, child) in children.iter().enumerate() {
                    let child_bounds = bounds.child(i as u8);
                    self.for_each_leaf_recursive(child, &child_bounds, depth + 1, f);
                }
            }
        }
    }
}
