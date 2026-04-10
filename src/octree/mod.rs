pub mod cell;
pub mod types;
pub mod morton;
pub mod build;
pub mod balance;

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

/// Compute the 6 face-neighbor coordinates of cell (cx, cy, cz) at depth d.
/// Returns None if the neighbor is outside the octree bounds [0, 2^d).
pub fn face_neighbors(cx: u32, cy: u32, cz: u32, depth: u8) -> [Option<(u32, u32, u32)>; 6] {
    let max_coord = 1u32 << depth;
    [
        if cx > 0 { Some((cx - 1, cy, cz)) } else { None },           // -X
        if cx + 1 < max_coord { Some((cx + 1, cy, cz)) } else { None }, // +X
        if cy > 0 { Some((cx, cy - 1, cz)) } else { None },           // -Y
        if cy + 1 < max_coord { Some((cx, cy + 1, cz)) } else { None }, // +Y
        if cz > 0 { Some((cx, cy, cz - 1)) } else { None },           // -Z
        if cz + 1 < max_coord { Some((cx, cy, cz + 1)) } else { None }, // +Z
    ]
}

impl Octree {
    /// Look up the cell at (cx, cy, cz) at the given depth.
    ///
    /// Navigates from the root, determining the octant at each level from the
    /// cell coordinates. If the path hits a Leaf/Empty/Full before reaching
    /// `depth`, returns that cell with its actual (coarser) depth.
    pub fn cell_at(&self, cx: u32, cy: u32, cz: u32, depth: u8) -> (&Cell, u8) {
        let mut cell = &self.root;
        for d in 0..depth {
            match cell {
                Cell::Branch { children_index } => {
                    let shift = depth - d - 1;
                    let oct_x = (cx >> shift) & 1;
                    let oct_y = (cy >> shift) & 1;
                    let oct_z = (cz >> shift) & 1;
                    let octant = (oct_x | (oct_y << 1) | (oct_z << 2)) as usize;
                    cell = &self.children[*children_index as usize][octant];
                }
                _ => return (cell, d),
            }
        }
        (cell, depth)
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bbox::CubicBounds;

    /// Build a minimal test octree: root is Branch with 8 children.
    /// Child 0 (octant 000) is a Leaf with a sign change.
    /// All other children are Empty.
    fn make_test_octree() -> Octree {
        let mut corner_values = [1.0f64; 8];
        corner_values[0] = -1.0; // corner 0 inside → sign change
        let leaf = LeafData::new(corner_values, 0.0);

        let children: [Cell; 8] = [
            Cell::Leaf(leaf),
            Cell::Empty,
            Cell::Empty,
            Cell::Empty,
            Cell::Empty,
            Cell::Empty,
            Cell::Empty,
            Cell::Empty,
        ];

        let mut children_storage = Vec::new();
        children_storage.push(children);

        Octree {
            bounds: CubicBounds { origin: [0.0, 0.0, 0.0], size: 1.0 },
            root: Cell::Branch { children_index: 0 },
            children: children_storage,
            vertices: Vec::new(),
            hermite_data: Vec::new(),
            min_depth: 1,
            max_depth: 1,
            iso_value: 0.0,
        }
    }

    /// Build a 2-level test octree: root → Branch → child 0 is Branch → 8 leaves.
    fn make_depth2_octree() -> Octree {
        let leaf_sign = LeafData::new({
            let mut v = [1.0; 8]; v[0] = -1.0; v
        }, 0.0);
        let leaf_empty = LeafData::new([1.0; 8], 0.0);

        // Level 2: 8 children of child 0
        let level2_children: [Cell; 8] = std::array::from_fn(|i| {
            if i == 0 {
                Cell::Leaf(leaf_sign.clone())
            } else {
                Cell::Leaf(leaf_empty.clone())
            }
        });

        let mut children_storage = Vec::new();
        // Index 0: level 2 children (children of octant 0)
        children_storage.push(level2_children);

        // Index 1: level 1 children (root's children)
        let level1_children: [Cell; 8] = [
            Cell::Branch { children_index: 0 }, // octant 0 has sub-children
            Cell::Empty,
            Cell::Empty,
            Cell::Empty,
            Cell::Empty,
            Cell::Empty,
            Cell::Empty,
            Cell::Empty,
        ];
        children_storage.push(level1_children);

        Octree {
            bounds: CubicBounds { origin: [0.0, 0.0, 0.0], size: 1.0 },
            root: Cell::Branch { children_index: 1 },
            children: children_storage,
            vertices: Vec::new(),
            hermite_data: Vec::new(),
            min_depth: 1,
            max_depth: 2,
            iso_value: 0.0,
        }
    }

    #[test]
    fn test_cell_at_depth1_leaf() {
        let octree = make_test_octree();
        // Octant 0 (0,0,0) at depth 1 should be the Leaf
        let (cell, depth) = octree.cell_at(0, 0, 0, 1);
        assert_eq!(depth, 1);
        assert!(matches!(cell, Cell::Leaf(_)));
    }

    #[test]
    fn test_cell_at_depth1_empty() {
        let octree = make_test_octree();
        // Octant 7 (1,1,1) at depth 1 should be Empty
        let (cell, depth) = octree.cell_at(1, 1, 1, 1);
        assert_eq!(depth, 1);
        assert!(matches!(cell, Cell::Empty));
    }

    #[test]
    fn test_cell_at_coarser_return() {
        let octree = make_test_octree();
        // Query depth 2 at (2, 2, 2): octant 1 at depth 1 is Empty.
        // cell_at should return early at depth 1 with Empty.
        let (cell, actual_depth) = octree.cell_at(2, 2, 2, 2);
        assert_eq!(actual_depth, 1);
        assert!(matches!(cell, Cell::Empty));
    }

    #[test]
    fn test_cell_at_depth2() {
        let octree = make_depth2_octree();
        // (0,0,0) at depth 2 should be the sign-change Leaf
        let (cell, depth) = octree.cell_at(0, 0, 0, 2);
        assert_eq!(depth, 2);
        assert!(matches!(cell, Cell::Leaf(_)));

        // (1,0,0) at depth 2: child of octant 0 at depth 1, octant 1 at depth 2
        let (cell, depth) = octree.cell_at(1, 0, 0, 2);
        assert_eq!(depth, 2);
        // This is the leaf_empty (no sign change)
        assert!(matches!(cell, Cell::Leaf(_)));
    }

    #[test]
    fn test_cell_at_mixed_depth() {
        let octree = make_depth2_octree();
        // (2, 0, 0) at depth 2: octant 1 at depth 1 is Empty → returns at depth 1
        let (cell, actual_depth) = octree.cell_at(2, 0, 0, 2);
        assert_eq!(actual_depth, 1);
        assert!(matches!(cell, Cell::Empty));
    }

    #[test]
    fn test_face_neighbors_interior() {
        let neighbors = face_neighbors(2, 3, 4, 4);
        assert_eq!(neighbors[0], Some((1, 3, 4))); // -X
        assert_eq!(neighbors[1], Some((3, 3, 4))); // +X
        assert_eq!(neighbors[2], Some((2, 2, 4))); // -Y
        assert_eq!(neighbors[3], Some((2, 4, 4))); // +Y
        assert_eq!(neighbors[4], Some((2, 3, 3))); // -Z
        assert_eq!(neighbors[5], Some((2, 3, 5))); // +Z
    }

    #[test]
    fn test_face_neighbors_boundary() {
        // At origin: -X, -Y, -Z are out of bounds
        let neighbors = face_neighbors(0, 0, 0, 3);
        assert_eq!(neighbors[0], None); // -X
        assert_eq!(neighbors[1], Some((1, 0, 0))); // +X
        assert_eq!(neighbors[2], None); // -Y
        assert_eq!(neighbors[4], None); // -Z

        // At max corner: +X, +Y, +Z are out of bounds (max = 2^3 = 8)
        let neighbors = face_neighbors(7, 7, 7, 3);
        assert_eq!(neighbors[0], Some((6, 7, 7))); // -X
        assert_eq!(neighbors[1], None); // +X
        assert_eq!(neighbors[3], None); // +Y
        assert_eq!(neighbors[5], None); // +Z
    }
}
