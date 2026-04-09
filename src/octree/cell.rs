/// A single octree cell.
#[derive(Clone)]
pub enum Cell {
    /// Entirely outside the surface (all corners positive / same sign, no surface nearby).
    Empty,
    /// Entirely inside the surface (all corners negative / same sign, no surface nearby).
    Full,
    /// Interior node with 8 children. Index into `Octree::children`.
    Branch {
        children_index: u32,
    },
    /// Leaf cell that may contain surface crossings.
    Leaf(LeafData),
}

/// Data stored at a leaf cell.
#[derive(Clone)]
pub struct LeafData {
    /// Bitmask: bit i = 1 if corner i is inside (value < iso_value).
    pub corner_mask: u8,
    /// Function values at the 8 corners.
    pub corner_values: [f64; 8],
    /// Gradient vectors at the 8 corners (estimated or analytic).
    pub corner_gradients: Option<[[f64; 3]; 8]>,
    /// Index into `Octree::vertices` for this cell's first vertex.
    /// Set during DC extraction (Phase 5-6).
    pub vertex_start: u32,
    /// Number of vertices in this cell (MDC: >1 per connected component).
    pub vertex_count: u8,
    /// Edge intersection data indices. Set during Hermite data computation (Phase 4).
    pub edge_intersections: [Option<u32>; 12],
}

impl LeafData {
    pub fn new(corner_values: [f64; 8], iso_value: f64) -> Self {
        let mut mask = 0u8;
        for i in 0..8 {
            if corner_values[i] < iso_value {
                mask |= 1 << i;
            }
        }
        Self {
            corner_mask: mask,
            corner_values,
            corner_gradients: None,
            vertex_start: 0,
            vertex_count: 0,
            edge_intersections: [None; 12],
        }
    }

    /// Returns true if this leaf has any sign change across its edges.
    pub fn has_sign_change(&self) -> bool {
        self.corner_mask != 0 && self.corner_mask != 0xFF
    }
}

impl Cell {
    pub fn is_leaf(&self) -> bool {
        matches!(self, Cell::Leaf(_) | Cell::Empty | Cell::Full)
    }

    pub fn is_branch(&self) -> bool {
        matches!(self, Cell::Branch { .. })
    }
}
