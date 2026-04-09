/// Axis-aligned bounding box with expand-to-cubic logic.
///
/// Octree cells must always be cubic. If the user's bounding box is non-cubic,
/// we expand it to the smallest enclosing cube.
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

/// A cubic bounding box used by the octree.
#[derive(Debug, Clone, Copy)]
pub struct CubicBounds {
    /// Minimum corner (origin).
    pub origin: [f64; 3],
    /// Side length (all axes equal).
    pub size: f64,
}

impl BoundingBox {
    pub fn new(min: [f64; 3], max: [f64; 3]) -> Self {
        debug_assert!(min[0] < max[0] && min[1] < max[1] && min[2] < max[2]);
        Self { min, max }
    }

    /// Expand this bounding box to the smallest enclosing cube.
    /// The cube is centered on the original bbox center.
    pub fn to_cubic(&self) -> CubicBounds {
        let center = [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ];
        let dx = self.max[0] - self.min[0];
        let dy = self.max[1] - self.min[1];
        let dz = self.max[2] - self.min[2];
        let size = dx.max(dy).max(dz);
        let half = size * 0.5;
        CubicBounds {
            origin: [center[0] - half, center[1] - half, center[2] - half],
            size,
        }
    }
}

impl CubicBounds {
    /// Get the center of this cubic region.
    pub fn center(&self) -> [f64; 3] {
        let h = self.size * 0.5;
        [self.origin[0] + h, self.origin[1] + h, self.origin[2] + h]
    }

    /// Get the child bounds for octant index (0-7).
    /// Octant indexing: bit 0 = X, bit 1 = Y, bit 2 = Z.
    pub fn child(&self, octant: u8) -> CubicBounds {
        debug_assert!(octant < 8);
        let half = self.size * 0.5;
        CubicBounds {
            origin: [
                self.origin[0] + if octant & 1 != 0 { half } else { 0.0 },
                self.origin[1] + if octant & 2 != 0 { half } else { 0.0 },
                self.origin[2] + if octant & 4 != 0 { half } else { 0.0 },
            ],
            size: half,
        }
    }

    /// Get the position of corner `idx` (0-7).
    /// Corner indexing: bit 0 = X, bit 1 = Y, bit 2 = Z (0 = min, 1 = max).
    pub fn corner(&self, idx: u8) -> [f64; 3] {
        debug_assert!(idx < 8);
        [
            self.origin[0] + if idx & 1 != 0 { self.size } else { 0.0 },
            self.origin[1] + if idx & 2 != 0 { self.size } else { 0.0 },
            self.origin[2] + if idx & 4 != 0 { self.size } else { 0.0 },
        ]
    }

    /// Get the midpoint of edge `idx` (0-11).
    /// Edges are grouped by parallel axis: 0-3 (X-axis), 4-7 (Y-axis), 8-11 (Z-axis).
    /// Within each group, the 4 edges are indexed by the 2 remaining axis bits.
    pub fn edge_midpoint(&self, idx: u8) -> [f64; 3] {
        debug_assert!(idx < 12);
        let half = self.size * 0.5;
        let axis = (idx / 4) as usize; // 0=X, 1=Y, 2=Z
        let bits = idx % 4;
        // The two non-axis dimensions
        let (a1, a2) = match axis {
            0 => (1usize, 2usize),
            1 => (0usize, 2usize),
            _ => (0usize, 1usize),
        };
        let mut pos = self.origin;
        pos[axis] += half; // midpoint along the edge axis
        pos[a1] += if bits & 1 != 0 { self.size } else { 0.0 };
        pos[a2] += if bits & 2 != 0 { self.size } else { 0.0 };
        pos
    }

    /// Get the two corner indices for edge `idx`.
    pub fn edge_corners(idx: u8) -> (u8, u8) {
        debug_assert!(idx < 12);
        EDGE_CORNERS[idx as usize]
    }

    /// Diagonal length of this cubic cell.
    pub fn diagonal(&self) -> f64 {
        self.size * std::f64::consts::SQRT_2 * (3.0f64 / 2.0).sqrt()
        // = size * sqrt(3)
    }

    pub fn contains(&self, p: [f64; 3]) -> bool {
        p[0] >= self.origin[0]
            && p[0] <= self.origin[0] + self.size
            && p[1] >= self.origin[1]
            && p[1] <= self.origin[1] + self.size
            && p[2] >= self.origin[2]
            && p[2] <= self.origin[2] + self.size
    }
}

/// Edge-to-corner lookup table.
/// edge_corners[i] = (corner_a, corner_b) for edge i.
/// Edges 0-3: parallel to X-axis, Edges 4-7: parallel to Y-axis, Edges 8-11: parallel to Z-axis.
const EDGE_CORNERS: [(u8, u8); 12] = [
    // X-axis edges (axis=0): varying bit 0 (X), fixed bits 1,2 (Y,Z)
    (0b000, 0b001), // edge 0: Y=0,Z=0
    (0b010, 0b011), // edge 1: Y=1,Z=0
    (0b100, 0b101), // edge 2: Y=0,Z=1
    (0b110, 0b111), // edge 3: Y=1,Z=1
    // Y-axis edges (axis=1): varying bit 1 (Y), fixed bits 0,2 (X,Z)
    (0b000, 0b010), // edge 4: X=0,Z=0
    (0b001, 0b011), // edge 5: X=1,Z=0
    (0b100, 0b110), // edge 6: X=0,Z=1
    (0b101, 0b111), // edge 7: X=1,Z=1
    // Z-axis edges (axis=2): varying bit 2 (Z), fixed bits 0,1 (X,Y)
    (0b000, 0b100), // edge 8: X=0,Y=0
    (0b001, 0b101), // edge 9: X=1,Y=0
    (0b010, 0b110), // edge 10: X=0,Y=1
    (0b011, 0b111), // edge 11: X=1,Y=1
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cubic_expansion_square() {
        let bb = BoundingBox::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let cb = bb.to_cubic();
        assert!((cb.size - 2.0).abs() < 1e-10);
        assert!((cb.origin[0] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cubic_expansion_rectangular() {
        let bb = BoundingBox::new([-1.0, -1.0, -1.0], [3.0, 1.0, 1.0]);
        let cb = bb.to_cubic();
        // max dimension is X: 4.0
        assert!((cb.size - 4.0).abs() < 1e-10);
        // center is (1, 0, 0), so origin = (1-2, 0-2, 0-2) = (-1, -2, -2)
        assert!((cb.origin[0] - (-1.0)).abs() < 1e-10);
        assert!((cb.origin[1] - (-2.0)).abs() < 1e-10);
        assert!((cb.origin[2] - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_child_bounds() {
        let cb = CubicBounds { origin: [0.0, 0.0, 0.0], size: 2.0 };
        let c0 = cb.child(0); // octant 0: all min
        assert!((c0.origin[0]).abs() < 1e-10);
        assert!((c0.size - 1.0).abs() < 1e-10);

        let c7 = cb.child(7); // octant 7: all max
        assert!((c7.origin[0] - 1.0).abs() < 1e-10);
        assert!((c7.origin[1] - 1.0).abs() < 1e-10);
        assert!((c7.origin[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_corner_positions() {
        let cb = CubicBounds { origin: [0.0, 0.0, 0.0], size: 1.0 };
        let c0 = cb.corner(0);
        assert_eq!(c0, [0.0, 0.0, 0.0]);
        let c7 = cb.corner(7);
        assert_eq!(c7, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_edge_corners_consistency() {
        // Each edge should connect two corners that differ in exactly one bit
        for i in 0..12u8 {
            let (a, b) = CubicBounds::edge_corners(i);
            let diff = a ^ b;
            assert!(diff.count_ones() == 1, "Edge {} corners {} and {} differ in {} bits", i, a, b, diff.count_ones());
        }
    }
}
