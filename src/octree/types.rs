/// Quantized 3D grid position for cache key deduplication.
/// At depth d, coordinates range from 0 to 2^d (inclusive).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridPos {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl GridPos {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }
}
