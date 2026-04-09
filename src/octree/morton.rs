/// Morton (Z-order) code utilities for 3D spatial indexing.
///
/// A Morton code interleaves the bits of (x, y, z) grid coordinates into a single u64.
/// This enables O(1) neighbor finding and efficient spatial sorting.
///
/// We use 21 bits per axis, supporting up to depth 21 (2M cells per axis).

/// Spread 21 bits of x into every 3rd bit position.
fn spread_bits(mut x: u64) -> u64 {
    x &= 0x1F_FFFF; // mask to 21 bits
    x = (x | (x << 32)) & 0x001F_0000_0000_FFFF;
    x = (x | (x << 16)) & 0x001F_0000_FF00_00FF;
    x = (x | (x << 8)) & 0x100F_00F0_0F00_F00F;
    x = (x | (x << 4)) & 0x10C3_0C30_C30C_30C3;
    x = (x | (x << 2)) & 0x1249_2492_4924_9249;
    x
}

/// Compact every 3rd bit back into contiguous 21 bits.
fn compact_bits(mut x: u64) -> u64 {
    x &= 0x1249_2492_4924_9249;
    x = (x | (x >> 2)) & 0x10C3_0C30_C30C_30C3;
    x = (x | (x >> 4)) & 0x100F_00F0_0F00_F00F;
    x = (x | (x >> 8)) & 0x001F_0000_FF00_00FF;
    x = (x | (x >> 16)) & 0x001F_0000_0000_FFFF;
    x = (x | (x >> 32)) & 0x1F_FFFF;
    x
}

/// Encode (x, y, z) grid coordinates into a Morton code.
pub fn encode(x: u32, y: u32, z: u32) -> u64 {
    spread_bits(x as u64) | (spread_bits(y as u64) << 1) | (spread_bits(z as u64) << 2)
}

/// Decode a Morton code back into (x, y, z) grid coordinates.
pub fn decode(code: u64) -> (u32, u32, u32) {
    (
        compact_bits(code) as u32,
        compact_bits(code >> 1) as u32,
        compact_bits(code >> 2) as u32,
    )
}

/// Compute the Morton code for a child octant (0-7) of a parent at given depth.
/// The child's code is formed by appending the octant bits to the parent's code.
pub fn child_code(parent: u64, octant: u8, parent_depth: u8) -> u64 {
    let shift = (parent_depth as u32) * 3;
    parent | ((octant as u64) << shift)
}

/// Extract the octant index of a cell at `depth` from its Morton code.
pub fn octant_at_depth(code: u64, depth: u8) -> u8 {
    let shift = ((depth - 1) as u32) * 3;
    ((code >> shift) & 7) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        for x in 0..16u32 {
            for y in 0..16u32 {
                for z in 0..16u32 {
                    let code = encode(x, y, z);
                    let (dx, dy, dz) = decode(code);
                    assert_eq!((dx, dy, dz), (x, y, z));
                }
            }
        }
    }

    #[test]
    fn test_origin() {
        assert_eq!(encode(0, 0, 0), 0);
    }

    #[test]
    fn test_ordering() {
        // Morton code preserves spatial locality
        let c1 = encode(0, 0, 0);
        let c2 = encode(1, 0, 0);
        assert!(c2 > c1);
    }

    #[test]
    fn test_child_code() {
        let parent = encode(0, 0, 0);
        // Root is at depth 0; its children use parent_depth=0
        let c = child_code(parent, 7, 0);
        assert_eq!(c, 7); // octant 7 = 0b111 at shift 0

        // A child at depth 1 has its sub-children at shift 3
        let c2 = child_code(c, 3, 1); // parent_depth=1, octant 3 = 0b011
        assert_eq!(c2, 7 | (3 << 3)); // = 7 + 24 = 31
    }

    #[test]
    fn test_large_coordinates() {
        let x = (1 << 21) - 1; // max 21-bit value
        let y = (1 << 21) - 1;
        let z = (1 << 21) - 1;
        let code = encode(x, y, z);
        let (dx, dy, dz) = decode(code);
        assert_eq!((dx, dy, dz), (x, y, z));
    }
}
