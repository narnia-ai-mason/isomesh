use crate::bbox::CubicBounds;

/// Connected component analysis result for a leaf cell's inside corners.
///
/// In Manifold Dual Contouring (Schaefer, Ju, Warren 2007), each leaf cell
/// may contain multiple disconnected groups of "inside" corners. Each group
/// (connected component) gets its own QEF vertex, preventing non-manifold
/// "bowtie" vertices that occur in basic DC.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellComponents {
    /// Component ID for each corner (0..num_components-1).
    /// Outside corners (bit not set in corner_mask) get `0xFF`.
    pub corner_component: [u8; 8],
    /// Number of distinct connected components of inside corners.
    pub num_components: u8,
}

/// Compute connected components of inside corners for a given corner_mask.
///
/// Two inside corners are in the same component if they share a cube edge
/// (differ in exactly 1 bit) and both are inside. Uses inline Union-Find
/// over the 8 corners.
pub fn compute_components(corner_mask: u8) -> CellComponents {
    if corner_mask == 0 || corner_mask == 0xFF {
        return CellComponents {
            corner_component: [0xFF; 8],
            num_components: 0,
        };
    }

    // Union-Find: parent[i] stores the parent of corner i.
    // Only meaningful for inside corners.
    let mut parent = [0u8; 8];
    for i in 0..8u8 {
        parent[i as usize] = i;
    }

    // Find with path compression
    fn find(parent: &mut [u8; 8], mut x: u8) -> u8 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }

    // Union two corners
    fn union(parent: &mut [u8; 8], a: u8, b: u8) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            // Always point higher-index root to lower-index root for determinism
            if ra < rb {
                parent[rb as usize] = ra;
            } else {
                parent[ra as usize] = rb;
            }
        }
    }

    // For each of the 12 cube edges, if both endpoints are inside, union them
    for edge in 0..12u8 {
        let (c0, c1) = CubicBounds::edge_corners(edge);
        let both_inside = (corner_mask >> c0) & 1 == 1 && (corner_mask >> c1) & 1 == 1;
        if both_inside {
            union(&mut parent, c0, c1);
        }
    }

    // Flatten: assign sequential component IDs
    let mut corner_component = [0xFFu8; 8];
    let mut num_components = 0u8;
    let mut root_to_id = [0xFFu8; 8]; // maps root corner → component ID

    for i in 0..8u8 {
        if (corner_mask >> i) & 1 == 1 {
            let root = find(&mut parent, i);
            if root_to_id[root as usize] == 0xFF {
                root_to_id[root as usize] = num_components;
                num_components += 1;
            }
            corner_component[i as usize] = root_to_id[root as usize];
        }
    }

    CellComponents {
        corner_component,
        num_components,
    }
}

/// For a sign-change edge, return the component ID of the inside corner.
///
/// Panics (debug) if the edge has no sign change or the inside corner
/// has no valid component.
#[inline]
pub fn edge_component(components: &CellComponents, corner_mask: u8, edge_idx: u8) -> u8 {
    let (c0, c1) = CubicBounds::edge_corners(edge_idx);
    let c0_inside = (corner_mask >> c0) & 1 == 1;
    let inside_corner = if c0_inside { c0 } else { c1 };
    let comp = components.corner_component[inside_corner as usize];
    debug_assert!(comp != 0xFF, "Inside corner {} should have a valid component", inside_corner);
    comp
}

/// Precomputed lookup table: `COMPONENT_TABLE[corner_mask]` gives the
/// connected component analysis for that corner configuration.
///
/// Since corner_mask is a u8 (256 values), this table is small (< 3 KB)
/// and fits entirely in L1 cache.
pub static COMPONENT_TABLE: [CellComponents; 256] = {
    let mut table = [CellComponents {
        corner_component: [0xFF; 8],
        num_components: 0,
    }; 256];

    // We must use const-compatible code here (no function calls to
    // non-const functions). Inline the Union-Find logic.
    let mut mask: usize = 0;
    while mask < 256 {
        if mask == 0 || mask == 0xFF {
            // Already initialized to 0 components
            mask += 1;
            continue;
        }

        let corner_mask = mask as u8;

        // Inline Union-Find
        let mut parent: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

        // Inline find (iterative, no recursion allowed in const)
        // We'll do a simplified approach: iterate edges, union, then
        // do multiple passes to flatten.

        // Edge corners table (must inline since we can't call functions in const)
        const EDGES: [(u8, u8); 12] = [
            (0, 1), (2, 3), (4, 5), (6, 7), // X-axis
            (0, 2), (1, 3), (4, 6), (5, 7), // Y-axis
            (0, 4), (1, 5), (2, 6), (3, 7), // Z-axis
        ];

        let mut e = 0;
        while e < 12 {
            let (c0, c1) = EDGES[e];
            let both_inside =
                (corner_mask >> c0) & 1 == 1 && (corner_mask >> c1) & 1 == 1;
            if both_inside {
                // Find root of c0
                let mut r0 = c0;
                while parent[r0 as usize] != r0 {
                    r0 = parent[r0 as usize];
                }
                // Find root of c1
                let mut r1 = c1;
                while parent[r1 as usize] != r1 {
                    r1 = parent[r1 as usize];
                }
                // Union: lower root wins
                if r0 != r1 {
                    if r0 < r1 {
                        parent[r1 as usize] = r0;
                    } else {
                        parent[r0 as usize] = r1;
                    }
                }
            }
            e += 1;
        }

        // Flatten: find root for each inside corner (multiple passes to handle chains)
        let mut pass = 0;
        while pass < 8 {
            let mut i = 0;
            while i < 8 {
                if (corner_mask >> i) & 1 == 1 {
                    let mut r = i as u8;
                    while parent[r as usize] != r {
                        r = parent[r as usize];
                    }
                    parent[i] = r;
                }
                i += 1;
            }
            pass += 1;
        }

        // Assign sequential component IDs
        let mut corner_component = [0xFFu8; 8];
        let mut num_components = 0u8;
        let mut root_to_id = [0xFFu8; 8];

        let mut i = 0;
        while i < 8 {
            if (corner_mask >> i) & 1 == 1 {
                let root = parent[i];
                if root_to_id[root as usize] == 0xFF {
                    root_to_id[root as usize] = num_components;
                    num_components += 1;
                }
                corner_component[i] = root_to_id[root as usize];
            }
            i += 1;
        }

        table[mask] = CellComponents {
            corner_component,
            num_components,
        };

        mask += 1;
    }

    table
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference BFS implementation for verifying the lookup table.
    fn reference_components(corner_mask: u8) -> (u8, [u8; 8]) {
        if corner_mask == 0 || corner_mask == 0xFF {
            return (0, [0xFF; 8]);
        }

        let edges: [(u8, u8); 12] = [
            (0, 1), (2, 3), (4, 5), (6, 7),
            (0, 2), (1, 3), (4, 6), (5, 7),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ];

        // Build adjacency list for inside corners
        let mut adj: [u8; 8] = [0; 8]; // bitmask of neighbors
        for &(c0, c1) in &edges {
            if (corner_mask >> c0) & 1 == 1 && (corner_mask >> c1) & 1 == 1 {
                adj[c0 as usize] |= 1 << c1;
                adj[c1 as usize] |= 1 << c0;
            }
        }

        // BFS to find components
        let mut visited = 0u8;
        let mut corner_comp = [0xFFu8; 8];
        let mut num_comp = 0u8;

        for start in 0..8u8 {
            if (corner_mask >> start) & 1 == 0 {
                continue; // outside
            }
            if (visited >> start) & 1 == 1 {
                continue; // already visited
            }

            // BFS from this corner
            let mut queue = [0u8; 8];
            let mut head = 0;
            let mut tail = 0;
            queue[tail] = start;
            tail += 1;
            visited |= 1 << start;
            corner_comp[start as usize] = num_comp;

            while head < tail {
                let cur = queue[head];
                head += 1;
                for nb in 0..8u8 {
                    if (adj[cur as usize] >> nb) & 1 == 1 && (visited >> nb) & 1 == 0 {
                        visited |= 1 << nb;
                        corner_comp[nb as usize] = num_comp;
                        queue[tail] = nb;
                        tail += 1;
                    }
                }
            }

            num_comp += 1;
        }

        (num_comp, corner_comp)
    }

    #[test]
    fn test_all_256_corner_masks() {
        for mask in 0..=255u8 {
            let computed = compute_components(mask);
            let table = COMPONENT_TABLE[mask as usize];
            let (ref_count, ref_comp) = reference_components(mask);

            // Verify compute_components matches reference
            assert_eq!(
                computed.num_components, ref_count,
                "mask 0x{:02X}: compute_components gave {} components, reference gave {}",
                mask, computed.num_components, ref_count
            );
            assert_eq!(
                computed.corner_component, ref_comp,
                "mask 0x{:02X}: component assignment mismatch",
                mask
            );

            // Verify COMPONENT_TABLE matches reference
            assert_eq!(
                table.num_components, ref_count,
                "mask 0x{:02X}: COMPONENT_TABLE gave {} components, reference gave {}",
                mask, table.num_components, ref_count
            );
            assert_eq!(
                table.corner_component, ref_comp,
                "mask 0x{:02X}: COMPONENT_TABLE assignment mismatch",
                mask
            );

            // Verify table matches compute_components
            assert_eq!(
                computed, table,
                "mask 0x{:02X}: compute_components and COMPONENT_TABLE disagree",
                mask
            );
        }
    }

    #[test]
    fn test_empty_and_full() {
        let empty = compute_components(0x00);
        assert_eq!(empty.num_components, 0);
        assert_eq!(empty.corner_component, [0xFF; 8]);

        let full = compute_components(0xFF);
        assert_eq!(full.num_components, 0);
        assert_eq!(full.corner_component, [0xFF; 8]);
    }

    #[test]
    fn test_single_corner() {
        for i in 0..8u8 {
            let mask = 1 << i;
            let c = compute_components(mask);
            assert_eq!(c.num_components, 1, "single corner {}: expected 1 component", i);
            assert_eq!(c.corner_component[i as usize], 0, "corner {} should be component 0", i);
            for j in 0..8u8 {
                if j != i {
                    assert_eq!(c.corner_component[j as usize], 0xFF);
                }
            }
        }
    }

    #[test]
    fn test_two_adjacent_corners() {
        // Corners 0 and 1 share edge 0 (X-axis) → 1 component
        let c = compute_components(0b00000011);
        assert_eq!(c.num_components, 1);
        assert_eq!(c.corner_component[0], 0);
        assert_eq!(c.corner_component[1], 0);
    }

    #[test]
    fn test_two_body_diagonal_corners() {
        // Corners 0 (000) and 7 (111) differ in 3 bits → no shared edge → 2 components
        let c = compute_components(0b10000001);
        assert_eq!(c.num_components, 2);
        assert_ne!(c.corner_component[0], c.corner_component[7]);
    }

    #[test]
    fn test_two_face_diagonal_corners() {
        // Corners 0 (000) and 3 (011) differ in 2 bits (Y,X) → no shared edge → 2 components
        let c = compute_components(0b00001001);
        assert_eq!(c.num_components, 2);
        assert_ne!(c.corner_component[0], c.corner_component[3]);
    }

    #[test]
    fn test_checkerboard_4_components() {
        // 0x69 = 0b01101001: corners {0, 3, 5, 6}
        // 0→3: differ in bits 0,1 (2 bits) → no edge
        // 0→5: differ in bits 0,2 (2 bits) → no edge
        // 0→6: differ in bits 1,2 (2 bits) → no edge
        // 3→5: differ in bits 0,1,2 (3 bits) → no edge
        // 3→6: differ in bits 0,2 (2 bits) → no edge
        // 5→6: differ in bits 0,1 (2 bits) → no edge
        // No pair shares an edge → 4 components
        let c = compute_components(0x69);
        assert_eq!(c.num_components, 4, "checkerboard 0x69 should have 4 components");
    }

    #[test]
    fn test_inverse_checkerboard_4_components() {
        // 0x96 = 0b10010110: corners {1, 2, 4, 7} — complement of 0x69
        let c = compute_components(0x96);
        assert_eq!(c.num_components, 4, "inverse checkerboard 0x96 should have 4 components");
    }

    #[test]
    fn test_face_connected() {
        // Corners 0,1,2,3 = bottom face (Z=0). All share edges. → 1 component
        let c = compute_components(0b00001111);
        assert_eq!(c.num_components, 1);
        for i in 0..4 {
            assert_eq!(c.corner_component[i], 0);
        }
    }

    #[test]
    fn test_edge_component_function() {
        // Mask 0b00000001 (only corner 0 inside)
        // Edge 0: corners (0, 1), corner 0 is inside
        let c = compute_components(0b00000001);
        assert_eq!(edge_component(&c, 0b00000001, 0), 0);

        // Edge 4: corners (0, 2), corner 0 is inside
        assert_eq!(edge_component(&c, 0b00000001, 4), 0);

        // Edge 8: corners (0, 4), corner 0 is inside
        assert_eq!(edge_component(&c, 0b00000001, 8), 0);
    }

    #[test]
    fn test_edge_component_multi() {
        // Mask 0x69: corners {0, 3, 5, 6}, each in its own component
        let c = compute_components(0x69);

        // Edge 0: corners (0, 1). Corner 0 is inside (comp 0), corner 1 outside
        let comp_e0 = edge_component(&c, 0x69, 0);
        assert_eq!(comp_e0, c.corner_component[0]);

        // Edge 5: corners (1, 3). Corner 1 outside, corner 3 inside
        let comp_e5 = edge_component(&c, 0x69, 5);
        assert_eq!(comp_e5, c.corner_component[3]);
    }

    #[test]
    fn test_every_sign_change_edge_has_valid_component() {
        for mask in 1..=254u8 {
            let c = COMPONENT_TABLE[mask as usize];
            if c.num_components == 0 {
                continue;
            }
            for edge in 0..12u8 {
                let (c0, c1) = CubicBounds::edge_corners(edge);
                let s0 = (mask >> c0) & 1;
                let s1 = (mask >> c1) & 1;
                if s0 != s1 {
                    // Sign change edge
                    let comp = edge_component(&c, mask, edge);
                    assert!(
                        comp < c.num_components,
                        "mask 0x{:02X} edge {}: component {} >= num_components {}",
                        mask, edge, comp, c.num_components
                    );
                }
            }
        }
    }

    #[test]
    fn test_outside_corners_always_0xff() {
        for mask in 0..=255u8 {
            let c = COMPONENT_TABLE[mask as usize];
            for i in 0..8u8 {
                if (mask >> i) & 1 == 0 {
                    assert_eq!(
                        c.corner_component[i as usize], 0xFF,
                        "mask 0x{:02X} corner {}: outside corner should be 0xFF",
                        mask, i
                    );
                }
            }
        }
    }

    #[test]
    fn test_inside_corners_have_valid_component() {
        for mask in 1..=254u8 {
            let c = COMPONENT_TABLE[mask as usize];
            for i in 0..8u8 {
                if (mask >> i) & 1 == 1 {
                    assert!(
                        c.corner_component[i as usize] < c.num_components,
                        "mask 0x{:02X} corner {}: component {} >= num_components {}",
                        mask, i, c.corner_component[i as usize], c.num_components
                    );
                }
            }
        }
    }

    #[test]
    fn test_component_count_distribution() {
        // Count how many masks produce each component count
        let mut counts = [0u32; 9]; // max 8 corners → max 8 components (theoretical)
        for mask in 0..=255u8 {
            let c = COMPONENT_TABLE[mask as usize];
            counts[c.num_components as usize] += 1;
        }
        // 0 components: mask=0x00 and mask=0xFF → 2
        assert_eq!(counts[0], 2);
        // 1 component should be the majority
        assert!(counts[1] > 100, "Expected most masks to have 1 component, got {}", counts[1]);
        // There should be some multi-component masks
        let multi: u32 = counts[2..].iter().sum();
        assert!(multi > 0, "Expected some multi-component masks");
        // Max should be 4 (checkerboard)
        assert_eq!(counts[5], 0);
        assert_eq!(counts[6], 0);
        assert_eq!(counts[7], 0);
        assert_eq!(counts[8], 0);
    }
}
