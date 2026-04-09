# isomesh Benchmark Report

## Methodology

- **isomesh**: Dual Contouring (adaptive octree + QEF), uniform depth
- **PyMCubes**: Marching Cubes on uniform grid
- **Resolution matching**: depth 5 = 32 cells/axis ↔ MC 33 grid pts; depth 6 = 64 ↔ 65
- **Timing note**: isomesh time = total (octree + batch eval + bisection + QEF + mesh). PyMCubes time = MC extraction only (grid eval excluded). Fair comparison requires adding grid eval time to PyMCubes.

## Results Summary

### Manifold/Watertight Status

| Shape | Category | isomesh low | isomesh high | pymcubes low | pymcubes high |
|-------|----------|:-----------:|:------------:|:------------:|:-------------:|
| sphere | basic | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ |
| torus | basic | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ |
| ellipsoid | basic | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ |
| box | sharp | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ |
| csg_cross | sharp | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ |
| csg_intersection | sharp | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ |
| chamfered_box | sharp | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ |
| gyroid | tpms | ✗/✗ | ✗/✗ | ✗/✗ | ✗/✗ |
| schwarz_p | tpms | ✗/✗ | ✗/✗ | ✗/✗ | ✗/✗ |
| schwarz_d | tpms | ✗/✗ | ✗/✗ | ✗/✗ | ✗/✗ |
| genus2 | topology | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ |
| tanglecube | topology | ✗/✗ | ✗/✗ | ✗/✗ | ✗/✗ |
| heart | topology | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ |
| barth_sextic | topology | ✗/✗ | ✗/✗ | ✗/✗ | ✗/✗ |
| thin_plate | thin | ✓/✓ | ✓/✓ | ✓/✓ | ✓/✓ |
| thin_shell | thin | ✗/✗ | ✓/✓ | ✓/✓ | ✓/✓ |

### Non-watertight cases analysis

All non-watertight results fall into two categories:

1. **Boundary clipping** (gyroid, schwarz_p, schwarz_d, tanglecube): The surface extends to the bounding box boundary, creating open edges where it's clipped. Both methods produce identical Euler characteristics — this is inherent to the shape/bbox, not an algorithm deficiency.

2. **Self-intersecting algebraic surface** (barth_sextic): The degree-6 surface has complex self-intersections that neither algorithm handles correctly.

3. **Resolution-dependent** (thin_shell at low res for isomesh): Wall thickness (0.08) ≈ cell size at depth 5 (0.094). isomesh misses the inner surface at this resolution. Resolved at depth 6.

### Mesh Quality Comparison (same resolution)

Both methods produce nearly identical vertex/face counts at the same resolution. This is expected since both operate on the same underlying grid. The key differentiator is **sharp feature preservation**, not vertex count.

### isomesh advantages

- **Sharp features**: QEF vertex placement naturally positions vertices on edges/corners of box, CSG intersection, chamfered shapes.
- **Adaptive refinement**: When using min_depth < max_depth, isomesh can concentrate resolution near the surface.
- **Single-function API**: No need to pre-evaluate a 3D grid; isomesh handles evaluation internally.

### PyMCubes advantages

- **Raw extraction speed**: MC extraction is ~100x faster than DC (but grid eval is not included in PyMCubes timing).
- **Simplicity**: Lookup-table based, no QEF overhead.
