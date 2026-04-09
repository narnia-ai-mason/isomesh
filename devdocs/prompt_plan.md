# Task

You are an expert systems architect specializing in computational geometry,
isosurface extraction, and high-performance Python/Rust interop.

Design a detailed implementation plan for a Python package called `isomesh`.

---

## Overview

`isomesh` is a Python package that extracts an isosurface mesh (default: zero-level-set)
from an arbitrary user-supplied implicit function.

The package must produce **manifold, watertight, consistently oriented** triangle meshes
that faithfully preserve both smooth organic shapes and sharp mechanical features
(edges, corners), while being robust, fast, and memory-efficient.

**Performance target**: faster, more robust, and higher quality than PyMCubes.
Typical execution should complete in seconds to tens of seconds.

---

## Core Requirements

### Implicit Function Interface

The user provides a Python callable that is **already vectorized** (numpy-based):

```python
def f(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Args:
        positions: (N, 3) float64 array of query points
    Returns:
        values: (N,) float64 array of implicit function values
        gradients: (N, 3) float64 array of gradients, or None
    """
```

- The function may represent an SDF, but any implicit function must be supported.
- The function is already vectorized — it expects `(N, 3)` input and returns
  `(N,)` values and optionally `(N, 3)` gradients in one call.
- When the Rust core needs function values (e.g., evaluating octree nodes,
  finding edge crossings), it must collect **all** query points for that stage
  and call the Python function **once** with the full batch.
  Do NOT call the function point-by-point from Rust — this would be catastrophically slow.
- If gradients are not provided (i.e., the function returns `None` for gradients),
  estimate them via **centered finite differences** (with a configurable step size).
- Optional: support `torch.Tensor` inputs/outputs natively.
  At minimum, provide transparent numpy <-> Tensor conversion;
  ideally detect and preserve Tensor throughout the pipeline
  (to enable autodiff gradient computation on GPU).

### Language & Binding

- Core algorithms (octree, contouring, mesh extraction) in **Rust**.
- Python bindings via **PyO3 + maturin**.
- **Zero-copy** numpy array exchange via `numpy` crate / PyO3 buffer protocol.
- Multi-threaded Rust execution via **Rayon**
  (release GIL during Rust-side computation).

---

## Algorithm Requirements

### Bounding Box

- User specifies `bbox_min` and `bbox_max` as 3D coordinates.
  If not provided, default to `[-1, -1, -1]` to `[1, 1, 1]`.
- **Non-cubic (rectangular) bounding boxes are supported**, but octree cells
  must always be **cubic** (equal side lengths).
- If the user's bounding box dimensions are not evenly divisible into
  equal-sized cubes at the root level, **expand (ceil) the bounding box**
  to the smallest enclosing box that is divisible into uniform cubes.
  Document this behavior clearly.

### Adaptive Octree

- Build an adaptive octree over the (possibly expanded) bounding box.
- User controls `min_depth` and `max_depth` (i.e., resolution range).
- **Subdivision criteria**: choose the best combination of criteria
  from the literature. The plan must justify the choices.
  At minimum, the criteria must ensure:
  - Thin features are **never** missed.
  - Sharp edges/corners trigger sufficient refinement.
  - Smooth regions are not over-refined.

### Isosurface Extraction — Dual Contouring Family

- Use a **Dual Contouring** variant (NOT Marching Cubes)
  to preserve sharp edges and corners.
- The chosen algorithm must **guarantee manifold, watertight output**.
  If standard Dual Contouring does not guarantee this,
  the plan must specify which variant does (e.g., Manifold Dual Contouring,
  Dual Marching Cubes) and why.
- Vertex placement via **QEF (Quadric Error Function)** minimization
  using Hermite data (intersection points + normals from gradients).
- Support arbitrary `iso_value` (default `0.0`).

### Sharp Feature Preservation

- Detect sharp edges/corners from gradient discontinuities
  (angle threshold between normals at edge crossings).
- QEF vertex placement must respect feature edges —
  do NOT clamp vertices to cell centers.

### Robustness (HIGHEST PRIORITY)

- **Thin features**: The algorithm must guarantee that thin structures
  (thickness approaching cell size) are never skipped.
  Specify the concrete technique used (e.g., interval arithmetic,
  redundant sign sampling, multi-scale sign checks).
- **Degenerate cases**: Handle exact-zero values at grid nodes,
  co-planar intersections, and topological ambiguities gracefully.
- **Multiple connected components**: Naturally supported.
- **Sign determination (inside/outside) must be extremely robust.**
  This is the #1 correctness priority.

---

## Output

### Mesh Data

- Return `vertices` as `(V, 3) float64` numpy array
  and `faces` as `(F, 3) int64` numpy array.
- **Consistent orientation** (outward-pointing normals, consistent face winding).
- Guarantee: **manifold** and **watertight**.

---

## API Design

```python
import isomesh
import numpy as np

def sdf_sphere(pos: np.ndarray):
    values = np.linalg.norm(pos, axis=1) - 1.0
    gradients = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    return values, gradients

vertices, faces = isomesh.extract(
    func=sdf_sphere,
    bbox_min=(-2, -2, -2),
    bbox_max=(2, 2, 2),
    min_depth=3,
    max_depth=7,
    angle_threshold=30.0,   # degrees — sharp feature detection
    iso_value=0.0,          # default 0.0, support arbitrary values
)

# vertices: np.ndarray (V, 3) float64
# faces: np.ndarray (F, 3) int64
```

---

## Performance KPIs (Priority Order)

1. **Robust** — Never miss thin features; always produce manifold/watertight mesh;
   extremely robust sign determination.
2. **Fast** — Minimize function evaluations via adaptive refinement;
   parallelize Rust-side work; beat PyMCubes in speed.
3. **Memory-efficient** — Zero-copy arrays; compact octree representation.

---

## Build & Distribution

- `pyproject.toml` with maturin backend.
- CI: build wheels for Linux (manylinux), macOS (x86_64 + arm64), Windows.
- Minimum Python 3.11, numpy >= 1.24.
- Optional dependency: `torch` (detected at runtime).

---

## Testing & Validation

Design a comprehensive test suite. The plan must define:

- **Analytic SDFs** for accuracy validation:
  sphere, torus, etc. — compare extracted mesh to ground truth
  (e.g., Hausdorff distance, volume error).
- **Sharp feature test shapes**: choose appropriate mechanical part geometries
  (e.g., cube, CSG box intersection, chamfered/filleted parts)
  that test edge and corner preservation. Justify the choices.
- **Thin feature test shapes**: choose geometries with thin walls or narrow gaps
  (e.g., thin plate, thin shell, two close parallel surfaces)
  that test the robustness of sign detection. Justify the choices.
- **Manifold/watertight validation**: programmatic verification
  (half-edge checks, Euler characteristic, etc.).
- **Performance benchmarks**: timing and peak memory at various depths.

---

## Deliverables

Produce a phased implementation plan with:

1. Project scaffolding (Rust + Python + maturin)
2. Octree construction & adaptive refinement
3. Dual contouring core (QEF, feature detection)
4. Mesh extraction & manifold guarantees
5. Python API & batch evaluation bridge
6. Torch support
7. Testing, benchmarking, CI/CD

For each phase, specify:
- Key files/modules to create
- Data structures and algorithms (with references to papers where applicable)
- Risks and mitigation strategies
- Estimated complexity (S/M/L)

---

## Reference Literature

The following papers and resources have been surveyed and are relevant.
Use them to inform and justify your design choices.

### Dual Contouring & Manifold Guarantees

- Ju, Losasso, Schaefer, Warren. "Dual Contouring of Hermite Data." SIGGRAPH 2002.
  — The foundational DC paper. One vertex per cell, QEF placement. Non-manifold.
- **Schaefer, Ju, Warren. "Manifold Dual Contouring." IEEE TVCG 13(3), 2007.**
  — Extends DC with multiple vertices per cell (one per connected component)
  and a manifold criterion check. Provably manifold/watertight. **Primary candidate.**
- Schaefer, Warren. "Dual Contouring: The Secret Sauce." Tech report.
  — Practical implementation details for QEF, sharp features.
- Nielson. "Dual Marching Cubes." IEEE VIS 2004.
  — Lookup-table based, no QEF. Simpler but no sharp feature support.
- Schaefer, Warren. "Dual Marching Cubes: Primal Contouring of Dual Grids." CGF 2005.
  — QEF on dual grid + MC topology. An alternative approach.

### QEF Minimization

- Garland, Heckbert. "Surface Simplification Using Quadric Error Metrics." SIGGRAPH 1997.
- Lindstrom. "Out-of-Core Simplification of Large Polygonal Models." SIGGRAPH 2000.
  — Robust SVD-based QEF with mass-point bias for rank-deficient cases.
- **Trettner, Kobbelt. "Fast and Robust QEF using Probabilistic Quadrics." CGF 2020.**
  — Always full rank, no SVD needed, 50x faster. Modern alternative.
- Matt Keeter. "QEF Explainer." — Best tutorial on QEF math and implementation.
- Matt Keeter. "A Simple Adversarial Model for Dual Contouring." 2023.
  — Documents vertex-outside-cell failure modes.

### Robust Octree & Thin Feature Detection

- **Duff. "Interval Arithmetic Recursive Subdivision for Implicit Functions." SIGGRAPH 1992.**
  — Foundational IA for guaranteed surface detection. Gold standard.
- **Plantinga, Vegter. "Isotopic Meshing of Implicit Surfaces." Visual Computer 2007.**
  — IA-based certified meshing with isotopy guarantee (strongest correctness).
- **Kalra, Barr. "Guaranteed Ray Intersections with Implicit Surfaces." SIGGRAPH 1989.**
  — Lipschitz-based bounds for guaranteed detection. For SDFs, L=1.
- Galin et al. "Segment Tracing Using Local Lipschitz Bounds." CGF/Eurographics 2020.
  — Local (not global) Lipschitz bounds, much tighter.
- Ban et al. "Generalized Lipschitz Tracing of Implicit Surfaces." CGF 2025.
  — Lipschitz field for black-box functions.
- Paiva et al. "Robust Adaptive Meshes for Implicit Surfaces." SIBGRAPI 2006.
  — Combines interval/affine arithmetic with topology criteria.
- Kobbelt et al. "Feature Sensitive Surface Extraction." SIGGRAPH 2001.
  — Curvature-driven refinement for mesh quality.

### Octree Balancing

- Sundar et al. "Bottom-Up Construction and 2:1 Balance Refinement." SIAM 2008.
- Kazhdan et al. "Unconstrained Isosurface Extraction on Arbitrary Octrees." SGP 2007.
  — Alternative: watertight extraction without 2:1 balance (more complex).

### Key Constraint: Black-Box Python Callable

The implicit function is an opaque Python callable — no expression tree or DSL.
This means:
- Full interval arithmetic is NOT feasible for the general case
  (would require operator overloading / DSL).
- **Recommended robustness strategy for black-box functions (Tier 2)**:
  1. Lipschitz-based refinement guard (estimate L via finite differences at corners).
  2. Edge midpoint supersampling (12 extra samples per cell, shared with neighbors).
  3. Gradient magnitude suspicion (force subdivision if gradient varies >4x in cell).
  4. User-configurable `max_depth` as safety net.
- Optionally offer a Tier 4 "certified" mode for users who provide an
  interval-evaluable function via a provided DSL/helper.

### Reference Implementations

Study these for architecture and algorithm details:
- **mkeeter/fidget** (Rust) — MDC + IA + JIT, by libfive author. Closest reference.
- **libfive** (C++) — Production MDC, Oracle interface for black-box functions.
- **nickgildea/qef** (C/C++) — Standalone QEF solver, Jacobi SVD, SIMD. Port target.
- **Philip-Trettner/probabilistic-quadrics** (C++17) — Fast QEF alternative.
- **PyO3/rust-numpy** — Zero-copy numpy <-> Rust ndarray.
