# isomesh Reference Survey

## 1. Isosurface Extraction Algorithm 선정

### 추천: Manifold Dual Contouring (MDC)

**논문**: Schaefer, Ju, Warren. "Manifold Dual Contouring." IEEE TVCG 13(3), 2007.
[PDF](https://people.engr.tamu.edu/schaefer/research/dualsimp_tvcg.pdf)

Original DC (Ju et al., 2002)는 cell당 vertex 1개 제약으로 non-manifold 출력 가능.
MDC는 cell당 **복수 vertex** (connected component별 1개)를 허용하고,
vertex clustering 시 **manifold criterion**을 검사하여 manifold/watertight를 보장한다.

- Manifold/watertight: **보장**
- Sharp feature (QEF): **지원**
- Adaptive octree: **설계 목적**
- 구현 난이도: 중상 (DC 대비 connected component 추적, manifold check 추가)

### 비교 대상

| Method | Manifold | Sharp Features | Adaptive Octree | 비고 |
|--------|----------|---------------|-----------------|------|
| **Manifold DC (2007)** | **Yes** | **Yes (QEF)** | **Yes** | **1순위 추천** |
| Original DC (2002) | No | Yes (QEF) | Yes | 기반 알고리즘 |
| Dual Marching Cubes - Nielson (2004) | Mostly | No | No (uniform) | QEF 없음, lookup-table 기반 |
| Dual MC - Schaefer & Warren (2005) | ~Yes | Yes (QEF) | Yes | Dual grid + MC |
| Tetrahedral DC (Rashid, 2016) | Yes | Limited | No | geometry 과다 생성 |
| FlexiCubes (Shen, 2023) | Weak | Yes | Yes | differentiable rendering 전용 |
| Neural DC (Chen, 2022) | No | Yes | - | 학습 기반, 범용 부적합 |
| Occupancy-Based DC (Hwang, 2024) | Yes | Yes | - | MDC 기반, occupancy 특화 |

### 최신 주목 논문

- **Carrera et al. "Dual Contouring of Signed Distance Data"** (arXiv 2604.00157, 2026.03)
  - Gradient/normal 없이 discrete SDF 데이터만으로 sharp feature 복원
  - 아직 너무 최신이라 구현 참조 어려움

### 구현 전략

1. Basic DC + adaptive octree + QEF 구현 (기반)
2. MDC 확장: cell당 복수 vertex, connected component 추적, manifold criterion
3. 선택적으로 DMC mode 제공 (lookup-table 기반, sharp feature 불필요 시)

---

## 2. Adaptive Octree Refinement

### 2.1 Sign-Change (기본)

8개 corner에서 sign 변화 감지. 구현 trivial, 반드시 포함.
**한계**: thin feature를 놓침 (모든 corner가 같은 sign일 때).

### 2.2 Thin Feature Detection (최우선)

**핵심 문제**: 셀 크기보다 얇은 구조가 8개 corner 사이를 빠져나감.

#### 접근법 비교

| 기법 | 보장 | 비용 | black-box 함수 지원 | 추천 |
|------|------|------|---------------------|------|
| **Interval Arithmetic** | 수학적 보장 | 2-10x | 제한적 (DSL 필요) | Tier 4 (optional) |
| **Affine Arithmetic** | 수학적 보장 | IA보다 높음 | 제한적 | 고급 옵션 |
| **Lipschitz bound** | L 정확 시 보장 | 낮음 | **Yes** (FD로 추정) | **Tier 1-2** |
| **Edge midpoint sampling** | 확률적 | ~3x | **Yes** | **Tier 2** |
| **27-point stencil** | 확률적 | ~5x | **Yes** | Tier 3 |
| **Neighbor consistency** | 휴리스틱 | 낮음 | **Yes** | Tier 3 보조 |

#### 추천 전략 (Tier 2 — 기본값)

1. **Lipschitz-based refinement guard**: corner에서 finite difference로 local L 추정.
   `min(|f(corners)|) < L_est * cell_diagonal * safety_factor` 이면 강제 subdivision.
2. **Edge midpoint supersampling**: 12개 edge midpoint에서 sign change 검사.
3. **Gradient magnitude suspicion**: gradient 크기가 cell 내에서 4x 이상 변하면 강제 subdivision.
4. `max_depth` parameter로 사용자 안전 제어.

**핵심 참고 논문**:
- Duff, "Interval Arithmetic Recursive Subdivision" (SIGGRAPH 1992)
- Plantinga & Vegter, "Isotopic Meshing of Implicit Surfaces" (Visual Computer 23, 2007)
- Kalra & Barr, "Guaranteed Ray Intersections with Implicit Surfaces" (SIGGRAPH 1989)
- Galin et al., "Segment Tracing Using Local Lipschitz Bounds" (Eurographics 2020)
- Ban et al., "Generalized Lipschitz Tracing of Implicit Surfaces" (CGF 2025)

### 2.3 Curvature-Based Refinement (품질)

Hessian 기반 곡률 추정 → 곡률 높은 영역 세분화.
Surface가 이미 감지된 cell에서 mesh 품질을 높이는 용도.
Thin feature 감지에는 기여하지 않음.

**참고**: Kobbelt et al., "Feature Sensitive Surface Extraction" (SIGGRAPH 2001)

### 2.4 Feature-Sensitive Refinement (sharp edge)

QEF residual 또는 normal angle spread가 threshold 초과 시 subdivision.
DC를 구현하면 QEF가 이미 계산되므로 추가 비용 negligible.

### 2.5 2:1 Balancing

인접 cell의 depth 차이를 최대 1로 제한.
Dual contouring에서 crack-free mesh를 위해 필요.
Ripple propagation (BFS queue)으로 구현. 셀 수 증가 ~10-30%.

**참고**:
- Sundar et al., "Bottom-Up Construction and 2:1 Balance Refinement" (SIAM 2008)
- Kazhdan et al., "Unconstrained Isosurface Extraction on Arbitrary Octrees" (SGP 2007) — 2:1 없이도 가능하지만 훨씬 복잡

---

## 3. QEF Minimization & Sharp Feature Preservation

### 3.1 QEF 공식

Edge-surface intersection의 Hermite data (p_i, n_i)에서:

```
E(x) = Σ_i (n_i · (x - p_i))^2 = (Ax - b)^T(Ax - b)
```

Compact 표현: `A^T A` (3x3, 6값), `A^T b` (3값), `b^T b` (1값), mass point (4값) = **14 floats/cell**.
Additively composable → octree simplification에서 parent = Σ children.

### 3.2 SVD 기반 풀이

A^T A의 eigendecomposition → pseudoinverse. Rank-deficient case 처리:
- sigma_i > threshold → 1/sigma_i
- sigma_i ≤ threshold → 0 (해당 방향은 mass point로 대체)

Threshold: `0.1` (unit normal 가정) 또는 `0.001 * sigma_max` (상대적).

3x3 symmetric matrix → **Jacobi rotation** 5 sweeps로 충분 (full SVD library 불필요).

### 3.3 Vertex containment 전략

QEF minimizer가 cell 밖에 나가는 경우:

| 전략 | 특징 |
|------|------|
| Mass point fallback | 간단, sharp feature 손실 |
| **Mass-point biased QEF** | 미결정 방향만 mass point로, **추천** |
| Constrained QEF | 위반 좌표 고정 후 축소 시스템 재풀이 |
| Cell splitting (MDC) | 복수 vertex 허용, manifold 보장 |
| Soft bias term | λ‖x - center‖² 추가, λ로 tradeoff 제어 |

**추천**: Mass-point biased QEF (Keeter 방식) + MDC의 cell splitting 조합.

### 3.4 현대적 대안: Probabilistic Quadrics

**Trettner & Kobbelt, "Fast and Robust QEF Minimization using Probabilistic Quadrics"** (CGF 2020)
- Gaussian noise model로 QEF를 확률적으로 재정의
- Always full rank → SVD 불필요, Cholesky로 풀이
- **50x faster** than SVD, noise에 더 강건
- Header-only C++17: [github.com/Philip-Trettner/probabilistic-quadrics](https://github.com/Philip-Trettner/probabilistic-quadrics)

---

## 4. 기존 구현체 조사

### 4.1 Python Packages

| Package | Algorithm | Adaptive | Sharp | 상태 |
|---------|-----------|----------|-------|------|
| **PyMCubes** | Marching Cubes | No (uniform) | No | 비활성, Cython/C++ |
| **scikit-image** | MC (Lewiner) | No (uniform) | No | 활성, watertight 미보장 |
| **isoext** | MC + DC (GPU) | Sparse grid | Yes | CUDA+PyTorch 필수 |
| **VTK** | MC variant | No | No | 거대 의존성 |

**사용자 불만 공통점**: 느림, sharp feature 없음, adaptive 없음, non-manifold, 거대 의존성.
→ **isomesh의 명확한 시장 gap**: CPU 기반, pip install 가능, adaptive + DC + sharp feature.

### 4.2 Rust Crates

| Crate | Algorithm | 성숙도 | 비고 |
|-------|-----------|--------|------|
| **fidget** (~42K downloads) | **Manifold DC** + IA + JIT | 높음 | libfive 저자(Keeter), **최우선 참조** |
| isosurface (~22K) | MC + DC 등 | Alpha | Zero-dependency, 교육적 |
| fast-surface-nets (~22K) | Surface Nets | 높음 | ~20M tri/sec, sharp feature 없음 |
| dual_contouring | DC | 초기 | Archived |
| splashsurf | MC (SPH용) | 높음 | Domain decomposition 참조 |

**핵심**: `fidget`는 libfive의 Rust 후속작으로, MDC + IA + tape simplification을 구현.
isomesh의 아키텍처 참조로 가장 적합.

### 4.3 C++ Implementations

| Project | 특징 |
|---------|------|
| **libfive** | Production-grade MDC, IA, tape eval, Oracle system |
| **OpenVDB** | Sparse VDB grid + DC variant, VFX 산업 표준 |
| **nickgildea/qef** | Standalone QEF solver, Jacobi SVD, SIMD, port 대상 |
| **probabilistic-quadrics** | Header-only C++17, 50x faster QEF |
| **dualmc** | Single-header DMC, Rust port 용이 |

---

## 5. PyO3 + NumPy Zero-Copy

### 핵심 타입

- `PyReadonlyArray<T, D>` → `.as_array()` → `ndarray::ArrayView` (zero-copy read)
- `PyReadwriteArray<T, D>` → `.as_array_mut()` → `ndarray::ArrayViewMut`
- `IntoPyArray` trait: `Vec<T>` / `ndarray::Array` → `PyArray` (소유권 이전, zero-copy)

### GIL 전략 (batch evaluation)

```
1. Python에서 Rust 함수 호출
2. Rust: GIL 해제 (py.allow_threads)
3. Rust: octree 순회, 평가 필요 점 수집 (Rayon 병렬)
4. Rust: GIL 재획득 (Python::with_gil)
5. Rust→Python: (N,3) numpy array로 Python 함수 호출
6. Python: vectorized 계산, (N,) values + (N,3) gradients 반환
7. Rust: GIL 해제, 결과로 계속 처리
```

**주의**: Rayon worker가 GIL을 기다리면서 GIL holder가 Rayon을 기다리면 **deadlock**.
반드시 GIL 해제 후 parallel section 진입.

### maturin 프로젝트 구조

```
isomesh/
├── Cargo.toml           # [lib] crate-type = ["cdylib"]
├── pyproject.toml       # maturin backend
├── python/isomesh/      # Python wrapper
│   └── __init__.py
├── src/                 # Rust core
│   └── lib.rs
└── .github/workflows/
    └── CI.yml           # maturin generate-ci github
```

CI: `maturin-action` (PyO3 공식) → manylinux, macOS (x86_64+arm64), Windows wheels.

---

## 6. 핵심 논문 목록

### Dual Contouring

1. Ju, Losasso, Schaefer, Warren. **"Dual Contouring of Hermite Data."** SIGGRAPH 2002.
   [PDF](https://www.cse.wustl.edu/~taoju/research/dualContour.pdf)

2. Schaefer, Warren. **"Dual Contouring: The Secret Sauce."** Tech report.
   [PDF](https://people.eecs.berkeley.edu/~jrs/meshpapers/SchaeferWarren2.pdf)

3. **Schaefer, Ju, Warren. "Manifold Dual Contouring." IEEE TVCG 13(3), 2007.**
   [PDF](https://people.engr.tamu.edu/schaefer/research/dualsimp_tvcg.pdf)

4. Nielson. "Dual Marching Cubes." IEEE VIS 2004.

5. Schaefer, Warren. "Dual Marching Cubes: Primal Contouring of Dual Grids." CGF/Eurographics 2005.

### QEF & Feature Preservation

6. Garland, Heckbert. **"Surface Simplification Using Quadric Error Metrics."** SIGGRAPH 1997.

7. Lindstrom. "Out-of-Core Simplification of Large Polygonal Models." SIGGRAPH 2000.

8. **Trettner, Kobbelt. "Fast and Robust QEF Minimization using Probabilistic Quadrics."** CGF 39(2), 2020.

9. Kobbelt et al. "Feature Sensitive Surface Extraction from Volume Data." SIGGRAPH 2001.

### Robustness & Thin Features

10. **Duff. "Interval Arithmetic Recursive Subdivision for Implicit Functions and CSG."** SIGGRAPH 1992.

11. **Plantinga, Vegter. "Isotopic Meshing of Implicit Surfaces."** Visual Computer 23, 2007.

12. **Kalra, Barr. "Guaranteed Ray Intersections with Implicit Surfaces."** SIGGRAPH 1989.

13. **Galin et al. "Segment Tracing Using Local Lipschitz Bounds."** CGF/Eurographics 2020.

14. Ban et al. "Generalized Lipschitz Tracing of Implicit Surfaces." CGF 2025.

15. Paiva et al. "Robust Adaptive Meshes for Implicit Surfaces." SIBGRAPI 2006.

16. Varadhan et al. "Topology Preserving Surface Extraction Using Adaptive Subdivision." SGP 2004.

### Octree

17. Sundar et al. "Bottom-Up Construction and 2:1 Balance Refinement of Linear Octrees in Parallel." SIAM 2008.

18. Kazhdan et al. "Unconstrained Isosurface Extraction on Arbitrary Octrees." SGP 2007.
    [PDF](https://hhoppe.com/unconstrainediso.pdf)

19. Ju. "Intersection-free Contouring on An Octree Grid." 2006.

### Systems & Implementation

20. Keeter. "Massively Parallel Rendering of Complex Closed-Form Implicit Surfaces." SIGGRAPH 2020.
    (libfive/fidget의 이론적 기반)

---

## 7. 오픈소스 참조 코드

| 프로젝트 | 언어 | 용도 |
|----------|------|------|
| [mkeeter/fidget](https://github.com/mkeeter/fidget) | Rust | **아키텍처 최우선 참조** (MDC + IA) |
| [libfive/libfive](https://github.com/libfive/libfive) | C++ | Production MDC, Oracle interface |
| [nickgildea/qef](https://github.com/nickgildea/qef) | C/C++ | QEF solver port 대상 |
| [Philip-Trettner/probabilistic-quadrics](https://github.com/Philip-Trettner/probabilistic-quadrics) | C++17 | 고속 QEF 대안 |
| [swiftcoder/isosurface](https://github.com/swiftcoder/isosurface) | Rust | Multi-algorithm 참조 |
| [dominikwodniok/dualmc](https://github.com/dominikwodniok/dualmc) | C++ | Single-header DMC |
| [PyO3/rust-numpy](https://github.com/PyO3/rust-numpy) | Rust | Zero-copy numpy binding |
| [bonsairobo/fast-surface-nets-rs](https://github.com/bonsairobo/fast-surface-nets-rs) | Rust | 성능 벤치마크 참조 |

---

## 8. Tutorials & Explainers

- Matt Keeter, [QEF Explainer](https://www.mattkeeter.com/projects/qef/) — QEF 수학 + 구현 최고 참조
- Matt Keeter, [Adversarial Model for DC](https://www.mattkeeter.com/blog/2023-04-23-adversarial/) — DC vertex 탈출 문제
- Boris the Brave, [Dual Contouring Tutorial](https://www.boristhebrave.com/2018/04/15/dual-contouring-tutorial/)
- Nick Gildea, [Implementing Dual Contouring](http://ngildea.blogspot.com/2014/11/implementing-dual-contouring.html)
