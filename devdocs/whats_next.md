# What's Next

현재 상태 (2026-04-10 기준):
- MDC Phase 1 (Connected Component Analysis) + adaptive octree + QEF (Probabilistic Quadrics) + Newton projection + Rayon 병렬화
- Manifold/watertight for closed shapes at uniform surface depth
- Multi-component cells에서 component별 별도 vertex 생성 (bowtie 방지)
- 80 Python tests + 37 Rust tests, 16 benchmark shapes

---

## 1. Rayon 병렬화

**난이도: 낮음 (S)**
**예상 효과: 4-8x speedup**

### 현재 상태
- QEF solve, Newton projection, DC quad generation 모두 single-thread
- GIL 프로토콜은 이미 올바르게 설계됨 (Python 호출은 main thread, Rust 연산만 병렬)

### 구현 방향
1. `extract_dc`에서 QEF accumulation + solve를 `rayon::par_iter`로 병렬화
2. Newton projection batch를 Rayon으로 (Python 호출 후 결과 처리만)
3. Quad generation: thread-local `Vec<Face>` → 마지막에 merge

### 주의사항
- **GIL deadlock 방지**: `Python::with_gil` 절대 Rayon closure 안에서 호출 금지
- Python 호출은 반드시 sequential, 결과 처리만 parallel
- 코드 grep 검증: `rayon::` 안에 `Python::with_gil` 없어야 함

### 핵심 파일
- `src/dc/extract.rs`: QEF solve loop, Newton projection loop

---

## 2. Probabilistic Quadrics (Trettner & Kobbelt 2020)

**난이도: 중간 (M)**
**예상 효과: QEF solve 50x 빠름, always full rank**

### 현재 상태
- Jacobi SVD 기반 3x3 eigendecomposition + mass-point bias
- Rank-deficient case: eigenvalue threshold로 처리
- 동작하지만 SVD가 상대적으로 느림

### 구현 방향
1. `src/qef/probabilistic.rs` 생성
2. Gaussian noise model로 QEF를 확률적 재정의
3. Cholesky decomposition으로 풀이 (always full rank → SVD 불필요)
4. 기존 `QefData` 인터페이스 유지, solver만 교체

### 핵심 참조
- 논문: Trettner, Kobbelt. "Fast and Robust QEF Minimization using Probabilistic Quadrics." CGF 39(2), 2020.
- 참조 구현: [Philip-Trettner/probabilistic-quadrics](https://github.com/Philip-Trettner/probabilistic-quadrics) (header-only C++17)
- `devdocs/references.md` §3.4

### 주의사항
- Noise variance σ² 파라미터 선택: 논문의 권장값 사용
- 기존 SVD solver도 옵션으로 유지 (비교용)

---

## 3. MDC (Manifold Dual Contouring)

**난이도: 높음 (L)**
**예상 효과: 모든 topology에서 manifold/watertight 보장**

### 현재 상태 (Phase 1 완료, Phase 2-3 남음)
- ✅ Phase 1: Connected Component Analysis (`src/dc/component.rs`)
  - 256개 corner_mask에 대한 precomputed lookup table (`COMPONENT_TABLE`)
  - Union-Find 기반 connected component 분석
  - 15개 Rust 단위 테스트 (256 전수 검증 포함)
- ✅ Phase 3 (부분): Integration (`src/dc/extract.rs`)
  - Component별 별도 QEF vertex 할당
  - Component-aware quad generation (`vertex_for_edge`)
  - Degenerate quad 필터링
  - 단일 component fast path 최적화
- ❌ Phase 2: Manifold Criterion Check (미구현)
  - 현재 한계: union-of-cylinders 같은 복잡한 intersection에서 여전히 non-manifold 가능
  - 이는 inside corners가 connected이지만 surface가 cell 내에서 여러 sheet를 형성하는 경우

### 남은 구현 방향

#### Phase 2: Manifold Criterion Check
- `src/dc/manifold.rs`
- Face/edge proc에서 vertex pairing이 non-manifold을 생성하는지 검사
- 위반 시 cell의 vertex를 split (component 세분화)
- Schaefer et al. 2007 §4의 collapsibility test 구현

### 핵심 참조
- 논문: Schaefer, Ju, Warren. "Manifold Dual Contouring." IEEE TVCG 13(3), 2007.
  [PDF](https://people.engr.tamu.edu/schaefer/research/dualsimp_tvcg.pdf)
- 참조 구현: `mkeeter/fidget` (Rust, MDC + IA)
- 참조 구현: `libfive` (C++, production MDC)

### 난관
- ✅ 256 corner configuration 전수 테스트 — 완료
- Manifold criterion의 edge case (특히 adaptive octree와 결합 시)
- fidget의 구현을 상세히 연구할 것 — 실전 검증된 MDC

---

## 4. True Adaptive Mesh (2:1 Balance + T-junction Stitching)

**난이도: 매우 높음 (XL)**
**예상 효과: 같은 품질에서 vertex 수 1/5~1/10, DC의 본질적 강점 실현**

### 현재 상태 (기본 구현 완료, depth boundary 개선 남음)
- ✅ `src/octree/balance.rs`: 2:1 balance BFS ripple + surface propagation
- ✅ `src/dc/extract.rs`: cross-depth neighbor lookup (same → coarser → finer)
- ✅ depth-aware cell_map (`HashMap<(u32,u32,u32,u8), usize>`)
- ✅ per-leaf cell_size for QEF sigma_p
- ✅ Python API: `adaptive=True` flag in `isomesh.extract()`
- ✅ 검증됨: sphere d3-5 77% vertex 감소, sphere d3-6 94% vertex 감소, manifold/watertight

### 알려진 한계: Depth Boundary Gaps
- Surface cell이 collapsed Empty cell (tree pruning으로 인해 낮은 depth에 존재)과 인접할 때 gap 발생
- 이는 adaptive와 non-adaptive 빌드 모두에 공통된 기존 한계
- Smooth shape (sphere, torus)에서는 문제 없음
- Sharp feature shape (box) 또는 좁은 depth 범위 (d4-5)에서 발생

### 남은 개선 방향

#### Depth Boundary Completion
- Surface edge를 공유하는 4개 cell이 모두 surface cell이 되도록 보장
- 접근 1: Balance에서 edge-adjacent (26-neighborhood) propagation 추가
- 접근 2: DC extraction 전에 surface edge 완전성 검증 pass 추가
- 접근 3: Kazhdan et al. 2007의 unconstrained extraction 방식 적용

### 핵심 참조
- Kazhdan et al. "Unconstrained Isosurface Extraction on Arbitrary Octrees." SGP 2007.
  [PDF](https://hhoppe.com/unconstrainediso.pdf)
- `mkeeter/fidget`: adaptive MDC 구현의 실전 참조

---

## 구현 권장 순서

```
1. ✅ Rayon 병렬화 (S)           — 완료
2. ✅ Probabilistic Quadrics (M) — 완료
3. ✅ MDC (L)                   — Phase 1 (Component Analysis) 완료
4. 🔶 True Adaptive (XL)        — 기본 구현 완료, depth boundary gap 개선 남음
```

Smooth shape에서 true adaptive 동작 확인 (sphere 77-94% vertex 감소). Sharp feature shape에서의 depth boundary gap은 향후 개선 과제.
