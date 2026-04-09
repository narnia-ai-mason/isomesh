# What's Next

현재 상태 (2026-04-10 기준):
- Basic DC + adaptive octree + QEF (Jacobi SVD) + Newton projection
- Manifold/watertight for closed shapes at uniform surface depth
- 80 tests, 16 benchmark shapes, PyMCubes 대비 동등 속도 + 우월한 sharp feature

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

### 현재 상태
- Basic DC: cell당 vertex 1개
- 단순 topology에서는 manifold, 복잡한 경우 (self-intersection 근처 등) non-manifold 가능

### 구현 방향 (Schaefer, Ju, Warren 2007)

#### Phase 1: Connected Component Analysis
- `src/dc/components.rs`
- 각 leaf cell에서 inside corners의 connected component 분석 (Union-Find)
- Corner graph: 두 inside corner가 sign change 없는 edge로 연결되면 같은 component
- 각 component에 별도 QEF vertex 할당

#### Phase 2: Manifold Criterion Check
- `src/dc/manifold.rs`
- Face/edge proc에서 vertex pairing이 non-manifold을 생성하는지 검사
- 위반 시 cell의 vertex를 split (component 세분화)
- Schaefer et al. 2007 §4의 collapsibility test 구현

#### Phase 3: Integration
- `src/dc/extract.rs` 수정: component별 vertex lookup, quad 생성 시 올바른 vertex 선택

### 핵심 참조
- 논문: Schaefer, Ju, Warren. "Manifold Dual Contouring." IEEE TVCG 13(3), 2007.
  [PDF](https://people.engr.tamu.edu/schaefer/research/dualsimp_tvcg.pdf)
- 참조 구현: `mkeeter/fidget` (Rust, MDC + IA)
- 참조 구현: `libfive` (C++, production MDC)

### 난관
- 256 corner configuration 전수 테스트 필요
- Manifold criterion의 edge case (특히 adaptive octree와 결합 시)
- fidget의 구현을 상세히 연구할 것 — 실전 검증된 MDC

---

## 4. True Adaptive Mesh (2:1 Balance + T-junction Stitching)

**난이도: 매우 높음 (XL)**
**예상 효과: 같은 품질에서 vertex 수 1/5~1/10, DC의 본질적 강점 실현**

### 현재 상태
- Adaptive refinement는 feature 근처 cell만 세분화
- 하지만 DC extraction 전에 **모든 surface cell을 max_depth로 강제 확장**
- → Surface 위에서는 사실상 uniform resolution
- True adaptive의 이점 (mesh 경량화) 미실현

### 구현 방향

#### Phase 1: 2:1 Balance
- `src/octree/balance.rs`
- BFS ripple propagation: 인접 cell depth 차이 ≤ 1 보장
- 새 cell의 corner 값 batch evaluation
- Sundar et al. 2008 참조

#### Phase 2: T-junction Stitching
- Coarse face ↔ fine face 4개의 경계 처리
- 2:1 balance 전제: coarse face 하나 = fine face 4개 (단일 수준 차이)
- Coarse cell의 edge가 fine cell 2개의 edge와 겹침
- 핵심: fine cell의 canonical edge iteration이 coarse cell의 vertex를 올바르게 참조

#### Phase 3: Cross-depth Quad Generation
- Per-leaf canonical edge 방식 유지
- Neighbor lookup: 같은 depth에서 못 찾으면 ±1 depth 탐색
- Coarse cell 1개와 fine cell 여러 개가 공유하는 edge에서 fan 형태 quad 생성
- **Manifold/watertight 검증이 가장 어려운 부분**

### 핵심 참조
- Kazhdan et al. "Unconstrained Isosurface Extraction on Arbitrary Octrees." SGP 2007.
  — 2:1 balance 없이도 watertight 보장하는 방법 (더 복잡하지만 더 유연)
  [PDF](https://hhoppe.com/unconstrainediso.pdf)
- Ju. "Intersection-free Contouring on An Octree Grid." 2006.
- `mkeeter/fidget`: adaptive MDC 구현의 실전 참조

### 난관
- T-junction에서 crack-free 보장이 핵심 난관
- Manifold/watertight 유지가 depth boundary에서 매우 어려움
- 2:1 balance cascade로 cell 수 증가 (~10-30%)
- **권장: MDC (항목 3)를 먼저 구현한 후 진행** — MDC의 multi-vertex-per-cell이 T-junction 처리를 단순화

---

## 구현 권장 순서

```
1. Rayon 병렬화 (S)     — 즉시 속도 개선, 다른 항목과 독립
2. Probabilistic Quadrics (M) — QEF 속도 50x, Rayon과 시너지
3. MDC (L)              — manifold 보장, True Adaptive의 선행 조건
4. True Adaptive (XL)   — MDC + 2:1 balance + T-junction, 최종 목표
```

항목 3과 4는 순서가 중요: MDC 없이 True Adaptive를 하면 T-junction에서 non-manifold 문제가 더 심해짐.
