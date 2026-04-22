# Known Issue: SIGSEGV with `adaptive=True, max_depth=8` on Ubuntu server

## Status
**Likely resolved — awaiting Ubuntu verification.** Static analysis of
the adaptive-path code uncovered a use-after-free in
`octree::balance::subdivide_cell_for_balance` (balance.rs ~line 500) that
matches the observed platform dependency exactly. Fix applied on branch
`fix/balance-uaf`; still need to run on the original Ubuntu server to
confirm the SIGSEGV is gone.

### Root cause (high confidence)
The offending function took a raw pointer into `octree.children[i][j]`,
then called `octree.children.push(child_cells)` which can reallocate the
Vec and invalidate that pointer, then wrote through the dangling pointer.
Textbook use-after-free.

```rust
let cell_ptr = { ... as *mut Cell };   // pointer into children[i][j]
unsafe {
    let idx = octree.children.len() as u32;
    octree.children.push(child_cells);  // may move the buffer!
    *cell_ptr = Cell::Branch { ... };   // dangling write → SIGSEGV
}
```

This is hit many times at `max_depth=8` because the `children` Vec grows
past its initial capacity repeatedly during 2:1 balance propagation,
and each capacity doubling reallocates. Why macOS didn't crash: Apple's
libmalloc keeps recently-freed regions mapped and reusable for a while,
so the dangling write lands on still-valid memory and silently succeeds
(the output is correct by luck). glibc's malloc unmaps large allocations
faster, so the same write hits an unmapped page → SIGSEGV.

The fix reorders the sequence to **check → push → re-navigate → write**,
so the reference used for the write is always obtained after the Vec is
stable. No more raw pointer, no more unsafe.

## Symptom
Running `long_train_mesh.py` on an Ubuntu server (exact version / kernel /
libc / Python / torch build not yet captured) with

```python
isomesh.extract(
    func=<DeepSDF wrapper with autograd gradients>,
    bbox_min=(-1.1, -1.1, -1.1),
    bbox_max=( 1.1,  1.1,  1.1),
    min_depth=4,
    max_depth=8,
    angle_threshold=30.0,
    iso_value=0.0,
    adaptive=True,
)
```

crashed with **SIGSEGV (exit code 139)** inside the Rust core **after**
successfully extracting `src.obj` (16,478 vertices). I.e. the first shape
completed, the second shape triggered the crash.

User workaround at the time: fall back to `adaptive=False, max_depth=7` plus
three safety guards in the Python wrapper (chunked evaluation, NaN / inf
sanitization via `np.nan_to_num`, and running the extractor in a subprocess
so a segfault can't take down the training loop).

## What we know
- On macOS (Apple Silicon, isomesh 0.1.0 local release build, CPU
  inference since there's no CUDA), the same checkpoint does **not**
  crash at either `max_depth=7` or `max_depth=8`, with either
  `adaptive=True` or `adaptive=False`. Every combination produces a
  watertight, manifold, single-component mesh. At `adaptive=True,
  max_depth=8` the src shape extracts 75,376 verts / 150,788 faces in
  ~13s (vs ~99s for the equivalent `adaptive=False, min=max=8` uniform
  extraction — adaptive is 7.5x faster once the hole bug was fixed).
  Bottom line: the SIGSEGV is specific to the Ubuntu build, not to the
  `adaptive=True, max_depth=8` algorithmic path itself.
- `adaptive=False, max_depth=7` on the same checkpoint previously had a
  different bug (boundary-edge holes from a Phase 3 surface-propagation
  gap). Fixed in commit 4a3270d on branch
  `fix/adaptive-false-force-to-max`.

## Plausible root causes (not yet verified)
Given that macOS works clean at the exact same algorithmic config, the
bug is almost certainly **platform-specific** rather than algorithmic:

1. glibc malloc (Ubuntu) vs macOS's libmalloc react differently to the
   allocation patterns of `Octree::children` at 256³. If there's a
   latent out-of-bounds or use-after-free in the `unsafe` block of
   `navigate_to_cell_mut`, only glibc's layout exposes it.
2. rayon thread count scales with core count — an Ubuntu server with
   many more cores than a MacBook may expose a race in parallel
   sections that serial-enough macOS doesn't hit.
3. CUDA-side autograd (Ubuntu only; macOS used CPU) produces gradient
   tensors that get copied to numpy via `.cpu().numpy()`. A stale CUDA
   stream or memory-pressure edge case could feed non-finite gradients
   that slip past `np.nan_to_num` (e.g. subnormals, or values that
   overflow only after the downstream QEF SVD).
4. Different Rust release build on Ubuntu (target-cpu=native) may use
   SIMD intrinsics in `qef/quadric.rs` that have UB triggered only on
   that microarchitecture.

## Verification playbook (on Ubuntu)
The `fix/balance-uaf` branch intentionally re-includes
`devdocs/deepsdf_debug/` and `long_run.ckpt` so the Ubuntu box can
reproduce in one clone. Remember to restore the original .gitignore
entries once verification is done (these shouldn't live on `main`).

### 1. Smoke test (fastest)
```bash
git clone <repo> && cd isomesh
git checkout fix/balance-uaf
maturin develop --release
cd devdocs/deepsdf_debug && python repro_highres.py
```
Pass criterion: `adaptive_d8` reports `watertight=True boundary=0`.

### 2. Run the full original workflow
```bash
python long_train_mesh.py   # uses the user's actual CUDA + autograd path
```
Pass criterion: no SIGSEGV; `meshes/*.obj` files all watertight.

### 3. If the SIGSEGV still triggers, collect a real stack
```bash
maturin develop            # debug build, not --release
RUST_BACKTRACE=full MALLOC_CHECK_=3 python long_train_mesh.py
```
- `MALLOC_CHECK_=3` makes glibc abort immediately on heap corruption.
- Backtrace pointing into a different code path means there's a
  second bug distinct from the one we just fixed.

### 4. gdb if you need to inspect locals at crash
```bash
gdb --args python long_train_mesh.py
(gdb) run
# on SIGSEGV:
(gdb) bt full
(gdb) info proc mappings    # is the faulting address unmapped?
```

### 5. ASAN if you still can't explain the crash
```bash
rustup toolchain install nightly
RUSTFLAGS="-Z sanitizer=address" \
    cargo +nightly build --release --target x86_64-unknown-linux-gnu
# swap the resulting .so into python/isomesh/ or rebuild with maturin
LD_PRELOAD=... python long_train_mesh.py
```
ASAN reports allocated-here / freed-here / use-here triplet that makes
UAFs trivial to pin down.

## Workaround (for end users)
- The `adaptive=False` hole bug is now fixed, so `adaptive=False`
  at any `min_depth/max_depth` is safe — but at `max_depth=8` it's
  ~7x slower than `adaptive=True` (full uniform vs feature-refined).
- If the Ubuntu SIGSEGV persists and adaptive speedup matters:
  try `adaptive=True, max_depth=7` (known clean on both platforms).
  For the SimJEB-class mechanical parts that inspired this issue,
  the extra resolution from d8 mainly cleans up spurious
  thin-wall artifacts in learned SDFs; d7 is often good enough.

## Related
- Hole bug in `adaptive=False, min_depth < max_depth`:
  `devdocs/deepsdf_debug/` and the fix on this branch.
