# Known Issue: SIGSEGV with `adaptive=True, max_depth=8` on Ubuntu server

## Status
Deferred — to investigate separately. Reproduction environment is not at hand on macOS.

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
- On macOS (isomesh 0.1.0 local build), the same checkpoint + same
  `adaptive=True, max_depth=7` does **not** crash and produces a clean
  watertight mesh. See `devdocs/deepsdf_debug/sweep2.py`.
- `adaptive=False, max_depth=7` on the *same* checkpoint has a different
  bug (boundary-edge holes) — that one is tracked and being fixed.

## Plausible root causes (not yet verified)
1. `max_depth=8` → 256³ grid → larger corner cache and bigger
   allocations. The bug may be a latent memory-safety issue in
   `Octree::children` growth, `navigate_to_cell_mut`'s `unsafe` block,
   or in `balance::enforce_balance_2to1` that only triggers at higher
   depths.
2. `autograd` on a `16-mixed`-trained model may produce non-finite
   gradients at some query points that slip past `np.nan_to_num` (for
   example, `-inf` in a gradient component that `nan_to_num` converts
   to a large finite value but still causes downstream float issues in
   the QEF solver — SVD on a matrix with extremely large entries).
3. Platform-specific: rayon thread-pool × glibc malloc × the SIMD
   paths in `qef/quadric.rs` may hit UB that only the Ubuntu build
   exposes. macOS ships with a different malloc and Apple Silicon's
   SIMD lanes are different.

## Reproduction to try (on Ubuntu)
1. Same checkpoint (`long_run.ckpt`) + the user's `long_train_mesh.py`
   with `adaptive=True, max_depth=8`.
2. Capture `RUST_BACKTRACE=full` output. Strip the Python subprocess
   wrapper so the segfault's call stack is actually visible.
3. Build isomesh with `maturin develop` (debug, not release) and
   `RUSTFLAGS="-C debuginfo=2"` so gdb can show a useful stack.
4. `gdb --args python long_train_mesh.py` then `run`, on SIGSEGV
   `bt full`.
5. If the backtrace points into `balance::enforce_balance_2to1` or
   `navigate_to_cell_mut`, treat as a tree-mutation UB issue. If it
   points into `qef::quadric`, check the input Hermite data for
   non-finite values.

## Workaround (for end users)
- Use `adaptive=False, min_depth=max_depth=N` (e.g. N=7). After the
  `adaptive=False` hole fix lands, uniform mode is what you want here.
- If you need max_depth=8 resolution and adaptivity, wait for this
  issue to be resolved.

## Related
- Hole bug in `adaptive=False, min_depth < max_depth`:
  `devdocs/deepsdf_debug/` and the fix on this branch.
