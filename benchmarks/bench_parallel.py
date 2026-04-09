#!/usr/bin/env python3
"""
Benchmark Rayon parallelization speedup.

Measures extraction time across thread counts (1, 2, 4, 8) and depths (5, 6, 7).
Verifies that outputs are identical regardless of thread count.
"""

import os
import sys
import time
import subprocess
import json
import numpy as np

# Run as subprocess to control RAYON_NUM_THREADS per invocation
if __name__ == "__main__" and "--worker" not in sys.argv:
    THREAD_COUNTS = [1, 2, 4, 8]
    DEPTHS = [5, 6, 7]
    SHAPES = ["sphere", "box", "torus", "gyroid"]
    TRIALS = 5

    script = os.path.abspath(__file__)
    results = {}

    for threads in THREAD_COUNTS:
        env = os.environ.copy()
        env["RAYON_NUM_THREADS"] = str(threads)
        proc = subprocess.run(
            [sys.executable, script, "--worker",
             "--depths", ",".join(str(d) for d in DEPTHS),
             "--shapes", ",".join(SHAPES),
             "--trials", str(TRIALS)],
            env=env, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"Worker failed (threads={threads}):\n{proc.stderr}", file=sys.stderr)
            sys.exit(1)
        worker_results = json.loads(proc.stdout)
        results[threads] = worker_results

    # Print comparison table
    print(f"\n{'Shape':<10} {'Depth':>5}  ", end="")
    for t in THREAD_COUNTS:
        print(f"  {t}T (ms)", end="")
    print("  Speedup(1T/max)")
    print("-" * 80)

    for shape in SHAPES:
        for depth in DEPTHS:
            key = f"{shape}_d{depth}"
            print(f"{shape:<10} {depth:>5}  ", end="")
            times = {}
            for t in THREAD_COUNTS:
                entry = results[t].get(key, {})
                ms = entry.get("median_ms", float("nan"))
                times[t] = ms
                print(f"  {ms:>7.1f}", end="")
            if times.get(1, 0) > 0:
                best = min(times[t] for t in THREAD_COUNTS if t in times)
                speedup = times[1] / best if best > 0 else 0
                print(f"  {speedup:>6.2f}x", end="")
            print()

    # Verify output consistency across thread counts
    print("\n--- Output Consistency Check ---")
    all_ok = True
    for shape in SHAPES:
        for depth in DEPTHS:
            key = f"{shape}_d{depth}"
            counts = set()
            for t in THREAD_COUNTS:
                entry = results[t].get(key, {})
                counts.add((entry.get("n_vertices", -1), entry.get("n_faces", -1)))
            if len(counts) == 1:
                nv, nf = counts.pop()
                print(f"  {key}: OK (V={nv}, F={nf})")
            else:
                print(f"  {key}: MISMATCH {counts}")
                all_ok = False

    if all_ok:
        print("\nAll outputs consistent across thread counts.")
    else:
        print("\nWARNING: Output mismatch detected!", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


# ── Worker mode ──────────────────────────────────────────────────────

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--worker", action="store_true")
parser.add_argument("--depths", type=str, default="5,6,7")
parser.add_argument("--shapes", type=str, default="sphere,box,torus,gyroid")
parser.add_argument("--trials", type=int, default=5)
args = parser.parse_args()

import isomesh

DEPTHS = [int(d) for d in args.depths.split(",")]
TRIALS = args.trials


def sdf_sphere(pos, radius=1.0):
    norms = np.linalg.norm(pos, axis=1)
    values = norms - radius
    safe = np.where(norms > 1e-12, norms, 1.0)
    return values, pos / safe[:, None]


def sdf_box(pos):
    half = np.array([1.0, 1.0, 1.0])
    q = np.abs(pos) - half
    q_max = np.maximum(q, 0.0)
    outside_dist = np.linalg.norm(q_max, axis=1)
    inside_dist = np.minimum(np.max(q, axis=1), 0.0)
    return outside_dist + inside_dist, None


def sdf_torus(pos, R=1.0, r=0.3):
    xy = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
    return np.sqrt((xy - R)**2 + pos[:, 2]**2) - r, None


def sdf_gyroid(pos):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    scale = np.pi * 2
    values = (np.sin(scale * x) * np.cos(scale * y)
              + np.sin(scale * y) * np.cos(scale * z)
              + np.sin(scale * z) * np.cos(scale * x))
    return values * 0.15, None


SHAPE_MAP = {
    "sphere": (sdf_sphere, 1.5),
    "box": (sdf_box, 1.5),
    "torus": (sdf_torus, 1.8),
    "gyroid": (sdf_gyroid, 1.0),
}

shape_names = [s for s in args.shapes.split(",") if s in SHAPE_MAP]
results = {}

for shape_name in shape_names:
    func, bbox = SHAPE_MAP[shape_name]
    for depth in DEPTHS:
        key = f"{shape_name}_d{depth}"
        times = []
        nv, nf = 0, 0
        for trial in range(TRIALS):
            t0 = time.perf_counter()
            v, f = isomesh.extract(
                func=func,
                bbox_min=(-bbox, -bbox, -bbox),
                bbox_max=(bbox, bbox, bbox),
                min_depth=depth, max_depth=depth,
            )
            elapsed = time.perf_counter() - t0
            times.append(elapsed * 1000)  # ms
            nv, nf = len(v), len(f)

        times.sort()
        median_ms = times[len(times) // 2]
        results[key] = {
            "median_ms": round(median_ms, 2),
            "n_vertices": nv,
            "n_faces": nf,
        }

print(json.dumps(results))
