#!/usr/bin/env python3
"""
Visual + quantitative comparison: isomesh vs PyMCubes.
Renders side-by-side mesh images and produces a summary table.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mcubes
import trimesh
import isomesh

OUTPUT = "/Users/minsikseo/workspace/2_work/1_active/aslanx-flow/1_code/isomesh/output"

# ── SDFs ──

def sdf_sphere(pos, r=1.0):
    norms = np.linalg.norm(pos, axis=1)
    return norms - r, pos / np.where(norms > 1e-12, norms, 1.0)[:, None]

def sdf_box(pos, half=None):
    if half is None: half = np.array([1.0, 1.0, 1.0])
    half = np.asarray(half)
    q = np.abs(pos) - half
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0), None

def sdf_csg_cross(pos):
    d1, _ = sdf_box(pos, np.array([0.25, 0.25, 1.2]))
    d2, _ = sdf_box(pos, np.array([1.2, 0.25, 0.25]))
    d3, _ = sdf_box(pos, np.array([0.25, 1.2, 0.25]))
    return np.minimum(np.minimum(d1, d2), d3), None

def sdf_torus(pos, R=1.0, r=0.35):
    xy = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
    return np.sqrt((xy - R)**2 + pos[:, 2]**2) - r, None

def sdf_chamfered_box(pos):
    db, _ = sdf_box(pos, np.array([0.8, 0.8, 0.8]))
    ds = np.linalg.norm(pos, axis=1) - 1.2
    return np.maximum(db, ds), None

# ── Helpers ──

def pymcubes_extract(func, bbox, res):
    b = bbox
    t0 = time.perf_counter()
    lin = np.linspace(-b, b, res)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    vals, _ = func(pts)
    vol = vals.reshape(res, res, res)
    v, f = mcubes.marching_cubes(vol, 0.0)
    v = v / (res - 1) * (2 * b) - b
    t = time.perf_counter() - t0
    # PyMCubes produces inverted normals — flip faces
    f = f[:, ::-1].copy()
    return v, f.astype(np.int64), t

def isomesh_extract(func, bbox, mind, maxd):
    b = bbox
    t0 = time.perf_counter()
    v, f = isomesh.extract(func=func, bbox_min=(-b,)*3, bbox_max=(b,)*3, min_depth=mind, max_depth=maxd)
    return v, f, time.perf_counter() - t0

def render_mesh(ax, vertices, faces, title, color="#4488cc"):
    """Render mesh as shaded polygons."""
    if len(faces) == 0:
        ax.set_title(title)
        return
    # Subsample faces for rendering if too many
    max_faces = 5000
    if len(faces) > max_faces:
        idx = np.random.default_rng(42).choice(len(faces), max_faces, replace=False)
        faces_sub = faces[idx]
    else:
        faces_sub = faces

    polys = vertices[faces_sub]
    collection = Poly3DCollection(polys, alpha=0.8, facecolor=color, edgecolor="#333333", linewidth=0.1)
    ax.add_collection3d(collection)

    # Set limits
    all_v = vertices
    mid = (all_v.max(axis=0) + all_v.min(axis=0)) / 2
    span = (all_v.max(axis=0) - all_v.min(axis=0)).max() * 0.6
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()

def mesh_stats(v, f, func=None):
    """Return dict of mesh quality metrics."""
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    d = {"V": len(v), "F": len(f), "watertight": bool(mesh.is_watertight)}
    if func is not None and len(v) > 0:
        vals, _ = func(v)
        d["hausdorff"] = float(np.max(np.abs(vals)))
        d["mean_sdf"] = float(np.mean(np.abs(vals)))
    return d

# ── Main comparison ──

SHAPES = {
    "sphere":       (sdf_sphere, 1.5),
    "box":          (sdf_box, 1.5),
    "csg_cross":    (sdf_csg_cross, 1.8),
    "torus":        (sdf_torus, 1.8),
    "chamfered_box": (sdf_chamfered_box, 1.5),
}

CONFIGS = [
    # (label, isomesh_min, isomesh_max, mc_res)
    ("low",  4, 4, 17),
    ("mid",  5, 5, 33),
    ("high", 6, 6, 65),
]

print("=" * 110)
print(f"{'Shape':<16} {'Config':<6} {'Method':<10} {'V':>6} {'F':>7} {'Time':>8} {'Hausdorff':>10} {'Watertight':>11}")
print("=" * 110)

for shape_name, (func, bbox) in SHAPES.items():
    fig = plt.figure(figsize=(18, 6 * len(CONFIGS)))

    for ci, (label, mind, maxd, mc_res) in enumerate(CONFIGS):
        # isomesh
        v_iso, f_iso, t_iso = isomesh_extract(func, bbox, mind, maxd)
        s_iso = mesh_stats(v_iso, f_iso, func)
        s_iso["time"] = t_iso

        # pymcubes
        v_mc, f_mc, t_mc = pymcubes_extract(func, bbox, mc_res)
        s_mc = mesh_stats(v_mc, f_mc, func)
        s_mc["time"] = t_mc

        for method, stats, v, f, color in [
            ("isomesh", s_iso, v_iso, f_iso, "#4488cc"),
            ("pymcubes", s_mc, v_mc, f_mc, "#cc6644"),
        ]:
            h_str = f"{stats.get('hausdorff', 0):.5f}" if "hausdorff" in stats else "N/A"
            wt = "✓" if stats["watertight"] else "✗"
            prefix = shape_name if method == "isomesh" else ""
            print(f"{prefix:<16} {label:<6} {method:<10} {stats['V']:>6} {stats['F']:>7} {stats['time']:>7.4f}s {h_str:>10} {wt:>11}")

        # Render side-by-side
        ax1 = fig.add_subplot(len(CONFIGS), 2, ci * 2 + 1, projection="3d")
        render_mesh(ax1, v_iso, f_iso,
                    f"isomesh ({label}) V={len(v_iso)} F={len(f_iso)} t={t_iso:.3f}s",
                    color="#4488cc")
        ax2 = fig.add_subplot(len(CONFIGS), 2, ci * 2 + 2, projection="3d")
        render_mesh(ax2, v_mc, f_mc,
                    f"PyMCubes ({label}) V={len(v_mc)} F={len(f_mc)} t={t_mc:.3f}s",
                    color="#cc6644")

    print("-" * 110)
    fig.suptitle(f"{shape_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT}/{shape_name}_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# ── Adaptive vs uniform comparison for box corners ──
print("\n" + "=" * 110)
print("SHARP FEATURE: Box corner distance to analytic corners")
print("=" * 110)

corners = np.array([[s0, s1, s2] for s0 in [-1, 1] for s1 in [-1, 1] for s2 in [-1, 1]], dtype=np.float64)

for label, mind, maxd, mc_res in CONFIGS:
    v_iso, _, t_iso = isomesh_extract(sdf_box, 1.5, mind, maxd)
    v_mc, _, t_mc = pymcubes_extract(sdf_box, 1.5, mc_res)

    iso_cd = np.mean([np.min(np.linalg.norm(v_iso - c, axis=1)) for c in corners])
    mc_cd = np.mean([np.min(np.linalg.norm(v_mc - c, axis=1)) for c in corners])

    winner = "isomesh" if iso_cd < mc_cd else "pymcubes"
    print(f"  {label}: isomesh={iso_cd:.4f}  pymcubes={mc_cd:.4f}  winner={winner}  "
          f"(improvement: {mc_cd/max(iso_cd,1e-10):.0f}x)")

print(f"\nRendered comparison images saved to {OUTPUT}/")
