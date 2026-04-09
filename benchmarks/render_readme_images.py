#!/usr/bin/env python3
"""Generate comparison images for README: isomesh vs PyMCubes."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mcubes
import isomesh

OUT = "/Users/minsikseo/workspace/2_work/1_active/aslanx-flow/1_code/isomesh/docs/images"

# ── SDFs ──

def sdf_sphere(pos, r=1.0):
    norms = np.linalg.norm(pos, axis=1)
    return norms - r, pos / np.where(norms > 1e-12, norms, 1.0)[:, None]

def sdf_box(pos):
    half = np.array([1.0, 1.0, 1.0])
    q = np.abs(pos) - half
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0), None

def sdf_chamfered_box(pos):
    half = np.array([0.8, 0.8, 0.8])
    q = np.abs(pos) - half
    db = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)
    ds = np.linalg.norm(pos, axis=1) - 1.2
    return np.maximum(db, ds), None

def sdf_csg_cross(pos):
    d1, _ = sdf_box_h(pos, np.array([0.25, 0.25, 1.2]))
    d2, _ = sdf_box_h(pos, np.array([1.2, 0.25, 0.25]))
    d3, _ = sdf_box_h(pos, np.array([0.25, 1.2, 0.25]))
    return np.minimum(np.minimum(d1, d2), d3), None

def sdf_box_h(pos, half):
    q = np.abs(pos) - half
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0), None

def sdf_torus(pos, R=1.0, r=0.35):
    xy = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
    return np.sqrt((xy - R)**2 + pos[:, 2]**2) - r, None

# ── Helpers ──

def mc_extract(func, bbox, res):
    b = bbox
    lin = np.linspace(-b, b, res)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    vals, _ = func(pts)
    v, f = mcubes.marching_cubes(vals.reshape(res, res, res), 0.0)
    v = v / (res - 1) * (2 * b) - b
    f = f[:, ::-1]  # fix inverted normals
    return v, f

def iso_extract(func, bbox, mind, maxd, angle=20.0):
    b = bbox
    v, f = isomesh.extract(func=func, bbox_min=(-b,)*3, bbox_max=(b,)*3,
                             min_depth=mind, max_depth=maxd, angle_threshold=angle)
    return v, f

def render(ax, verts, faces, color, elev=25, azim=-50, max_polys=8000):
    if len(faces) == 0:
        return
    if len(faces) > max_polys:
        idx = np.random.default_rng(42).choice(len(faces), max_polys, replace=False)
        faces = faces[idx]
    polys = verts[faces]
    col = Poly3DCollection(polys, alpha=0.95, facecolor=color, edgecolor=color,
                            linewidth=0.05)
    ax.add_collection3d(col)
    mid = (verts.max(0) + verts.min(0)) / 2
    span = (verts.max(0) - verts.min(0)).max() * 0.55
    ax.set_xlim(mid[0]-span, mid[0]+span)
    ax.set_ylim(mid[1]-span, mid[1]+span)
    ax.set_zlim(mid[2]-span, mid[2]+span)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

# ── Render comparisons ──

SHAPES = [
    ("Box (sharp features)", sdf_box, 1.5, 5, 6, 33, 65),
    ("Chamfered Box", sdf_chamfered_box, 1.5, 4, 7, 33, 129),
    ("Sphere", sdf_sphere, 1.5, 5, 6, 33, 65),
    ("CSG Cross", sdf_csg_cross, 1.8, 4, 7, 33, 129),
    ("Torus", sdf_torus, 1.8, 4, 7, 33, 129),
]

# 1. Side-by-side comparison grid
fig, axes = plt.subplots(len(SHAPES), 2, figsize=(10, 5 * len(SHAPES)),
                          subplot_kw={"projection": "3d"})

for i, (name, func, bbox, mind, maxd, mc_lo, mc_hi) in enumerate(SHAPES):
    v_iso, f_iso = iso_extract(func, bbox, mind, maxd)
    v_mc, f_mc = mc_extract(func, bbox, mc_hi)

    render(axes[i, 0], v_iso, f_iso, "#3d7ec7")
    axes[i, 0].set_title(f"isomesh ({mind}→{maxd})\nV={len(v_iso):,}  F={len(f_iso):,}", fontsize=10)

    render(axes[i, 1], v_mc, f_mc, "#c76b3d")
    axes[i, 1].set_title(f"PyMCubes ({mc_hi})\nV={len(v_mc):,}  F={len(f_mc):,}", fontsize=10)

fig.suptitle("isomesh (Dual Contouring) vs PyMCubes (Marching Cubes)", fontsize=14, y=1.0)
fig.tight_layout()
fig.savefig(f"{OUT}/comparison_grid.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUT}/comparison_grid.png")

# 2. Box corner close-up comparison at LOW resolution to show the difference
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={"projection": "3d"})

v_iso_lo, f_iso_lo = iso_extract(sdf_box, 1.5, 4, 4)
v_mc_lo, f_mc_lo = mc_extract(sdf_box, 1.5, 17)

render(axes2[0], v_iso_lo, f_iso_lo, "#3d7ec7", elev=30, azim=-45)
axes2[0].set_title("isomesh — depth 4\nSharp corners preserved", fontsize=11)

render(axes2[1], v_mc_lo, f_mc_lo, "#c76b3d", elev=30, azim=-45)
axes2[1].set_title("PyMCubes — 17³ grid\nCorners rounded by MC", fontsize=11)

fig2.suptitle("Sharp Feature Preservation: Box at Low Resolution", fontsize=13)
fig2.tight_layout()
fig2.savefig(f"{OUT}/box_sharp_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUT}/box_sharp_comparison.png")

# 3. Individual shape renders for README hero image
for name_short, func, bbox, mind, maxd in [
    ("sphere", sdf_sphere, 1.5, 5, 6),
    ("box", sdf_box, 1.5, 4, 6),
    ("chamfered_box", sdf_chamfered_box, 1.5, 4, 7),
    ("torus", sdf_torus, 1.8, 4, 7),
    ("csg_cross", sdf_csg_cross, 1.8, 4, 7),
]:
    v, f = iso_extract(func, bbox, mind, maxd)
    fig3, ax3 = plt.subplots(1, 1, figsize=(5, 5), subplot_kw={"projection": "3d"})
    render(ax3, v, f, "#3d7ec7")
    ax3.set_title(f"{name_short}\nV={len(v):,}  F={len(f):,}", fontsize=11)
    fig3.tight_layout()
    fig3.savefig(f"{OUT}/{name_short}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT}/{name_short}.png")

print("\nDone!")
