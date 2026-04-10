#!/usr/bin/env python3
"""
Isomesh Benchmark Suite
=======================

Comprehensive comparison of isomesh (Dual Contouring) vs PyMCubes (Marching Cubes).

Shapes (9):
  Category 1 — Basic (smooth, analytic ground truth)
    1. sphere              Unit sphere
    2. torus               Standard torus (R=1.0, r=0.35)

  Category 2 — Sharp features (CAD-like)
    3. box                 Flat faces, sharp edges & corners
    4. chamfered_sphere    Curved face + sharp circular edges
    5. cylinder            Flat + curved + 2 circular sharp edges
    6. mechanical_part     Box + through-rod − 4 holes (complex CSG)

  Category 3 — Thin features
    7. thin_shell_hemi     Hemisphere shell (wall=0.08), open boundary

  Category 4 — Complex geometry (mesh-based SDF via point-cloud-utils)
    8. bunny               Stanford Bunny (organic, smooth curvature)
    9. simjeb_148          SimJEB jet engine bracket (sharp CAD)

Methods:
    pymcubes         Marching Cubes, res=65
    uniform          isomesh DC, depth=6 (64 cells/axis — matches MC)
    adaptive         isomesh DC adaptive, min_depth=4, max_depth=6
    adaptive_fine    isomesh DC adaptive, min_depth=4, max_depth=7

Usage:
    uv run python benchmarks/run_benchmark.py
    uv run python benchmarks/run_benchmark.py --shapes sphere box
    uv run python benchmarks/run_benchmark.py --methods uniform adaptive
    uv run python benchmarks/run_benchmark.py --no-render
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
import mcubes
import isomesh

try:
    import point_cloud_utils as pcu

    HAS_PCU = True
except ImportError:
    HAS_PCU = False

try:
    import pyrender
    from PIL import Image, ImageDraw, ImageFont

    HAS_PYRENDER = True
except ImportError:
    HAS_PYRENDER = False


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
BENCHMARKS_DIR = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════════
#  SDF Library
# ═══════════════════════════════════════════════════════════════════


def sdf_sphere(pos, radius=1.0):
    norms = np.linalg.norm(pos, axis=1)
    values = norms - radius
    safe = np.where(norms > 1e-12, norms, 1.0)
    return values, pos / safe[:, None]


def sdf_torus(pos, R=1.0, r=0.35):
    xy = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    return np.sqrt((xy - R) ** 2 + pos[:, 2] ** 2) - r, None


def sdf_box(pos, half=None):
    if half is None:
        half = np.array([1.0, 1.0, 1.0])
    half = np.asarray(half, dtype=np.float64)
    q = np.abs(pos) - half
    qm = np.maximum(q, 0.0)
    return np.linalg.norm(qm, axis=1) + np.minimum(np.max(q, axis=1), 0.0), None


def _rotation_matrix(axis, angle):
    """Rodrigues' rotation: axis must be unit vector, angle in radians."""
    ax = np.asarray(axis, dtype=np.float64)
    ax = ax / np.linalg.norm(ax)
    c, s = np.cos(angle), np.sin(angle)
    K = np.array([[0, -ax[2], ax[1]],
                  [ax[2], 0, -ax[0]],
                  [-ax[1], ax[0], 0]])
    return np.eye(3) * c + (1 - c) * np.outer(ax, ax) + s * K


def sdf_rotated_box(pos, half=None, axis=None, angle=np.pi / 4):
    """Box rotated around an arbitrary axis — all edges become non-axis-aligned."""
    if axis is None:
        axis = np.array([1.0, 1.0, 1.0])
    R_inv = _rotation_matrix(axis, -angle)
    local_pos = pos @ R_inv.T
    return sdf_box(local_pos, half)


def sdf_chamfered_sphere(pos, R=1.0, d=0.8):
    """Sphere with 6 axis-aligned plane cuts — curved faces + sharp circular edges."""
    ds, _ = sdf_sphere(pos, radius=R)
    db, _ = sdf_box(pos, np.array([d, d, d]))
    return np.maximum(ds, db), None


def _sdf_capped_cylinder(pos, radius, half_height, axis):
    """Capped cylinder along the given axis (0=x, 1=y, 2=z)."""
    radial_axes = [a for a in range(3) if a != axis]
    r_dist = np.sqrt(pos[:, radial_axes[0]] ** 2 + pos[:, radial_axes[1]] ** 2) - radius
    h_dist = np.abs(pos[:, axis]) - half_height
    outside = np.sqrt(np.maximum(r_dist, 0) ** 2 + np.maximum(h_dist, 0) ** 2)
    inside = np.minimum(np.maximum(r_dist, h_dist), 0)
    return outside + inside


def sdf_cylinder(pos, radius=0.5, half_height=1.0):
    """Capped cylinder along z-axis."""
    return _sdf_capped_cylinder(pos, radius, half_height, axis=2), None


def sdf_mechanical_part(pos):
    """Box ∪ through-rod − 4 cylindrical holes. Typical mechanical part."""
    d_box, _ = sdf_box(pos, np.array([0.4, 0.4, 0.4]))
    d_rod = _sdf_capped_cylinder(pos, radius=0.2, half_height=1.0, axis=0)
    d_union = np.minimum(d_box, d_rod)
    for ox, oy in [(0.18, 0.18), (0.18, -0.18), (-0.18, 0.18), (-0.18, -0.18)]:
        shifted = pos - np.array([ox, oy, 0.0])
        d_hole = _sdf_capped_cylinder(shifted, radius=0.1, half_height=0.5, axis=2)
        d_union = np.maximum(d_union, -d_hole)
    return d_union, None


def sdf_thin_shell_hemi(pos, R=1.0, wall=0.08):
    """Hemisphere shell — thin wall, open boundary, inner surface visible."""
    r = np.linalg.norm(pos, axis=1)
    d_shell = np.maximum(r - R, (R - wall) - r)
    d_plane = pos[:, 2]  # keep z < 0 half (lower hemisphere, interior visible)
    return np.maximum(d_shell, d_plane), None


# ═══════════════════════════════════════════════════════════════════
#  Mesh-based SDF (point-cloud-utils)
# ═══════════════════════════════════════════════════════════════════


class MeshSDF:
    """Wraps a triangle mesh as an SDF callable for isomesh."""

    def __init__(self, mesh_path, target_bbox=1.4):
        mesh = trimesh.load(str(mesh_path), force="mesh")
        # Center and normalize to [-target_bbox, target_bbox]
        center = mesh.bounds.mean(axis=0)
        mesh.vertices -= center
        scale = target_bbox / np.abs(mesh.vertices).max()
        mesh.vertices *= scale
        self.vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
        self.mesh = mesh

    def __call__(self, pos):
        sdf, _, _ = pcu.signed_distance_to_mesh(
            np.ascontiguousarray(pos, dtype=np.float64),
            self.vertices,
            self.faces,
        )
        return sdf.astype(np.float64), None


# ═══════════════════════════════════════════════════════════════════
#  Shape & Method Definitions
# ═══════════════════════════════════════════════════════════════════


def build_shapes():
    """Return ordered dict of benchmark shapes."""
    shapes = {
        # Category 1: Basic
        "sphere": {
            "func": sdf_sphere,
            "bbox": 1.5,
            "category": "basic",
            "desc": "Unit sphere",
        },
        "torus": {
            "func": sdf_torus,
            "bbox": 1.8,
            "category": "basic",
            "desc": "Torus (R=1.0, r=0.35)",
        },
        # Category 2: Sharp features
        "box": {
            "func": sdf_box,
            "bbox": 1.5,
            "category": "sharp",
            "desc": "Unit box",
        },
        "rotated_box": {
            "func": sdf_rotated_box,
            "bbox": 2.0,
            "category": "sharp",
            "desc": "Box rotated 45° around (1,1,1) axis",
        },
        "chamfered_sphere": {
            "func": sdf_chamfered_sphere,
            "bbox": 1.5,
            "category": "sharp",
            "desc": "Sphere ∩ cube (R=1.0, d=0.8)",
        },
        "cylinder": {
            "func": sdf_cylinder,
            "bbox": 1.5,
            "category": "sharp",
            "desc": "Capped cylinder (r=0.5, h=2.0)",
        },
        "mechanical_part": {
            "func": sdf_mechanical_part,
            "bbox": 1.3,
            "category": "sharp",
            "desc": "Box + rod − 4 holes",
        },
        # Category 3: Thin features
        "thin_shell_hemi": {
            "func": sdf_thin_shell_hemi,
            "bbox": 1.3,
            "category": "thin",
            "desc": "Hemisphere shell (wall=0.08)",
        },
    }

    # Category 4: Mesh-based
    if HAS_PCU:
        for name, filename in [("bunny", "bunny.obj"), ("simjeb_148", "148.obj")]:
            path = BENCHMARKS_DIR / filename
            if path.exists():
                shapes[name] = {
                    "func": MeshSDF(path),
                    "bbox": 1.5,
                    "category": "complex",
                    "desc": f"Mesh-based SDF ({filename})",
                }
            else:
                print(f"  [SKIP] {filename} not found at {path}")
    else:
        print("  [SKIP] point-cloud-utils not installed — mesh-based shapes disabled")

    return shapes


METHODS = {
    "pymcubes": {
        "label": "MC (res=65)",
        "type": "mc",
        "mc_res": 65,
    },
    "uniform": {
        "label": "DC Uniform (d=6)",
        "type": "isomesh",
        "min_depth": 6,
        "max_depth": 6,
        "adaptive": False,
    },
    "adaptive": {
        "label": "DC Adaptive (4→6)",
        "type": "isomesh",
        "min_depth": 4,
        "max_depth": 6,
        "adaptive": True,
    },
    "adaptive_fine": {
        "label": "DC Adaptive (4→7)",
        "type": "isomesh",
        "min_depth": 4,
        "max_depth": 7,
        "adaptive": True,
    },
}


# ═══════════════════════════════════════════════════════════════════
#  Extraction
# ═══════════════════════════════════════════════════════════════════


def extract_pymcubes(func, bbox, res):
    b = bbox
    lin = np.linspace(-b, b, res)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    t0 = time.perf_counter()
    vals, _ = func(pts)
    volume = vals.reshape(res, res, res)
    vertices, faces = mcubes.marching_cubes(volume, 0.0)
    elapsed = time.perf_counter() - t0

    vertices = vertices / (res - 1) * (2 * b) - b
    return vertices, faces.astype(np.int64), elapsed


def extract_isomesh(func, bbox, min_depth, max_depth, adaptive):
    b = bbox
    t0 = time.perf_counter()
    v, f = isomesh.extract(
        func=func,
        bbox_min=(-b, -b, -b),
        bbox_max=(b, b, b),
        min_depth=min_depth,
        max_depth=max_depth,
        adaptive=adaptive,
    )
    elapsed = time.perf_counter() - t0
    return v, f, elapsed


# ═══════════════════════════════════════════════════════════════════
#  Quality Metrics
# ═══════════════════════════════════════════════════════════════════


def triangle_quality(vertices, faces):
    """Min angle per triangle — mean, 5th percentile, minimum."""
    if len(faces) == 0:
        return {"min_angle_mean": 0.0, "min_angle_p5": 0.0, "min_angle_min": 0.0}

    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    e01, e02, e12 = v1 - v0, v2 - v0, v2 - v1

    def _angle(a, b):
        cos = np.sum(a * b, axis=1) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-15
        )
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    a0 = _angle(e01, e02)
    a1 = _angle(-e01, e12)
    a2 = _angle(-e02, -e12)
    min_angles = np.minimum(np.minimum(a0, a1), a2)

    return {
        "min_angle_mean": round(float(np.mean(min_angles)), 2),
        "min_angle_p5": round(float(np.percentile(min_angles, 5)), 2),
        "min_angle_min": round(float(np.min(min_angles)), 2),
    }


def sdf_error(vertices, func):
    """Surface accuracy: how close are vertices to the true zero-level set."""
    if len(vertices) == 0:
        return {"mean_sdf_err": float("inf"), "max_sdf_err": float("inf")}
    vals, _ = func(vertices)
    av = np.abs(vals)
    return {
        "mean_sdf_err": float(np.mean(av)),
        "max_sdf_err": float(np.max(av)),
    }


def analyze_mesh(vertices, faces, func=None):
    """Full mesh analysis — topology, quality, accuracy."""
    r = {"V": len(vertices), "F": len(faces)}

    if len(vertices) == 0 or len(faces) == 0:
        r.update(watertight=False, manifold=False)
        return r

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    r["watertight"] = bool(mesh.is_watertight)

    # Edge topology
    edges = {}
    for face in faces:
        for i in range(3):
            e = tuple(sorted((int(face[i]), int(face[(i + 1) % 3]))))
            edges[e] = edges.get(e, 0) + 1

    boundary = sum(1 for c in edges.values() if c == 1)
    non_manifold = sum(1 for c in edges.values() if c != 2)
    E = len(edges)

    r["manifold"] = non_manifold == 0
    r["boundary_edges"] = boundary
    r["non_manifold_edges"] = non_manifold
    r["euler"] = len(vertices) - E + len(faces)

    # Triangle quality
    r.update(triangle_quality(vertices, faces))

    # Surface accuracy
    if func is not None:
        r.update(sdf_error(vertices, func))

    return r


# ═══════════════════════════════════════════════════════════════════
#  Rendering (pyrender + PIL)
# ═══════════════════════════════════════════════════════════════════

METHOD_COLORS_RGB = {
    "pymcubes": (0.91, 0.66, 0.49),
    "uniform": (0.25, 0.70, 0.64),
    "adaptive": (0.40, 0.62, 0.74),
    "adaptive_fine": (0.52, 0.40, 0.48),
}

VIEW_ANGLES = {
    "thin_shell_hemi": (15, -50),
    "mechanical_part": (25, -45),
    "torus": (30, -60),
    "bunny": (25, 120),
}
DEFAULT_VIEW = (25, -60)

# Per-shape up-axis override (default: Z-up)
UP_VECTORS = {
    "bunny": np.array([0.0, 1.0, 0.0]),  # Y-up
}
DEFAULT_UP = np.array([0.0, 0.0, 1.0])  # Z-up

CELL_W, CELL_H = 500, 500
LABEL_H = 70
TITLE_H = 40


def _get_font(size=14):
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _look_at(eye, target, up=np.array([0.0, 0.0, 1.0])):
    """4x4 camera pose matrix (pyrender convention: camera looks along -Z)."""
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, up)
        rn = np.linalg.norm(right)
    right = right / rn
    true_up = np.cross(right, fwd)
    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = true_up
    mat[:3, 2] = -fwd
    mat[:3, 3] = eye
    return mat


def _cam_pose(center, extent, elev, azim, up=None):
    """Camera pose from elevation/azimuth angles, looking at center."""
    if up is None:
        up = DEFAULT_UP
    dist = extent * 1.8
    er, ar = np.radians(elev), np.radians(azim)
    # Compute eye position using the up-axis convention
    cos_e, sin_e = np.cos(er), np.sin(er)
    cos_a, sin_a = np.cos(ar), np.sin(ar)
    if up[1] > 0.5:  # Y-up
        offset = np.array([
            cos_e * cos_a,
            sin_e,
            cos_e * sin_a,
        ])
    else:  # Z-up (default)
        offset = np.array([
            cos_e * cos_a,
            cos_e * sin_a,
            sin_e,
        ])
    eye = center + dist * offset
    return _look_at(eye, center, up=up)


def _render_one(vertices, faces, color_rgb, renderer, elev=25, azim=-60,
                 up=None):
    """Render a single mesh via pyrender with 3-point lighting."""
    if up is None:
        up = DEFAULT_UP
    w, h = renderer.viewport_width, renderer.viewport_height
    if len(vertices) == 0 or len(faces) == 0:
        return Image.new("RGB", (w, h), (255, 255, 255))

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.fix_normals()
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[*color_rgb, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.8,
    )
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)

    scene = pyrender.Scene(
        bg_color=[1.0, 1.0, 1.0, 1.0],
        ambient_light=[0.3, 0.3, 0.3],
    )
    scene.add(pr_mesh)

    center = (mesh.bounds[0] + mesh.bounds[1]) / 2
    extent = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])

    # Camera
    cam_p = _cam_pose(center, extent, elev, azim, up=up)
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    scene.add(cam, pose=cam_p)

    # Key light — upper-left of camera
    key_p = _cam_pose(center, extent, elev + 30, azim - 40, up=up)
    scene.add(pyrender.DirectionalLight(
        color=[1.0, 0.98, 0.95], intensity=2.5,
    ), pose=key_p)

    # Fill light — lower-right of camera, cooler
    fill_p = _cam_pose(center, extent, elev - 10, azim + 55, up=up)
    scene.add(pyrender.DirectionalLight(
        color=[0.92, 0.95, 1.0], intensity=1.2,
    ), pose=fill_p)

    # Rim light — behind and above
    rim_p = _cam_pose(center, extent, elev + 50, azim + 160, up=up)
    scene.add(pyrender.DirectionalLight(
        color=[1.0, 1.0, 1.0], intensity=0.8,
    ), pose=rim_p)

    color_img, _ = renderer.render(scene)
    return Image.fromarray(color_img)


def _labeled_cell(img, lines, font):
    """Image + centered text label below."""
    w, h = img.size
    canvas = Image.new("RGB", (w, h + LABEL_H), (255, 255, 255))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    y = h + 4
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((w - tw) // 2, y), line, fill=(30, 30, 30), font=font)
        y += bbox[3] - bbox[1] + 3
    return canvas


def render_comparison(shape_name, shape_results, output_dir):
    """Per-shape side-by-side comparison (pyrender)."""
    n = len(shape_results)
    renderer = pyrender.OffscreenRenderer(CELL_W, CELL_H)
    font = _get_font(14)
    title_font = _get_font(20)
    elev, azim = VIEW_ANGLES.get(shape_name, DEFAULT_VIEW)
    up = UP_VECTORS.get(shape_name, DEFAULT_UP)

    cells = []
    for method_name, r in shape_results.items():
        m = r["metrics"]
        color = METHOD_COLORS_RGB.get(method_name, (0.5, 0.5, 0.5))
        img = _render_one(r["vertices"], r["faces"], color, renderer, elev, azim,
                          up=up)
        lines = [
            METHODS[method_name]["label"],
            f"V={m['V']:,}  F={m['F']:,}",
            f"t={r['time']:.3f}s",
        ]
        cells.append(_labeled_cell(img, lines, font))

    renderer.delete()

    gap = 6
    total_w = sum(c.width for c in cells) + gap * (n - 1)
    total_h = TITLE_H + cells[0].height
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))

    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), shape_name, font=title_font)
    tw = bbox[2] - bbox[0]
    draw.text(((total_w - tw) // 2, 6), shape_name, fill=(0, 0, 0), font=title_font)

    x = 0
    for cell in cells:
        canvas.paste(cell, (x, TITLE_H))
        x += cell.width + gap

    path = output_dir / f"comparison_{shape_name}.png"
    canvas.save(path)
    return path


def render_overview_grid(all_shape_results, method_names, output_dir):
    """All shapes x all methods overview grid (pyrender)."""
    G = 350
    G_LABEL = 40
    HDR_H = 30
    ROW_W = 160

    renderer = pyrender.OffscreenRenderer(G, G)
    font = _get_font(12)
    hdr_font = _get_font(14)

    shape_names = list(all_shape_results.keys())
    n_s, n_m = len(shape_names), len(method_names)
    cell_h = G + G_LABEL
    total_w = ROW_W + n_m * G
    total_h = HDR_H + n_s * cell_h

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Column headers
    for col, mname in enumerate(method_names):
        label = METHODS[mname]["label"]
        bb = draw.textbbox((0, 0), label, font=hdr_font)
        tw = bb[2] - bb[0]
        x = ROW_W + col * G + (G - tw) // 2
        draw.text((x, 6), label, fill=(0, 0, 0), font=hdr_font)

    for row, sname in enumerate(shape_names):
        y0 = HDR_H + row * cell_h
        elev, azim = VIEW_ANGLES.get(sname, DEFAULT_VIEW)
        up = UP_VECTORS.get(sname, DEFAULT_UP)

        # Row label
        bb = draw.textbbox((0, 0), sname, font=hdr_font)
        th = bb[3] - bb[1]
        draw.text((8, y0 + G // 2 - th // 2), sname, fill=(0, 0, 0), font=hdr_font)

        shape_res = all_shape_results[sname]
        for col, mname in enumerate(method_names):
            if mname not in shape_res:
                continue
            r = shape_res[mname]
            m = r["metrics"]
            color = METHOD_COLORS_RGB.get(mname, (0.5, 0.5, 0.5))
            img = _render_one(r["vertices"], r["faces"], color, renderer, elev, azim,
                              up=up)

            x = ROW_W + col * G
            canvas.paste(img, (x, y0))

            info = f"V={m['V']:,}  F={m['F']:,}"
            bb = draw.textbbox((0, 0), info, font=font)
            tw = bb[2] - bb[0]
            draw.text((x + (G - tw) // 2, y0 + G + 4), info, fill=(80, 80, 80), font=font)

    renderer.delete()

    path = output_dir / "overview_grid.png"
    canvas.save(path)
    return path


# ═══════════════════════════════════════════════════════════════════
#  Report
# ═══════════════════════════════════════════════════════════════════


def generate_markdown_report(all_results, output_dir):
    """Generate a summary markdown report."""
    lines = ["# Isomesh Benchmark Results\n"]

    # Group by category
    categories = {}
    for r in all_results:
        cat = r.get("category", "other")
        categories.setdefault(cat, []).append(r)

    cat_labels = {
        "basic": "Basic (Smooth)",
        "sharp": "Sharp Features",
        "thin": "Thin Features",
        "complex": "Complex Geometry (Mesh-based SDF)",
    }

    for cat, cat_label in cat_labels.items():
        results = categories.get(cat, [])
        if not results:
            continue

        lines.append(f"\n## {cat_label}\n")
        lines.append(
            "| Shape | Method | V | F | Time (s) | Manifold | Watertight | "
            "Euler | Mean SDF Err | Min∠ Mean | Min∠ P5 |"
        )
        lines.append(
            "|-------|--------|--:|--:|--------:|:--------:|:----------:|"
            "-----:|-----------:|--------:|------:|"
        )

        # Group by shape
        shapes_in_cat = {}
        for r in results:
            shapes_in_cat.setdefault(r["shape"], []).append(r)

        for sname, sresults in shapes_in_cat.items():
            for i, r in enumerate(sresults):
                shape_col = sname if i == 0 else ""
                ok_m = "✓" if r.get("manifold") else "✗"
                ok_w = "✓" if r.get("watertight") else "✗"
                sdf_e = f"{r.get('mean_sdf_err', 0):.2e}" if "mean_sdf_err" in r else "—"
                ang = f"{r.get('min_angle_mean', 0):.1f}°" if r.get("min_angle_mean") else "—"
                ang5 = f"{r.get('min_angle_p5', 0):.1f}°" if r.get("min_angle_p5") else "—"
                lines.append(
                    f"| {shape_col} | {r['method_label']} | "
                    f"{r['V']:,} | {r['F']:,} | {r['time']:.3f} | "
                    f"{ok_m} | {ok_w} | {r.get('euler', '?')} | "
                    f"{sdf_e} | {ang} | {ang5} |"
                )

    path = output_dir / "benchmark_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(description="Isomesh Benchmark Suite")
    p.add_argument("--shapes", nargs="*", help="Run only these shapes")
    p.add_argument("--methods", nargs="*", help="Run only these methods")
    p.add_argument("--no-render", action="store_true", help="Skip image rendering")
    p.add_argument("--no-stl", action="store_true", help="Skip STL export")
    p.add_argument("--output", type=str, default=None, help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output) if args.output else OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building benchmark shapes...")
    shapes = build_shapes()

    # Filter
    if args.shapes:
        shapes = {k: v for k, v in shapes.items() if k in args.shapes}
    methods = METHODS
    if args.methods:
        methods = {k: v for k, v in METHODS.items() if k in args.methods}

    print(f"  Shapes:  {', '.join(shapes.keys())}")
    print(f"  Methods: {', '.join(methods.keys())}")
    print()

    # Header
    hdr = (
        f"{'Shape':<20} {'Method':<22} {'V':>7} {'F':>7} {'Time':>8} "
        f"{'Mfld':>5} {'Water':>5} {'Euler':>6} {'MeanSDF':>10} {'Min∠':>7}"
    )
    sep = "=" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    all_results = []
    all_shape_results = {}  # for rendering

    for shape_name, shape_info in shapes.items():
        func = shape_info["func"]
        bbox = shape_info["bbox"]
        cat = shape_info["category"]
        shape_results = {}

        for method_name, method_cfg in methods.items():
            try:
                if method_cfg["type"] == "mc":
                    v, f, t = extract_pymcubes(func, bbox, method_cfg["mc_res"])
                else:
                    v, f, t = extract_isomesh(
                        func, bbox,
                        method_cfg["min_depth"],
                        method_cfg["max_depth"],
                        method_cfg["adaptive"],
                    )

                metrics = analyze_mesh(v, f, func=func)
                metrics["time"] = t
                metrics["method"] = method_name
                metrics["method_label"] = method_cfg["label"]
                metrics["shape"] = shape_name
                metrics["category"] = cat

                shape_results[method_name] = {
                    "vertices": v,
                    "faces": f,
                    "metrics": metrics,
                    "time": t,
                }

                # STL export
                if not args.no_stl and len(v) > 0 and len(f) > 0:
                    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
                    mesh.export(str(output_dir / f"{shape_name}_{method_name}.stl"))

                # Console row
                row_shape = shape_name if method_name == list(methods.keys())[0] else ""
                ok_m = "✓" if metrics.get("manifold") else "✗"
                ok_w = "✓" if metrics.get("watertight") else "✗"
                sdf_e = (
                    f"{metrics['mean_sdf_err']:.2e}"
                    if "mean_sdf_err" in metrics
                    else "—"
                )
                ang = (
                    f"{metrics['min_angle_mean']:.1f}°"
                    if metrics.get("min_angle_mean")
                    else "—"
                )
                print(
                    f"{row_shape:<20} {method_cfg['label']:<22} "
                    f"{metrics['V']:>7} {metrics['F']:>7} {t:>7.3f}s "
                    f"{ok_m:>5} {ok_w:>5} {metrics.get('euler', '?'):>6} "
                    f"{sdf_e:>10} {ang:>7}"
                )

                all_results.append(metrics)

            except Exception as e:
                print(
                    f"{shape_name:<20} {method_cfg['label']:<22} "
                    f"{'FAILED':>7} — {e}"
                )

        all_shape_results[shape_name] = shape_results

        # Per-shape comparison image
        if not args.no_render and HAS_PYRENDER and shape_results:
            img = render_comparison(shape_name, shape_results, output_dir)
            print(f"  → {img.name}")

        print("-" * len(hdr))

    # Overview grid
    if not args.no_render and HAS_PYRENDER and all_shape_results:
        active_methods = [m for m in methods if any(m in sr for sr in all_shape_results.values())]
        grid_path = render_overview_grid(all_shape_results, active_methods, output_dir)
        print(f"\nOverview grid: {grid_path}")

    # JSON results (strip numpy types)
    json_results = []
    for r in all_results:
        jr = {}
        for k, v in r.items():
            if isinstance(v, (np.integer, np.int64)):
                jr[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                jr[k] = float(v)
            elif isinstance(v, np.bool_):
                jr[k] = bool(v)
            else:
                jr[k] = v
        json_results.append(jr)

    json_path = output_dir / "benchmark_results.json"
    with open(json_path, "w") as fp:
        json.dump(json_results, fp, indent=2)

    # Markdown report
    md_path = generate_markdown_report(all_results, output_dir)

    # Summary
    print_summary(all_results, methods)
    print(f"\nOutput: {output_dir}/")
    print(f"  JSON:     {json_path.name}")
    print(f"  Report:   {md_path.name}")


def print_summary(all_results, methods):
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for method_name, method_cfg in methods.items():
        results = [r for r in all_results if r["method"] == method_name]
        if not results:
            continue

        n = len(results)
        manifold = sum(1 for r in results if r.get("manifold"))
        watertight = sum(1 for r in results if r.get("watertight"))
        total_time = sum(r.get("time", 0) for r in results)
        total_v = sum(r["V"] for r in results)
        total_f = sum(r["F"] for r in results)

        sdf_errs = [
            r["mean_sdf_err"]
            for r in results
            if "mean_sdf_err" in r and r["mean_sdf_err"] < float("inf")
        ]
        avg_sdf = np.mean(sdf_errs) if sdf_errs else float("nan")

        angles = [
            r["min_angle_mean"]
            for r in results
            if r.get("min_angle_mean", 0) > 0
        ]
        avg_angle = np.mean(angles) if angles else float("nan")

        print(f"\n  {method_cfg['label']}:")
        print(f"    Topology:   {manifold}/{n} manifold, {watertight}/{n} watertight")
        print(f"    Total mesh: {total_v:,} V, {total_f:,} F")
        print(f"    Total time: {total_time:.2f}s")
        print(f"    Avg SDF error: {avg_sdf:.2e}")
        print(f"    Avg min-angle: {avg_angle:.1f}°")


if __name__ == "__main__":
    main()
