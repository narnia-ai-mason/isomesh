#!/usr/bin/env python3
"""
Comprehensive isomesh benchmark suite.

Compares isomesh (Dual Contouring) vs PyMCubes (Marching Cubes) across
standard benchmark shapes from the isosurface extraction literature.

Shapes:
  Category 1 — Basic analytic (ground truth available)
    sphere, torus, ellipsoid

  Category 2 — Sharp features (CAD-like)
    box, csg_cross (3-bar union), csg_intersection, chamfered_box

  Category 3 — TPMS (Triply Periodic Minimal Surfaces)
    gyroid, schwarz_p, schwarz_d

  Category 4 — Complex topology & stress tests
    genus2, tanglecube, heart, barth_sextic

  Category 5 — Thin features / robustness
    thin_plate, thin_shell
"""

import time
import json
import numpy as np
import trimesh
import mcubes
import isomesh

OUTPUT = "/Users/minsikseo/workspace/2_work/1_active/aslanx-flow/1_code/isomesh/output"

# ─────────────────────────── SDF library ───────────────────────────

def sdf_sphere(pos, radius=1.0):
    norms = np.linalg.norm(pos, axis=1)
    values = norms - radius
    safe = np.where(norms > 1e-12, norms, 1.0)
    return values, pos / safe[:, None]

def sdf_torus(pos, R=1.0, r=0.35):
    xy = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
    return np.sqrt((xy - R)**2 + pos[:, 2]**2) - r, None

def sdf_ellipsoid(pos, a=1.0, b=0.6, c=0.4):
    # Approximate SDF for ellipsoid (exact near surface)
    scaled = pos / np.array([a, b, c])
    norms = np.linalg.norm(scaled, axis=1)
    values = norms - 1.0
    # Scale by average radius for better SDF approximation
    values *= (a * b * c) ** (1.0/3.0)
    return values, None

def sdf_box(pos, half=None):
    if half is None:
        half = np.array([1.0, 1.0, 1.0])
    half = np.asarray(half)
    q = np.abs(pos) - half
    qm = np.maximum(q, 0.0)
    return np.linalg.norm(qm, axis=1) + np.minimum(np.max(q, axis=1), 0.0), None

def sdf_csg_cross(pos):
    """Union of 3 orthogonal bars."""
    d1, _ = sdf_box(pos, np.array([0.25, 0.25, 1.2]))
    d2, _ = sdf_box(pos, np.array([1.2, 0.25, 0.25]))
    d3, _ = sdf_box(pos, np.array([0.25, 1.2, 0.25]))
    return np.minimum(np.minimum(d1, d2), d3), None

def sdf_csg_intersection(pos):
    """Intersection of sphere and box — rounded cube."""
    ds, _ = sdf_sphere(pos, radius=1.3)
    db, _ = sdf_box(pos, np.array([1.0, 1.0, 1.0]))
    return np.maximum(ds, db), None

def sdf_chamfered_box(pos):
    """Box with chamfered edges (intersection of box and larger sphere)."""
    db, _ = sdf_box(pos, np.array([0.8, 0.8, 0.8]))
    ds, _ = sdf_sphere(pos, radius=1.2)
    return np.maximum(db, ds), None

# --- TPMS ---
def sdf_gyroid(pos, scale=2.0 * np.pi):
    x, y, z = pos[:, 0] * scale, pos[:, 1] * scale, pos[:, 2] * scale
    values = np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)
    return values, None

def sdf_schwarz_p(pos, scale=2.0 * np.pi):
    x, y, z = pos[:, 0] * scale, pos[:, 1] * scale, pos[:, 2] * scale
    return np.cos(x) + np.cos(y) + np.cos(z), None

def sdf_schwarz_d(pos, scale=2.0 * np.pi):
    x, y, z = pos[:, 0] * scale, pos[:, 1] * scale, pos[:, 2] * scale
    values = (np.sin(x) * np.sin(y) * np.sin(z)
              + np.sin(x) * np.cos(y) * np.cos(z)
              + np.cos(x) * np.sin(y) * np.cos(z)
              + np.cos(x) * np.cos(y) * np.sin(z))
    return values, None

# --- Complex topology ---
def sdf_genus2(pos):
    """Genus-2 surface: 2-(cos(x)+cos(y)+cos(z)) ... algebraic approximation."""
    x, y, z = pos[:, 0] * 2, pos[:, 1] * 2, pos[:, 2] * 2
    # Implicit genus-2: two merged tori
    R, r = 0.8, 0.3
    # Torus 1 centered at (0.5, 0, 0)
    p1 = pos.copy(); p1[:, 0] -= 0.45
    xy1 = np.sqrt(p1[:, 0]**2 + p1[:, 1]**2)
    d1 = np.sqrt((xy1 - R)**2 + p1[:, 2]**2) - r
    # Torus 2 centered at (-0.5, 0, 0)
    p2 = pos.copy(); p2[:, 0] += 0.45
    xy2 = np.sqrt(p2[:, 0]**2 + p2[:, 1]**2)
    d2 = np.sqrt((xy2 - R)**2 + p2[:, 2]**2) - r
    # Smooth union
    k = 0.15
    h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return d1 * h + d2 * (1 - h) - k * h * (1 - h), None

def sdf_tanglecube(pos):
    """Tanglecube: x^4 - 5x^2 + y^4 - 5y^2 + z^4 - 5z^2 + 11.8 = 0."""
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    return x**4 - 5*x**2 + y**4 - 5*y**2 + z**4 - 5*z**2 + 11.8, None

def sdf_heart(pos):
    """Heart surface: (x^2 + 9/4*y^2 + z^2 - 1)^3 - x^2*z^3 - 9/80*y^2*z^3 = 0."""
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    t = x**2 + 9.0/4.0 * y**2 + z**2 - 1.0
    return t**3 - x**2 * z**3 - 9.0/80.0 * y**2 * z**3, None

def sdf_barth_sextic(pos):
    """Barth sextic: degree-6 algebraic surface with icosahedral symmetry."""
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    x2, y2, z2 = x**2, y**2, z**2
    w2 = 1.0  # homogeneous coordinate w=1
    s = x2 + y2 + z2
    values = (4 * (phi**2 * x2 - y2) * (phi**2 * y2 - z2) * (phi**2 * z2 - x2)
              - (1 + 2*phi) * (s - w2)**2 * w2**2) # but we want it as implicit
    # Normalize for better behavior
    return values * 0.01, None

def sdf_thin_plate(pos, thickness=0.08):
    half = np.array([0.9, 0.9, thickness / 2.0])
    return sdf_box(pos, half)

def sdf_thin_shell(pos, R=1.0, wall=0.08):
    r = np.linalg.norm(pos, axis=1)
    return np.maximum(r - R, (R - wall) - r), None


# ─────────────────────── Benchmark definitions ─────────────────────

BENCHMARKS = {
    # Category 1: Basic analytic
    "sphere":     {"func": sdf_sphere, "bbox": 1.5, "category": "basic"},
    "torus":      {"func": sdf_torus, "bbox": 1.8, "category": "basic"},
    "ellipsoid":  {"func": sdf_ellipsoid, "bbox": 1.5, "category": "basic"},

    # Category 2: Sharp features
    "box":              {"func": sdf_box, "bbox": 1.5, "category": "sharp"},
    "csg_cross":        {"func": sdf_csg_cross, "bbox": 1.8, "category": "sharp"},
    "csg_intersection": {"func": sdf_csg_intersection, "bbox": 1.8, "category": "sharp"},
    "chamfered_box":    {"func": sdf_chamfered_box, "bbox": 1.5, "category": "sharp"},

    # Category 3: TPMS
    "gyroid":    {"func": sdf_gyroid, "bbox": 1.0, "category": "tpms"},
    "schwarz_p": {"func": sdf_schwarz_p, "bbox": 1.0, "category": "tpms"},
    "schwarz_d": {"func": sdf_schwarz_d, "bbox": 1.0, "category": "tpms"},

    # Category 4: Complex topology
    "genus2":       {"func": sdf_genus2, "bbox": 2.0, "category": "topology"},
    "tanglecube":   {"func": sdf_tanglecube, "bbox": 2.2, "category": "topology"},
    "heart":        {"func": sdf_heart, "bbox": 1.5, "category": "topology"},
    "barth_sextic": {"func": sdf_barth_sextic, "bbox": 1.5, "category": "topology"},

    # Category 5: Thin features
    "thin_plate": {"func": sdf_thin_plate, "bbox": 1.5, "category": "thin"},
    "thin_shell": {"func": sdf_thin_shell, "bbox": 1.5, "category": "thin"},
}


# ─────────────────────── PyMCubes wrapper ──────────────────────────

def run_pymcubes(func_scalar, bbox, resolution):
    """Run PyMCubes on a uniform grid."""
    n = resolution
    b = bbox
    lin = np.linspace(-b, b, n)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    vals, _ = func_scalar(pts)
    volume = vals.reshape(n, n, n)

    t0 = time.perf_counter()
    vertices, faces = mcubes.marching_cubes(volume, 0.0)
    elapsed = time.perf_counter() - t0

    # Rescale vertices from [0, n-1] to [-b, b]
    vertices = vertices / (n - 1) * (2 * b) - b

    return vertices, faces.astype(np.int64), elapsed


def run_isomesh(func, bbox, depth):
    """Run isomesh extraction."""
    b = bbox
    t0 = time.perf_counter()
    v, f = isomesh.extract(
        func=func,
        bbox_min=(-b, -b, -b), bbox_max=(b, b, b),
        min_depth=depth, max_depth=depth,
    )
    elapsed = time.perf_counter() - t0
    return v, f, elapsed


# ──────────────────────── Analysis helpers ─────────────────────────

def analyze_mesh(vertices, faces, name=""):
    """Compute mesh quality metrics."""
    if len(vertices) == 0 or len(faces) == 0:
        return {"name": name, "V": 0, "F": 0, "watertight": False, "manifold": False}

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    result = {
        "name": name,
        "V": len(vertices),
        "F": len(faces),
        "watertight": bool(mesh.is_watertight),
    }

    # Edge manifold check
    edges = {}
    for fi, face in enumerate(faces):
        for i in range(3):
            e = tuple(sorted((int(face[i]), int(face[(i+1) % 3]))))
            edges[e] = edges.get(e, 0) + 1
    non_manifold = sum(1 for c in edges.values() if c != 2)
    boundary = sum(1 for c in edges.values() if c == 1)
    result["manifold"] = non_manifold == 0
    result["boundary_edges"] = boundary
    result["non_manifold_edges"] = non_manifold
    result["E"] = len(edges)
    result["euler"] = len(vertices) - len(edges) + len(faces)

    return result


# ────────────────────────── Main ───────────────────────────────────

def main():
    # Matching resolutions: isomesh depth 5 = 32 cells/axis, PyMCubes 33 grid points
    # isomesh depth 6 = 64 cells/axis, PyMCubes 65 grid points
    configs = [
        {"depth": 5, "mc_res": 33, "label": "low"},
        {"depth": 6, "mc_res": 65, "label": "high"},
    ]

    all_results = []

    print("=" * 100)
    print(f"{'Shape':<20} {'Method':<10} {'Res':<6} {'V':>7} {'F':>7} {'Time':>8} {'Manifold':>9} {'Water':>6} {'Euler':>6}")
    print("=" * 100)

    for bname, binfo in BENCHMARKS.items():
        func = binfo["func"]
        bbox = binfo["bbox"]
        cat = binfo["category"]

        for cfg in configs:
            # --- isomesh ---
            try:
                v_iso, f_iso, t_iso = run_isomesh(func, bbox, cfg["depth"])
                m_iso = analyze_mesh(v_iso, f_iso, f"isomesh_{bname}")
                m_iso["time"] = t_iso
                m_iso["method"] = "isomesh"
                m_iso["shape"] = bname
                m_iso["category"] = cat
                m_iso["resolution"] = cfg["label"]

                # Export STL
                if len(v_iso) > 0 and len(f_iso) > 0:
                    mesh = trimesh.Trimesh(vertices=v_iso, faces=f_iso, process=False)
                    mesh.export(f"{OUTPUT}/{bname}_isomesh_{cfg['label']}.stl")

                ok_m = "✓" if m_iso["manifold"] else "✗"
                ok_w = "✓" if m_iso["watertight"] else "✗"
                print(f"{bname:<20} {'isomesh':<10} {cfg['label']:<6} {m_iso['V']:>7} {m_iso['F']:>7} {t_iso:>7.3f}s {ok_m:>9} {ok_w:>6} {m_iso['euler']:>6}")
                all_results.append(m_iso)
            except Exception as e:
                print(f"{bname:<20} {'isomesh':<10} {cfg['label']:<6} {'FAILED':>7} — {e}")

            # --- PyMCubes ---
            try:
                v_mc, f_mc, t_mc = run_pymcubes(func, bbox, cfg["mc_res"])
                m_mc = analyze_mesh(v_mc, f_mc, f"pymcubes_{bname}")
                m_mc["time"] = t_mc
                m_mc["method"] = "pymcubes"
                m_mc["shape"] = bname
                m_mc["category"] = cat
                m_mc["resolution"] = cfg["label"]

                if len(v_mc) > 0 and len(f_mc) > 0:
                    mesh = trimesh.Trimesh(vertices=v_mc, faces=f_mc, process=False)
                    mesh.export(f"{OUTPUT}/{bname}_pymcubes_{cfg['label']}.stl")

                ok_m = "✓" if m_mc["manifold"] else "✗"
                ok_w = "✓" if m_mc["watertight"] else "✗"
                print(f"{'':<20} {'pymcubes':<10} {cfg['label']:<6} {m_mc['V']:>7} {m_mc['F']:>7} {t_mc:>7.3f}s {ok_m:>9} {ok_w:>6} {m_mc['euler']:>6}")
                all_results.append(m_mc)
            except Exception as e:
                print(f"{'':<20} {'pymcubes':<10} {cfg['label']:<6} {'FAILED':>7} — {e}")

        print("-" * 100)

    # Save results as JSON
    with open(f"{OUTPUT}/benchmark_results.json", "w") as fp:
        json.dump(all_results, fp, indent=2, default=str)

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    for cfg in configs:
        label = cfg["label"]
        iso_results = [r for r in all_results if r["method"] == "isomesh" and r["resolution"] == label]
        mc_results = [r for r in all_results if r["method"] == "pymcubes" and r["resolution"] == label]

        iso_manifold = sum(1 for r in iso_results if r["manifold"])
        iso_water = sum(1 for r in iso_results if r["watertight"])
        mc_manifold = sum(1 for r in mc_results if r["manifold"])
        mc_water = sum(1 for r in mc_results if r["watertight"])

        iso_time = sum(r["time"] for r in iso_results)
        mc_time = sum(r["time"] for r in mc_results)

        print(f"\n[{label} resolution]")
        print(f"  isomesh:  {iso_manifold}/{len(iso_results)} manifold, {iso_water}/{len(iso_results)} watertight, total {iso_time:.2f}s")
        print(f"  pymcubes: {mc_manifold}/{len(mc_results)} manifold, {mc_water}/{len(mc_results)} watertight, total {mc_time:.2f}s (MC extraction only, excludes grid eval)")

    print(f"\nAll STL files exported to {OUTPUT}/")
    print(f"Full results in {OUTPUT}/benchmark_results.json")


if __name__ == "__main__":
    main()
