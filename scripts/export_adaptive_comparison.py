"""Export STL meshes comparing uniform vs adaptive mode for visual evaluation."""

import numpy as np
import trimesh
import isomesh

from tests.helpers.sdf_library import (
    sdf_sphere, sdf_box, sdf_torus,
    sdf_two_spheres_touching, sdf_hollow_cube, sdf_cross_pipes, sdf_csg_cross,
)
from tests.helpers.mesh_checks import check_manifold_watertight

OUTPUT = "output/adaptive_comparison"

SHAPES = [
    ("sphere",      sdf_sphere,              (-2,-2,-2), (2,2,2)),
    ("torus",       sdf_torus,               (-2,-2,-2), (2,2,2)),
    ("box",         sdf_box,                 (-2,-2,-2), (2,2,2)),
    ("csg_cross",   sdf_csg_cross,           (-2,-2,-2), (2,2,2)),
    ("two_spheres", sdf_two_spheres_touching, (-3,-3,-3), (3,3,3)),
]

CONFIGS = [
    # (label, min_depth, max_depth, adaptive)
    ("uniform_d5",   5, 5, False),
    ("uniform_d6",   6, 6, False),
    ("adaptive_3_5", 3, 5, True),
    ("adaptive_3_6", 3, 6, True),
    ("adaptive_3_7", 3, 7, True),
]


def save_stl(vertices, faces, path):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(path)


def main():
    import os
    os.makedirs(OUTPUT, exist_ok=True)

    report_lines = ["# Adaptive vs Uniform Comparison\n"]
    report_lines.append(f"| Shape | Config | Vertices | Faces | Manifold | Watertight | File |")
    report_lines.append(f"|-------|--------|----------|-------|----------|------------|------|")

    for shape_name, sdf_fn, bmin, bmax in SHAPES:
        for label, mind, maxd, adaptive in CONFIGS:
            try:
                v, f = isomesh.extract(
                    func=sdf_fn,
                    bbox_min=bmin, bbox_max=bmax,
                    min_depth=mind, max_depth=maxd,
                    adaptive=adaptive,
                )
            except Exception as e:
                report_lines.append(f"| {shape_name} | {label} | ERROR | - | - | - | - |")
                print(f"  ERROR {shape_name}/{label}: {e}")
                continue

            if len(v) == 0:
                report_lines.append(f"| {shape_name} | {label} | 0 | 0 | - | - | - |")
                continue

            r = check_manifold_watertight(v, f)
            mf = "✅" if r["is_manifold"] else "❌"
            wt = "✅" if r["is_watertight"] else "❌"

            fname = f"{shape_name}_{label}.stl"
            fpath = f"{OUTPUT}/{fname}"
            save_stl(v, f, fpath)

            report_lines.append(
                f"| {shape_name} | {label} | {len(v)} | {len(f)} | {mf} | {wt} | {fname} |"
            )
            print(f"  {shape_name}/{label}: V={len(v):6d} F={len(f):6d} mf={r['is_manifold']} wt={r['is_watertight']}")

    report = "\n".join(report_lines) + "\n"
    report_path = f"{OUTPUT}/COMPARISON_REPORT.md"
    with open(report_path, "w") as fp:
        fp.write(report)
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
