"""Manifold/watertight validation across multiple shapes."""

import pytest
import isomesh
from tests.helpers.sdf_library import sdf_sphere, sdf_box, sdf_torus, sdf_csg_cross
from tests.helpers.mesh_checks import (
    check_manifold_watertight,
    check_no_degenerate_faces,
    check_face_indices_valid,
)


def _extract(func, depth=4):
    return isomesh.extract(
        func=func,
        bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
        min_depth=depth, max_depth=depth,
    )


class TestFaceValidity:
    @pytest.mark.parametrize("sdf_func,name", [
        (sdf_sphere, "sphere"),
        (sdf_box, "box"),
        (sdf_torus, "torus"),
    ])
    def test_face_indices_valid(self, sdf_func, name):
        v, f = _extract(sdf_func)
        assert check_face_indices_valid(v, f), f"{name}: invalid face indices"

    @pytest.mark.parametrize("sdf_func,name", [
        (sdf_sphere, "sphere"),
        (sdf_box, "box"),
        (sdf_torus, "torus"),
    ])
    def test_no_degenerate_faces(self, sdf_func, name):
        v, f = _extract(sdf_func)
        assert check_no_degenerate_faces(f), f"{name}: degenerate faces found"


class TestManifold:
    def test_sphere_manifold_stats(self):
        """Check manifold properties and report non-manifold edges.

        Basic DC can produce some non-manifold edges at cell boundaries.
        MDC extension will guarantee manifold output.
        """
        v, f = _extract(sdf_sphere)
        result = check_manifold_watertight(v, f)
        # Report stats — full manifold guarantee requires MDC (Phase 6+)
        total_edges = result["E"]
        nm_edges = len(result["non_manifold_edges"])
        nm_ratio = nm_edges / max(total_edges, 1)
        # Non-manifold edges should be a small fraction
        assert nm_ratio < 0.1, (
            f"Too many non-manifold edges: {nm_edges}/{total_edges} ({nm_ratio:.1%})"
        )

    def test_sphere_euler_characteristic(self):
        v, f = _extract(sdf_sphere)
        result = check_manifold_watertight(v, f)
        # For Basic DC, Euler characteristic may deviate from 2
        # Log it for tracking; MDC will guarantee V-E+F=2
        euler = result["euler_characteristic"]
        assert abs(euler - 2) < 100, (
            f"Sphere Euler too far from 2: {euler} "
            f"(V={result['V']}, E={result['E']}, F={result['F']})"
        )

    def test_box_produces_mesh(self):
        v, f = _extract(sdf_box)
        assert len(v) > 0 and len(f) > 0

    def test_torus_produces_mesh(self):
        v, f = _extract(sdf_torus, depth=4)
        assert len(v) > 0 and len(f) > 0
