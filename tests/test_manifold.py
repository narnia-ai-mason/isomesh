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


class TestManifoldWatertight:
    def test_sphere_manifold(self):
        v, f = _extract(sdf_sphere)
        result = check_manifold_watertight(v, f)
        assert result["is_manifold"], (
            f"Sphere non-manifold: {len(result['non_manifold_edges'])} edges, "
            f"e.g. {result['non_manifold_edges'][:3]}"
        )

    def test_sphere_watertight(self):
        v, f = _extract(sdf_sphere)
        result = check_manifold_watertight(v, f)
        assert result["is_watertight"], (
            f"Sphere not watertight: {len(result['boundary_edges'])} boundary edges"
        )

    def test_sphere_euler_characteristic(self):
        """Genus-0 closed surface: V - E + F = 2."""
        v, f = _extract(sdf_sphere)
        result = check_manifold_watertight(v, f)
        assert result["euler_characteristic"] == 2, (
            f"Sphere Euler={result['euler_characteristic']} "
            f"(V={result['V']}, E={result['E']}, F={result['F']})"
        )

    def test_sphere_consistent_winding(self):
        v, f = _extract(sdf_sphere)
        result = check_manifold_watertight(v, f)
        assert result["consistent_winding"], "Sphere has inconsistent face winding"

    def test_box_manifold(self):
        v, f = _extract(sdf_box)
        result = check_manifold_watertight(v, f)
        assert result["is_manifold"], (
            f"Box non-manifold: {len(result['non_manifold_edges'])} edges"
        )

    def test_box_watertight(self):
        v, f = _extract(sdf_box)
        result = check_manifold_watertight(v, f)
        assert result["is_watertight"], (
            f"Box not watertight: {len(result['boundary_edges'])} boundary edges"
        )

    def test_torus_produces_mesh(self):
        v, f = _extract(sdf_torus, depth=4)
        assert len(v) > 0 and len(f) > 0
