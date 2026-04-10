"""MDC (Manifold Dual Contouring) specific tests.

Tests that MDC correctly handles multi-component cells and maintains
manifold/watertight guarantees on adversarial shapes, while preserving
accuracy on standard shapes (non-regression).
"""

import numpy as np
import pytest

import isomesh
from tests.helpers.sdf_library import (
    sdf_sphere, sdf_box, sdf_torus,
    sdf_two_spheres_touching, sdf_hollow_cube, sdf_cross_pipes,
)
from tests.helpers.mesh_checks import (
    check_manifold_watertight, check_no_degenerate_faces, check_face_indices_valid,
)
from tests.helpers.metrics import hausdorff_to_analytic, volume_relative_error


EXTRACT_KWARGS = dict(
    bbox_min=(-3.0, -3.0, -3.0),
    bbox_max=(3.0, 3.0, 3.0),
    min_depth=4,
    max_depth=4,
)


class TestMDCManifold:
    """Manifold/watertight guarantees on adversarial shapes."""

    def test_two_spheres_manifold(self):
        v, f = isomesh.extract(func=sdf_two_spheres_touching, **EXTRACT_KWARGS)
        assert len(v) > 0
        assert len(f) > 0
        result = check_manifold_watertight(v, f)
        assert result["is_manifold"], f"non-manifold edges: {result['non_manifold_edges']}"
        assert result["is_watertight"], f"boundary edges: {result['boundary_edges']}"

    def test_two_spheres_no_degenerate(self):
        v, f = isomesh.extract(func=sdf_two_spheres_touching, **EXTRACT_KWARGS)
        assert check_no_degenerate_faces(f)
        assert check_face_indices_valid(v, f)

    def test_hollow_cube_manifold(self):
        # wall=0.3 ensures wall > cell_size at depth 5 (cell=4/32=0.125)
        v, f = isomesh.extract(
            func=lambda pos: sdf_hollow_cube(pos, wall=0.3),
            bbox_min=(-2.0, -2.0, -2.0),
            bbox_max=(2.0, 2.0, 2.0),
            min_depth=5, max_depth=5,
        )
        assert len(v) > 0
        result = check_manifold_watertight(v, f)
        assert result["is_manifold"], f"non-manifold edges: {result['non_manifold_edges']}"
        assert result["is_watertight"], f"boundary edges: {result['boundary_edges']}"

    def test_hollow_cube_no_degenerate(self):
        v, f = isomesh.extract(
            func=lambda pos: sdf_hollow_cube(pos, wall=0.3),
            bbox_min=(-2.0, -2.0, -2.0),
            bbox_max=(2.0, 2.0, 2.0),
            min_depth=5, max_depth=5,
        )
        assert check_no_degenerate_faces(f)
        assert check_face_indices_valid(v, f)

    def test_cross_pipes_produces_mesh(self):
        """Cross-pipes produces a valid mesh. Full manifold guarantee for
        union-of-cylinders requires MDC Phase 2 (manifold criterion check)."""
        v, f = isomesh.extract(
            func=sdf_cross_pipes,
            bbox_min=(-2.0, -2.0, -2.0),
            bbox_max=(2.0, 2.0, 2.0),
            min_depth=4, max_depth=4,
        )
        assert len(v) > 0
        assert len(f) > 0

    def test_cross_pipes_no_degenerate(self):
        v, f = isomesh.extract(
            func=sdf_cross_pipes,
            bbox_min=(-2.0, -2.0, -2.0),
            bbox_max=(2.0, 2.0, 2.0),
            min_depth=4, max_depth=4,
        )
        assert check_no_degenerate_faces(f)
        assert check_face_indices_valid(v, f)


class TestMDCNonRegression:
    """Verify MDC doesn't degrade accuracy on standard shapes."""

    def test_sphere_accuracy(self):
        v, f = isomesh.extract(
            func=sdf_sphere,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=4, max_depth=4,
        )
        h = hausdorff_to_analytic(v, sdf_sphere)
        cell_size = 4.0 / (2 ** 4)
        assert h < 2 * cell_size, f"Hausdorff {h} > 2*cell_size {2*cell_size}"

    def test_sphere_volume(self):
        v, f = isomesh.extract(
            func=sdf_sphere,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=4, max_depth=4,
        )
        err = volume_relative_error(v, f, 4.0 / 3.0 * np.pi)
        assert err < 0.15, f"Volume error {err:.3f} > 0.15"

    def test_box_sharp_features(self):
        v, f = isomesh.extract(
            func=sdf_box,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=4, max_depth=4,
        )
        # Check that box corners are detected
        corners = np.array([
            [s0, s1, s2]
            for s0 in [-1, 1] for s1 in [-1, 1] for s2 in [-1, 1]
        ], dtype=float)
        found = 0
        for c in corners:
            dists = np.linalg.norm(v - c, axis=1)
            if dists.min() < 0.3:
                found += 1
        assert found >= 6, f"Only found {found}/8 box corners"

    @pytest.mark.parametrize("sdf_func,name", [
        (sdf_sphere, "sphere"), (sdf_box, "box"), (sdf_torus, "torus"),
    ])
    def test_manifold_preserved(self, sdf_func, name):
        v, f = isomesh.extract(
            func=sdf_func,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=4, max_depth=4,
        )
        result = check_manifold_watertight(v, f)
        assert result["is_manifold"], f"{name}: non-manifold edges"
        assert result["is_watertight"], f"{name}: boundary edges"
        assert result["consistent_winding"], f"{name}: inconsistent winding"

    def test_determinism(self):
        """MDC must remain deterministic under Rayon."""
        results = []
        for _ in range(3):
            v, f = isomesh.extract(
                func=sdf_sphere,
                bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
                min_depth=4, max_depth=4,
            )
            results.append((v.copy(), f.copy()))
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0][0], results[i][0])
            np.testing.assert_array_equal(results[0][1], results[i][1])
