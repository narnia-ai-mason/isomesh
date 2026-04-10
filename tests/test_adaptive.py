"""True Adaptive Mesh tests.

Tests that adaptive mode (2:1 balance + cross-depth quad generation)
produces manifold/watertight meshes with significantly fewer vertices
than uniform-depth extraction.
"""

import numpy as np
import pytest

import isomesh
from tests.helpers.sdf_library import sdf_sphere, sdf_box, sdf_torus
from tests.helpers.mesh_checks import (
    check_manifold_watertight, check_no_degenerate_faces, check_face_indices_valid,
)
from tests.helpers.metrics import hausdorff_to_analytic, volume_relative_error


class TestAdaptiveNonRegression:
    """When min_depth == max_depth, adaptive must produce identical results."""

    @pytest.mark.parametrize("depth", [3, 4, 5])
    def test_uniform_identity(self, depth):
        """adaptive=True with min==max should equal adaptive=False."""
        v_u, f_u = isomesh.extract(
            func=sdf_sphere,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=depth, max_depth=depth, adaptive=False,
        )
        v_a, f_a = isomesh.extract(
            func=sdf_sphere,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=depth, max_depth=depth, adaptive=True,
        )
        np.testing.assert_array_equal(v_u, v_a)
        np.testing.assert_array_equal(f_u, f_a)


class TestAdaptiveManifold:
    """Adaptive mode should produce manifold/watertight meshes."""

    def test_sphere_d3_5_manifold(self):
        v, f = isomesh.extract(
            func=sdf_sphere,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=3, max_depth=5, adaptive=True,
        )
        assert len(v) > 0
        r = check_manifold_watertight(v, f)
        assert r["is_manifold"], f"non-manifold edges: {r['non_manifold_edges']}"
        assert r["is_watertight"], f"boundary edges: {r['boundary_edges']}"
        assert r["consistent_winding"]

    def test_torus_d3_5_manifold(self):
        v, f = isomesh.extract(
            func=sdf_torus,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=3, max_depth=5, adaptive=True,
        )
        assert len(v) > 0
        r = check_manifold_watertight(v, f)
        assert r["is_manifold"]
        assert r["is_watertight"]

    def test_sphere_d3_6_manifold(self):
        v, f = isomesh.extract(
            func=sdf_sphere,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=3, max_depth=6, adaptive=True,
        )
        assert len(v) > 0
        r = check_manifold_watertight(v, f)
        assert r["is_manifold"]
        assert r["is_watertight"]

    def test_box_d3_5_manifold(self):
        v, f = isomesh.extract(
            func=sdf_box,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=3, max_depth=5, adaptive=True,
        )
        assert len(v) > 0
        r = check_manifold_watertight(v, f)
        assert r["is_manifold"]
        assert r["is_watertight"]

    def test_box_d3_6_manifold(self):
        v, f = isomesh.extract(
            func=sdf_box,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=3, max_depth=6, adaptive=True,
        )
        assert len(v) > 0
        r = check_manifold_watertight(v, f)
        assert r["is_manifold"]
        assert r["is_watertight"]

    def test_torus_d4_5_manifold(self):
        v, f = isomesh.extract(
            func=sdf_torus,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=4, max_depth=5, adaptive=True,
        )
        assert len(v) > 0
        r = check_manifold_watertight(v, f)
        assert r["is_manifold"]
        assert r["is_watertight"]


class TestAdaptiveVertexReduction:
    """Adaptive mode should use fewer vertices than uniform max_depth."""

    def test_sphere_vertex_reduction(self):
        """Smooth sphere should benefit most from adaptive (few features)."""
        v_uniform, _ = isomesh.extract(
            func=sdf_sphere,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=5, max_depth=5,
        )
        v_adaptive, _ = isomesh.extract(
            func=sdf_sphere,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=3, max_depth=5, adaptive=True,
        )
        ratio = len(v_adaptive) / len(v_uniform)
        assert ratio < 0.5, f"Adaptive vertex ratio {ratio:.2f} should be < 0.5"

    def test_sphere_accuracy_preserved(self):
        """Adaptive mode should maintain surface accuracy."""
        v, f = isomesh.extract(
            func=sdf_sphere,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=3, max_depth=5, adaptive=True,
        )
        h = hausdorff_to_analytic(v, sdf_sphere)
        # Accuracy should be at least as good as the effective depth
        max_cell = 4.0 / (2 ** 3)  # min_depth cell size
        assert h < 2 * max_cell, f"Hausdorff {h:.4f} > {2*max_cell:.4f}"

    def test_determinism(self):
        """Adaptive mode must be deterministic under Rayon."""
        results = []
        for _ in range(3):
            v, f = isomesh.extract(
                func=sdf_sphere,
                bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
                min_depth=3, max_depth=5, adaptive=True,
            )
            results.append((v.copy(), f.copy()))
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0][0], results[i][0])
            np.testing.assert_array_equal(results[0][1], results[i][1])
