"""Phase 2 tests: octree construction, bbox expansion, corner deduplication."""

import numpy as np
import pytest


def sdf_sphere(pos: np.ndarray):
    """Sphere SDF centered at origin, radius 1."""
    norms = np.linalg.norm(pos, axis=1)
    values = norms - 1.0
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    gradients = pos / safe_norms[:, np.newaxis]
    return values, gradients


def sdf_all_outside(pos: np.ndarray):
    """Function that is positive everywhere (no surface)."""
    return np.ones(len(pos)), None


def sdf_all_inside(pos: np.ndarray):
    """Function that is negative everywhere (no surface)."""
    return -np.ones(len(pos)), None


class TestBuildUniformOctree:
    def test_depth_3_leaf_count(self):
        """Depth 3 uniform octree for sphere should have leaves near surface."""
        from isomesh._isomesh_rs import build_octree_uniform

        leaf_count, branch_count, empty_count, full_count, max_depth, total_evals = \
            build_octree_uniform(sdf_sphere, [-2, -2, -2], [2, 2, 2], 3)

        # At depth 3, there are 8^3 = 512 potential leaf cells
        # Cells far from surface are Empty or Full, remaining are Leaf
        total_leaves = leaf_count + empty_count + full_count
        assert total_leaves > 0
        assert leaf_count > 0  # sphere should have some surface-crossing leaves
        assert max_depth == 3

    def test_depth_3_corner_count(self):
        """At depth 3 with bbox [-2,2]^3, unique corners = 9^3 = 729."""
        from isomesh._isomesh_rs import build_octree_uniform

        _, _, _, _, _, total_evals = \
            build_octree_uniform(sdf_sphere, [-2, -2, -2], [2, 2, 2], 3)

        assert total_evals == 9 ** 3  # 729 unique corners

    def test_depth_1_simple(self):
        """Depth 1 = 8 cells, 27 unique corners."""
        from isomesh._isomesh_rs import build_octree_uniform

        _, _, _, _, _, total_evals = \
            build_octree_uniform(sdf_sphere, [-2, -2, -2], [2, 2, 2], 1)

        assert total_evals == 3 ** 3  # 27 unique corners

    def test_depth_2_corner_count(self):
        """Depth 2 = 64 cells, 125 unique corners."""
        from isomesh._isomesh_rs import build_octree_uniform

        _, _, _, _, _, total_evals = \
            build_octree_uniform(sdf_sphere, [-2, -2, -2], [2, 2, 2], 2)

        assert total_evals == 5 ** 3  # 125 unique corners

    def test_all_outside(self):
        """Function positive everywhere: all cells should be Empty."""
        from isomesh._isomesh_rs import build_octree_uniform

        leaf_count, _, empty_count, full_count, _, _ = \
            build_octree_uniform(sdf_all_outside, [-1, -1, -1], [1, 1, 1], 2)

        assert leaf_count == 0  # no sign changes
        assert empty_count > 0
        assert full_count == 0

    def test_all_inside(self):
        """Function negative everywhere: all cells should be Full."""
        from isomesh._isomesh_rs import build_octree_uniform

        leaf_count, _, empty_count, full_count, _, _ = \
            build_octree_uniform(sdf_all_inside, [-1, -1, -1], [1, 1, 1], 2)

        assert leaf_count == 0  # no sign changes
        assert empty_count == 0
        assert full_count > 0

    def test_non_cubic_bbox_expansion(self):
        """Non-cubic bbox should still produce correct corner count."""
        from isomesh._isomesh_rs import build_octree_uniform

        # bbox is 4x2x2, should expand to 4x4x4 cube
        _, _, _, _, _, total_evals = \
            build_octree_uniform(sdf_sphere, [-2, -1, -1], [2, 1, 1], 2)

        # After expansion to 4x4x4 cube, at depth 2:
        # cells_per_axis = 4, corners_per_axis = 5
        assert total_evals == 5 ** 3  # 125

    def test_iso_value_nonzero(self):
        """iso_value shifts the surface: f(x)=iso_value."""
        from isomesh._isomesh_rs import build_octree_uniform

        # sphere SDF: f(x) = |x| - 1
        # iso_value=0.0 → |x| = 1 (radius 1)
        # iso_value=0.5 → |x| = 1.5 (radius 1.5, larger surface)
        leaf_count_0, _, _, _, _, _ = \
            build_octree_uniform(sdf_sphere, [-2, -2, -2], [2, 2, 2], 3, 0.0)
        leaf_count_05, _, _, _, _, _ = \
            build_octree_uniform(sdf_sphere, [-2, -2, -2], [2, 2, 2], 3, 0.5)

        # Both should produce non-empty octrees
        assert leaf_count_0 > 0
        assert leaf_count_05 > 0
        # They should produce different leaf counts (different surfaces)
        assert leaf_count_0 != leaf_count_05

    def test_batch_eval_single_call(self):
        """Verify the function is called exactly once for all corners."""
        call_count = 0

        def counting_sdf(pos):
            nonlocal call_count
            call_count += 1
            norms = np.linalg.norm(pos, axis=1)
            return norms - 1.0, None

        from isomesh._isomesh_rs import build_octree_uniform
        build_octree_uniform(counting_sdf, [-2, -2, -2], [2, 2, 2], 3)

        assert call_count == 1, f"Expected 1 batch call, got {call_count}"


class TestBuildAdaptiveOctree:
    def test_adaptive_has_more_leaves_near_surface(self):
        """Adaptive refinement should create more leaves than uniform at same min_depth."""
        from isomesh._isomesh_rs import build_octree_uniform, build_octree_adaptive

        # Uniform at depth 3
        leaf_u, _, _, _, max_d_u, evals_u = \
            build_octree_uniform(sdf_sphere, [-2, -2, -2], [2, 2, 2], 3)

        # Adaptive from depth 3 to 5
        leaf_a, _, _, _, max_d_a, evals_a = \
            build_octree_adaptive(sdf_sphere, [-2, -2, -2], [2, 2, 2], 3, 5)

        # Adaptive should refine deeper near the surface
        assert max_d_a >= max_d_u
        # More evaluations due to additional refinement
        assert evals_a >= evals_u
        # More leaves near the surface
        assert leaf_a >= leaf_u

    def test_adaptive_min_equals_max_is_uniform(self):
        """When min_depth == max_depth, adaptive should equal uniform."""
        from isomesh._isomesh_rs import build_octree_uniform, build_octree_adaptive

        leaf_u, br_u, emp_u, full_u, _, evals_u = \
            build_octree_uniform(sdf_sphere, [-2, -2, -2], [2, 2, 2], 3)
        leaf_a, br_a, emp_a, full_a, _, evals_a = \
            build_octree_adaptive(sdf_sphere, [-2, -2, -2], [2, 2, 2], 3, 3)

        assert leaf_u == leaf_a
        assert evals_u == evals_a

    def test_adaptive_all_outside_no_refinement(self):
        """No refinement needed when function has no zero crossing."""
        from isomesh._isomesh_rs import build_octree_adaptive

        leaf_count, _, _, _, max_d, evals = \
            build_octree_adaptive(sdf_all_outside, [-1, -1, -1], [1, 1, 1], 2, 5)

        # No surface → no refinement beyond min_depth
        assert leaf_count == 0
        assert max_d == 0  # no leaves, so max_depth stays at 0

    def test_adaptive_depth_increases(self):
        """Adaptive should reach deeper than min_depth for sphere."""
        from isomesh._isomesh_rs import build_octree_adaptive

        _, _, _, _, max_d, _ = \
            build_octree_adaptive(sdf_sphere, [-2, -2, -2], [2, 2, 2], 2, 6)

        assert max_d > 2, f"Expected depth > 2, got {max_d}"

    def test_extract_with_adaptive(self):
        """extract() should work with adaptive octree (min != max depth)."""
        import isomesh

        v, f = isomesh.extract(
            func=sdf_sphere,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=2, max_depth=5,
        )
        assert v.ndim == 2 and v.shape[1] == 3
        assert f.ndim == 2 and f.shape[1] == 3


class TestExtractWithOctree:
    def test_extract_produces_mesh(self):
        """extract() should produce a non-empty triangle mesh for sphere."""
        import isomesh

        v, f = isomesh.extract(
            func=sdf_sphere,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=3, max_depth=3,
        )
        assert v.ndim == 2 and v.shape[1] == 3
        assert f.ndim == 2 and f.shape[1] == 3
        assert v.dtype == np.float64
        assert f.dtype == np.int64
        assert len(v) > 0, "Expected non-empty vertices"
        assert len(f) > 0, "Expected non-empty faces"

    def test_extract_sphere_accuracy(self):
        """Sphere mesh vertices should lie near the unit sphere surface."""
        import isomesh

        v, f = isomesh.extract(
            func=sdf_sphere,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=4, max_depth=4,
        )
        dists = np.abs(np.linalg.norm(v, axis=1) - 1.0)
        assert dists.max() < 0.05, f"Max distance from sphere: {dists.max():.4f}"
        assert dists.mean() < 0.02, f"Mean distance from sphere: {dists.mean():.4f}"

    def test_extract_face_indices_valid(self):
        """All face indices should reference valid vertices."""
        import isomesh

        v, f = isomesh.extract(
            func=sdf_sphere,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=3, max_depth=3,
        )
        if len(f) > 0:
            assert f.min() >= 0
            assert f.max() < len(v)
