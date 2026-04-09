"""Sharp feature preservation tests."""

import numpy as np
import pytest
import isomesh
from tests.helpers.sdf_library import sdf_box, sdf_csg_cross
from tests.helpers.mesh_checks import check_face_indices_valid


class TestBoxSharpEdges:
    def test_box_produces_mesh(self):
        """Box SDF should produce a non-empty mesh."""
        v, f = isomesh.extract(
            func=sdf_box,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=4, max_depth=4,
        )
        assert len(v) > 0, "Box produced no vertices"
        assert len(f) > 0, "Box produced no faces"
        assert check_face_indices_valid(v, f)

    def test_box_vertices_near_surface(self):
        """Box vertices should have small SDF values."""
        v, f = isomesh.extract(
            func=sdf_box,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=4, max_depth=4,
        )
        vals, _ = sdf_box(v)
        assert np.max(np.abs(vals)) < 0.1, f"Max SDF at vertices: {np.max(np.abs(vals)):.4f}"

    def test_box_corners_present(self):
        """Box mesh should have vertices near the 8 analytic corners."""
        v, f = isomesh.extract(
            func=sdf_box,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=4, max_depth=4,
        )
        corners = np.array([
            [s0, s1, s2]
            for s0 in [-1, 1] for s1 in [-1, 1] for s2 in [-1, 1]
        ], dtype=np.float64)

        found = 0
        for corner in corners:
            dists = np.linalg.norm(v - corner, axis=1)
            if dists.min() < 0.3:  # within 0.3 of analytic corner
                found += 1

        assert found >= 6, f"Only {found}/8 corners found near analytic positions"


class TestCSGCross:
    def test_csg_cross_produces_mesh(self):
        v, f = isomesh.extract(
            func=sdf_csg_cross,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=4, max_depth=4,
        )
        assert len(v) > 0, "CSG cross produced no vertices"
        assert len(f) > 0, "CSG cross produced no faces"
