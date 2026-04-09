"""Thin feature robustness tests."""

import numpy as np
import pytest
import isomesh
from tests.helpers.sdf_library import sdf_thin_plate, sdf_thin_shell, sdf_narrow_gap


class TestThinPlate:
    def test_thin_plate_detected(self):
        """Thin plate (thickness=0.1) should produce a non-empty mesh."""
        def sdf(pos):
            return sdf_thin_plate(pos, thickness=0.1)

        v, f = isomesh.extract(
            func=sdf,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=4, max_depth=5,
        )
        assert len(v) > 0, "Thin plate (0.1) produced no vertices"
        assert len(f) > 0, "Thin plate (0.1) produced no faces"

    def test_thin_plate_two_sided(self):
        """Thin plate mesh should have vertices on both sides of z=0."""
        def sdf(pos):
            return sdf_thin_plate(pos, thickness=0.1)

        v, f = isomesh.extract(
            func=sdf,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=4, max_depth=5,
        )
        if len(v) > 0:
            has_positive_z = np.any(v[:, 2] > 0.01)
            has_negative_z = np.any(v[:, 2] < -0.01)
            assert has_positive_z and has_negative_z, (
                f"Expected vertices on both sides of z=0, "
                f"z range: [{v[:, 2].min():.3f}, {v[:, 2].max():.3f}]"
            )


class TestThinShell:
    def test_thin_shell_detected(self):
        """Thin shell (wall=0.1) should produce a non-empty mesh."""
        def sdf(pos):
            return sdf_thin_shell(pos, wall=0.1)

        v, f = isomesh.extract(
            func=sdf,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=4, max_depth=5,
        )
        assert len(v) > 0, "Thin shell produced no vertices"


class TestNarrowGap:
    def test_narrow_gap_produces_mesh(self):
        """Two plates with narrow gap should produce a mesh.

        Gap=0.3, plate_thickness=0.5 — large enough for depth 4 to resolve.
        """
        def sdf(pos):
            return sdf_narrow_gap(pos, gap=0.3, plate_thickness=0.5)

        v, f = isomesh.extract(
            func=sdf,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=4, max_depth=5,
        )
        assert len(v) > 0, "Narrow gap produced no vertices"
        assert len(f) > 0, "Narrow gap produced no faces"
