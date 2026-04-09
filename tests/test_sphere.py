"""Sphere SDF accuracy tests — analytic validation baseline."""

import numpy as np
import pytest
import isomesh
from tests.helpers.sdf_library import sdf_sphere
from tests.helpers.metrics import hausdorff_to_analytic, mesh_volume, volume_relative_error
from tests.helpers.mesh_checks import check_face_indices_valid, check_no_degenerate_faces


@pytest.mark.parametrize("max_depth", [3, 4, 5])
def test_sphere_produces_mesh(max_depth):
    v, f = isomesh.extract(
        func=sdf_sphere,
        bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
        min_depth=max_depth, max_depth=max_depth,
    )
    assert len(v) > 0, f"No vertices at depth {max_depth}"
    assert len(f) > 0, f"No faces at depth {max_depth}"


@pytest.mark.parametrize("max_depth", [3, 4, 5])
def test_sphere_hausdorff_bounded(max_depth):
    v, f = isomesh.extract(
        func=sdf_sphere,
        bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
        min_depth=max_depth, max_depth=max_depth,
    )
    h = hausdorff_to_analytic(v, sdf_sphere)
    cell_size = 4.0 / (2 ** max_depth)
    assert h < cell_size * 2, f"depth={max_depth}: hausdorff={h:.4f}, cell_size={cell_size:.4f}"


def test_sphere_hausdorff_decreases_with_depth():
    results = []
    for depth in [3, 4, 5]:
        v, f = isomesh.extract(
            func=sdf_sphere,
            bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
            min_depth=depth, max_depth=depth,
        )
        results.append(hausdorff_to_analytic(v, sdf_sphere))
    for i in range(len(results) - 1):
        # Allow absolute tolerance for near-zero Hausdorff (Newton projection
        # can achieve machine precision, making relative comparison meaningless)
        assert results[i + 1] <= results[i] * 1.1 + 1e-3, (
            f"Hausdorff did not decrease: depth {i+3}={results[i]:.6f} -> {i+4}={results[i+1]:.6f}"
        )


def test_sphere_volume_accuracy():
    """Mesh volume should be close to analytic 4/3*pi."""
    v, f = isomesh.extract(
        func=sdf_sphere,
        bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
        min_depth=4, max_depth=4,
    )
    analytic_vol = (4.0 / 3.0) * np.pi
    rel_err = volume_relative_error(v, f, analytic_vol)
    assert rel_err < 0.15, f"Volume relative error: {rel_err:.3f} (mesh_vol={abs(mesh_volume(v, f)):.3f})"


def test_sphere_face_validity(sphere_mesh_d4):
    v, f = sphere_mesh_d4
    assert check_face_indices_valid(v, f)
    assert check_no_degenerate_faces(f)


def test_sphere_no_gradient_mode():
    def sdf_no_grad(pos):
        return np.linalg.norm(pos, axis=1) - 1.0, None

    v, f = isomesh.extract(
        func=sdf_no_grad,
        bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
        min_depth=3, max_depth=3,
    )
    assert len(v) > 0
    assert len(f) > 0
    h = hausdorff_to_analytic(v, sdf_sphere)
    assert h < 0.2, f"No-gradient hausdorff: {h:.4f}"


def test_sphere_custom_iso_value():
    v, f = isomesh.extract(
        func=sdf_sphere,
        bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
        min_depth=3, max_depth=3,
        iso_value=0.5,
    )
    assert len(v) > 0
    mean_r = np.linalg.norm(v, axis=1).mean()
    assert abs(mean_r - 1.5) < 0.1, f"Expected radius ~1.5, got {mean_r:.3f}"
