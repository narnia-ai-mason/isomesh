"""Mesh quality metrics for validation against analytic SDFs."""

import numpy as np


def hausdorff_to_analytic(vertices: np.ndarray, analytic_sdf) -> float:
    """One-sided Hausdorff: max |sdf(vertex)| over all vertices.

    For an SDF, |sdf(p)| is the distance from p to the surface.
    """
    values, _ = analytic_sdf(vertices)
    return float(np.max(np.abs(values)))


def mean_distance_to_analytic(vertices: np.ndarray, analytic_sdf) -> float:
    """Mean |sdf(vertex)| over all vertices."""
    values, _ = analytic_sdf(vertices)
    return float(np.mean(np.abs(values)))


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Signed mesh volume via divergence theorem.

    V = (1/6) * sum_faces (v0 . (v1 x v2))
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return float(np.sum(v0 * np.cross(v1, v2)) / 6.0)


def volume_relative_error(
    vertices: np.ndarray,
    faces: np.ndarray,
    analytic_volume: float,
) -> float:
    """|V_mesh - V_analytic| / V_analytic."""
    v_mesh = abs(mesh_volume(vertices, faces))
    return abs(v_mesh - analytic_volume) / analytic_volume
