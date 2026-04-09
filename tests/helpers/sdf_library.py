"""Analytic SDF functions for testing."""

import numpy as np


def sdf_sphere(pos: np.ndarray, radius: float = 1.0):
    norms = np.linalg.norm(pos, axis=1)
    values = norms - radius
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    gradients = pos / safe_norms[:, np.newaxis]
    return values, gradients


def sdf_box(pos: np.ndarray, half_extents: np.ndarray = None):
    if half_extents is None:
        half_extents = np.array([1.0, 1.0, 1.0])
    half_extents = np.asarray(half_extents)
    q = np.abs(pos) - half_extents
    q_max = np.maximum(q, 0.0)
    outside_dist = np.linalg.norm(q_max, axis=1)
    inside_dist = np.minimum(np.max(q, axis=1), 0.0)
    values = outside_dist + inside_dist
    # Approximate gradient via subgradient (exact except at edges/corners)
    return values, None


def sdf_torus(pos: np.ndarray, R: float = 1.0, r: float = 0.3):
    xy_dist = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    values = np.sqrt((xy_dist - R) ** 2 + pos[:, 2] ** 2) - r
    return values, None


def sdf_thin_plate(pos: np.ndarray, thickness: float = 0.05, size: float = 1.0):
    half = np.array([size, size, thickness / 2.0])
    return sdf_box(pos, half_extents=half)


def sdf_thin_shell(pos: np.ndarray, R_outer: float = 1.0, wall: float = 0.05):
    r = np.linalg.norm(pos, axis=1)
    R_inner = R_outer - wall
    d_outer = r - R_outer
    d_inner = R_inner - r
    values = np.maximum(d_outer, d_inner)
    return values, None


def sdf_narrow_gap(pos: np.ndarray, gap: float = 0.1, plate_thickness: float = 0.3):
    offset = (plate_thickness + gap) / 2.0
    half = np.array([1.0, 1.0, plate_thickness / 2.0])
    pos_upper = pos.copy()
    pos_upper[:, 2] -= offset
    pos_lower = pos.copy()
    pos_lower[:, 2] += offset
    d1, _ = sdf_box(pos_upper, half)
    d2, _ = sdf_box(pos_lower, half)
    values = np.minimum(d1, d2)  # union
    return values, None


def sdf_csg_cross(pos: np.ndarray):
    box_a = np.array([0.3, 0.3, 1.0])
    box_b = np.array([1.0, 0.3, 0.3])
    d_a, _ = sdf_box(pos, box_a)
    d_b, _ = sdf_box(pos, box_b)
    values = np.maximum(d_a, d_b)  # intersection
    return values, None
