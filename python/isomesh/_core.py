from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from isomesh._validation import (
    validate_func,
    validate_bbox,
    validate_depth,
    validate_angle_threshold,
)


def extract(
    func: object,
    *,
    bbox_min: tuple[float, float, float] = (-1.0, -1.0, -1.0),
    bbox_max: tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_depth: int = 3,
    max_depth: int = 7,
    angle_threshold: float = 30.0,
    iso_value: float = 0.0,
    fd_step: float = 1e-5,
    adaptive: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Extract an isosurface mesh from an implicit function.

    Parameters
    ----------
    func : callable
        Vectorized implicit function:
        (N,3) float64 -> ((N,) float64, (N,3) float64 | None).
    bbox_min, bbox_max : tuple of 3 floats
        Axis-aligned bounding box. If non-cubic, it will be expanded to the
        smallest enclosing cube (centered on the original bbox center).
    min_depth, max_depth : int
        Octree depth range. Cell count per axis = 2^depth.
    angle_threshold : float
        Degrees. Normals differing by more than this trigger sharp-feature handling.
    iso_value : float
        Isosurface level (default 0.0).
    fd_step : float
        Finite difference step for gradient estimation when func returns None gradients.

    Returns
    -------
    vertices : ndarray (V, 3) float64
    faces : ndarray (F, 3) int64
    """
    # Validation
    validate_func(func)
    bbox_min_arr, bbox_max_arr = validate_bbox(bbox_min, bbox_max)
    validate_depth(min_depth, max_depth)
    validate_angle_threshold(angle_threshold)

    # Warn if bbox will be expanded to cubic
    extents = bbox_max_arr - bbox_min_arr
    if not np.allclose(extents, extents[0]):
        max_extent = extents.max()
        center = (bbox_min_arr + bbox_max_arr) * 0.5
        expanded_min = center - max_extent / 2
        expanded_max = center + max_extent / 2
        warnings.warn(
            f"Non-cubic bounding box expanded to cubic: "
            f"[{expanded_min.tolist()}] to [{expanded_max.tolist()}]",
            stacklevel=2,
        )

    # Build the callback adapter that always returns numpy arrays
    def _eval_batch(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Called by Rust. Returns (values, gradients) as numpy arrays."""
        values, gradients = func(positions)
        values = np.asarray(values, dtype=np.float64).ravel()
        if gradients is not None:
            gradients = np.asarray(gradients, dtype=np.float64).reshape(-1, 3)
        return values, gradients

    # Call Rust core
    from isomesh._isomesh_rs import extract_mesh

    vertices, faces = extract_mesh(
        eval_fn=_eval_batch,
        bbox_min=bbox_min_arr.tolist(),
        bbox_max=bbox_max_arr.tolist(),
        min_depth=min_depth,
        max_depth=max_depth,
        angle_threshold_deg=angle_threshold,
        iso_value=iso_value,
        adaptive=adaptive,
    )

    # Ensure correct dtypes
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    # Ensure (V, 3) and (F, 3) shapes even if empty
    if vertices.ndim == 1 and vertices.size == 0:
        vertices = vertices.reshape(0, 3)
    if faces.ndim == 1 and faces.size == 0:
        faces = faces.reshape(0, 3)

    return vertices, faces
