from __future__ import annotations

import numpy as np


def validate_func(func: object) -> None:
    """Probe the function with 2 test points to catch signature errors early."""
    if not callable(func):
        raise ValueError(f"func must be callable, got {type(func).__name__}")

    probe = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    try:
        result = func(probe)
    except Exception as e:
        raise ValueError(
            f"func({probe.shape}) raised {type(e).__name__}: {e}\n"
            "Expected signature: func(positions: ndarray(N,3)) -> (values(N,), gradients(N,3)|None)"
        ) from e

    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError(
            f"func must return a 2-tuple (values, gradients), got {type(result)}"
        )

    values, gradients = result
    values = np.asarray(values)
    if values.shape != (2,):
        raise ValueError(
            f"func returned values with shape {values.shape}, expected (N,) where N=2"
        )
    if gradients is not None:
        gradients = np.asarray(gradients)
        if gradients.shape != (2, 3):
            raise ValueError(
                f"func returned gradients with shape {gradients.shape}, expected (N,3) where N=2"
            )


def validate_bbox(
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert bounding box to float64 arrays."""
    bbox_min_arr = np.asarray(bbox_min, dtype=np.float64)
    bbox_max_arr = np.asarray(bbox_max, dtype=np.float64)

    if bbox_min_arr.shape != (3,) or bbox_max_arr.shape != (3,):
        raise ValueError(
            f"bbox_min and bbox_max must be 3-element tuples, "
            f"got shapes {bbox_min_arr.shape} and {bbox_max_arr.shape}"
        )

    if not np.all(bbox_min_arr < bbox_max_arr):
        raise ValueError(
            f"bbox_min must be strictly less than bbox_max in all dimensions.\n"
            f"  bbox_min={bbox_min_arr.tolist()}\n"
            f"  bbox_max={bbox_max_arr.tolist()}"
        )

    return bbox_min_arr, bbox_max_arr


def validate_depth(min_depth: int, max_depth: int) -> None:
    """Validate octree depth parameters."""
    if not isinstance(min_depth, int) or not isinstance(max_depth, int):
        raise ValueError(
            f"min_depth and max_depth must be integers, "
            f"got {type(min_depth).__name__} and {type(max_depth).__name__}"
        )
    if min_depth < 1:
        raise ValueError(f"min_depth must be >= 1, got {min_depth}")
    if max_depth > 12:
        raise ValueError(f"max_depth must be <= 12 (safety cap), got {max_depth}")
    if min_depth > max_depth:
        raise ValueError(
            f"min_depth ({min_depth}) must be <= max_depth ({max_depth})"
        )


def validate_angle_threshold(angle_threshold: float) -> None:
    """Validate sharp feature angle threshold."""
    if not (0.0 < angle_threshold < 180.0):
        raise ValueError(
            f"angle_threshold must be in (0, 180) degrees, got {angle_threshold}"
        )
