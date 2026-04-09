"""Verify that Rayon parallelization produces deterministic (bit-identical) output."""

import numpy as np
import pytest
import isomesh
from tests.helpers.sdf_library import sdf_sphere, sdf_box, sdf_torus


@pytest.mark.parametrize("sdf_func,name", [
    (sdf_sphere, "sphere"),
    (sdf_box, "box"),
    (sdf_torus, "torus"),
])
def test_deterministic_output(sdf_func, name):
    """Multiple runs with the same input must produce bit-identical vertices and faces."""
    results = []
    for _ in range(5):
        v, f = isomesh.extract(
            func=sdf_func,
            bbox_min=(-2, -2, -2),
            bbox_max=(2, 2, 2),
            min_depth=4,
            max_depth=4,
        )
        results.append((v.copy(), f.copy()))

    for i in range(1, len(results)):
        np.testing.assert_array_equal(
            results[0][0], results[i][0],
            err_msg=f"{name}: vertices differ between run 0 and {i}",
        )
        np.testing.assert_array_equal(
            results[0][1], results[i][1],
            err_msg=f"{name}: faces differ between run 0 and {i}",
        )
