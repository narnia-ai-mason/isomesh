"""Shared fixtures for isomesh tests."""

import pytest
import numpy as np

from tests.helpers.sdf_library import sdf_sphere


@pytest.fixture
def default_kwargs():
    return dict(
        bbox_min=(-2.0, -2.0, -2.0),
        bbox_max=(2.0, 2.0, 2.0),
        min_depth=3,
        max_depth=3,
    )


@pytest.fixture
def sphere_mesh_d4():
    """Sphere mesh at depth 4."""
    import isomesh

    return isomesh.extract(
        func=sdf_sphere,
        bbox_min=(-2, -2, -2),
        bbox_max=(2, 2, 2),
        min_depth=4,
        max_depth=4,
    )
