"""Phase 1 smoke tests: import, batch eval round-trip, validation."""

import numpy as np
import pytest


def sdf_sphere(pos: np.ndarray):
    """Simple sphere SDF with analytic gradient."""
    norms = np.linalg.norm(pos, axis=1)
    values = norms - 1.0
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    gradients = pos / safe_norms[:, np.newaxis]
    return values, gradients


def sdf_sphere_no_grad(pos: np.ndarray):
    """Sphere SDF without gradient."""
    norms = np.linalg.norm(pos, axis=1)
    values = norms - 1.0
    return values, None


class TestImport:
    def test_import_isomesh(self):
        import isomesh
        assert hasattr(isomesh, "extract")
        assert hasattr(isomesh, "__version__")

    def test_import_native_module(self):
        from isomesh._isomesh_rs import extract_mesh, test_batch_eval
        assert callable(extract_mesh)
        assert callable(test_batch_eval)


class TestBatchEvalBridge:
    def test_batch_eval_with_gradients(self):
        from isomesh._isomesh_rs import test_batch_eval

        points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        values, has_grads, n = test_batch_eval(sdf_sphere, points)

        assert n == 3
        assert has_grads is True
        assert len(values) == 3
        # Point at origin: |[0,0,0]| - 1 = -1
        assert abs(values[0] - (-1.0)) < 1e-10
        # Point at [1,0,0]: |[1,0,0]| - 1 = 0
        assert abs(values[1] - 0.0) < 1e-10

    def test_batch_eval_without_gradients(self):
        from isomesh._isomesh_rs import test_batch_eval

        points = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        values, has_grads, n = test_batch_eval(sdf_sphere_no_grad, points)

        assert n == 2
        assert has_grads is False
        assert abs(values[0] - (-1.0)) < 1e-10
        assert abs(values[1] - 1.0) < 1e-10

    def test_batch_eval_large_batch(self):
        from isomesh._isomesh_rs import test_batch_eval

        rng = np.random.default_rng(42)
        points = rng.standard_normal((10000, 3)).tolist()
        values, has_grads, n = test_batch_eval(sdf_sphere, points)

        assert n == 10000
        assert has_grads is True
        assert len(values) == 10000

    def test_batch_eval_empty(self):
        from isomesh._isomesh_rs import test_batch_eval

        values, has_grads, n = test_batch_eval(sdf_sphere, [])
        assert n == 0
        assert len(values) == 0


class TestFDGradients:
    def test_fd_gradients_accuracy(self):
        from isomesh._isomesh_rs import test_fd_gradients

        points = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        # Use only-values function for FD
        fd_grads = test_fd_gradients(sdf_sphere_no_grad, points, 1e-6)

        # Analytic gradient of sphere SDF at [1,0,0] is [1,0,0]
        assert len(fd_grads) == 3
        assert abs(fd_grads[0][0] - 1.0) < 1e-4
        assert abs(fd_grads[0][1] - 0.0) < 1e-4
        assert abs(fd_grads[0][2] - 0.0) < 1e-4

        # At [0,1,0] gradient is [0,1,0]
        assert abs(fd_grads[1][0] - 0.0) < 1e-4
        assert abs(fd_grads[1][1] - 1.0) < 1e-4
        assert abs(fd_grads[1][2] - 0.0) < 1e-4


class TestExtractAPI:
    def test_extract_returns_correct_types(self):
        import isomesh

        v, f = isomesh.extract(func=sdf_sphere)
        assert isinstance(v, np.ndarray)
        assert isinstance(f, np.ndarray)
        assert v.dtype == np.float64
        assert f.dtype == np.int64
        assert v.ndim == 2 and v.shape[1] == 3
        assert f.ndim == 2 and f.shape[1] == 3
        assert len(v) > 0  # should produce actual mesh
        assert len(f) > 0

    def test_extract_with_custom_bbox(self):
        import isomesh

        v, f = isomesh.extract(
            func=sdf_sphere,
            bbox_min=(-2.0, -2.0, -2.0),
            bbox_max=(2.0, 2.0, 2.0),
        )
        assert v.shape[1] == 3
        assert f.shape[1] == 3

    def test_extract_keyword_only(self):
        import isomesh

        # All args except func must be keyword-only
        with pytest.raises(TypeError):
            isomesh.extract(sdf_sphere, (-1, -1, -1), (1, 1, 1))


class TestValidation:
    def test_invalid_func_not_callable(self):
        import isomesh

        with pytest.raises(ValueError, match="callable"):
            isomesh.extract(func="not a function")

    def test_invalid_func_wrong_return(self):
        import isomesh

        def bad_func(pos):
            return np.zeros(len(pos))  # returns single array, not tuple

        with pytest.raises(ValueError, match="2-tuple"):
            isomesh.extract(func=bad_func)

    def test_invalid_bbox(self):
        import isomesh

        with pytest.raises(ValueError, match="strictly less"):
            isomesh.extract(func=sdf_sphere, bbox_min=(1, 1, 1), bbox_max=(0, 0, 0))

    def test_invalid_depth(self):
        import isomesh

        with pytest.raises(ValueError, match="min_depth"):
            isomesh.extract(func=sdf_sphere, min_depth=0)

        with pytest.raises(ValueError, match="max_depth"):
            isomesh.extract(func=sdf_sphere, max_depth=20)

        with pytest.raises(ValueError, match="min_depth.*max_depth"):
            isomesh.extract(func=sdf_sphere, min_depth=5, max_depth=3)

    def test_invalid_angle_threshold(self):
        import isomesh

        with pytest.raises(ValueError, match="angle_threshold"):
            isomesh.extract(func=sdf_sphere, angle_threshold=0.0)

        with pytest.raises(ValueError, match="angle_threshold"):
            isomesh.extract(func=sdf_sphere, angle_threshold=180.0)
