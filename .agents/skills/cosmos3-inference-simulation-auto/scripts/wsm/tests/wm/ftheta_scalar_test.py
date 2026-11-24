import numpy as np

from cosmos_auto.wm import pcd_utils
from cosmos_auto.wm.camera.ftheta import FThetaCamera, eval_polynomial_derivative
from cosmos_auto.wm.graphics_utils import WorldSpaceLines


def test_synthetic_ftheta_camera_has_scalar_shader_angle():
    camera = FThetaCamera.from_numpy(
        np.array([640, 360, 1280, 720, 0, 1 / 600, 0, 0, 0, 0, 1], dtype=np.float32),
        device="cpu",
    )
    assert np.ndim(camera._max_ray_angle) == 0
    assert np.isfinite(float(camera._max_ray_angle))
    geometry = WorldSpaceLines(
        np.array([[[0, 0, 10], [1, 0, 10]]]),
        colors=np.ones((2, 3)),
        line_width=1,
        camera_model=camera,
    )
    assert geometry._cam_params["max_ray_angle"] == float(camera._max_ray_angle)


def test_ftheta_polynomial_derivative() -> None:
    # The derivative of 3 + 2x + 5x² is 2 + 10x.
    coeffs = np.array([3.0, 2.0, 5.0])
    xs = np.array([0.0, 2.0])
    np.testing.assert_allclose(eval_polynomial_derivative(xs, coeffs), [2.0, 22.0])


def test_scalar_ftheta_polynomial_derivative() -> None:
    # The derivative of 3 + 2x + 5x² is 2 + 10x.
    coeffs = [3.0, 2.0, 5.0]
    assert pcd_utils.eval_polynomial_derivative(0.0, coeffs) == 2.0
    assert pcd_utils.eval_polynomial_derivative(2.0, coeffs) == 22.0
