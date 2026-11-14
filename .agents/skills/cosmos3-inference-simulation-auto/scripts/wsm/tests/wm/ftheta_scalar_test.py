import numpy as np

from cosmos_auto.wm.camera.ftheta import FThetaCamera
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
