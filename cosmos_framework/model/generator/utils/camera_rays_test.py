# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -----------------------------------------------------------------------------

from typing import Any

import cv2
import numpy as np
import pytest
import torch

from cosmos_framework.model.generator.utils.camera_rays import (
    token_pixel_grid,
    unproject_calibration,
    validate_token_geometry,
)
from cosmos_framework.model.generator.utils.ftheta import ftheta_unproject

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

# DeformCam360 and GrandTour sidecar values.
_OPENCV_COEFFICIENTS = [-0.38748353719711304, 0.11847664415836334, -0.00010451977868797258, -0.0036122221499681473]
_FISHEYE_COEFFICIENTS = [-0.03843622792783702, -0.002634177248264113, 0.002422949539367233, -0.001382691294699419]
# OpenCV's 12- and 14-term layouts append thin-prism s1..s4, then the sensor tilt (tau_x, tau_y) in radians.
_OPENCV_RATIONAL_COEFFICIENTS = [*_OPENCV_COEFFICIENTS, 0.01, 0.002, -0.001, 0.0005]
_OPENCV_THIN_PRISM_COEFFICIENTS = [*_OPENCV_RATIONAL_COEFFICIENTS, 0.002, -0.001, 0.0015, -0.0005]
_OPENCV_TILTED_COEFFICIENTS = [*_OPENCV_THIN_PRISM_COEFFICIENTS, 0.01, -0.02]


def _calibration(camera_model: str, distortion_model: str, coefficients: list[float]) -> dict[str, Any]:
    return {
        "camera_model": camera_model,
        "distortion_model": distortion_model,
        "distortion_coefficients": coefficients,
        "fx_px": 700.0,
        "fy_px": 690.0,
        "cx_px": 640.0,
        "cy_px": 360.0,
    }


def _camera_points() -> np.ndarray:  # returns [N,3]
    theta = np.linspace(0.0, 0.9, 7)  # [A]
    phi = np.linspace(0.0, 2 * np.pi, 9, endpoint=False)  # [B]
    tt, pp = np.meshgrid(theta, phi, indexing="ij")  # [A,B], [A,B]
    return np.stack((np.sin(tt) * np.cos(pp), np.sin(tt) * np.sin(pp), np.cos(tt)), axis=-1).reshape(-1, 3)  # [N,3]


def _matrix(calibration: dict[str, Any]) -> np.ndarray:  # returns [3,3]
    return np.array(
        [[calibration["fx_px"], 0, calibration["cx_px"]], [0, calibration["fy_px"], calibration["cy_px"]], [0, 0, 1]]
    )


@pytest.mark.parametrize(
    ("camera_model", "distortion_model", "coefficients"),
    [
        ("pinhole", "none", []),
        ("pinhole", "opencv", _OPENCV_COEFFICIENTS),
        ("pinhole", "opencv", [*_OPENCV_COEFFICIENTS, 0.01]),
        ("pinhole", "opencv", _OPENCV_RATIONAL_COEFFICIENTS),
        ("pinhole", "opencv", _OPENCV_THIN_PRISM_COEFFICIENTS),
        ("pinhole", "opencv", _OPENCV_TILTED_COEFFICIENTS),
        ("fisheye", "equidistant", _FISHEYE_COEFFICIENTS),
    ],
)
def test_unprojection_inverts_opencv_projection(
    camera_model: str, distortion_model: str, coefficients: list[float]
) -> None:
    calibration = _calibration(camera_model, distortion_model, coefficients)
    points = _camera_points()  # [N,3]
    distortion = np.array(coefficients or [0.0, 0.0, 0.0, 0.0], dtype=np.float64)  # [K]
    if camera_model == "fisheye":
        pixels, _ = cv2.fisheye.projectPoints(
            points[:, None], np.zeros(3), np.zeros(3), _matrix(calibration), distortion
        )
    else:
        pixels, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), _matrix(calibration), distortion)
    rays = unproject_calibration(torch.from_numpy(pixels[:, 0]), calibration)  # [N,3]
    torch.testing.assert_close(rays, torch.from_numpy(points), atol=1e-7, rtol=0)


def test_ftheta_dispatch_matches_native_unprojection() -> None:
    coefficients = [16.0, 16.0, 32.0, 32.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pixels = torch.tensor([[16.0, 16.0], [8.0, 24.0], [30.0, 2.0]], dtype=torch.float64)  # [N,2]
    calibration = {"camera_model": "ftheta", "distortion_model": "ftheta", "distortion_coefficients": coefficients}
    torch.testing.assert_close(unproject_calibration(pixels, calibration), ftheta_unproject(pixels, coefficients))


def test_optical_axis_and_principal_point_agree_across_models() -> None:
    center = torch.tensor([[640.0, 360.0]], dtype=torch.float64)  # [1,2]
    for calibration in (
        _calibration("pinhole", "none", []),
        _calibration("pinhole", "opencv", _OPENCV_COEFFICIENTS),
        _calibration("fisheye", "equidistant", _FISHEYE_COEFFICIENTS),
    ):
        torch.testing.assert_close(unproject_calibration(center, calibration), torch.tensor([[0.0, 0.0, 1.0]]).double())


def test_opencv_outside_invertible_domain_fails_closed() -> None:
    calibration = _calibration("pinhole", "opencv", [-0.5, 0.0, 0.0, 0.0])
    # r - 0.5 r^3 peaks at 0.544; a distorted radius of 0.6 has no undistorted preimage.
    pixels = torch.tensor([[640.0 + 0.6 * 700.0, 360.0]], dtype=torch.float64)  # [1,2]
    with pytest.raises(ValueError, match="OpenCV"):
        unproject_calibration(pixels, calibration)


def test_equidistant_outside_monotonic_domain_fails_closed() -> None:
    calibration = _calibration("fisheye", "equidistant", [-0.5, 0.0, 0.0, 0.0])
    pixels = torch.tensor([[640.0 + 1.5 * 700.0, 360.0]], dtype=torch.float64)  # [1,2]
    with pytest.raises(ValueError, match="fisheye"):
        unproject_calibration(pixels, calibration)


def test_token_grid_uses_integer_pixel_centers() -> None:
    grid = token_pixel_grid(2, 3, 16)  # [2,3,3]
    torch.testing.assert_close(grid[0, 0], torch.tensor([7.5, 7.5, 1.0]).double())
    torch.testing.assert_close(grid[1, 2], torch.tensor([39.5, 23.5, 1.0]).double())


def test_token_geometry_validation_rejects_unsupported_and_out_of_domain_models() -> None:
    affine = torch.eye(3, dtype=torch.float64)  # [3,3]
    validate_token_geometry(_calibration("pinhole", "none", []), affine, image_hw=(720, 1280), stride=16)
    with pytest.raises(ValueError, match="Unsupported camera model"):
        validate_token_geometry(_calibration("orthographic", "none", []), affine, image_hw=(720, 1280), stride=16)
    # The image corners reach a distorted radius beyond the model's invertible range.
    with pytest.raises(ValueError, match="OpenCV"):
        validate_token_geometry(
            _calibration("pinhole", "opencv", [-0.9, 0.0, 0.0, 0.0]), affine, image_hw=(720, 1280), stride=16
        )
