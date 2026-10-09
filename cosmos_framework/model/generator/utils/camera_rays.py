# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -----------------------------------------------------------------------------

"""Unproject native calibration pixels to camera rays for every supported lens model.

Pixel coordinates use the calibration's own integer-center convention. Every inverse is
checked by reprojection so a token center outside a lens model's invertible domain fails
closed instead of producing a plausible but wrong ray.
"""

import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import torch

from cosmos_framework.model.generator.utils.ftheta import ftheta_unproject

# Lens models RigRoPE can turn into rays; ``_validate_calibration`` owns their coefficient layouts.
SUPPORTED_RAY_CAMERA_MODELS = frozenset({"ftheta", "pinhole", "fisheye"})
_NEWTON_ITERATIONS = 40
_REPROJECTION_TOLERANCE = 1e-9
_JACOBIAN_STEP = 1e-7


def _matmul3(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


@lru_cache(maxsize=64)
def _opencv_tilt_matrix(tau_x: float, tau_y: float) -> tuple[tuple[float, ...], ...]:  # returns [3,3]
    """OpenCV's tilted-sensor projection (``computeTiltProjectionMatrix``) for angles in radians."""
    cx, sx, cy, sy = math.cos(tau_x), math.sin(tau_x), math.cos(tau_y), math.sin(tau_y)
    rotation = _matmul3(
        [[cy, 0.0, -sy], [0.0, 1.0, 0.0], [sy, 0.0, cy]], [[1.0, 0.0, 0.0], [0.0, cx, sx], [0.0, -sx, cx]]
    )
    projection = [
        [rotation[2][2], 0.0, -rotation[0][2]],
        [0.0, rotation[2][2], -rotation[1][2]],
        [0.0, 0.0, 1.0],
    ]
    return tuple(tuple(row) for row in _matmul3(projection, rotation))


def _opencv_distort(points: torch.Tensor, coefficients: Sequence[float]) -> torch.Tensor:  # [...,2] -> [...,2]
    """Apply OpenCV distortion (4, 5, 8, 12, or 14 coefficients) to normalized points.

    The layout is ``k1 k2 p1 p2 [k3 [k4 k5 k6 [s1 s2 s3 s4 [tau_x tau_y]]]]``: radial,
    tangential, thin-prism and tilted-sensor terms, as in ``cv2.projectPoints``.
    """
    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tau_x, tau_y = (*coefficients, *([0.0] * 10))[:14]
    x, y = points[..., 0], points[..., 1]  # [...], [...]
    r2 = x * x + y * y  # [...]
    r4 = r2 * r2  # [...]
    radial = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))  # [...]
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) + s1 * r2 + s2 * r4  # [...]
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y + s3 * r2 + s4 * r4  # [...]
    if tau_x or tau_y:
        tilt = _opencv_tilt_matrix(tau_x, tau_y)
        depth = tilt[2][0] * xd + tilt[2][1] * yd + tilt[2][2]  # [...]
        xd, yd = (
            (tilt[0][0] * xd + tilt[0][1] * yd + tilt[0][2]) / depth,
            (tilt[1][0] * xd + tilt[1][1] * yd + tilt[1][2]) / depth,
        )  # [...], [...]
    return torch.stack((xd, yd), dim=-1)  # [...,2]


def _opencv_undistort(distorted: torch.Tensor, coefficients: Sequence[float]) -> torch.Tensor:  # [...,2] -> [...,2]
    """Invert OpenCV distortion by Newton iteration from the distorted point."""
    if len(coefficients) not in (4, 5, 8, 12, 14):
        raise ValueError(
            f"RigRoPE supports OpenCV distortion with 4, 5, 8, 12, or 14 coefficients, got {len(coefficients)}"
        )
    points = distorted.clone()  # [...,2]
    step_x = distorted.new_tensor([_JACOBIAN_STEP, 0.0])  # [2]
    step_y = distorted.new_tensor([0.0, _JACOBIAN_STEP])  # [2]
    for _ in range(_NEWTON_ITERATIONS):
        residual = _opencv_distort(points, coefficients) - distorted  # [...,2]
        column_x = (_opencv_distort(points + step_x, coefficients) - _opencv_distort(points - step_x, coefficients)) / (
            2 * _JACOBIAN_STEP
        )  # [...,2]
        column_y = (_opencv_distort(points + step_y, coefficients) - _opencv_distort(points - step_y, coefficients)) / (
            2 * _JACOBIAN_STEP
        )  # [...,2]
        det = column_x[..., 0] * column_y[..., 1] - column_y[..., 0] * column_x[..., 1]  # [...]
        if (det <= 0).any():
            raise ValueError("Token centers fall outside the invertible OpenCV distortion domain")
        dx = (column_y[..., 1] * residual[..., 0] - column_y[..., 0] * residual[..., 1]) / det  # [...]
        dy = (column_x[..., 0] * residual[..., 1] - column_x[..., 1] * residual[..., 0]) / det  # [...]
        points = points - torch.stack((dx, dy), dim=-1)  # [...,2]
    error = torch.linalg.vector_norm(_opencv_distort(points, coefficients) - distorted, dim=-1)  # [...]
    if not torch.isfinite(points).all() or (error > _REPROJECTION_TOLERANCE).any():
        raise ValueError("OpenCV undistortion did not converge at the token centers")
    return points  # [...,2]


def _equidistant_unproject(distorted: torch.Tensor, coefficients: Sequence[float]) -> torch.Tensor:
    """Invert OpenCV fisheye ``theta_d = theta (1 + k1 theta^2 + ... + k4 theta^8)``; [...,2] -> [...,3]."""
    if len(coefficients) != 4:
        raise ValueError(f"Equidistant fisheye calibration requires 4 coefficients, got {len(coefficients)}")
    k1, k2, k3, k4 = coefficients
    theta_d = torch.linalg.vector_norm(distorted, dim=-1)  # [...]
    theta = theta_d.clone()  # [...]
    for _ in range(_NEWTON_ITERATIONS):
        t2 = theta * theta  # [...]
        value = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - theta_d  # [...]
        slope = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))  # [...]
        if (slope <= 0).any():
            raise ValueError("Token centers fall outside the monotonic equidistant fisheye domain")
        theta = theta - value / slope  # [...]
    t2 = theta * theta  # [...]
    error = (theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - theta_d).abs()  # [...]
    if not torch.isfinite(theta).all() or (error > _REPROJECTION_TOLERANCE).any():
        raise ValueError("Equidistant fisheye inversion did not converge at the token centers")
    if (theta < 0).any() or (theta >= math.pi).any():
        raise ValueError("Equidistant fisheye token ray angles must lie in [0, pi)")
    scale = torch.sin(theta) / theta_d.clamp_min(1e-15)  # [...]
    rays = torch.cat((distorted * scale[..., None], torch.cos(theta)[..., None]), dim=-1)  # [...,3]
    optical_axis = rays.new_tensor([0.0, 0.0, 1.0])  # [3]
    return torch.where((theta_d <= 1e-12)[..., None], optical_axis, rays)  # [...,3]


def unproject_calibration(pixels: torch.Tensor, calibration: Mapping[str, Any]) -> torch.Tensor:
    """Return unit camera rays (OpenCV axes) for native pixels; pixels [...,2] -> rays [...,3]."""
    if pixels.shape[-1] != 2 or not pixels.is_floating_point() or not torch.isfinite(pixels).all():
        raise ValueError("Calibration pixels must be finite floating-point coordinates ending in two values")
    model = calibration["camera_model"]
    coefficients = [float(value) for value in calibration["distortion_coefficients"]]
    if model == "ftheta":
        return ftheta_unproject(pixels, coefficients)  # [...,3]
    original_dtype = pixels.dtype
    pixels = pixels.double()  # [...,2]
    focal = pixels.new_tensor([calibration["fx_px"], calibration["fy_px"]])  # [2]
    center = pixels.new_tensor([calibration["cx_px"], calibration["cy_px"]])  # [2]
    distorted = (pixels - center) / focal  # [...,2]
    if model == "pinhole":
        distortion = calibration["distortion_model"]
        if distortion == "none":
            points = distorted  # [...,2]
        elif distortion == "opencv":
            points = _opencv_undistort(distorted, coefficients)  # [...,2]
        else:
            raise ValueError(f"Unsupported pinhole distortion model for RigRoPE rays: {distortion}")
        rays = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)  # [...,3]
        rays = rays / torch.linalg.vector_norm(rays, dim=-1, keepdim=True)  # [...,3]
    elif model == "fisheye":
        if calibration["distortion_model"] != "equidistant":
            raise ValueError(
                f"Unsupported fisheye distortion model for RigRoPE rays: {calibration['distortion_model']}"
            )
        rays = _equidistant_unproject(distorted, coefficients)  # [...,3]
    else:
        raise ValueError(f"Unsupported camera model for RigRoPE rays: {model}")
    return rays.to(dtype=original_dtype)  # [...,3]


def token_pixel_grid(height: int, width: int, stride: int) -> torch.Tensor:  # returns [H,W,3]
    """Homogeneous centers of ``stride``-pixel tokens in integer-center pixel coordinates."""
    ys = (torch.arange(height, dtype=torch.float64) + 0.5) * stride - 0.5  # [H]
    xs = (torch.arange(width, dtype=torch.float64) + 0.5) * stride - 0.5  # [W]
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [H,W], [H,W]
    return torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)  # [H,W,3]


def _calibration_key(calibration: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        calibration["camera_model"],
        calibration["distortion_model"],
        tuple(float(value) for value in calibration["distortion_coefficients"]),
        *(float(calibration[key]) for key in ("fx_px", "fy_px", "cx_px", "cy_px")),
    )


@lru_cache(maxsize=256)
def _validate_cached(
    key: tuple[Any, ...], affine_values: tuple[float, ...], image_hw: tuple[int, int], stride: int
) -> None:
    model, distortion, coefficients, fx, fy, cx, cy = key
    calibration = {
        "camera_model": model,
        "distortion_model": distortion,
        "distortion_coefficients": coefficients,
        "fx_px": fx,
        "fy_px": fy,
        "cx_px": cx,
        "cy_px": cy,
    }
    height, width = image_hw
    pixels = token_pixel_grid(math.ceil(height / stride), math.ceil(width / stride), stride)  # [H,W,3]
    affine = torch.tensor(affine_values, dtype=torch.float64).reshape(3, 3)  # [3,3]
    if not torch.isfinite(affine).all() or abs(float(torch.linalg.det(affine))) < 1e-12:
        raise ValueError("image_from_calibration must be finite and invertible")
    native = pixels @ torch.linalg.inv(affine).T  # [H,W,3]
    unproject_calibration(native[..., :2] / native[..., 2:], calibration)  # [H,W,3]


def validate_token_geometry(
    calibration: Mapping[str, Any],
    image_from_calibration: torch.Tensor,  # [3,3]
    *,
    image_hw: tuple[int, int],
    stride: int,
) -> None:
    """Validate the pre-interpolation latent-center rays RigRoPE will consume."""
    if calibration["camera_model"] not in SUPPORTED_RAY_CAMERA_MODELS:
        raise ValueError(f"Unsupported camera model for RigRoPE rays: {calibration['camera_model']}")
    _validate_cached(
        _calibration_key(calibration),
        tuple(float(value) for value in image_from_calibration.flatten()),
        image_hw,
        stride,
    )
