# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# -----------------------------------------------------------------------------

"""Native F-theta unprojection, independent of the attention encoding."""

import math
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import torch


def _polynomial(values: torch.Tensor, coefficients: Sequence[float]) -> torch.Tensor:  # [...], [K] -> [...]
    result = torch.zeros_like(values)  # [...]
    for coefficient in reversed(coefficients):
        result = result * values + coefficient  # [...]
    return result


@lru_cache(maxsize=256)
def _forward_domain(coefficients: tuple[float, ...]) -> float:
    """Return the first positive turning point, bounded by the ray sphere."""
    derivative = [index * value for index, value in enumerate(coefficients)][1:]
    if derivative[0] <= 0:
        raise ValueError("F-theta forward polynomial must increase at the optical axis")
    roots = np.polynomial.polynomial.polyroots(derivative)
    boundaries = [float(root.real) for root in roots if abs(root.imag) < 1e-8 and root.real > 0]
    return min([math.pi - 1e-8, *boundaries])


def ftheta_unproject(
    pixels: torch.Tensor, coefficients: Sequence[float]
) -> torch.Tensor:  # pixels: [...,2], returns [...,3]
    """Unproject native pixel centers using either direction of the provider polynomial.

    The optional final three coefficients describe A=[[c,d],[e,1]]. Forward
    polynomials are inverted on their first monotonic interval rather than using
    an equivalent pinhole K or an unconstrained fitted inverse polynomial.
    """
    values = tuple(float(value) for value in coefficients)
    if len(values) not in (11, 14) or not all(math.isfinite(value) for value in values):
        raise ValueError("F-theta calibration requires 11 or 14 finite coefficients")
    if pixels.shape[-1] != 2 or min(values[2:4]) <= 0:
        raise ValueError("F-theta pixels must end in two coordinates and image dimensions must be positive")
    if not pixels.is_floating_point() or not torch.isfinite(pixels).all():
        raise ValueError("F-theta pixels must have finite floating-point coordinates")
    original_dtype = pixels.dtype
    pixels = pixels.double()  # [...,2]
    center = pixels.new_tensor(values[:2])  # [2]
    c, d, e = values[11:] if len(values) == 14 else (1.0, 0.0, 0.0)
    if abs(c - d * e) < 1e-12:
        raise ValueError("F-theta affine lens matrix is singular")
    inverse_affine = pixels.new_tensor([[1.0, -d], [-e, c]]) / (c - d * e)  # [2,2]
    offset = (pixels - center) @ inverse_affine.T  # [...,2]
    radius = torch.linalg.vector_norm(offset, dim=-1)  # [...]
    polynomial = values[4:10]
    if values[10] > 0:
        angle = _polynomial(radius, polynomial)  # [...]
        derivative = _polynomial(radius, [index * value for index, value in enumerate(polynomial)][1:])  # [...]
        if (derivative <= 0).any():
            raise ValueError("Token centers fall outside the monotonic F-theta backward domain")
    else:
        upper_angle = _forward_domain(polynomial)
        upper = torch.full_like(radius, upper_angle)  # [...]
        lower = torch.zeros_like(radius)  # [...]
        if (radius > _polynomial(upper, polynomial) + 1e-7).any() or (radius < polynomial[0] - 1e-7).any():
            raise ValueError("Token centers fall outside the invertible F-theta forward domain")
        for _ in range(48):
            midpoint = (lower + upper) * 0.5  # [...]
            below = _polynomial(midpoint, polynomial) < radius  # [...]
            lower = torch.where(below, midpoint, lower)  # [...]
            upper = torch.where(below, upper, midpoint)  # [...]
        angle = (lower + upper) * 0.5  # [...]
    if not torch.isfinite(angle).all() or (angle < -1e-8).any() or (angle >= math.pi).any():
        raise ValueError("F-theta token ray angles must lie in [0, pi)")
    scale = torch.sin(angle) / radius.clamp_min(1e-15)  # [...]
    rays = torch.cat((offset * scale[..., None], torch.cos(angle)[..., None]), dim=-1)  # [...,3]
    optical_axis = rays.new_tensor([0.0, 0.0, 1.0])  # [3]
    rays = torch.where((radius <= 1e-12)[..., None], optical_axis, rays)  # [...,3]
    return rays.to(dtype=original_dtype)  # [...,3]


@lru_cache(maxsize=128)
def _validate_cached(
    coefficients: tuple[float, ...], affine_values: tuple[float, ...], image_hw: tuple[int, int], stride: int
) -> None:
    height, width = image_hw
    ys = (torch.arange(math.ceil(height / stride), dtype=torch.float64) + 0.5) * stride  # [H]
    xs = (torch.arange(math.ceil(width / stride), dtype=torch.float64) + 0.5) * stride  # [W]
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [H,W], [H,W]
    pixels = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)  # [H,W,3]
    affine = torch.tensor(affine_values, dtype=torch.float64).reshape(3, 3)  # [3,3]
    if not torch.isfinite(affine).all() or abs(float(torch.linalg.det(affine))) < 1e-12:
        raise ValueError("image_from_calibration must be finite and invertible")
    native = pixels @ torch.linalg.inv(affine).T  # [H,W,3]
    ftheta_unproject(native[..., :2] / native[..., 2:], coefficients)  # [H,W,3]


def validate_ftheta_token_geometry(
    coefficients: Sequence[float],
    image_from_calibration: torch.Tensor,  # [3,3]
    *,
    image_hw: tuple[int, int],
    stride: int,
) -> None:
    """Validate pre-interpolation latent-center rays consumed by RigRoPE."""
    _validate_cached(
        tuple(float(value) for value in coefficients),
        tuple(float(value) for value in image_from_calibration.flatten()),
        image_hw,
        stride,
    )
