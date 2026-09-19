# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Blur kernel generators for the Real-ESRGAN style degradation pipeline.

Adapted from BasicSR ``basicsr/data/degradations.py`` (Apache-2.0)
https://github.com/XPixelGroup/BasicSR/blob/8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a/basicsr/data/degradations.py
via the Cosmos Transfer1 corruptors. Every random draw goes through an explicit
``numpy.random.Generator`` so a clip's kernel is reproducible from its seed.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import special

KERNEL_TYPES = ("iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso")


def sigma_matrix2(sig_x: float, sig_y: float, theta: float) -> np.ndarray:  # returns [2,2]
    """Rotated covariance matrix of a bivariate Gaussian."""
    d_matrix = np.array([[sig_x**2, 0], [0, sig_y**2]])  # [2,2]
    u_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])  # [2,2]
    return np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))  # [2,2]


def mesh_grid(kernel_size: int) -> np.ndarray:  # returns [K,K,2]
    """Coordinate grid centred at zero."""
    ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)  # [K]
    xx, yy = np.meshgrid(ax, ax)  # [K,K] each
    return np.stack([xx, yy], axis=-1)  # [K,K,2]


def _quadratic_form(sigma_matrix: np.ndarray, grid: np.ndarray) -> np.ndarray:  # returns [K,K]
    inverse_sigma = np.linalg.inv(sigma_matrix)  # [2,2]
    return np.sum(np.dot(grid, inverse_sigma) * grid, 2)  # [K,K]


def _sigma_matrix(sig_x: float, sig_y: float, theta: float, isotropic: bool) -> np.ndarray:  # returns [2,2]
    if isotropic:
        return np.array([[sig_x**2, 0], [0, sig_x**2]])  # [2,2]
    return sigma_matrix2(sig_x, sig_y, theta)  # [2,2]


def bivariate_gaussian(
    kernel_size: int, sig_x: float, sig_y: float, theta: float, isotropic: bool = True
) -> np.ndarray:  # returns [K,K]
    """Normalised isotropic or anisotropic Gaussian kernel."""
    grid = mesh_grid(kernel_size)  # [K,K,2]
    kernel = np.exp(-0.5 * _quadratic_form(_sigma_matrix(sig_x, sig_y, theta, isotropic), grid))  # [K,K]
    return kernel / np.sum(kernel)  # [K,K]


def bivariate_generalized_gaussian(
    kernel_size: int, sig_x: float, sig_y: float, theta: float, beta: float, isotropic: bool = True
) -> np.ndarray:  # returns [K,K]
    """Normalised generalized Gaussian kernel; ``beta == 1`` is the plain Gaussian."""
    grid = mesh_grid(kernel_size)  # [K,K,2]
    q = _quadratic_form(_sigma_matrix(sig_x, sig_y, theta, isotropic), grid)  # [K,K]
    kernel = np.exp(-0.5 * np.power(q, beta))  # [K,K]
    return kernel / np.sum(kernel)  # [K,K]


def bivariate_plateau(
    kernel_size: int, sig_x: float, sig_y: float, theta: float, beta: float, isotropic: bool = True
) -> np.ndarray:  # returns [K,K]
    """Normalised plateau-shaped kernel ``1 / (1 + q^beta)``."""
    grid = mesh_grid(kernel_size)  # [K,K,2]
    q = _quadratic_form(_sigma_matrix(sig_x, sig_y, theta, isotropic), grid)  # [K,K]
    kernel = np.reciprocal(np.power(q, beta) + 1)  # [K,K]
    return kernel / np.sum(kernel)  # [K,K]


def _sample_sigma_rotation(
    rng: np.random.Generator,
    sigma_x_range: Sequence[float],
    sigma_y_range: Sequence[float],
    rotation_range: Sequence[float],
    isotropic: bool,
) -> tuple[float, float, float]:
    assert sigma_x_range[0] < sigma_x_range[1], "Wrong sigma_x_range."
    sigma_x = float(rng.uniform(sigma_x_range[0], sigma_x_range[1]))
    if isotropic:
        return sigma_x, sigma_x, 0.0
    assert sigma_y_range[0] < sigma_y_range[1], "Wrong sigma_y_range."
    assert rotation_range[0] < rotation_range[1], "Wrong rotation_range."
    sigma_y = float(rng.uniform(sigma_y_range[0], sigma_y_range[1]))
    rotation = float(rng.uniform(rotation_range[0], rotation_range[1]))
    return sigma_x, sigma_y, rotation


def _sample_beta(rng: np.random.Generator, beta_range: Sequence[float]) -> float:
    # Real-ESRGAN draws below or above 1 with equal probability so both regimes are covered.
    if rng.uniform() < 0.5:
        return float(rng.uniform(beta_range[0], 1))
    return float(rng.uniform(1, beta_range[1]))


def random_mixed_kernel(
    rng: np.random.Generator,
    kernel_list: Sequence[str],
    kernel_prob: Sequence[float],
    kernel_size: int,
    sigma_x_range: Sequence[float],
    sigma_y_range: Sequence[float],
    rotation_range: Sequence[float] = (-math.pi, math.pi),
    betag_range: Sequence[float] = (0.5, 8),
    betap_range: Sequence[float] = (0.5, 8),
) -> tuple[np.ndarray, dict]:  # returns ([K,K], sampled parameters)
    """Sample one kernel type from ``kernel_list`` and its parameters, seeded by ``rng``."""
    assert kernel_size % 2 == 1, "Kernel size must be an odd number."
    assert len(kernel_list) == len(kernel_prob), "kernel_list and kernel_prob must have equal length."
    prob = np.asarray(kernel_prob, dtype=np.float64)  # [N]
    kernel_type = str(rng.choice(np.asarray(kernel_list), p=prob / prob.sum()))
    if kernel_type not in KERNEL_TYPES:
        raise ValueError(f"Unknown kernel type {kernel_type}; supported: {KERNEL_TYPES}")
    isotropic = kernel_type.endswith("iso") and not kernel_type.endswith("aniso")
    sigma_x, sigma_y, rotation = _sample_sigma_rotation(rng, sigma_x_range, sigma_y_range, rotation_range, isotropic)
    info = {"kernel_type": kernel_type, "kernel_size": kernel_size, "sigma_x": sigma_x, "sigma_y": sigma_y}
    if not isotropic:
        info["rotation"] = rotation
    if kernel_type in ("iso", "aniso"):
        kernel = bivariate_gaussian(kernel_size, sigma_x, sigma_y, rotation, isotropic)  # [K,K]
    elif kernel_type in ("generalized_iso", "generalized_aniso"):
        beta = _sample_beta(rng, betag_range)
        info["beta"] = beta
        kernel = bivariate_generalized_gaussian(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic)  # [K,K]
    else:
        beta = _sample_beta(rng, betap_range)
        info["beta"] = beta
        kernel = bivariate_plateau(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic)  # [K,K]
    return kernel, info


def circular_lowpass_kernel(cutoff: float, kernel_size: int, pad_to: int = 0) -> np.ndarray:  # returns [P,P]
    """2D circularly symmetric sinc low-pass filter.

    Reference: https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter

    Args:
        cutoff: cutoff frequency in radians; ``pi`` is the maximum.
        kernel_size: odd spatial size of the kernel.
        pad_to: zero-pad the kernel to this odd size when larger than ``kernel_size``.
    """
    assert kernel_size % 2 == 1, "Kernel size must be an odd number."
    centre = (kernel_size - 1) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = np.fromfunction(
            lambda x, y: cutoff
            * special.j1(cutoff * np.sqrt((x - centre) ** 2 + (y - centre) ** 2))
            / (2 * np.pi * np.sqrt((x - centre) ** 2 + (y - centre) ** 2)),
            [kernel_size, kernel_size],
        )  # [K,K]
    kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = cutoff**2 / (4 * np.pi)
    kernel = kernel / np.sum(kernel)  # [K,K]
    if pad_to > kernel_size:
        pad_size = (pad_to - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))  # [P,P]
    return kernel


def random_sinc_kernel(
    rng: np.random.Generator, kernel_size: int, cutoff_range: Sequence[float] | None = None
) -> tuple[np.ndarray, dict]:  # returns ([K,K], info)
    """Sinc kernel. ``cutoff_range`` bounds omega_c; ``None`` uses the Real-ESRGAN prior, which widens the range for
    kernels of size 13 and above."""
    if cutoff_range is None:
        cutoff_range = (np.pi / 3, np.pi) if kernel_size < 13 else (np.pi / 5, np.pi)
    omega_c = float(rng.uniform(*cutoff_range))
    kernel = circular_lowpass_kernel(omega_c, kernel_size)  # [K,K]
    return kernel, {"kernel_type": "sinc", "kernel_size": kernel_size, "omega_c": omega_c}


def scale_kernel_size(kernel_size: int, factor: float, max_kernel_size: int) -> int:
    """Scale an odd kernel size by ``factor`` and return the nearest odd size within ``[3, max_kernel_size]``."""
    scaled = int(round(kernel_size * factor))
    scaled = max(3, min(scaled, max_kernel_size))
    if scaled % 2 == 0:
        scaled = scaled - 1 if scaled >= max_kernel_size else scaled + 1
    return scaled
