# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Pixel-space degradation primitives.

All functions take frames as ``[T,C,H,W]`` float tensors in [0, 1] on any device and return the
same layout. Parameters are passed in explicitly (they are sampled once per clip by
``degrade.py``), so the only randomness here is the per-pixel noise realisation, drawn from a
caller-owned ``torch.Generator`` that lives on the frame device. Outputs are therefore
deterministic per device; CPU and GPU agree on every sampled parameter but not bit-for-bit on
noise.

Performance notes (1080p, measured on an L4 + 4 CPU threads):
- Blur runs through FFT convolution once the kernel is larger than ``_DIRECT_CONV_MAX_KERNEL``,
  which makes cost independent of kernel size (direct conv2d at k=61 was 40x slower on CPU).
- JPEG uses libjpeg through OpenCV on CPU (about 30 ms per 1080p frame) and the torch DiffJPEG
  simulator on GPU.
- Exact Poisson sampling costs about 300 ms per 1080p frame on CPU with either torch or numpy, so
  ``poisson_mode="gaussian_approx"`` (signal-dependent Gaussian) is provided for CPU workers.

Noise code follows BasicSR ``basicsr/data/degradations.py`` (Apache-2.0).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.augmentors.hr_lr_degradation.diffjpeg import DiffJPEG

_ANTIALIAS_KERNELS = {"bicubic_antialias": "bicubic", "bilinear_antialias": "bilinear"}
_DIRECT_CONV_MAX_KERNEL = 9
JPEG_BACKENDS = ("auto", "cv2", "diffjpeg")
POISSON_MODES = ("auto", "exact", "gaussian_approx")


def make_generator(seed: int, device: torch.device | str) -> torch.Generator:
    """Seeded generator on ``device`` (CUDA generators must live on the tensor device)."""
    dev = torch.device(device)
    gen = torch.Generator(device=dev if dev.type == "cuda" else "cpu")
    gen.manual_seed(int(seed) & 0x7FFF_FFFF_FFFF_FFFF)
    return gen


def _filter2d_direct(frames: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:  # frames: [T,C,H,W]; kernel: [K,K]
    k = kernel.shape[-1]
    t, c, h, w = frames.shape
    pad = k // 2
    padded = F.pad(frames.reshape(t * c, 1, h, w), (pad, pad, pad, pad), mode="reflect")  # [T*C,1,H+2p,W+2p]
    out = F.conv2d(padded, kernel.reshape(1, 1, k, k))  # [T*C,1,H,W]
    return out.reshape(t, c, h, w)  # [T,C,H,W]


def _filter2d_fft(frames: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:  # frames: [T,C,H,W]; kernel: [K,K]
    """Same correlation as ``_filter2d_direct`` (reflect padding, kernel not flipped) via rfft2."""
    k = kernel.shape[-1]
    h, w = frames.shape[-2:]
    pad = k // 2
    padded = F.pad(frames, (pad, pad, pad, pad), mode="reflect")  # [T,C,Hp,Wp]
    hp, wp = padded.shape[-2:]
    # conv2d computes correlation; FFT multiplication computes convolution, so flip the kernel.
    kernel_flipped = torch.flip(kernel, dims=(0, 1))  # [K,K]
    kernel_padded = torch.zeros(hp, wp, device=frames.device, dtype=frames.dtype)  # [Hp,Wp]
    kernel_padded[:k, :k] = kernel_flipped
    kernel_padded = torch.roll(kernel_padded, shifts=(-pad, -pad), dims=(0, 1))  # [Hp,Wp] centred at origin
    spectrum = torch.fft.rfft2(padded) * torch.fft.rfft2(kernel_padded)  # [T,C,Hp,Wp/2+1]
    out = torch.fft.irfft2(spectrum, s=(hp, wp))  # [T,C,Hp,Wp]
    return out[..., pad : pad + h, pad : pad + w]  # [T,C,H,W]


def filter2d(frames: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:  # frames: [T,C,H,W]; kernel: [K,K]
    """Depthwise 2D correlation with one shared odd kernel and reflect padding (torch ``cv2.filter2D``)."""
    k = kernel.shape[-1]
    if k % 2 != 1:
        raise ValueError(f"Kernel size must be odd, got {k}")
    kernel = kernel.to(dtype=frames.dtype, device=frames.device)  # [K,K]
    if k <= _DIRECT_CONV_MAX_KERNEL:
        return _filter2d_direct(frames, kernel)  # [T,C,H,W]
    return _filter2d_fft(frames, kernel)  # [T,C,H,W]


def blur(frames: torch.Tensor, kernel: np.ndarray) -> torch.Tensor:  # frames: [T,C,H,W]; kernel: [K,K]
    """Blur every frame with the same kernel."""
    kernel_t = torch.from_numpy(np.ascontiguousarray(kernel, dtype=np.float32))  # [K,K]
    return filter2d(frames, kernel_t)  # [T,C,H,W]


def resize(frames: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:  # frames: [T,C,H,W]
    """Resize with the Real-ESRGAN interpolation modes (no antialiasing; aliasing is part of the degradation)."""
    if mode not in ("area", "bilinear", "bicubic"):
        raise ValueError(f"Unsupported resize mode {mode!r}")
    if tuple(frames.shape[-2:]) == tuple(size):
        return frames
    if mode == "area":
        return F.interpolate(frames, size=size, mode="area")  # [T,C,h,w]
    return F.interpolate(frames, size=size, mode=mode, align_corners=False)  # [T,C,h,w]


def resize_clean(frames: torch.Tensor, size: tuple[int, int], kernel: str) -> torch.Tensor:  # frames: [T,C,H,W]
    """Antialiased resize used for the clean P0 profile and for benchmark-style LR construction."""
    if kernel == "area":
        return F.interpolate(frames, size=size, mode="area")  # [T,C,h,w]
    if kernel in _ANTIALIAS_KERNELS:
        return F.interpolate(
            frames, size=size, mode=_ANTIALIAS_KERNELS[kernel], align_corners=False, antialias=True
        )  # [T,C,h,w]
    raise ValueError(f"Unsupported clean resize kernel {kernel!r}")


def _randn_like_shape(shape: tuple[int, ...], gen: torch.Generator, like: torch.Tensor) -> torch.Tensor:
    # returns [*shape] on like.device
    return torch.randn(shape, generator=gen, dtype=like.dtype, device=like.device)


def add_gaussian_noise(
    frames: torch.Tensor, sigma: float, gray: bool, gen: torch.Generator
) -> torch.Tensor:  # frames: [T,C,H,W]
    """Additive Gaussian noise with standard deviation ``sigma`` in 8-bit units."""
    t, c, h, w = frames.shape
    if gray:
        noise = _randn_like_shape((t, 1, h, w), gen, frames).expand(t, c, h, w)  # [T,C,H,W]
    else:
        noise = _randn_like_shape((t, c, h, w), gen, frames)  # [T,C,H,W]
    return (frames + noise * (sigma / 255.0)).clamp_(0.0, 1.0)  # [T,C,H,W]


def _levels_per_frame(img_q: torch.Tensor) -> torch.Tensor:  # img_q: [T,C,H,W] quantised to 1/255; returns [T,1,1,1]
    """Number of distinct 8-bit levels per frame rounded up to a power of two (BasicSR ``vals``)."""
    t = img_q.shape[0]
    codes = (img_q * 255.0).round().long().reshape(t, -1)  # [T,C*H*W]
    counts = [int(torch.bincount(codes[i], minlength=256).count_nonzero().item()) for i in range(t)]
    levels = [2 ** int(np.ceil(np.log2(max(n, 1)))) for n in counts]
    return img_q.new_tensor(levels).view(t, 1, 1, 1)  # [T,1,1,1]


def _poisson_noise(img: torch.Tensor, gen: torch.Generator, mode: str) -> torch.Tensor:  # img: [T,C,H,W]
    """Shot noise whose rate scales with the number of distinct 8-bit levels per frame, as in BasicSR."""
    img_q = (img * 255.0).round().clamp(0.0, 255.0) / 255.0  # [T,C,H,W]
    vals = _levels_per_frame(img_q)  # [T,1,1,1]
    rates = img_q * vals  # [T,C,H,W]
    if mode == "exact":
        sampled = torch.poisson(rates, generator=gen)  # [T,C,H,W]
    elif mode == "gaussian_approx":
        # Poisson(lambda) ~ N(lambda, lambda) for moderate lambda; keeps the signal-dependent variance.
        sampled = rates + rates.sqrt() * _randn_like_shape(tuple(rates.shape), gen, rates)  # [T,C,H,W]
        sampled = sampled.round().clamp_(min=0.0)  # [T,C,H,W]
    else:
        raise ValueError(f"Unknown poisson mode {mode!r}; expected one of {POISSON_MODES[1:]}")
    return sampled / vals - img_q  # [T,C,H,W]


def resolve_poisson_mode(mode: str, device: torch.device) -> str:
    if mode == "auto":
        return "exact" if device.type == "cuda" else "gaussian_approx"
    if mode not in POISSON_MODES:
        raise ValueError(f"Unknown poisson mode {mode!r}")
    return mode


def add_poisson_noise(
    frames: torch.Tensor, scale: float, gray: bool, gen: torch.Generator, mode: str = "auto"
) -> torch.Tensor:  # frames: [T,C,H,W]
    """Poisson (shot) noise scaled by ``scale``; grey noise is computed on the luminance and shared across channels."""
    mode = resolve_poisson_mode(mode, frames.device)
    t, c, h, w = frames.shape
    if gray:
        weights = frames.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)  # [1,3,1,1]
        luma = (frames * weights).sum(dim=1, keepdim=True)  # [T,1,H,W]
        noise = _poisson_noise(luma, gen, mode).expand(t, c, h, w)  # [T,C,H,W]
    else:
        noise = _poisson_noise(frames, gen, mode)  # [T,C,H,W]
    return (frames + noise * scale).clamp_(0.0, 1.0)  # [T,C,H,W]


def _jpeg_cv2(frames: torch.Tensor, quality: float) -> torch.Tensor:  # frames: [T,C,H,W] CPU float in [0,1]
    """libjpeg round trip per frame through OpenCV (RGB <-> BGR handled here)."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(round(quality))]
    frames_u8 = (frames.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()  # [T,H,W,C]
    out = np.empty_like(frames_u8)  # [T,H,W,C]
    for i in range(frames_u8.shape[0]):
        bgr = cv2.cvtColor(np.ascontiguousarray(frames_u8[i]), cv2.COLOR_RGB2BGR)  # [H,W,C]
        ok, encoded = cv2.imencode(".jpg", bgr, encode_param)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)  # [H,W,C]
        out[i] = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(out).permute(0, 3, 1, 2).to(frames.dtype) / 255.0  # [T,C,H,W]


def resolve_jpeg_backend(backend: str, device: torch.device) -> str:
    if backend == "auto":
        return "cv2" if device.type == "cpu" else "diffjpeg"
    if backend not in JPEG_BACKENDS:
        raise ValueError(f"Unknown JPEG backend {backend!r}")
    if backend == "cv2" and device.type != "cpu":
        raise ValueError("JPEG backend 'cv2' requires CPU tensors")
    return backend


def jpeg(frames: torch.Tensor, quality: float, jpeger: DiffJPEG, backend: str = "auto") -> torch.Tensor:
    # frames: [T,C,H,W]; returns [T,C,H,W]
    """JPEG round trip at one quality for the whole chunk."""
    backend = resolve_jpeg_backend(backend, frames.device)
    if backend == "cv2":
        return _jpeg_cv2(frames, quality)  # [T,C,H,W]
    if jpeger.y_table.device != frames.device:
        jpeger.to(frames.device)
    return jpeger(frames.clamp(0.0, 1.0), quality=quality)  # [T,C,H,W]


def to_uint8(frames: torch.Tensor) -> torch.Tensor:  # frames: [T,C,H,W] float in [0,1], returns uint8
    return (frames.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)  # [T,C,H,W]
