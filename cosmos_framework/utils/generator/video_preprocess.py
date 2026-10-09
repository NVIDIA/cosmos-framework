# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import math

import numpy as np
import torch
import torchvision.transforms.functional as torchvision_F
from PIL import Image


def resize_and_center_crop_tensor(frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Aspect-ratio-preserving resize followed by center crop."""
    if frames.dim() < 2:
        raise ValueError(f"Expected image-like tensor with trailing H,W dimensions, got shape {tuple(frames.shape)}.")
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Target size must be positive, got H={target_h}, W={target_w}.")
    orig_h, orig_w = int(frames.shape[-2]), int(frames.shape[-1])
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError(f"Source size must be positive, got H={orig_h}, W={orig_w}.")
    scaling_ratio = max(target_w / orig_w, target_h / orig_h)
    resize_h = int(math.ceil(scaling_ratio * orig_h))
    resize_w = int(math.ceil(scaling_ratio * orig_w))
    resized_frames = torchvision_F.resize(frames, [resize_h, resize_w], antialias=True)  # [...,resize_h,resize_w]
    return torchvision_F.center_crop(resized_frames, [target_h, target_w])  # [...,target_h,target_w]


def resize_and_center_crop_affine(source_hw: tuple[int, int], target_hw: tuple[int, int]) -> torch.Tensor:
    """Source-to-output pixel affine of ``resize_and_center_crop_tensor`` in integer-center coordinates."""
    (orig_h, orig_w), (target_h, target_w) = source_hw, target_hw
    if min(orig_h, orig_w, target_h, target_w) <= 0:
        raise ValueError(f"Image sizes must be positive, got source={source_hw}, target={target_hw}.")
    scaling_ratio = max(target_w / orig_w, target_h / orig_h)
    resize_h = int(math.ceil(scaling_ratio * orig_h))
    resize_w = int(math.ceil(scaling_ratio * orig_w))
    # Matches torchvision center_crop offsets, which round half to even.
    top, left = int(round((resize_h - target_h) / 2.0)), int(round((resize_w - target_w) / 2.0))
    sx, sy = resize_w / orig_w, resize_h / orig_h
    return torch.tensor(  # [3,3]
        [[sx, 0.0, (sx - 1) / 2 - left], [0.0, sy, (sy - 1) / 2 - top], [0.0, 0.0, 1.0]], dtype=torch.float64
    )


def _resize_and_crop_window(
    source_hw: tuple[int, int], target_hw: tuple[int, int], zoom: float, crop_center: tuple[float, float]
) -> tuple[int, int, int, int]:
    """Resized size and crop origin ``(resize_h, resize_w, top, left)`` of ``resize_and_crop_tensor``."""
    (orig_h, orig_w), (target_h, target_w) = source_hw, target_hw
    if min(orig_h, orig_w, target_h, target_w) <= 0:
        raise ValueError(f"Image sizes must be positive, got source={source_hw}, target={target_hw}.")
    if not (math.isfinite(zoom) and zoom >= 1.0):
        raise ValueError(f"Crop zoom must be finite and at least 1, got {zoom}.")
    if not all(0.0 <= fraction <= 1.0 for fraction in crop_center):
        raise ValueError(f"Crop center fractions must lie in [0, 1], got {crop_center}.")
    scaling_ratio = max(target_w / orig_w, target_h / orig_h) * zoom
    resize_h = int(math.ceil(scaling_ratio * orig_h))
    resize_w = int(math.ceil(scaling_ratio * orig_w))
    # Round half to even, as torchvision center_crop does, so (0.5, 0.5) is its center crop.
    top = int(round(crop_center[0] * (resize_h - target_h)))
    left = int(round(crop_center[1] * (resize_w - target_w)))
    return resize_h, resize_w, top, left


def resize_and_crop_tensor(
    frames: torch.Tensor,
    target_h: int,
    target_w: int,
    *,
    zoom: float = 1.0,
    crop_center: tuple[float, float] = (0.5, 0.5),
) -> torch.Tensor:
    """Aspect-ratio-preserving resize to ``zoom`` times the covering size, then a target-size crop.

    ``crop_center`` places the crop along the slack in each axis, (top, left) fractions from 0
    (first rows or columns) to 1 (last); ``zoom=1`` at (0.5, 0.5) is ``resize_and_center_crop_tensor``.
    """
    if frames.dim() < 2:
        raise ValueError(f"Expected image-like tensor with trailing H,W dimensions, got shape {tuple(frames.shape)}.")
    resize_h, resize_w, top, left = _resize_and_crop_window(
        (int(frames.shape[-2]), int(frames.shape[-1])), (target_h, target_w), zoom, crop_center
    )
    resized_frames = torchvision_F.resize(frames, [resize_h, resize_w], antialias=True)  # [...,resize_h,resize_w]
    return resized_frames[..., top : top + target_h, left : left + target_w]  # [...,target_h,target_w]


def resize_and_crop_affine(
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
    *,
    zoom: float = 1.0,
    crop_center: tuple[float, float] = (0.5, 0.5),
) -> torch.Tensor:
    """Source-to-output pixel affine of ``resize_and_crop_tensor`` in integer-center coordinates."""
    resize_h, resize_w, top, left = _resize_and_crop_window(source_hw, target_hw, zoom, crop_center)
    sx, sy = resize_w / source_hw[1], resize_h / source_hw[0]
    return torch.tensor(  # [3,3]
        [[sx, 0.0, (sx - 1) / 2 - left], [0.0, sy, (sy - 1) / 2 - top], [0.0, 0.0, 1.0]], dtype=torch.float64
    )


def tensor_to_pil_images(video_tensor: torch.Tensor, *, channels_first: bool | None = None) -> list[Image.Image]:
    """Convert a video tensor of shape (C, T, H, W) or (T, C, H, W) into a list of PIL images.

    Args:
        video_tensor: Video tensor with shape (C, T, H, W) or (T, C, H, W).
        channels_first: Specify the layout when it cannot be inferred, such as for three-frame RGB videos.
            If omitted, the helper retains its original layout inference.

    Returns:
        One PIL image per frame.
    """
    if channels_first is None:
        channels_first = video_tensor.shape[0] == 3 and video_tensor.shape[1] > 3
    # (C, T, H, W) -> (T, C, H, W)
    if channels_first:
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # [T,C,H,W]

    # (T, C, H, W) -> (T, H, W, C) and detach to CPU numpy.
    video_np = video_tensor.permute(0, 2, 3, 1).cpu().numpy()

    # PIL expects uint8 with values in [0, 255]; rescale floats accordingly.
    if video_np.dtype == np.float32 or video_np.dtype == np.float64:
        if video_np.max() <= 1.0:
            video_np = (video_np * 255).astype(np.uint8)
        else:
            video_np = video_np.astype(np.uint8)

    return [Image.fromarray(frame) for frame in video_np]
