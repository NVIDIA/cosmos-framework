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
