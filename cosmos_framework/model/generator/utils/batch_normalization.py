# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Normalize ragged raw vision batches at the model input boundary.

Legacy image/video dictionaries already implement the model's input contract and
are left intact. A unified ``vision`` column carries ragged per-sample items and
explicit image identities; this adapter flattens it into the same canonical
vision layout. Model stages consume sample metadata, never a loader-class flag.
"""

from __future__ import annotations

from typing import Any

import torch

from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan


def _items(value: Any) -> list[torch.Tensor]:
    if value is None:
        return []
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    return [item for item in values if item is not None]


def _scalar(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Expected scalar sample metadata.")
        return float(value.item())
    return float(value)


def normalize_vision_batch_inplace(
    batch: dict[str, Any],
    *,
    video_key: str,
    tensor_kwargs_fp32: dict[str, Any],
) -> None:
    """Convert unified raw vision items once, preserving sample ownership.

    Each canonical item has shape [1,C,T,H,W]. uint8 pixels use the existing
    [-1,1] map; floating-point pixels retain their values. Per-camera uint8 items
    remain unnormalized until the VAE encodes each camera, avoiding a full-size
    normalized multiview allocation. Empty vision lists are valid for samples
    with LiDAR. Optional sample columns keep their original aligned positions.
    """
    if "vision" not in batch:
        return
    grouped = [_items(value) for value in batch["vision"]]
    image_flags = batch.get("sample_is_image")
    if not isinstance(image_flags, list) or len(image_flags) != len(grouped):
        raise ValueError("Unified vision requires one sample_is_image flag per sample.")
    if not all(isinstance(flag, bool) for flag in image_flags):
        raise ValueError("sample_is_image entries must be boolean.")
    batch_size = len(grouped)
    plans = batch.get("sequence_plan", [None] * batch_size)
    if len(plans) != batch_size:
        raise ValueError("Sequence plans must align with the sample list.")
    frames_per_view = batch.get("num_video_frames_per_view", [None] * batch_size)
    views_per_sample = batch.get("sample_n_views", [None] * batch_size)
    per_camera = "enable_per_camera_vae_encoding" in batch
    sizes = batch.get("image_size", [None] * batch_size)
    for column in (frames_per_view, views_per_sample, sizes):
        if len(column) != batch_size:
            raise ValueError("Vision metadata must align with the logical sample list.")
    flat_sizes: list[torch.Tensor] = []
    sample_sizes: list[torch.Tensor | None] = []
    flat_vision: list[torch.Tensor] = []
    normalized_plans: list[SequencePlan] = []
    view_counts: list[torch.Tensor] = []
    frame_counts: list[torch.Tensor] = []

    for sample_index, (items, is_image, plan) in enumerate(zip(grouped, image_flags, plans, strict=True)):
        if plan is None:
            for key in ("action", "sound", "lidar"):
                if _items(batch.get(key, [None] * batch_size)[sample_index]):
                    raise ValueError(f"Samples with {key} must supply an explicit SequencePlan.")
            plan = SequencePlan(has_text=True, has_vision=bool(items))
        if bool(plan.has_vision) != bool(items):
            raise ValueError("SequencePlan.has_vision must match its sample's vision items.")
        normalized_plans.append(plan)
        sample_size = None
        for item_index, item in enumerate(items):
            if not isinstance(item, torch.Tensor):
                raise TypeError("Vision items must be tensors.")
            if is_image and item.ndim == 3:
                item = item.unsqueeze(0).unsqueeze(2)  # [1,C,1,H,W]
            elif item.ndim == 4:
                item = item.unsqueeze(0)  # [1,C,T,H,W]
            if item.ndim != 5 or item.shape[0] != 1 or (is_image and item.shape[2] != 1):
                raise ValueError(f"Invalid vision item shape {tuple(item.shape)} for is_image={is_image}.")
            if item.dtype == torch.uint8:
                if not per_camera:
                    item = item.to(**tensor_kwargs_fp32) / 127.5 - 1.0  # [1,C,T,H,W]
            elif torch.is_floating_point(item):
                item = item.to(**tensor_kwargs_fp32)  # [1,C,T,H,W]
            else:
                raise TypeError(f"Vision pixels must be uint8 or normalized floats, got {item.dtype}.")
            flat_vision.append(item)
            supplied_sizes = _items(sizes[sample_index])
            if supplied_sizes:
                size = supplied_sizes[min(item_index, len(supplied_sizes) - 1)]  # [4] or [1,4]
            else:
                height, width = item.shape[-2:]
                size = torch.tensor([height, width, height, width])  # [4]
            flat_sizes.append(size)
            if sample_size is None:
                sample_size = size  # [4] or [1,4]
        sample_sizes.append(sample_size)
        if per_camera:
            views = int(_scalar(views_per_sample[sample_index], 1))
            if views < 1:
                raise ValueError("Camera counts must be positive.")
            default_frames = items[0].shape[-3] // views if items and not is_image else 1
            frames = int(_scalar(frames_per_view[sample_index], default_frames))
            view_counts.append(torch.tensor(views))  # []
            frame_counts.append(torch.tensor(frames))  # []

    batch[video_key] = flat_vision
    batch["num_vision_items_per_sample"] = [len(items) for items in grouped]
    batch["sequence_plan"] = normalized_plans
    batch["image_size"] = flat_sizes
    batch["sample_image_size"] = sample_sizes
    batch["is_preprocessed"] = True
    if per_camera:
        batch["sample_n_views"] = view_counts
        batch["num_video_frames_per_view"] = frame_counts
    if "conditioning_fps" in batch:
        batch["conditioning_fps"] = [
            torch.tensor(_scalar(value, 1.0 if is_image else 24.0))  # []
            for value, is_image in zip(batch["conditioning_fps"], image_flags, strict=True)
        ]
    if "conditioning_fps_action" in batch:
        fallback = batch.get("conditioning_fps", [24.0] * batch_size)
        batch["conditioning_fps_action"] = [
            torch.tensor(_scalar(value, _scalar(fps, 24.0)))  # []
            for value, fps in zip(batch["conditioning_fps_action"], fallback, strict=True)
        ]
    if "lidar" in batch:
        batch["lidar"] = [_items(value) for value in batch["lidar"]]
    for name in ("action", "sound", "domain_id", "raw_action_dim", "action_valid_mask"):
        if name in batch:
            values = [_items(value) for value in batch[name]]
            if any(len(items) > 1 for items in values):
                raise ValueError(f"Each logical sample supports at most one {name} item.")
            batch[name] = [items[0] if items else None for items in values]
    if "control_weights" in batch:
        batch["control_weights"] = [value if value is not None else [] for value in batch["control_weights"]]
    # Mark completion by consuming the raw-format key. Calling the boundary again
    # (e.g. an online sampling callback) must not normalize pixels a second time.
    del batch["vision"]
