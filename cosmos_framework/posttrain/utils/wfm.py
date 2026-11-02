# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared construction and batching rules for typed WFM conditioning."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import torch

from cosmos_framework.posttrain.schema import WFMConditioningInput

_CONDITION_INDEX_ATTRS = (
    "condition_frame_indexes_vision",
    "condition_frame_indexes_lidar",
    "condition_frame_indexes_action",
    "condition_frame_indexes_sound",
)
_T2V_TEMPLATE_IDENTITY_VERSION = 1
_T2I_TEMPLATE_IDENTITY_VERSION = 1


def _t2v_template_identity(
    *,
    source_fingerprint: str,
    source_id: str,
    video_key: str,
    caption_key: str,
) -> str:
    """Bind one source transaction and its conditioning conversion without reading tensor bytes."""
    if not isinstance(source_fingerprint, str):
        raise TypeError(f"WFM conditioning source_fingerprint must be a string, got {source_fingerprint!r}")
    identity = {
        "version": _T2V_TEMPLATE_IDENTITY_VERSION,
        "source_fingerprint": source_fingerprint,
        "source_id": source_id,
        "video_key": video_key,
        "caption_key": caption_key,
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return f"wfm-t2v-template-v{_T2V_TEMPLATE_IDENTITY_VERSION}:{hashlib.sha256(payload).hexdigest()}"


def _t2i_template_identity(
    *,
    source_fingerprint: str,
    source_id: str,
    image_key: str,
    caption_key: str,
) -> str:
    """Bind one image-source transaction and its conditioning conversion without reading tensor bytes."""
    if not isinstance(source_fingerprint, str):
        raise TypeError(f"WFM conditioning source_fingerprint must be a string, got {source_fingerprint!r}")
    identity = {
        "version": _T2I_TEMPLATE_IDENTITY_VERSION,
        "source_fingerprint": source_fingerprint,
        "source_id": source_id,
        "image_key": image_key,
        "caption_key": caption_key,
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return f"wfm-t2i-template-v{_T2I_TEMPLATE_IDENTITY_VERSION}:{hashlib.sha256(payload).hexdigest()}"


def _zero_tensor_tree(value: Any, *, template_name: str = "T2V video") -> Any:
    """Preserve source structure while removing ungenerated target pixels."""
    if isinstance(value, torch.Tensor):
        return torch.zeros_like(value)  # same shape/dtype/device
    if isinstance(value, list):
        return [_zero_tensor_tree(item, template_name=template_name) for item in value]
    if isinstance(value, tuple):
        return tuple(_zero_tensor_tree(item, template_name=template_name) for item in value)
    if isinstance(value, dict):
        return {key: _zero_tensor_tree(item, template_name=template_name) for key, item in value.items()}
    raise TypeError(
        f"{template_name} template must contain only tensors and nested list/tuple/dict containers, "
        f"got {type(value).__name__}"
    )


def _snapshot_tree(value: Any) -> Any:
    """Recursively detach a producer-owned model-batch snapshot from its source."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()  # same shape/dtype/device, independent graph-free storage
    if isinstance(value, list):
        return [_snapshot_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_tree(item) for item in value)
    if isinstance(value, dict):
        return {key: _snapshot_tree(item) for key, item in value.items()}
    return copy.deepcopy(value)


def make_t2v_conditioning_input(
    source_sample: dict[str, Any],
    *,
    source_id: str,
    source_fingerprint: str,
    video_key: str = "video",
    caption_key: str = "ai_caption",
) -> WFMConditioningInput:
    """Build one shared T2V model-batch template for a source transaction."""
    if not source_id:
        raise ValueError("WFM conditioning source_id must be non-empty")
    if video_key not in source_sample:
        raise ValueError(f"T2V source sample has no {video_key!r} shape template")
    if caption_key not in source_sample:
        raise ValueError(f"T2V source sample has no {caption_key!r} prompt")
    prompt_value = source_sample[caption_key]
    if isinstance(prompt_value, list) and len(prompt_value) == 1 and isinstance(prompt_value[0], str):
        prompt = prompt_value[0]
    elif isinstance(prompt_value, str):
        prompt = prompt_value
    else:
        raise ValueError(f"T2V {caption_key!r} must be a string or one-string list, got {prompt_value!r}")
    sequence_plans = source_sample.get("sequence_plan", ())
    if not isinstance(sequence_plans, (list, tuple)):
        sequence_plans = (sequence_plans,)
    for plan in sequence_plans:
        for attr in _CONDITION_INDEX_ATTRS:
            conditioned_frames = getattr(plan, attr, ())
            if conditioned_frames:
                raise NotImplementedError(
                    "online WFM v1 is T2V-only and cannot zero a source with conditioned modality frames; "
                    f"got {attr}={list(conditioned_frames)!r}"
                )
    model_batch = {
        key: _snapshot_tree(value) for key, value in source_sample.items() if key not in (video_key, caption_key)
    }
    model_batch[caption_key] = prompt
    model_batch[video_key] = _zero_tensor_tree(source_sample[video_key])
    template_identity = _t2v_template_identity(
        source_fingerprint=source_fingerprint,
        source_id=source_id,
        video_key=video_key,
        caption_key=caption_key,
    )
    return WFMConditioningInput(
        id=source_id,
        template_identity=template_identity,
        model_batch=model_batch,
        prompt=prompt,
        modality="video",
    )


def make_t2i_conditioning_input(
    source_sample: dict[str, Any],
    *,
    source_id: str,
    source_fingerprint: str,
    image_key: str = "images",
    caption_key: str = "ai_caption",
) -> WFMConditioningInput:
    """Build one shared prompt-only T2I model-batch template for a rollout group."""
    if not source_id:
        raise ValueError("WFM conditioning source_id must be non-empty")
    if image_key not in source_sample:
        raise ValueError(f"T2I source sample has no {image_key!r} shape template")
    if caption_key not in source_sample:
        raise ValueError(f"T2I source sample has no {caption_key!r} prompt")
    prompt_value = source_sample[caption_key]
    if isinstance(prompt_value, list) and len(prompt_value) == 1 and isinstance(prompt_value[0], str):
        prompt = prompt_value[0]
    elif isinstance(prompt_value, str):
        prompt = prompt_value
    else:
        raise ValueError(f"T2I {caption_key!r} must be a string or one-string list, got {prompt_value!r}")
    model_batch = _snapshot_tree(source_sample)
    model_batch[caption_key] = prompt
    model_batch[image_key] = _zero_tensor_tree(source_sample[image_key], template_name="T2I image")
    template_identity = _t2i_template_identity(
        source_fingerprint=source_fingerprint,
        source_id=source_id,
        image_key=image_key,
        caption_key=caption_key,
    )
    return WFMConditioningInput(
        id=source_id,
        template_identity=template_identity,
        model_batch=model_batch,
        prompt=prompt,
        modality="image",
    )


def repeat_conditioning_batch(conditioning: WFMConditioningInput, count: int) -> dict[str, Any]:
    """Expand one batch-size-one model template to ``count`` local members."""
    if count < 1:
        raise ValueError(f"online NFT local group must be >= 1, got {count}")

    def repeat(value: Any) -> Any:
        # Upstream batches use either a leading tensor axis or a one-element list
        # axis. Preserve that convention instead of adding a second batch axis.
        if isinstance(value, torch.Tensor):
            if value.ndim == 0 or value.shape[0] != 1:
                raise ValueError(
                    "online NFT tensor conditioning fields must have a leading singleton batch axis, "
                    f"got shape {tuple(value.shape)}"
                )
            return torch.cat([value] * count, dim=0)  # [G_local,...]
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(
                    f"online NFT list conditioning fields must have exactly one batch member, got length {len(value)}"
                )
            return value * count
        return [value] * count

    return {key: repeat(value) for key, value in conditioning.model_batch.items()}
