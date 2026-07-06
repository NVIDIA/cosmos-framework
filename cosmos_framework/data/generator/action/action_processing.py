# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared Action processing records and normalization helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch

from cosmos_framework.utils import log

ActionNormalizationMethod = Literal["quantile", "quantile_rot", "meanstd", "minmax"]


class ActionNormalizer(Protocol):
    """Tensor-level action normalization interface used by ActionProcessor."""

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:  # action: [...,D], returns [...,D]
        """Map raw action values into model-space values."""
        ...

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:  # action: [...,D], returns [...,D]
        """Invert model-space action values back into raw action values."""
        ...


@dataclass(frozen=True)
class ActionAffineNormalization:
    """Resolved affine action normalizer.

    Forward normalization is ``(raw - offset) / scale``.  Inverse
    denormalization is ``normalized * scale + offset``.

    ``forward_clamp`` records lossy range-style forward clamping.  When
    ``forward_clamp_mask`` is provided, only channels with a ``True`` mask
    entry are clamped; this represents mixed UMI normalizers where some fields
    are range-clamped and others are plain affine transforms.
    """

    offset: torch.Tensor
    scale: torch.Tensor
    forward_clamp: tuple[float, float] | None = None
    forward_clamp_mask: torch.Tensor | None = None

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:  # action: [...,D], returns [...,D]
        """Normalize raw action values with resolved affine parameters."""
        offset = self.offset.to(device=action.device, dtype=action.dtype)  # [D]
        scale = self.scale.to(device=action.device, dtype=action.dtype)  # [D]
        normalized = (action - offset) / scale  # [...,D]
        if self.forward_clamp is not None:
            lo, hi = self.forward_clamp
            clamped = normalized.clamp(lo, hi)  # [...,D]
            if self.forward_clamp_mask is None:
                normalized = clamped  # [...,D]
            else:
                clamp_mask = self.forward_clamp_mask.to(device=action.device, dtype=torch.bool)  # [D]
                normalized = torch.where(clamp_mask, clamped, normalized)  # [...,D]
        return normalized  # [...,D]

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:  # action: [...,D], returns [...,D]
        """Invert action normalization with resolved affine parameters."""
        offset = self.offset.to(device=action.device, dtype=action.dtype)  # [D]
        scale = self.scale.to(device=action.device, dtype=action.dtype)  # [D]
        return action * scale + offset  # [...,D]


def load_action_stats(stats_path: str, stats_key: str = "global") -> dict[str, np.ndarray]:
    """Load pre-computed action normalization stats from a JSON file."""
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Action normalization stats not found at {stats_path}.")
    log.info(f"Loading action normalization stats from {stats_path}")
    with path.open("r") as f:
        raw = json.load(f)
    stat_keys = {"mean", "std", "min", "max", "q01", "q99"}
    if stats_key in raw:
        raw = raw[stats_key]
        if not isinstance(raw, dict):
            raise TypeError(f"Action normalization stats block {stats_key!r} in {stats_path} must be a dict.")
    elif stats_key != "global" and not any(key in raw for key in stat_keys):
        raise KeyError(f"Action normalization stats block {stats_key!r} not found in {stats_path}.")
    return {k: np.array(v, dtype=np.float32) for k, v in raw.items() if k in stat_keys}


def resolve_action_normalization(
    method: ActionNormalizationMethod,
    stats: dict[str, torch.Tensor],
    apply_forward_clamp: bool = False,
) -> ActionAffineNormalization:
    """Resolve configured action stats into affine forward/inverse parameters."""
    if method == "meanstd":
        offset = stats["mean"]  # [D]
        scale = stats["std"].clamp(min=1e-8)  # [D]
        return ActionAffineNormalization(offset=offset, scale=scale)

    if method == "minmax":
        lo = stats["min"]  # [D]
        hi = stats["max"]  # [D]
    elif method in ("quantile", "quantile_rot"):
        lo = stats["q01"]  # [D]
        hi = stats["q99"]  # [D]
    else:
        raise ValueError(f"Unknown normalization method: {method!r}")

    offset = (hi + lo) / 2.0  # [D]
    scale = (hi - lo).clamp(min=1e-8) / 2.0  # [D]

    if apply_forward_clamp:
        # Ideally this hardcode should be removed, but for now we keep it so we can be aligned with mid-training checkpoints.
        forward_clamp = (-1.0, 1.0)
    else:
        forward_clamp = None

    return ActionAffineNormalization(
        offset=offset,
        scale=scale,
        forward_clamp=forward_clamp,
    )


def make_pose_action_scale_normalizer(
    action_dim: int,
    *,
    translation_scale: float = 1.0,
    rotation_scale: float = 1.0,
) -> ActionAffineNormalization:
    """Build a normalizer that maps raw pose deltas into scaled model space.

    Pose actions use the shared layout ``[translation(3), rotation(...)]``.
    The returned normalizer multiplies translation channels by
    ``translation_scale`` and rotation channels by ``rotation_scale`` during
    preprocessing, then inverts those factors during postprocessing.
    """
    if action_dim < 3:
        raise ValueError(f"Pose action_dim must be at least 3, got {action_dim}")
    if translation_scale == 0:
        raise ValueError("translation_scale must be non-zero")
    if rotation_scale == 0:
        raise ValueError("rotation_scale must be non-zero")

    offset = torch.zeros(action_dim, dtype=torch.float32)  # [D]
    scale = torch.ones(action_dim, dtype=torch.float32)  # [D]
    scale[:3] = 1.0 / float(translation_scale)  # [D]
    if action_dim > 3:
        scale[3:] = 1.0 / float(rotation_scale)  # [D]
    return ActionAffineNormalization(offset=offset, scale=scale)


@dataclass(frozen=True)
class ActionProcessingRecord:
    """Per-sample metadata needed to invert Action model-space preprocessing."""

    raw_action_dim: int
    action_normalizer: ActionNormalizer | None


def pad_action_to_max_dim(
    action: torch.Tensor, max_action_dim: int
) -> torch.Tensor:  # action: [T,D], returns [T,D_model]
    """Pad action tensor to max_action_dim along the last dimension.

    Args:
        action: Action tensor of shape (T, D) where D is the current action dimension.
        max_action_dim: Target action dimension to pad to.

    Returns:
        Padded action tensor of shape (T, max_action_dim).
    """
    if action.shape[-1] > max_action_dim:
        raise ValueError(f"Action dimension {action.shape[-1]} is greater than max_action_dim {max_action_dim}")
    if action.shape[-1] == max_action_dim:
        return action  # [T,D_model]
    padding_size = max_action_dim - action.shape[-1]
    zero_padding = torch.zeros(*action.shape[:-1], padding_size, dtype=action.dtype, device=action.device)  # [T,D_pad]
    return torch.cat([action, zero_padding], dim=-1)  # [T,D_model]


def make_batched_action_processing_fields(
    record: ActionProcessingRecord,
    batch_size: int,
    *,
    action_channel_masking: bool = True,
) -> dict[str, list[torch.Tensor | ActionProcessingRecord | None]]:
    """Build batch-list fields whose action width and inverse record cannot drift apart."""
    raw_action_dim = torch.tensor(record.raw_action_dim, dtype=torch.long) if action_channel_masking else None  # []
    return {
        "raw_action_dim": [raw_action_dim] * batch_size,
        "action_processing_record": [record] * batch_size,
    }


class ActionProcessor:
    """Forward and inverse Action tensor processing for a single sample."""

    def __init__(self, max_action_dim: int, action_channel_masking: bool = True) -> None:
        self.max_action_dim = int(max_action_dim)
        self.action_channel_masking = bool(action_channel_masking)

    def preprocess_action(
        self,
        data_dict: dict[str, Any],
        action: torch.Tensor,
        *,
        action_normalizer: ActionNormalizer | None,
    ) -> dict[str, Any]:
        """Return a sample with normalized, padded action fields and the inverse record."""

        raw_action_dim = int(action.shape[-1])
        if action_normalizer is not None:
            action = action_normalizer.normalize_action(action)  # [T,D]
            if int(action.shape[-1]) != raw_action_dim:
                raise ValueError(
                    f"Action normalizer changed action width from {raw_action_dim} to {int(action.shape[-1])}"
                )

        processed_data_dict = dict(data_dict)
        processed_data_dict["action"] = pad_action_to_max_dim(action, self.max_action_dim)  # [T,D_model]
        record = ActionProcessingRecord(
            raw_action_dim=raw_action_dim,
            action_normalizer=action_normalizer,
        )
        processed_data_dict["raw_action_dim"] = (
            torch.tensor(record.raw_action_dim, dtype=torch.long) if self.action_channel_masking else None
        )  # []
        processed_data_dict["action_processing_record"] = record
        return processed_data_dict

    @staticmethod
    def _unpad_action(action: torch.Tensor, raw_action_dim: int) -> torch.Tensor:
        """Drop model-only padded action channels."""
        if action.shape[-1] < raw_action_dim:
            raise ValueError(f"invalid raw_action_dim={raw_action_dim} for action with shape {tuple(action.shape)}")
        return action[..., :raw_action_dim].contiguous()  # [...,D_raw]

    @staticmethod
    def postprocess_action(
        action: torch.Tensor,
        record: ActionProcessingRecord,
    ) -> torch.Tensor:
        """Unpad and denormalize a model-space action tensor."""
        action = ActionProcessor._unpad_action(action, record.raw_action_dim)  # [...,D_raw]
        if record.action_normalizer is not None:
            action = record.action_normalizer.denormalize_action(action)  # [...,D_raw]
        return action  # [...,D_raw]


def get_action_processing_records(data_batch: dict[str, Any]) -> list[ActionProcessingRecord | None]:
    """Read all per-sample processing records from a collated Action batch."""
    records = data_batch.get("action_processing_record")
    if records is None:
        return []
    if isinstance(records, ActionProcessingRecord):
        return [records]
    if isinstance(records, list):
        for record in records:
            if record is not None and not isinstance(record, ActionProcessingRecord):
                raise TypeError(f"Unexpected action_processing_record entry type: {type(record).__name__}")
        return records
    raise TypeError(f"Unexpected action_processing_record type: {type(records).__name__}")
