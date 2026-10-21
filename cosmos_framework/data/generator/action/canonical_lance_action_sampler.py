# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from __future__ import annotations

from typing import Any

import torch

from cosmos_framework.data.generator.action.canonical_lance_dataset import (
    CANONICAL_CAMERA_MASK_NATIVE_ACTION_KEY,
    CANONICAL_CAMERA_MASK_TRANSLATION_SCALE_KEY,
)
from cosmos_framework.data.generator.action.utils.action_processing import ActionNormalizer
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
from cosmos_framework.data.generator.action.utils.unified_action_schema import UNIFIED_ACTION_DIM, UnifiedActionCodec


def _project_unified_camera_rows(
    sample: dict[str, Any],
    native_action: torch.Tensor,
    normalized_action: torch.Tensor,
    action_codec: UnifiedActionCodec,
    translation_scale: float,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Place relabeled camera poses in unified ego slots after source mapping."""
    domain_ids = sample.get("domain_id")
    if not isinstance(domain_ids, torch.Tensor) or domain_ids.ndim != 1:
        return sample, normalized_action
    if normalized_action.ndim != 2 or normalized_action.shape != (domain_ids.numel(), UNIFIED_ACTION_DIM):
        raise ValueError(
            "Camera-masked unified action/domain mismatch: "
            f"action={tuple(normalized_action.shape)} domain_id={tuple(domain_ids.shape)}"
        )
    camera_rows = domain_ids.to(device=normalized_action.device).eq(get_domain_id("camera_pose"))
    if not bool(camera_rows.any()):
        return sample, normalized_action
    if native_action.ndim != 2 or native_action.shape[0] != domain_ids.numel() or native_action.shape[1] < 9:
        raise ValueError(
            "Camera-masked native action must align with domain IDs and contain a 9D pose; "
            f"got action={tuple(native_action.shape)} domain_id={tuple(domain_ids.shape)}."
        )
    unified_raw = sample.get("action")
    if not isinstance(unified_raw, torch.Tensor) or unified_raw.shape != normalized_action.shape:
        raise ValueError("Camera-masked unified raw and normalized actions must have matching [T,59] shapes.")

    native_camera_rows = camera_rows.to(device=native_action.device)
    camera_pose = native_action[native_camera_rows, :9].to(device=unified_raw.device, dtype=unified_raw.dtype)
    camera_raw = torch.zeros(
        (camera_pose.shape[0], UNIFIED_ACTION_DIM),
        device=unified_raw.device,
        dtype=unified_raw.dtype,
    )
    camera_raw[:, :9] = camera_pose
    unified_raw = unified_raw.clone()
    unified_raw[camera_rows.to(device=unified_raw.device)] = camera_raw
    sample = dict(sample)
    sample["action"] = unified_raw

    if action_codec.unified_normalizer is not None:
        if translation_scale != 1.0:
            raise ValueError(
                "Unified global normalization requires translation_scale=1.0 for camera-mask-enabled data; "
                f"got {translation_scale}."
            )
        camera_normalized = action_codec.unified_normalizer.normalize_action(camera_raw)
    elif action_codec.native_normalizer is not None:
        # Native camera-row normalization includes the dataset's configured
        # translation scale. Preserve it while relocating the 9D pose from the
        # source mapping (for example YAM wrist slots) into unified ego slots.
        native_normalized = action_codec.native_normalizer.normalize_action(native_action)
        camera_normalized = torch.zeros_like(camera_raw)
        camera_normalized[:, :9] = native_normalized[native_camera_rows, :9].to(
            device=camera_normalized.device,
            dtype=camera_normalized.dtype,
        )
    else:
        camera_normalized = camera_raw.clone()
    camera_normalized[:, 9:] = 0
    normalized_action = normalized_action.clone()
    normalized_action[camera_rows] = camera_normalized.to(
        device=normalized_action.device,
        dtype=normalized_action.dtype,
    )
    return sample, normalized_action


def _apply_mixed_domain_validity_mask(
    sample: dict[str, Any],
    normalized_action: torch.Tensor,
    source_valid_mask: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Apply source-defined validity to each row in a mixed-domain sequence."""
    unified_raw = sample.get("action")
    if not isinstance(unified_raw, torch.Tensor) or unified_raw.ndim != 2:
        raise ValueError("Mixed-domain validity masking requires raw unified action with shape [T,59].")
    if unified_raw.shape != normalized_action.shape or unified_raw.shape[1] != UNIFIED_ACTION_DIM:
        raise ValueError(
            "Mixed-domain raw and normalized actions must have matching [T,59] shapes; "
            f"got raw={tuple(unified_raw.shape)} normalized={tuple(normalized_action.shape)}."
        )
    if source_valid_mask.shape != (UNIFIED_ACTION_DIM,):
        raise ValueError(f"Source validity mask must have shape [59], got {tuple(source_valid_mask.shape)}.")

    domain_id = sample.get("domain_id")
    if not isinstance(domain_id, torch.Tensor) or domain_id.ndim not in (0, 1):
        raise ValueError("Mixed-domain validity masking requires scalar or framewise domain_id.")
    domain_ids = domain_id.expand(unified_raw.shape[0]) if domain_id.ndim == 0 else domain_id
    if domain_ids.shape != (unified_raw.shape[0],):
        raise ValueError(
            f"Framewise domain_id must have shape [{unified_raw.shape[0]}], got {tuple(domain_ids.shape)}."
        )

    # The downstream ActionProcessingRecord only supports one static [D]
    # validity mask. Mixed-domain clips need a different [T,D] mask, so apply
    # it to both raw and normalized values here and do not expose it downstream.
    dynamic_valid_mask = (
        source_valid_mask.to(device=unified_raw.device, dtype=torch.bool)
        .unsqueeze(0)
        .expand(unified_raw.shape[0], -1)
        .clone()
    )
    camera_rows = domain_ids.to(device=unified_raw.device).eq(get_domain_id("camera_pose"))
    dynamic_valid_mask[camera_rows] = False
    dynamic_valid_mask[camera_rows, :9] = True

    sample = dict(sample)
    sample["action"] = torch.where(dynamic_valid_mask, unified_raw, torch.zeros_like(unified_raw))
    normalized_valid_mask = dynamic_valid_mask.to(device=normalized_action.device)
    normalized_action = torch.where(
        normalized_valid_mask,
        normalized_action,
        torch.zeros_like(normalized_action),
    )
    return sample, normalized_action


class CanonicalMixedDomainActionTransform:
    """Decorate the standard Action transform for canonical mixed-domain and multiview rows."""

    def __init__(
        self,
        transform: ActionTransformPipeline,
        robot_multiview_max_frames: int | None = None,
    ) -> None:
        self._transform = transform
        self.robot_multiview_max_frames: int | None = robot_multiview_max_frames
        if robot_multiview_max_frames is not None and (
            robot_multiview_max_frames < 5 or (robot_multiview_max_frames - 1) % 4
        ):
            raise ValueError("Robot multiview smoke prefixes must contain 4k+1 frames.")

    def __getattr__(self, name: str) -> Any:
        """Preserve the standard transform's public configuration surface."""
        transform = object.__getattribute__(self, "_transform")
        return getattr(transform, name)

    def __call__(
        self,
        data_dict: dict[str, Any],
        resolution: str | None,
        action_normalizer: ActionNormalizer | None = None,
        *,
        action_valid_mask: torch.Tensor | None = None,
        normalized_action: torch.Tensor | None = None,
        history_normalized_action: torch.Tensor | None = None,
        action_codec: UnifiedActionCodec | None = None,
    ) -> dict[str, Any]:
        """Finish mixed-domain unified projection without changing shared code."""
        data_dict = dict(data_dict)
        multiview_videos = data_dict.pop("_robot_multiview_videos", None)
        if multiview_videos is not None and self.robot_multiview_max_frames is not None:
            frames = self.robot_multiview_max_frames
            multiview_videos = [video[:, :frames] for video in multiview_videos]
            data_dict["video"] = multiview_videos[0]
            data_dict["action"] = data_dict["action"][: frames - 1]
            action_frames = data_dict["action"].shape[0]
            domain_id = data_dict.get("domain_id")
            if isinstance(domain_id, torch.Tensor) and domain_id.ndim == 1:
                data_dict["domain_id"] = domain_id[:action_frames]
            native_action = data_dict.get(CANONICAL_CAMERA_MASK_NATIVE_ACTION_KEY)
            if isinstance(native_action, torch.Tensor):
                data_dict[CANONICAL_CAMERA_MASK_NATIVE_ACTION_KEY] = native_action[:action_frames]
            if normalized_action is not None:
                normalized_action = normalized_action[:action_frames]
        native_action = data_dict.pop(CANONICAL_CAMERA_MASK_NATIVE_ACTION_KEY, None)
        translation_scale = data_dict.pop(CANONICAL_CAMERA_MASK_TRANSLATION_SCALE_KEY, None)
        if native_action is not None:
            if not isinstance(native_action, torch.Tensor):
                raise TypeError("Canonical camera-mask native action must be a tensor.")
            if not isinstance(translation_scale, (float, int)):
                raise TypeError("Canonical camera-mask translation scale must be numeric.")
            if normalized_action is not None:
                if action_valid_mask is None or not isinstance(action_codec, UnifiedActionCodec):
                    raise ValueError("Unified canonical camera masking requires its source mask and action codec.")
                data_dict, normalized_action = _project_unified_camera_rows(
                    data_dict,
                    native_action,
                    normalized_action,
                    action_codec,
                    float(translation_scale),
                )
                data_dict, normalized_action = _apply_mixed_domain_validity_mask(
                    data_dict,
                    normalized_action,
                    action_valid_mask,
                )
            # A clip can change domains frame-by-frame, while the model-facing
            # field only accepts one static [D] mask. The dynamic mask is fully
            # applied above, so replace the static source mask with a neutral
            # all-valid mask. Unified samples then share one collatable schema;
            # native samples without a source mask keep their existing behavior.
            action_valid_mask = torch.ones_like(action_valid_mask) if action_valid_mask is not None else None

        result = self._transform(
            data_dict,
            resolution,
            action_normalizer,
            action_valid_mask=action_valid_mask,
            normalized_action=normalized_action,
            history_normalized_action=history_normalized_action,
            action_codec=action_codec,
        )
        if multiview_videos is not None:
            if resolution != "480" or len(multiview_videos) != 3:
                raise ValueError("Robot multiview expects a 480p head and two 256p arm views.")
            # The standard transform above resizes the head to the 480 bucket;
            # resize each arm independently to the 256 bucket below. These are
            # aspect-ratio-dependent bucket names, not fixed pixel heights:
            # a 4:3 arm has 320x240 content on a reflection-padded 320x256 canvas (WxH).
            videos = [result["video"]]
            image_sizes = [result["image_size"]]
            for video in multiview_videos[1:]:
                resized = self._transform.video_resize({"video": video}, "256")
                videos.append(resized["video"])
                image_sizes.append(resized["image_size"])
            if len({video.shape[1] for video in videos}) != 1:
                raise ValueError("Robot multiview RGB streams must have the same frame count.")
            result["video"] = videos
            result["image_size"] = image_sizes
        return result
