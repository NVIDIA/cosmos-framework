# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cosmos_framework.data.generator.action.lance.lance_action_dataset import (
    DEFAULT_DENSE_CAPTION_PROBABILITY,
    DEFAULT_LANCE_ACTION_MODES,
    BaseLanceActionDataset,
    MaterializedActionWindow,
    TargetFps,
    resolve_lance_action_mode,
)
from cosmos_framework.data.generator.action.utils.action_processing import (
    ActionNormalizer,
    make_pose_action_scale_normalizer,
)
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.utils.pose_utils import (
    PoseConvention,
    RotationConvention,
    build_abs_pose_from_components,
    pose_abs_to_rel,
)
from cosmos_framework.data.generator.action.utils.unified_action_schema import UnifiedActionMapping, ego_pose_mapping
from cosmos_framework.data.generator.lance.lance_action_source import (
    ACTION_SOURCE_CAPTION_COLUMNS,
    LanceActionDatasetSpec,
)

_CAMERA_VIEW: str = "observation.images.main"


_GENERIC_CAMERA_PROMPT: str = "The camera moves in a scene."


_SUPPORTED_POSE_CONVENTIONS: tuple[PoseConvention, ...] = (
    "backward_framewise",
    "backward_anchored",
    "backward_chunk_anchored_8f",
    "backward_chunk_anchored_16f",
)


_CAMERA_VIDEO_DECODE_THREADS: int = 16


_CAMERA_METADATA_BLOCK_SIZE: int = 4


_ROTATION_DIMS: dict[RotationConvention, int] = {
    "euler_xyz": 3,
    "quat_xyzw": 4,
    "quat_wxyz": 4,
    "rot6d": 6,
    "axisangle": 3,
    "rot9d": 9,
}


def _extract_camera_caption(value: Any) -> str | None:
    """Extract the text from a plain caption or one ``chunk_0_*`` payload."""
    if value is None:
        return None
    caption = str(value).strip()
    if not caption:
        return None
    try:
        payload = json.loads(caption)
    except (json.JSONDecodeError, TypeError):
        return caption
    if not isinstance(payload, dict) or len(payload) != 1:
        return caption
    chunk_key, chunk = next(iter(payload.items()))
    if not str(chunk_key).startswith("chunk_0_") or not isinstance(chunk, dict):
        return caption
    inner_caption = chunk.get("caption")
    if inner_caption is None:
        return None
    resolved = str(inner_caption).strip()
    return resolved or None


def _resolve_camera_caption(source_row: dict[str, Any]) -> str:
    """Prefer the final caption, then source text, then the generic prompt."""
    for column in ACTION_SOURCE_CAPTION_COLUMNS:
        caption = _extract_camera_caption(source_row.get(column))
        if caption is not None:
            return caption
    return _GENERIC_CAMERA_PROMPT


class LanceCameraDataset(BaseLanceActionDataset):
    """Materialize camera trajectories from Lance.

    A ``chunk_length`` of 16 represents 16 actions and 17 aligned frames.
    ``fps=None`` keeps native timing, a scalar uses one target rate, and a tuple
    deterministically samples a target rate per training draw. Targets above the
    source rate fall back to native timing rather than upsampling.

    By default, every complete overlapping fixed-length window is indexed.
    ``variable_length=True`` instead emits one start-aligned window per source
    row, truncates it to at most ``chunk_length + 1`` frames, and snaps shorter
    windows down to the tokenizer-compatible ``4N+1`` frame grid.
    ``first_chunk_only=True`` instead requires a complete native-FPS prefix of
    ``chunk_length + 1`` frames, excluding short rows and rows whose keep ranges
    do not contain that prefix. It requires fixed length and short captions.
    """

    def __init__(
        self,
        lance_dataset_spec: LanceActionDatasetSpec,
        creds_path: str | Path,
        chunk_length: int = 16,
        fps: TargetFps = 10,
        mode: str = "forward_dynamics",
        split: str = "train",
        val_ratio: float = 0.01,
        seed: int = 0,
        rotation_format: RotationConvention = "rot6d",
        pose_convention: PoseConvention = "backward_framewise",
        embodiment_type: str = "camera_pose",
        translation_scale: float = 10.0,
        rotation_scale: float = 1.0,
        skip_video_loading: bool = False,
        enable_video_decode_resize: bool = False,
        video_decode_threads: int = _CAMERA_VIDEO_DECODE_THREADS,
        source_dataset_name: str | None = None,
        metadata_block_size: int = _CAMERA_METADATA_BLOCK_SIZE,
        variable_length: bool = False,
        retention_quality_score_threshold: float | None = 0.0,
        dense_caption_probability: float = DEFAULT_DENSE_CAPTION_PROBABILITY,
        first_chunk_only: bool = False,
    ) -> None:
        if first_chunk_only and (variable_length or fps is not None or chunk_length < 4 or chunk_length % 4):
            raise ValueError("first_chunk_only requires variable_length=False, fps=None, and chunk_length=4N >= 4.")
        valid_modes = {"joint", "fd_id", *DEFAULT_LANCE_ACTION_MODES}
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {sorted(valid_modes)}, got {mode!r}.")
        if rotation_format not in _ROTATION_DIMS:
            raise ValueError(f"Unsupported rotation_format={rotation_format!r}.")
        if pose_convention not in _SUPPORTED_POSE_CONVENTIONS:
            raise ValueError(
                f"Unsupported pose_convention={pose_convention!r}. Expected one of {_SUPPORTED_POSE_CONVENTIONS}."
            )
        for name, scale in (("translation_scale", translation_scale), ("rotation_scale", rotation_scale)):
            if not math.isfinite(scale) or scale == 0:
                raise ValueError(f"{name} must be non-zero and finite, got {scale}.")
        self.mode: str = mode
        self.rotation_format: RotationConvention = rotation_format
        self.pose_convention: PoseConvention = pose_convention
        self.embodiment_type: str = embodiment_type
        self.domain_id: int = get_domain_id(embodiment_type)
        self.translation_scale: float = translation_scale
        self.rotation_scale: float = rotation_scale
        self.variable_length: bool = variable_length
        self.first_chunk_only: bool = first_chunk_only
        self.action_normalizer: ActionNormalizer = make_pose_action_scale_normalizer(
            self.action_dim,
            translation_scale=translation_scale,
            rotation_scale=rotation_scale,
        )

        super().__init__(
            lance_dataset_spec,
            creds_path,
            view_names=(_CAMERA_VIEW,),
            chunk_length=chunk_length,
            fps=fps,
            split=split,
            val_ratio=val_ratio,
            seed=seed,
            skip_video_loading=skip_video_loading,
            enable_video_decode_resize=enable_video_decode_resize,
            video_decode_threads=video_decode_threads,
            source_dataset_name=source_dataset_name,
            metadata_block_size=metadata_block_size,
            one_anchor_per_row=variable_length or first_chunk_only,
            pad_end=False,
            retention_quality_score_threshold=retention_quality_score_threshold,
            dense_caption_probability=dense_caption_probability,
        )

    def _build_training_item(
        self,
        global_index: int,
        aux_seed: int,
        source_rows: dict[str, dict[str, Any]],
        window: MaterializedActionWindow,
    ) -> dict[str, Any]:
        """Convert one complete materialized window to the legacy camera contract."""
        del global_index
        missing_fields = sorted({"camera_position", "camera_rotation"} - window.per_frame_fields.keys())
        if missing_fields:
            raise ValueError(f"Action binary is missing required fields {missing_fields}.")

        position = np.asarray(window.per_frame_fields["camera_position"], dtype=np.float32)
        rotation = np.asarray(window.per_frame_fields["camera_rotation"], dtype=np.float32)
        expected_frames = (
            int(position.shape[0]) if self.variable_length and position.ndim == 2 else self.chunk_length + 1
        )
        if position.shape != (expected_frames, 3) or rotation.shape != (expected_frames, 4):
            raise ValueError(
                f"Expected camera_position [T+1,3] and camera_rotation [T+1,4] with {expected_frames} frames; "
                f"got camera_position={position.shape}, camera_rotation={rotation.shape}."
            )
        if not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError("camera_position and camera_rotation must contain only finite values.")

        poses_abs = build_abs_pose_from_components(
            position,
            rotation,
            rotation_input_format="quat_xyzw",
        )  # [T+1,4,4]

        action_array = pose_abs_to_rel(
            poses_abs,
            rotation_format=self.rotation_format,
            pose_convention=self.pose_convention,
        )  # [T,D]
        expected_actions = expected_frames - 1
        if action_array.shape != (expected_actions, self.action_dim) or not np.isfinite(action_array).all():
            raise ValueError(f"Unexpected camera action shape or values: {action_array.shape}.")
        action = torch.from_numpy(np.ascontiguousarray(action_array, dtype=np.float32))  # [T,D]
        if self.skip_video_loading:
            return {"action": action}

        source_video = window.videos.get(_CAMERA_VIEW)
        if source_video is None:
            raise ValueError(f"Materialized window is missing view {_CAMERA_VIEW!r}.")
        if source_video.ndim != 4 or source_video.shape[0] != expected_frames or source_video.shape[-1] != 3:
            raise ValueError(
                f"Expected video {_CAMERA_VIEW!r} [T+1,H,W,3] aligned with action fields, "
                f"got {tuple(source_video.shape)}."
            )
        if source_video.dtype != np.uint8:
            raise ValueError(f"Expected uint8 decoded video, got {source_video.dtype}.")
        video = torch.from_numpy(np.ascontiguousarray(source_video)).permute(3, 0, 1, 2).contiguous()  # [C,T+1,H,W]

        resolved_mode = resolve_lance_action_mode(self.mode, aux_seed)
        source_row = source_rows[_CAMERA_VIEW]
        caption = _resolve_camera_caption(source_row)
        return {
            "source_repo_id": str(source_row["source_dataset"]),
            "video": video,
            "action": action,
            "conditioning_fps": torch.tensor(self._window_fps(window), dtype=torch.float32),
            "prompt": caption,
            "ai_caption": caption,
            "mode": resolved_mode,
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "viewpoint": "ego_view",
        }

    @property
    def action_dim(self) -> int:
        """Return the pose action width for the configured rotation format."""
        return 3 + _ROTATION_DIMS[self.rotation_format]

    def get_action_normalizer(self, _sample: dict[str, Any] | None = None) -> ActionNormalizer:
        """Return the configured pose scale normalizer."""
        return self.action_normalizer

    def get_unified_action_mapping(self) -> UnifiedActionMapping:
        """Map 9D camera pose actions into canonical ego slots."""
        if self.rotation_format != "rot6d":
            raise ValueError(f"unified_v1 requires Lance camera rotation_format='rot6d', got {self.rotation_format!r}")
        return ego_pose_mapping()
