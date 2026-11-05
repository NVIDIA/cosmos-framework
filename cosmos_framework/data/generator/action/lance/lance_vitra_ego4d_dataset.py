# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from cosmos_framework.data.generator.action.hand_pose_dataset_config import (
    CAM_POSITION_KEY,
    CAM_ROTATION_KEY,
    FINGERTIP_JOINT_IDXS,
    HAND_LEFT_POSITION_KEY,
    HAND_LEFT_ROTATION_KEY,
    HAND_RIGHT_POSITION_KEY,
    HAND_RIGHT_ROTATION_KEY,
    NUM_JOINTS,
    QUAT_DIM_PER_JOINT,
    WRIST_JOINT_IDX,
)
from cosmos_framework.data.generator.action.hand_pose_semantics import (
    build_hand_pose_action,
    compute_raw_action_dim,
    load_hand_pose_action_normalizer,
)
from cosmos_framework.data.generator.action.lance.lance_action_dataset import (
    DEFAULT_DENSE_CAPTION_PROBABILITY,
    DEFAULT_SOURCE_METADATA_BLOCK_SIZE,
    DEFAULT_VIDEO_DECODE_THREADS,
    BaseLanceActionDataset,
    MaterializedActionWindow,
    TargetFps,
    require_action_caption,
    resolve_lance_action_mode,
)
from cosmos_framework.data.generator.action.training_view_keys import HAND_POSE_MAIN_VIEW
from cosmos_framework.data.generator.action.utils.action_processing import ActionNormalizationMethod, ActionNormalizer
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.utils.pose_utils import RotationConvention
from cosmos_framework.data.generator.action.utils.unified_action_schema import (
    UnifiedActionMapping,
    hand_pose_57d_mapping,
)
from cosmos_framework.data.generator.action.utils.viewpoint_utils import Viewpoint
from cosmos_framework.data.generator.drop_sample_contract import DROP_SAMPLE_KEY, DROP_SAMPLE_REASON_KEY
from cosmos_framework.data.generator.lance.lance_action_source import LanceActionDatasetSpec
from cosmos_framework.utils import log

_MAIN_VIEW = HAND_POSE_MAIN_VIEW


_ACT2_HAND_LEFT_POSITION = "hand_left_cam"


_ACT2_HAND_RIGHT_POSITION = "hand_right_cam"


_ACT2_HAND_LEFT_ROTATION = "hand_left_cam_rotation"


_ACT2_HAND_RIGHT_ROTATION = "hand_right_cam_rotation"


_ACT2_CAMERA_POSITION = "camera_position"


_ACT2_CAMERA_ROTATION = "camera_rotation"


_ZERO_NORM_QUATERNION_DROP_WARNING_LIMIT: int = 100


class LanceVITRAEgo4DDataset(BaseLanceActionDataset):
    """Reconstruct VITRA Ego4D hand-pose items from Lance-materialized Action windows."""

    def __init__(
        self,
        lance_dataset_spec: LanceActionDatasetSpec,
        creds_path: str | Path,
        chunk_length: int = 16,
        fps: TargetFps = 15,
        mode: str = "joint",
        split: str = "train",
        val_ratio: float = 0.005,
        seed: int = 42,
        rotation_format: RotationConvention = "rot6d",
        pose_convention: Literal["backward_anchored", "backward_framewise"] = "backward_framewise",
        embodiment_type: str = "hand_pose",
        viewpoint: Viewpoint = "ego_view",
        action_normalization: ActionNormalizationMethod | None = None,
        apply_forward_clamp: bool = False,
        action_stats_path: str | None = None,
        skip_video_loading: bool = False,
        video_decode_threads: int = DEFAULT_VIDEO_DECODE_THREADS,
        source_dataset_name: str | None = None,
        metadata_block_size: int = DEFAULT_SOURCE_METADATA_BLOCK_SIZE,
        snap_to_subtask: bool = False,
        max_subtasks_per_episode: int | None = None,
        retention_quality_score_threshold: float | None = 0.0,
        action_caption_dataset_name: str | None = "vitra_ego4d",
        dense_caption_probability: float = DEFAULT_DENSE_CAPTION_PROBABILITY,
    ) -> None:
        self.mode: str = mode
        self.rotation_format: RotationConvention = rotation_format
        self.pose_convention: Literal["backward_anchored", "backward_framewise"] = pose_convention
        self.embodiment_type: str = embodiment_type
        self.viewpoint: Viewpoint = viewpoint
        self.wrist_frame_alignment: np.ndarray | None = None
        self.position_joint_indices: tuple[int, ...] = (WRIST_JOINT_IDX, *FINGERTIP_JOINT_IDXS)
        self._raw_action_dim: int = compute_raw_action_dim(rotation_format, self.position_joint_indices)
        self._zero_norm_quaternion_drop_warnings: int = 0

        super().__init__(
            lance_dataset_spec,
            creds_path,
            view_names=(_MAIN_VIEW,),
            chunk_length=chunk_length,
            fps=fps,
            split=split,
            val_ratio=val_ratio,
            seed=seed,
            skip_video_loading=skip_video_loading,
            video_decode_threads=video_decode_threads,
            source_dataset_name=source_dataset_name,
            metadata_block_size=metadata_block_size,
            one_anchor_per_row=snap_to_subtask,
            max_subtasks_per_episode=max_subtasks_per_episode,
            retention_quality_score_threshold=retention_quality_score_threshold,
            action_caption_dataset_name=action_caption_dataset_name,
            dense_caption_probability=dense_caption_probability,
        )

        self.action_normalizer: ActionNormalizer | None = load_hand_pose_action_normalizer(
            action_normalization,
            embodiment_type=self.embodiment_type,
            pose_convention=self.pose_convention,
            rotation_format=self.rotation_format,
            apply_forward_clamp=apply_forward_clamp,
            action_stats_path=action_stats_path,
        )

    @staticmethod
    def _field(
        window: MaterializedActionWindow,
        name: str,
        width: int,
    ) -> torch.Tensor:  # returns [T+1,D]
        """Return one required per-frame ACT2 field as a tensor."""
        value = window.per_frame_fields.get(name)
        if value is None:
            raise ValueError(f"Action binary is missing required field {name!r}.")
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != width:
            raise ValueError(f"Expected action field {name!r} to have shape [T+1,{width}], got {array.shape}.")
        return torch.from_numpy(np.ascontiguousarray(array))  # [T+1,D]

    def _build_action_input(self, window: MaterializedActionWindow) -> dict[str, torch.Tensor]:
        """Map VITRA ACT2 fields into the shared hand-pose action builder keys."""
        return {
            HAND_LEFT_POSITION_KEY: self._field(window, _ACT2_HAND_LEFT_POSITION, 63),  # [T+1,63]
            HAND_RIGHT_POSITION_KEY: self._field(window, _ACT2_HAND_RIGHT_POSITION, 63),  # [T+1,63]
            HAND_LEFT_ROTATION_KEY: self._field(
                window, _ACT2_HAND_LEFT_ROTATION, NUM_JOINTS * QUAT_DIM_PER_JOINT
            ),  # [T+1,84]
            HAND_RIGHT_ROTATION_KEY: self._field(
                window, _ACT2_HAND_RIGHT_ROTATION, NUM_JOINTS * QUAT_DIM_PER_JOINT
            ),  # [T+1,84]
            CAM_POSITION_KEY: self._field(window, _ACT2_CAMERA_POSITION, 3),  # [T+1,3]
            CAM_ROTATION_KEY: self._field(window, _ACT2_CAMERA_ROTATION, 4),  # [T+1,4]
        }

    def _has_zero_norm_quaternion(self, action_input: dict[str, torch.Tensor]) -> bool:
        """Return whether any camera or wrist quaternion would fail action construction."""
        camera_norms = torch.linalg.vector_norm(action_input[CAM_ROTATION_KEY], dim=-1)  # [T+1]
        if bool((camera_norms < 1e-8).any()):
            return True
        for rotation_key in (HAND_LEFT_ROTATION_KEY, HAND_RIGHT_ROTATION_KEY):
            rotations = action_input[rotation_key].reshape(
                action_input[rotation_key].shape[0],
                NUM_JOINTS,
                QUAT_DIM_PER_JOINT,
            )  # [T+1,J,4]
            wrist_rotation_norms = torch.linalg.vector_norm(rotations[:, WRIST_JOINT_IDX], dim=-1)  # [T+1]
            if bool((wrist_rotation_norms < 1e-8).any()):
                return True
        return False

    @staticmethod
    def _mark_drop(item: dict[str, Any], reason: str | None) -> dict[str, Any]:
        """Attach the generic packer-side drop marker when needed."""
        if reason is not None:
            item[DROP_SAMPLE_KEY] = True
            item[DROP_SAMPLE_REASON_KEY] = reason
        return item

    def _warn_zero_norm_quaternion_drop(self, global_index: int, source_rows: dict[str, dict[str, Any]]) -> None:
        """Warn for early runtime catches that should be removed by offline quality filtering."""
        warning_count = getattr(self, "_zero_norm_quaternion_drop_warnings", 0)
        if warning_count >= _ZERO_NORM_QUATERNION_DROP_WARNING_LIMIT:
            return
        warning_count += 1
        self._zero_norm_quaternion_drop_warnings = warning_count
        source_row = source_rows.get(_MAIN_VIEW, {})
        source_dataset = source_row.get("source_dataset", getattr(self, "source_dataset_name", "unknown"))
        log.warning(
            "Lance VITRA/embodiment_a: dropped sample with zero-norm quaternion. "
            "This should disappear once offline quality filtering writes/uses retention_quality_score. "
            f"source_dataset={source_dataset!r}, global_index={global_index}, "
            f"warning_count={warning_count}/{_ZERO_NORM_QUATERNION_DROP_WARNING_LIMIT}",
            rank0_only=False,
        )

    def _video_tensor(self, window: MaterializedActionWindow, expected_frames: int) -> torch.Tensor:
        """Convert the decoded HWC view to trainer CHW-time layout."""
        return self._video_view_tchw(window, _MAIN_VIEW, expected_frames).permute(1, 0, 2, 3).contiguous()

    def _training_item_from_action(
        self,
        *,
        action: torch.Tensor,
        aux_seed: int,
        source_rows: dict[str, dict[str, Any]],
        window: MaterializedActionWindow,
        expected_frames: int,
    ) -> dict[str, Any]:
        """Wrap an already-built action tensor in the trainer item contract."""
        if self.skip_video_loading:
            return {"action": action}

        source_row = source_rows[_MAIN_VIEW]
        return {
            "source_repo_id": str(source_row["source_dataset"]),
            "ai_caption": require_action_caption(window.caption, dataset_name="VITRA/embodiment_a"),
            "video": self._video_tensor(window, expected_frames),  # [C,T+1,H,W]
            "action": action,
            "conditioning_fps": torch.tensor(self._window_fps(window), dtype=torch.float32),  # []
            "mode": resolve_lance_action_mode(self.mode, aux_seed),
            "domain_id": torch.tensor(get_domain_id(self.embodiment_type), dtype=torch.long),  # []
            "viewpoint": self.viewpoint,
        }

    def _build_training_item(
        self,
        global_index: int,
        aux_seed: int,
        source_rows: dict[str, dict[str, Any]],
        window: MaterializedActionWindow,
    ) -> dict[str, Any]:
        """Convert a materialized storage window into the VITRA hand-pose trainer contract."""
        action_input = self._build_action_input(window)
        expected_frames = action_input[CAM_POSITION_KEY].shape[0]
        for name, tensor in action_input.items():
            if tensor.shape[0] != expected_frames:
                raise ValueError(
                    f"Expected all VITRA action fields to have {expected_frames} frames; "
                    f"field {name!r} has {tensor.shape[0]}."
                )
        if expected_frames < 2:
            raise ValueError(f"Expected at least two VITRA frames, got {expected_frames}.")

        # Do not resample quality-rejected windows here. The dataloader process
        # has already assigned this Lance draw, so we mark it and let the packer
        # report the dropped draw count for deterministic checkpoint resume.
        drop_reason = None
        if not all(bool(torch.isfinite(tensor).all()) for tensor in action_input.values()):
            drop_reason = "non-finite action input"
        elif self._has_zero_norm_quaternion(action_input):
            drop_reason = "zero-norm quaternion"
            self._warn_zero_norm_quaternion_drop(global_index, source_rows)
        if drop_reason is not None:
            action = torch.zeros((expected_frames - 1, self.action_dim), dtype=torch.float32)  # [T,D]
            item = self._training_item_from_action(
                action=action,
                aux_seed=aux_seed,
                source_rows=source_rows,
                window=window,
                expected_frames=expected_frames,
            )
            return self._mark_drop(item, drop_reason)

        action = build_hand_pose_action(
            action_input,
            pose_convention=self.pose_convention,
            rotation_format=self.rotation_format,
            position_keys=(HAND_LEFT_POSITION_KEY, HAND_RIGHT_POSITION_KEY),
            rotation_keys=(HAND_LEFT_ROTATION_KEY, HAND_RIGHT_ROTATION_KEY),
            position_joint_indices=self.position_joint_indices,
            wrist_frame_alignment=self.wrist_frame_alignment,
        )  # [T,D]
        action_is_finite = torch.isfinite(action).all()  # []
        if not bool(action_is_finite):
            drop_reason = "non-finite action"
            action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)  # [T,D]

        item = self._training_item_from_action(
            action=action,
            aux_seed=aux_seed,
            source_rows=source_rows,
            window=window,
            expected_frames=expected_frames,
        )
        return self._mark_drop(item, drop_reason)

    @property
    def action_dim(self) -> int:
        """Return the unpadded VITRA hand-pose action width."""
        return self._raw_action_dim

    def get_action_normalizer(
        self,
        _sample: dict[str, Any] | None = None,
    ) -> ActionNormalizer | None:
        """Return the configured action normalizer for transform-time preprocessing."""
        return self.action_normalizer

    def get_unified_action_mapping(self) -> UnifiedActionMapping:
        """Map the shared 57D hand-pose layout into ``unified_v1``."""
        if self._raw_action_dim != 57:
            raise ValueError(f"unified_v1 requires the 57D rot6d hand layout, got {self._raw_action_dim}D")
        return hand_pose_57d_mapping()
