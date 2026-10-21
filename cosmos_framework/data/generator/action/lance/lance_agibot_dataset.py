# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from cosmos_framework.data.generator.action.action_binary import get_action_binary_spec_name
from cosmos_framework.data.generator.action.embodiment_c_spec import AGIBOT_GEAR_VIDEO_KEY
from cosmos_framework.data.generator.action.agibot_grippers import action_episode_gripper, calibrated_gripper_fields
from cosmos_framework.data.generator.action.agibot_semantics import (
    AGIBOT_ACTION_DIM,
    AGIBOT_LEFT_HAND_VIEW,
    AGIBOT_MULTIVIEW_DESCRIPTION,
    AGIBOT_RIGHT_HAND_VIEW,
    AgiBotFkEmbodiment,
    agibot_normalizer_filename,
    assemble_embodiment_c_state,
    build_agibot_action_spec,
    build_agibot_fk_action,
)
from cosmos_framework.data.generator.action.agibot_video import AGIBOT_NATIVE_VIDEO_SIZE_HW, resize_to_640x360
from cosmos_framework.data.generator.action.lance.lance_action_dataset import (
    ACTION_NORMALIZERS_DIR,
    DEFAULT_DENSE_CAPTION_PROBABILITY,
    BaseLanceActionDataset,
    InvalidActionSampleError,
    MaterializedActionWindow,
    TargetFps,
    require_action_caption,
    resolve_lance_action_mode,
)
from cosmos_framework.data.generator.action.training_view_keys import AGIBOT_BETA_HEAD_VIEW
from cosmos_framework.data.generator.action.utils.action_processing import (
    ActionNormalizationMethod,
    ActionNormalizer,
    load_action_normalizer,
)
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.utils.pose_utils import PoseConvention, compute_framewise_idle_frames
from cosmos_framework.data.generator.action.utils.unified_action_schema import (
    UnifiedActionMapping,
    humanoid_pose_mapping,
)
from cosmos_framework.data.generator.action.utils.viewpoint_utils import Viewpoint, compose_multiview
from cosmos_framework.data.generator.lance.lance_action_source import LanceActionDatasetSpec

AGIBOT_GEAR_HEAD_VIEW = AGIBOT_GEAR_VIDEO_KEY


AGIBOTWORLD2026_SOURCE_FAMILIES = frozenset({"agibot_2026_a40_s162", "agibot_2026_a40_s169"})


_AGIBOT_FULLY_OPEN_GRIPPER_WIDTH_M: float = 0.12


_ACTION_SPEC = build_agibot_action_spec()


@dataclass(frozen=True)
class _AgiBotSourceProfile:
    """Per-source AgiBot runtime semantics."""

    source_embodiment: str
    fk_embodiment_type: AgiBotFkEmbodiment
    state_layout: str
    state_width: int | None
    primary_view: str


def _agibot_source_profile(source_dataset_name: str) -> _AgiBotSourceProfile:
    """Return the AgiBot semantic profile for one physical source dataset."""
    source_family = get_action_binary_spec_name(source_dataset_name)
    if source_family == "embodiment_c":
        return _AgiBotSourceProfile(
            source_embodiment="embodiment_c_gripper",
            fk_embodiment_type="embodiment_c_gripper",
            state_layout="gear",
            state_width=32,
            primary_view=AGIBOT_GEAR_HEAD_VIEW,
        )
    if source_family == "embodiment_c_ext":
        return _AgiBotSourceProfile(
            source_embodiment="embodiment_c_gripper_ext",
            fk_embodiment_type="embodiment_c_gripper_ext",
            state_layout="gear",
            state_width=94,
            primary_view=AGIBOT_GEAR_HEAD_VIEW,
        )
    if source_family == "agibot_world":
        return _AgiBotSourceProfile(
            source_embodiment="agibotworld",
            fk_embodiment_type="embodiment_c_gripper",
            state_layout="beta",
            state_width=None,
            primary_view=AGIBOT_BETA_HEAD_VIEW,
        )
    raise ValueError(f"Unsupported AgiBot source_dataset_name={source_dataset_name!r}.")


def _agibot_view_names(viewpoint: Viewpoint, primary_view: str) -> tuple[str, ...]:
    """Return the source views needed for one AgiBot viewpoint."""
    if viewpoint == "ego_view":
        return (primary_view,)
    if viewpoint == "concat_view":
        return (primary_view, AGIBOT_LEFT_HAND_VIEW, AGIBOT_RIGHT_HAND_VIEW)
    raise ValueError(f"Unsupported viewpoint={viewpoint!r}. Expected 'ego_view' or 'concat_view'.")


def _resize_agibot_lance_view(
    video_tchw: torch.Tensor,  # [T,C,H,W]
) -> torch.Tensor:  # returns [T,C,H_out,W_out]
    """Correct the aspect ratio of AgiBot views stored at 640x480."""

    if tuple(video_tchw.shape[-2:]) == AGIBOT_NATIVE_VIDEO_SIZE_HW:
        return resize_to_640x360(video_tchw)  # [T,C,360,640]
    return video_tchw  # [T,C,H,W]


def load_agibot_action_normalizer(
    action_normalization: ActionNormalizationMethod | None,
    *,
    pose_convention: PoseConvention,
    apply_forward_clamp: bool,
) -> ActionNormalizer | None:
    if action_normalization is None:
        return None
    return load_action_normalizer(
        action_normalization,
        stats_path=ACTION_NORMALIZERS_DIR / agibot_normalizer_filename(pose_convention),
        apply_forward_clamp=apply_forward_clamp,
        expected_dim=AGIBOT_ACTION_DIM,
    )


class LanceAgiBotDataset(BaseLanceActionDataset):
    """Reconstruct the shared 29D AgiBot FK-pose contract from three ACT2 layouts.

    ``legacy`` preserves historical grippers and sampler checkpoint identity.
    ``hardware_v1`` calibrates using audited episode hardware and requires a
    cache with complete label coverage, as selected by ``LANCE_V2P1_MIXTURE``.
    """

    fully_open_gripper_width_m: float = _AGIBOT_FULLY_OPEN_GRIPPER_WIDTH_M

    def __init__(
        self,
        lance_dataset_spec: LanceActionDatasetSpec,
        creds_path: str | Path,
        chunk_length: int = 16,
        fps: TargetFps = 10,
        mode: str = "joint",
        split: str = "train",
        val_ratio: float | None = None,
        seed: int = 42,
        viewpoint: Viewpoint = "concat_view",
        pose_convention: PoseConvention = "backward_framewise",
        action_normalization: ActionNormalizationMethod | None = None,
        apply_forward_clamp: bool = False,
        skip_video_loading: bool = False,
        source_dataset_name: str | None = None,
        retention_quality_score_threshold: float | None = 0.0,
        dense_caption_probability: float = DEFAULT_DENSE_CAPTION_PROBABILITY,
        gripper_calibration: Literal["legacy", "hardware_v1"] = "legacy",
    ) -> None:
        if gripper_calibration not in {"legacy", "hardware_v1"}:
            raise ValueError(f"Unsupported AgiBot gripper_calibration={gripper_calibration!r}.")
        self.gripper_calibration: Literal["legacy", "hardware_v1"] = gripper_calibration
        resolved_source_dataset_name = self._resolve_source_dataset_name(lance_dataset_spec, source_dataset_name)
        selected_source_datasets = (
            lance_dataset_spec.source_datasets
            if lance_dataset_spec.combine_source_datasets
            else (resolved_source_dataset_name,)
        )
        source_profiles = tuple(_agibot_source_profile(name) for name in selected_source_datasets)
        default_profile = source_profiles[0]
        primary_view = (
            AGIBOT_GEAR_HEAD_VIEW if lance_dataset_spec.combine_source_datasets else default_profile.primary_view
        )
        view_names = _agibot_view_names(viewpoint, primary_view)
        if pose_convention not in {
            "backward_framewise",
            "backward_anchored",
            "backward_chunk_anchored_8f",
            "backward_chunk_anchored_16f",
        }:
            raise ValueError(f"Unsupported pose_convention={pose_convention!r}. Expected a backward pose convention.")

        self.mode: str = mode
        self.viewpoint: Viewpoint = viewpoint
        self.pose_convention: PoseConvention = pose_convention
        self.embodiment_type: str = default_profile.source_embodiment
        self.fk_embodiment_type: AgiBotFkEmbodiment = default_profile.fk_embodiment_type
        self.state_layout: str = default_profile.state_layout
        self.state_width: int | None = default_profile.state_width
        self.primary_view: str = primary_view
        self._combined_source_dataset: bool = lance_dataset_spec.combine_source_datasets
        self._default_source_profile: _AgiBotSourceProfile = default_profile
        super().__init__(
            lance_dataset_spec,
            creds_path,
            view_names=view_names,
            chunk_length=chunk_length,
            fps=fps,
            split=split,
            val_ratio=(0.05 if default_profile.state_layout == "beta" else 0.005) if val_ratio is None else val_ratio,
            seed=seed,
            skip_video_loading=skip_video_loading,
            source_dataset_name=resolved_source_dataset_name,
            retention_quality_score_threshold=retention_quality_score_threshold,
            dense_caption_probability=dense_caption_probability,
        )

        self.action_normalizer: ActionNormalizer | None = load_agibot_action_normalizer(
            action_normalization,
            pose_convention=pose_convention,
            apply_forward_clamp=apply_forward_clamp,
        )

    def _calibrate_gripper_window(
        self,
        source_rows: dict[str, dict[str, Any]],
        window: MaterializedActionWindow,
        profile: _AgiBotSourceProfile,
    ) -> MaterializedActionWindow:
        """Calibrate both observed grippers once, before FK and idle detection."""
        if self.gripper_calibration == "legacy":
            return window
        columns = self.lance_dataset_spec.action_spans
        identities = {
            (str(row[columns.source_dataset]), str(row[columns.shard_id]), int(row[columns.episode_index]))
            for row in source_rows.values()
        }
        if len(identities) != 1:
            raise ValueError("AgiBot gripper calibration requires one consistent source episode across views.")
        hardware = action_episode_gripper(*identities.pop())
        return replace(
            window,
            per_frame_fields=calibrated_gripper_fields(
                window.per_frame_fields,
                hardware=hardware,
                state_layout=profile.state_layout,
                state_width=profile.state_width,
            ),
        )

    def _source_profile_from_rows(self, source_rows: dict[str, dict[str, Any]]) -> _AgiBotSourceProfile:
        """Resolve per-sample semantics from the materialized source row."""
        if not source_rows:
            return self._default_source_profile
        action_spans = self.lance_dataset_spec.action_spans
        primary_row = source_rows.get(self.primary_view, next(iter(source_rows.values())))
        return _agibot_source_profile(str(primary_row[action_spans.source_dataset]))

    @staticmethod
    def _field(window: MaterializedActionWindow, name: str, width: int) -> np.ndarray:  # [T+1,D]
        """Return one finite ACT2 per-frame field with its exact source width."""
        value = window.per_frame_fields.get(name)
        if value is None:
            raise ValueError(f"Action binary is missing required field {name!r}.")
        array = np.asarray(value, dtype=np.float32)  # [T+1,D]
        if array.ndim != 2 or array.shape[1] != width or array.shape[0] < 2:
            raise ValueError(
                f"Expected action field {name!r} to have shape [T+1,{width}] with T >= 1, got {array.shape}."
            )
        if not np.isfinite(array).all():
            raise InvalidActionSampleError(f"Action field {name!r} contains NaN or Inf values.")
        return np.ascontiguousarray(array)  # [T+1,D]

    def _build_fk_state(
        self,
        window: MaterializedActionWindow,
        profile: _AgiBotSourceProfile | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:  # [T+1,S],[T+1,3]|None,[T+1,4]|None
        """Map one source ACT2 layout to FK state plus optional Beta base pose."""
        state_layout = self.state_layout if profile is None else profile.state_layout
        state_width = self.state_width if profile is None else profile.state_width
        if state_layout == "gear":
            if state_width is None:
                raise RuntimeError("GEAR state layout requires a concrete state width.")
            states = self._field(window, "state", state_width)  # [T+1,32|94]
            return states, None, None

        joint_position = self._field(window, "state_joint", 14)  # [T+1,14]
        end_effector = self._field(window, "state_end_effector", 20)  # [T+1,20]
        robot_pose = self._field(window, "state_robot", 7)  # [T+1,7]
        expected_frames = joint_position.shape[0]
        if end_effector.shape[0] != expected_frames or robot_pose.shape[0] != expected_frames:
            raise ValueError(
                "AgiBotWorld-Beta state fields must have matching T+1 lengths; "
                f"got state_joint={joint_position.shape}, state_end_effector={end_effector.shape}, "
                f"state_robot={robot_pose.shape}."
            )

        effector_position = end_effector[:, 0:2]  # [T+1,2]
        head_position = end_effector[:, 16:18]  # [T+1,2]
        waist_position = end_effector[:, 18:20]  # [T+1,2]
        states = assemble_embodiment_c_state(
            effector_position,
            joint_position,
            head_position,
            waist_position,
        )  # [T+1,20]
        robot_position = robot_pose[:, 0:3]  # [T+1,3]
        robot_orientation = robot_pose[:, 3:7]  # [T+1,4]
        return states, robot_position, robot_orientation

    def _build_fk_action(
        self,
        window: MaterializedActionWindow,
        profile: _AgiBotSourceProfile | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:  # [T,29], extras
        """Build legacy-compatible FK action and initial poses."""
        states, robot_position, robot_orientation = self._build_fk_state(window, profile)
        fk_embodiment_type = self.fk_embodiment_type if profile is None else profile.fk_embodiment_type
        return build_agibot_fk_action(
            states,
            fk_embodiment_type=fk_embodiment_type,
            pose_convention=self.pose_convention,
            robot_position=robot_position,
            robot_orientation=robot_orientation,
        )  # [T,29], extras

    def _build_video(
        self,
        window: MaterializedActionWindow,
        expected_frames: int,
        profile: _AgiBotSourceProfile | None = None,
    ) -> torch.Tensor:  # [C,T+1,H,W]
        """Build the ego or legacy top-plus-two-wrists video layout."""
        del profile
        primary = self._video_view_tchw(window, self.primary_view, expected_frames)  # [T+1,C,H,W]
        primary = _resize_agibot_lance_view(primary)  # [T+1,C,H_p,W_p]
        if self.viewpoint == "concat_view":
            left = self._video_view_tchw(window, AGIBOT_LEFT_HAND_VIEW, expected_frames)  # [T+1,C,H_l,W_l]
            right = self._video_view_tchw(window, AGIBOT_RIGHT_HAND_VIEW, expected_frames)  # [T+1,C,H_r,W_r]
            left = _resize_agibot_lance_view(left)  # [T+1,C,H_l',W_l']
            right = _resize_agibot_lance_view(right)  # [T+1,C,H_r',W_r']
            video_tchw = compose_multiview(primary, left, right)  # [T+1,C,3H/2,W]
        else:
            video_tchw = primary  # [T+1,C,H,W]
        return video_tchw.permute(1, 0, 2, 3).contiguous()  # [C,T+1,H',W]

    def _caption(self, value: Any, profile: _AgiBotSourceProfile | None = None) -> tuple[str, str]:
        """Return the trainer caption and Beta's optional debug-only suffix."""
        state_layout = self.state_layout if profile is None else profile.state_layout
        caption = require_action_caption(value, dataset_name="AgiBot")
        debug_caption = ""
        if state_layout == "beta":
            caption, separator, debug_caption = caption.partition("|")
            debug_caption = debug_caption.strip() if separator else ""
        caption = caption.strip()
        caption = require_action_caption(caption, dataset_name="AgiBot")
        return caption, debug_caption

    def _build_training_item(
        self,
        global_index: int,
        aux_seed: int,
        source_rows: dict[str, dict[str, Any]],
        window: MaterializedActionWindow,
    ) -> dict[str, Any]:
        """Convert one Lance-materialized source window to the AgiBot trainer contract."""
        del global_index
        profile = self._source_profile_from_rows(source_rows)
        window = self._calibrate_gripper_window(source_rows, window, profile)
        action, action_extras = self._build_fk_action(window, profile)  # [T,29], extras
        idle_frames = compute_framewise_idle_frames(
            action,
            _ACTION_SPEC,
            fps=self._window_fps(window),
            pose_convention=self.pose_convention,
        )  # [] | None
        if self.skip_video_loading:
            result: dict[str, Any] = {"action": action}
            if idle_frames is not None:
                result["idle_frames"] = idle_frames
            return result

        video = self._build_video(window, action.shape[0] + 1, profile)  # [C,T+1,H,W]
        caption, debug_caption = self._caption(window.caption, profile)
        resolved_mode = resolve_lance_action_mode(self.mode, aux_seed)
        result = {
            "ai_caption": caption,
            "video": video,
            "action": action,
            "conditioning_fps": torch.tensor(self._window_fps(window), dtype=torch.float32),  # []
            "mode": resolved_mode,
            "domain_id": torch.tensor(
                get_domain_id(
                    {
                        "embodiment_c_gripper": "embodiment_c_gripper",
                        "embodiment_c_gripper_ext": "embodiment_c_gripper_ext",
                    }.get(profile.source_embodiment, profile.source_embodiment)
                ),
                dtype=torch.long,
            ),  # []
            "viewpoint": self.viewpoint,
            **action_extras,
        }
        if idle_frames is not None:
            result["idle_frames"] = idle_frames
        if self.viewpoint == "concat_view":
            result["additional_view_description"] = AGIBOT_MULTIVIEW_DESCRIPTION
        if debug_caption:
            result["debug_caption"] = debug_caption
        return result

    @property
    def action_dim(self) -> int:
        """Return the shared FK action width."""
        return AGIBOT_ACTION_DIM

    def get_action_normalizer(self, _sample: dict[str, Any] | None = None) -> ActionNormalizer | None:
        """Return the configured shared AgiBot normalizer."""
        return self.action_normalizer

    def get_unified_action_mapping(self) -> UnifiedActionMapping:
        """Declare AgiBot ego/right/left 29D semantics for ``unified_v1``."""
        return humanoid_pose_mapping()
