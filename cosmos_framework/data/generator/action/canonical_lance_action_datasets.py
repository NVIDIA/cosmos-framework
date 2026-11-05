# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch

from cosmos_framework.data.generator.action.embodiment_c_fk import extract_gripper_states
from cosmos_framework.data.generator.action.agibot_grippers import calibrated_gripper_fields, episode_gripper
from cosmos_framework.data.generator.action.behavior1k_lerobot_dataset_config import (
    render_custom_prompt,
    state61_to_action23,
)
from cosmos_framework.data.generator.action.canonical_lance_dataset import (
    CANONICAL_SOURCE_SAMPLE_ID_KEY,
    CanonicalLanceDatasetMixin,
)
from cosmos_framework.data.generator.action.lance.lance_action_dataset import (
    MaterializedActionWindow,
    require_action_caption,
    resolve_lance_action_mode,
)
from cosmos_framework.data.generator.action.lance.lance_agibot_dataset import (
    LanceAgiBotDataset,
    _AgiBotSourceProfile,
    _resize_agibot_lance_view,
)
from cosmos_framework.data.generator.action.lance.lance_behavior1k_dataset import LanceBehavior1KDataset
from cosmos_framework.data.generator.action.lance.lance_embodiment_a_dataset import Lanceembodiment_aDataset
from cosmos_framework.data.generator.action.lance.lance_yam_dataset import LanceYAMDataset
from cosmos_framework.data.generator.action.utils.domain_utils import get_action_dim, get_domain_id
from cosmos_framework.data.generator.action.utils.image_augmentation import apply_synchronized_image_augmentation
from cosmos_framework.data.generator.action.utils.pose_utils import (
    compute_framewise_idle_frames,
    convert_rotation,
    pose_abs_to_rel,
    pose_rel_to_abs,
)
from cosmos_framework.data.generator.action.yam_semantics import (
    YAMKinematics,
    build_yam_image_augmentor,
    compose_yam_concat_view,
)

_SEMANTIC_ROTATION_MOVEMENT_WEIGHT: float = 0.3


_AGIBOT_ACTION_DIM: int = get_action_dim("agibotworld")


_INITIAL_GRIPPER_STATE_KEY: str = "_canonical_camera_mask_initial_grippers"


_BEHAVIOR1K_ZED_LINK_TO_OPENCV_CAMERA: np.ndarray = np.eye(4, dtype=np.float32)


_BEHAVIOR1K_ZED_LINK_TO_OPENCV_CAMERA[:3, 3] = np.asarray([0.06, 0.0, 0.01], dtype=np.float32)


YAMViewMode = Literal["concat_view", "ego_view", "third_person_view", "wrist_view", "random_side"]


_YAM_CAMERAS_BY_VIEW_MODE: dict[YAMViewMode, tuple[str, ...]] = {
    "concat_view": ("top", "left", "right"),
    "ego_view": ("top",),
    "third_person_view": ("left",),
    "wrist_view": ("right",),
    "random_side": ("left", "right"),
}


_YAM_SINGLE_CAMERA_BY_VIEW_MODE: dict[YAMViewMode, str] = {
    "ego_view": "top",
    "third_person_view": "left",
    "wrist_view": "right",
}


_YAM_CONCAT_VIEW_DESCRIPTION: str = (
    "The top row shows the overhead/top camera. The bottom row contains the left camera "
    "on the left and the right camera on the right."
)


def _pose_movement_score(action: torch.Tensor, pose_slice: slice, pose_convention: str) -> torch.Tensor:
    """Convert one anchored rot6d pose block back to framewise physical movement."""
    pose_np = action[:, pose_slice].detach().to(device="cpu", dtype=torch.float32).numpy()
    poses_abs = pose_rel_to_abs(pose_np, rotation_format="rot6d", pose_convention=pose_convention)
    framewise = pose_abs_to_rel(
        poses_abs,
        rotation_format="rot6d",
        pose_convention="backward_framewise",
    )
    translation = np.linalg.norm(framewise[:, :3], axis=-1)
    rotation = np.asarray(
        convert_rotation(framewise[:, 3:], input_format="rot6d", output_format="matrix"),
        dtype=np.float32,
    )
    angle = np.arccos(np.clip((np.trace(rotation, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0))
    score = np.maximum(translation, _SEMANTIC_ROTATION_MOVEMENT_WEIGHT * angle)
    return torch.from_numpy(score.astype(np.float32, copy=False)).to(device=action.device, dtype=action.dtype)


def _semantic_humanoid_movement_score(action: torch.Tensor, pose_convention: str) -> torch.Tensor:
    """Score arm motion in the shared 29D head/right/gripper/left/gripper action."""
    if action.ndim != 2 or action.shape[1] != _AGIBOT_ACTION_DIM:
        raise ValueError(f"Expected semantic humanoid action [T,{_AGIBOT_ACTION_DIM}], got {tuple(action.shape)}.")
    return torch.maximum(
        _pose_movement_score(action, slice(9, 18), pose_convention),
        _pose_movement_score(action, slice(19, 28), pose_convention),
    )


def _gripper_change(
    action: torch.Tensor,
    indices: tuple[int, ...],
    threshold: float,
    *,
    initial_grippers: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return rows whose scalar gripper state changed beyond the common guard threshold."""
    grippers = action[:, indices]
    prepend = grippers[:1]
    if initial_grippers is not None:
        if initial_grippers.shape != (len(indices),):
            raise ValueError(
                f"Initial gripper state must have shape {(len(indices),)}, got {tuple(initial_grippers.shape)}."
            )
        prepend = initial_grippers.to(device=action.device, dtype=action.dtype).unsqueeze(0)
    delta = torch.diff(grippers, dim=0, prepend=prepend)
    return (~torch.isfinite(delta).all(dim=-1)) | (delta.abs().amax(dim=-1) > threshold)


def _required_initial_grippers(item: dict[str, Any], action: torch.Tensor) -> torch.Tensor:
    """Consume the internal pre-action gripper observation used by camera masking."""
    initial_grippers = item.pop(_INITIAL_GRIPPER_STATE_KEY, None)
    if not isinstance(initial_grippers, torch.Tensor) or initial_grippers.shape != (2,):
        raise ValueError("Camera-masked humanoid samples require two initial gripper observations.")
    return initial_grippers.to(device=action.device, dtype=action.dtype)


def _static_camera_pose(action: torch.Tensor) -> torch.Tensor:
    """Return zero translation plus identity rot6d in the configured relative convention."""
    pose = action.new_zeros((action.shape[0], 9))
    pose[:, 3] = 1
    pose[:, 7] = 1
    return pose


def _humanoid_initial_pose(item: dict[str, Any], key: str) -> np.ndarray:
    """Read one finite 4x4 pose required by the shared 29D humanoid layout."""
    value = item.get(key)
    if not isinstance(value, torch.Tensor) or value.shape != (4, 4):
        raise ValueError(f"29D humanoid camera masking requires {key!r} with shape [4,4].")
    pose = value.detach().to(device="cpu", dtype=torch.float32).numpy()
    if not np.isfinite(pose).all():
        raise ValueError(f"29D humanoid camera-mask initial pose {key!r} contains NaN or Inf.")
    return pose.astype(np.float32, copy=False)


def _humanoid_action_in_head_frame(
    item: dict[str, Any],
    action: torch.Tensor,
    pose_convention: str,
    *,
    head_to_reference: np.ndarray | None = None,
) -> torch.Tensor:
    """Express both 29D wrists in the moving head or calibrated camera frame."""
    if action.ndim != 2 or action.shape[1] != _AGIBOT_ACTION_DIM:
        raise ValueError(f"Expected humanoid action [T,{_AGIBOT_ACTION_DIM}], got {tuple(action.shape)}.")
    action_np = action.detach().to(device="cpu", dtype=torch.float32).numpy()
    head_abs = pose_rel_to_abs(
        action_np[:, 0:9],
        rotation_format="rot6d",
        pose_convention=pose_convention,
        initial_pose=_humanoid_initial_pose(item, "initial_pose"),
    )
    right_abs = pose_rel_to_abs(
        action_np[:, 9:18],
        rotation_format="rot6d",
        pose_convention=pose_convention,
        initial_pose=_humanoid_initial_pose(item, "initial_pose_right"),
    )
    left_abs = pose_rel_to_abs(
        action_np[:, 19:28],
        rotation_format="rot6d",
        pose_convention=pose_convention,
        initial_pose=_humanoid_initial_pose(item, "initial_pose_left"),
    )
    reference_abs = head_abs if head_to_reference is None else head_abs @ head_to_reference
    world_to_head = np.linalg.inv(reference_abs)
    mask_action = action.clone()
    mask_action[:, 9:18] = torch.from_numpy(
        pose_abs_to_rel(
            world_to_head @ right_abs,
            rotation_format="rot6d",
            pose_convention=pose_convention,
        )
    ).to(device=action.device, dtype=action.dtype)
    mask_action[:, 19:28] = torch.from_numpy(
        pose_abs_to_rel(
            world_to_head @ left_abs,
            rotation_format="rot6d",
            pose_convention=pose_convention,
        )
    ).to(device=action.device, dtype=action.dtype)
    return mask_action


class CanonicalLanceAgiBotDataset(CanonicalLanceDatasetMixin, LanceAgiBotDataset):
    """AgiBot Beta/GEAR semantics over preselected canonical samples."""

    CAMERA_MASK_THRESHOLD: float = 1e-3
    CAMERA_MASK_MIN_FRAMES: int = 11
    CAMERA_MASK_GRIPPER_THRESHOLD: float = 2e-3
    CAMERA_MASK_GRIPPER_GUARD_FRAMES: int = 15

    def __init__(
        self,
        *args: Any,
        camera_mask_threshold: float | None = CAMERA_MASK_THRESHOLD,
        camera_mask_min_frames: int = CAMERA_MASK_MIN_FRAMES,
        camera_mask_gripper_threshold: float = CAMERA_MASK_GRIPPER_THRESHOLD,
        camera_mask_gripper_guard_frames: int = CAMERA_MASK_GRIPPER_GUARD_FRAMES,
        **kwargs: Any,
    ) -> None:
        # Canonical AgiBot already uses audited calibration for every sample.
        kwargs["gripper_calibration"] = "hardware_v1"
        super().__init__(
            *args,
            camera_mask_threshold=camera_mask_threshold,
            camera_mask_min_frames=camera_mask_min_frames,
            camera_mask_gripper_threshold=camera_mask_gripper_threshold,
            camera_mask_gripper_guard_frames=camera_mask_gripper_guard_frames,
            **kwargs,
        )

    @staticmethod
    def _video_view_tchw(
        window: MaterializedActionWindow,
        view_name: str,
        expected_frames: int,
    ) -> torch.Tensor:
        """Apply AgiBot's existing 640x360 preprocessing to every RGB view."""
        return _resize_agibot_lance_view(LanceAgiBotDataset._video_view_tchw(window, view_name, expected_frames))

    def _caption(self, value: Any, profile: Any = None) -> tuple[str, str]:
        """Use the configured full Lance caption verbatim, including Beta details."""
        del profile
        caption = "" if value is None else str(value).strip()
        return caption, ""

    def _calibrate_gripper_window(
        self,
        source_rows: dict[str, dict[str, Any]],
        window: MaterializedActionWindow,
        profile: _AgiBotSourceProfile,
    ) -> MaterializedActionWindow:
        """Resolve canonical episode identity for the parent's calibration hook."""
        primary_row = source_rows.get(self.primary_view, next(iter(source_rows.values())))
        hardware = episode_gripper(primary_row[CANONICAL_SOURCE_SAMPLE_ID_KEY])
        # The parent still owns FK, idle detection and native 29D construction.
        # Calibrate its input, not just the final action: the initial grippers
        # used below and the later unified 59D conversion must agree as well.
        return replace(
            window,
            per_frame_fields=calibrated_gripper_fields(
                window.per_frame_fields,
                hardware=hardware,
                state_layout=profile.state_layout,
                state_width=profile.state_width,
            ),
        )

    def _build_training_item(
        self,
        global_index: int,
        aux_seed: int,
        source_rows: dict[str, dict[str, Any]],
        window: MaterializedActionWindow,
    ) -> dict[str, Any]:
        """Use the parent's calibrated action and the same initial camera guard."""
        result = super()._build_training_item(global_index, aux_seed, source_rows, window)
        if self.camera_mask:
            profile = self._source_profile_from_rows(source_rows)
            window = self._calibrate_gripper_window(source_rows, window, profile)
            states, _robot_position, _robot_orientation = self._build_fk_state(window, profile)
            right_gripper, left_gripper = extract_gripper_states(states[:1], profile.fk_embodiment_type)
            if right_gripper is None or left_gripper is None:
                raise ValueError("AgiBot camera masking requires two scalar gripper observations.")
            result[_INITIAL_GRIPPER_STATE_KEY] = torch.tensor(
                [right_gripper[0], left_gripper[0]],
                dtype=result["action"].dtype,
            )
        return result

    def _camera_mask_source_action(self, item: dict[str, Any], action: torch.Tensor) -> torch.Tensor:
        """Express both wrists in the moving head frame before scoring motion."""
        return _humanoid_action_in_head_frame(item, action, self.pose_convention)

    def _camera_mask_movement_score(self, action: torch.Tensor) -> torch.Tensor:
        return _semantic_humanoid_movement_score(action, self.pose_convention)

    def _camera_mask_initial_grippers(self, item: dict[str, Any], action: torch.Tensor) -> torch.Tensor:
        return _required_initial_grippers(item, action)

    def _camera_mask_gripper_change(
        self,
        action: torch.Tensor,
        *,
        initial_grippers: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _gripper_change(
            action,
            (18, 28),
            self.camera_mask_gripper_threshold,
            initial_grippers=initial_grippers,
        )


class CanonicalLanceembodiment_aDataset(CanonicalLanceDatasetMixin, Lanceembodiment_aDataset):
    """embodiment_a hand-pose semantics over preselected canonical samples."""

    def __init__(self, *args: Any, camera_mask: bool = False, **kwargs: Any) -> None:
        if camera_mask:
            raise ValueError("embodiment_a camera masking is disabled because its 57D wrist poses include camera motion.")
        super().__init__(*args, camera_mask=False, **kwargs)

    def _build_training_item(
        self,
        global_index: int,
        aux_seed: int,
        source_rows: dict[str, dict[str, Any]],
        window: MaterializedActionWindow,
    ) -> dict[str, Any]:
        """Match HandPoseDataset's task-name whitespace normalization."""
        result = super()._build_training_item(global_index, aux_seed, source_rows, window)
        caption = result.get("ai_caption")
        if isinstance(caption, str):
            result["ai_caption"] = caption.replace("_", " ")
        return result


class CanonicalLanceYAMDataset(CanonicalLanceDatasetMixin, LanceYAMDataset):
    """Preserve the legacy interactive YAM contract on canonical samples.

    The registered interactive experiments predate the consolidated YAM
    semantics. They use the ``grasp_site`` frame, no YAM-to-OpenCV rotation,
    and their original per-population normalizers. All five legacy view modes
    are available while the registered experiments continue to default to the
    top-camera ``ego_view``.
    """

    CAMERA_MASK_THRESHOLD: float = 1e-3
    CAMERA_MASK_MIN_FRAMES: int = 11
    CAMERA_MASK_GRIPPER_THRESHOLD: float = 5e-3
    CAMERA_MASK_GRIPPER_GUARD_FRAMES: int = 15

    def __init__(
        self,
        *args: Any,
        viewpoint: str = "ego_view",
        ee_site_name: str = "grasp_site",
        apply_opencv_correction: bool = False,
        camera_mask_threshold: float | None = CAMERA_MASK_THRESHOLD,
        camera_mask_min_frames: int = CAMERA_MASK_MIN_FRAMES,
        camera_mask_gripper_threshold: float = CAMERA_MASK_GRIPPER_THRESHOLD,
        camera_mask_gripper_guard_frames: int = CAMERA_MASK_GRIPPER_GUARD_FRAMES,
        **kwargs: Any,
    ) -> None:
        if viewpoint not in _YAM_CAMERAS_BY_VIEW_MODE:
            raise ValueError(
                f"Unsupported canonical interactive YAM viewpoint={viewpoint!r}; "
                f"expected one of {tuple(_YAM_CAMERAS_BY_VIEW_MODE)}."
            )
        if not ee_site_name.strip():
            raise ValueError("ee_site_name must not be empty.")
        view_mode = cast(YAMViewMode, viewpoint)
        super().__init__(
            *args,
            viewpoint="concat_view",
            camera_mask_threshold=camera_mask_threshold,
            camera_mask_min_frames=camera_mask_min_frames,
            camera_mask_gripper_threshold=camera_mask_gripper_threshold,
            camera_mask_gripper_guard_frames=camera_mask_gripper_guard_frames,
            **kwargs,
        )
        self.viewpoint: YAMViewMode = view_mode
        self._view_mode: YAMViewMode = view_mode
        self.view_names: tuple[str, ...] = tuple(
            self._camera_views[name] for name in _YAM_CAMERAS_BY_VIEW_MODE[view_mode]
        )
        if self.robot_multiview:
            self.view_names = tuple(self.canonical_dataset_spec.view_stream_id_by_name)
        self._ee_site_name: str = ee_site_name
        self._to_opencv = self._to_opencv if apply_opencv_correction else np.eye(3, dtype=np.float32)  # [3,3]

    def _get_kinematics(self) -> YAMKinematics:
        """Select the legacy experiment's named end-effector site locally."""
        kinematics = super()._get_kinematics()
        import mujoco  # type: ignore[import-not-found]

        site_id = mujoco.mj_name2id(kinematics.model, mujoco.mjtObj.mjOBJ_SITE, self._ee_site_name)
        if site_id < 0:
            raise RuntimeError(f"YAM MJCF is missing site {self._ee_site_name!r}.")
        kinematics.ee_site_id = int(site_id)
        return kinematics

    def _build_action(
        self,
        state: np.ndarray,  # [T+1,S]
        raw_action: np.ndarray,  # [T+1,S]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Drop the terminal ACT2 action to match LeRobot's T-action query."""
        if state.ndim != 2 or raw_action.ndim != 2 or state.shape[0] != raw_action.shape[0]:
            raise ValueError(
                "Canonical YAM requires aligned 2D state/action records; "
                f"got state={tuple(state.shape)}, action={tuple(raw_action.shape)}."
            )
        return super()._build_action(state, raw_action[:-1])

    def _build_training_item(
        self,
        global_index: int,
        aux_seed: int,
        source_rows: dict[str, dict[str, Any]],
        window: MaterializedActionWindow,
    ) -> dict[str, Any]:
        """Emit the same camera selection and layout as the legacy YAM loaders."""
        del global_index, source_rows
        state_values = window.per_frame_fields.get("state")
        action_values = window.per_frame_fields.get("action")
        if state_values is None or action_values is None:
            raise ValueError("Action binary is missing required fields 'state' and 'action'.")
        state = np.asarray(state_values, dtype=np.float32)  # [T+1,S]
        raw_action = np.asarray(action_values, dtype=np.float32)  # [T+1,S]
        expected_frames = state.shape[0] if state.ndim else 0
        if state.ndim != 2 or expected_frames < 2 or raw_action.shape[0] != expected_frames:
            raise ValueError(
                "Expected aligned YAM state/action arrays with T+1>=2; "
                f"got state={tuple(state.shape)}, action={tuple(raw_action.shape)}."
            )

        action, extras = self._build_action(state, raw_action)  # [T,20] | [T,14]
        idle_frames = compute_framewise_idle_frames(
            action,
            self.action_spec,
            fps=self._window_fps(window),
            pose_convention=self.pose_convention,
        )  # [] or None
        if self.skip_video_loading:
            result: dict[str, Any] = {"action": action}
            if idle_frames is not None:
                result["idle_frames"] = idle_frames
            return result

        views = [
            self._video_view_tchw(window, view_name, expected_frames) for view_name in self.view_names
        ]  # each [T+1,C,H,W]
        if self.use_image_augmentation:
            if self._image_augmentor is None:
                _, _, height, width = views[0].shape
                self._image_augmentor = build_yam_image_augmentor(height, width)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(aux_seed)
                views = apply_synchronized_image_augmentation(views, self._image_augmentor)  # each [T+1,C,H,W]
        view_by_name = dict(zip(self.view_names, views, strict=True))  # values [T+1,C,H,W]
        if self._view_mode == "concat_view":
            video_tchw = compose_yam_concat_view(
                view_by_name[self._camera_views["top"]],
                view_by_name[self._camera_views["left"]],
                view_by_name[self._camera_views["right"]],
            )  # [T+1,C,3H/2,W]
        elif self._view_mode == "random_side":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(aux_seed)
                choose_left = torch.rand(()) < 0.5  # []
            camera_name = "left" if bool(choose_left.item()) else "right"
            video_tchw = view_by_name[self._camera_views[camera_name]]  # [T+1,C,H,W]
        else:
            camera_name = _YAM_SINGLE_CAMERA_BY_VIEW_MODE[self._view_mode]
            video_tchw = view_by_name[self._camera_views[camera_name]]  # [T+1,C,H,W]
        video = video_tchw.permute(1, 0, 2, 3).contiguous()  # [C,T+1,H_out,W]
        result = {
            "ai_caption": require_action_caption(window.caption, dataset_name=self._dataset_label),
            "video": video,
            "action": action,
            "conditioning_fps": torch.tensor(self._window_fps(window), dtype=torch.float32),  # []
            "mode": resolve_lance_action_mode(self.mode, aux_seed),
            "domain_id": torch.tensor(get_domain_id(self._embodiment_type), dtype=torch.long),  # []
            "viewpoint": self.viewpoint,
            **extras,
        }
        if self._view_mode == "concat_view":
            result["additional_view_description"] = _YAM_CONCAT_VIEW_DESCRIPTION
        if idle_frames is not None:
            result["idle_frames"] = idle_frames
        return result

    def _camera_mask_movement_score(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim != 2 or action.shape[1] != 20:
            raise ValueError(f"Expected YAM end-effector action [T,20], got {tuple(action.shape)}.")
        return torch.maximum(
            _pose_movement_score(action, slice(0, 9), self.pose_convention),
            _pose_movement_score(action, slice(10, 19), self.pose_convention),
        )

    def _camera_mask_gripper_change(
        self,
        action: torch.Tensor,
        *,
        initial_grippers: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del initial_grippers
        return _gripper_change(action, (9, 19), self.camera_mask_gripper_threshold)

    def _camera_mask_camera_pose(self, item: dict[str, Any], action: torch.Tensor) -> torch.Tensor:
        """YAM's fixed external camera has zero translation and identity rotation."""
        del item
        return _static_camera_pose(action)


class CanonicalLanceBehavior1KDataset(
    CanonicalLanceDatasetMixin,
    LanceBehavior1KDataset,
):
    """Apply prompts and head-relative camera masking to B1K's 29D FK action."""

    CAMERA_MASK_THRESHOLD: float = 2e-3
    CAMERA_MASK_MIN_FRAMES: int = 11
    CAMERA_MASK_GRIPPER_THRESHOLD: float = 1e-2
    CAMERA_MASK_GRIPPER_GUARD_FRAMES: int = 15

    def __init__(
        self,
        *args: Any,
        prompt_map_path: str | Path | None = None,
        prompt_version: str = "custom",
        viewpoint: str = "ego_view",
        camera_mask: bool = False,
        camera_mask_threshold: float | None = CAMERA_MASK_THRESHOLD,
        camera_mask_min_frames: int = CAMERA_MASK_MIN_FRAMES,
        camera_mask_gripper_threshold: float = CAMERA_MASK_GRIPPER_THRESHOLD,
        camera_mask_gripper_guard_frames: int = CAMERA_MASK_GRIPPER_GUARD_FRAMES,
        **kwargs: Any,
    ) -> None:
        if prompt_version != "custom":
            raise ValueError("Canonical interactive Behavior-1K currently supports prompt_version='custom'.")
        if camera_mask and viewpoint != "ego_view":
            raise ValueError("Behavior-1K camera masking requires viewpoint='ego_view' to match the head trajectory.")
        self._prompt_map: dict[str, dict[str, Any]] = {}
        if prompt_map_path is not None:
            with Path(prompt_map_path).open() as file:
                loaded_prompt_map = json.load(file)
            if not isinstance(loaded_prompt_map, dict):
                raise TypeError(f"Behavior-1K prompt map at {prompt_map_path!r} must be a JSON object.")
            self._prompt_map = loaded_prompt_map
        super().__init__(
            *args,
            viewpoint=viewpoint,
            camera_mask=camera_mask,
            camera_mask_threshold=camera_mask_threshold,
            camera_mask_min_frames=camera_mask_min_frames,
            camera_mask_gripper_threshold=camera_mask_gripper_threshold,
            camera_mask_gripper_guard_frames=camera_mask_gripper_guard_frames,
            **kwargs,
        )
        # Keep the unrelated flat-table quality filter disabled. Canonical
        # materialization owns source selection for Interactive training.
        self.retention_quality_score_threshold = None

    def _caption(self, raw_task_name: str) -> str:
        """Render the same custom task prompt as the LeRobot experiment."""
        entry = self._prompt_map.get(raw_task_name)
        if entry is None:
            return raw_task_name.replace("_", " ").strip()
        return render_custom_prompt(entry, raw_task_name)

    def _build_training_item(
        self,
        global_index: int,
        aux_seed: int,
        source_rows: dict[str, dict[str, Any]],
        window: MaterializedActionWindow,
    ) -> dict[str, Any]:
        """Keep the parent's 29D FK action and replace only its prompt."""
        result = super()._build_training_item(global_index, aux_seed, source_rows, window)
        caption = result.get("ai_caption")
        if isinstance(caption, str):
            result["ai_caption"] = self._caption(caption)
        if self.camera_mask:
            state = result.get("state")
            if not isinstance(state, torch.Tensor) or state.ndim != 2 or state.shape[1] != 61:
                raise ValueError("Behavior-1K camera masking requires the parent state tensor with shape [T+1,61].")
            initial_native_action = state61_to_action23(state[0], base_vel_state_scale=1.0)
            result[_INITIAL_GRIPPER_STATE_KEY] = torch.stack(
                (
                    (initial_native_action[22] + 1.0) / 2.0,
                    (initial_native_action[14] + 1.0) / 2.0,
                )
            ).clamp_(0.0, 1.0)
        return result

    def _camera_mask_source_action(self, item: dict[str, Any], action: torch.Tensor) -> torch.Tensor:
        """Express both B1K wrists in the rendered ZED camera frame."""
        return _humanoid_action_in_head_frame(
            item,
            action,
            self.pose_convention,
            head_to_reference=_BEHAVIOR1K_ZED_LINK_TO_OPENCV_CAMERA,
        )

    def _camera_mask_camera_pose(self, item: dict[str, Any], action: torch.Tensor) -> torch.Tensor:
        """Convert the parent's 29D ``zed_link`` trajectory to the rendered camera."""
        if action.ndim != 2 or action.shape[1] != _AGIBOT_ACTION_DIM:
            raise ValueError(f"Expected B1K FK action [T,{_AGIBOT_ACTION_DIM}], got {tuple(action.shape)}.")
        action_np = action.detach().to(device="cpu", dtype=torch.float32).numpy()
        zed_link_abs = pose_rel_to_abs(
            action_np[:, :9],
            rotation_format="rot6d",
            pose_convention=self.pose_convention,
            initial_pose=_humanoid_initial_pose(item, "initial_pose"),
        )
        camera_abs = zed_link_abs @ _BEHAVIOR1K_ZED_LINK_TO_OPENCV_CAMERA
        camera_pose = pose_abs_to_rel(
            camera_abs,
            rotation_format="rot6d",
            pose_convention=self.pose_convention,
        )
        return torch.from_numpy(camera_pose).to(device=action.device, dtype=action.dtype)

    def _camera_mask_movement_score(self, action: torch.Tensor) -> torch.Tensor:
        return _semantic_humanoid_movement_score(action, self.pose_convention)

    def _camera_mask_initial_grippers(self, item: dict[str, Any], action: torch.Tensor) -> torch.Tensor:
        return _required_initial_grippers(item, action)

    def _camera_mask_gripper_change(
        self,
        action: torch.Tensor,
        *,
        initial_grippers: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _gripper_change(
            action,
            (18, 28),
            self.camera_mask_gripper_threshold,
            initial_grippers=initial_grippers,
        )


__all__ = [
    "CanonicalLanceAgiBotDataset",
    "CanonicalLanceBehavior1KDataset",
    "CanonicalLanceembodiment_aDataset",
    "CanonicalLanceYAMDataset",
    "YAMViewMode",
]
