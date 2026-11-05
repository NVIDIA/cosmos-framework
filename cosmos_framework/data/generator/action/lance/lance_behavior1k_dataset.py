# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from cosmos_framework.data.generator.action.action_binary import get_action_binary_spec_name
from cosmos_framework.data.generator.action.behavior1k_fk import (
    BEHAVIOR1K_R1PRO_FULLY_OPEN_GRIPPER_WIDTH_M,
    build_behavior1k_state_fk_action,
)
from cosmos_framework.data.generator.action.behavior1k_lerobot_dataset_config import (
    BEHAVIOR1K_BENCHMARK_SOURCE_DATASET,
    load_behavior1k_holdout_episode_indices,
    state61_to_action23,
)
from cosmos_framework.data.generator.action.lance.lance_action_dataset import (
    DEFAULT_DENSE_CAPTION_PROBABILITY,
    DEFAULT_SOURCE_METADATA_CACHE_BLOCKS,
    BaseLanceActionDataset,
    InvalidActionSampleError,
    MaterializedActionWindow,
    TargetFps,
    require_action_caption,
    resolve_lance_action_mode,
)
from cosmos_framework.data.generator.action.utils.action_spec import (
    ActionSpec,
    Gripper,
    Joint,
    Pos,
    Rot,
    build_action_spec,
)
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.utils.pose_utils import PoseConvention, compute_framewise_idle_frames
from cosmos_framework.data.generator.action.utils.unified_action_schema import (
    UnifiedActionMapping,
    humanoid_pose_mapping,
)
from cosmos_framework.data.generator.action.utils.viewpoint_utils import Viewpoint, compose_multiview
from cosmos_framework.data.generator.lance.lance_action_source import LanceActionDatasetSpec
from cosmos_framework.utils import log


def normalize_task_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Validate and normalize explicit task probabilities; zero disables a task."""
    values = {str(task): float(weight) for task, weight in weights.items()}
    if not values or any(not task or not math.isfinite(w) or w < 0 for task, w in values.items()):
        raise ValueError("Task weights must be nonempty, finite, and nonnegative.")
    total = math.fsum(values.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Task weights must have a finite positive total.")
    return {task: values[task] / total for task in sorted(values) if values[task] > 0}


_SOURCE_SPEC_NAMES = frozenset(
    {
        "behavior_1k_demos",
        "behavior_1k_demos_subtask",
    }
)


_HEAD_VIEW = "observation.rgb.zed_link_camera_0"


_LEFT_WRIST_VIEW = "observation.rgb.left_realsense_link_camera_0"


_RIGHT_WRIST_VIEW = "observation.rgb.right_realsense_link_camera_0"


_VIEW_NAMES = (_HEAD_VIEW, _LEFT_WRIST_VIEW, _RIGHT_WRIST_VIEW)


_EMBODIMENT_TYPE = "behavior1k_lerobot"


_ACTION_DIM = 29


_ACTION_SPEC = build_action_spec(
    Pos(prefix="head"),
    Rot("rot6d", prefix="head"),
    Pos(prefix="right"),
    Rot("rot6d", prefix="right"),
    Gripper(prefix="right"),
    Pos(prefix="left"),
    Rot("rot6d", prefix="left"),
    Gripper(prefix="left"),
)


_JOINT_ACTION_DIM = 23


_JOINT_ACTION_SPEC = build_action_spec(
    Joint(n=3, label="base"),
    Joint(n=4, label="trunk"),
    Joint(n=7, label="arm", prefix="left"),
    Gripper(prefix="left"),
    Joint(n=7, label="arm", prefix="right"),
    Gripper(prefix="right"),
)


def _load_official_prompts(demos_root: str) -> dict[str, str]:
    opath = f"{demos_root.rstrip('/')}/meta/tasks.jsonl"
    with open(opath) as _f:
        return {r["task_name"]: r["task"] for r in (json.loads(_ln) for _ln in _f if _ln.strip())}


class LanceBehavior1KDataset(BaseLanceActionDataset):
    """Load B1K as state-derived SE(3) or native joint commands.

    ``action_representation="native_joint"`` reads the ACT2 command field,
    optionally prepending state mapped to action coordinates (base scale 0.75).
    The default ``state_fk`` preserves the existing 29-D SE(3) contract.
    """

    # Keep default schema queries available without initializing storage or FK assets.
    action_representation: Literal["state_fk", "native_joint"] = "state_fk"

    # Calibrated against the R1Pro URDF limits and the B1K LeRobot state
    # manifest: all four finger qpos channels reach 0.05 m, and each gripper
    # has two opposing fingers.  B1K action channels remain [0,1] openness;
    # the metric-gripper sampler uses this width to normalize them.
    fully_open_gripper_width_m: float = BEHAVIOR1K_R1PRO_FULLY_OPEN_GRIPPER_WIDTH_M

    def __init__(
        self,
        lance_dataset_spec: LanceActionDatasetSpec,
        creds_path: str | Path,
        urdf_path: str | Path = "",
        chunk_length: int = 16,
        fps: TargetFps = 10,
        mode: str = "joint",
        split: str = "train",
        val_ratio: float = 0.03,
        seed: int = 42,
        excluded_episode_manifest_path: str | Path | None = None,
        viewpoint: Viewpoint = "concat_view",
        pose_convention: PoseConvention = "backward_framewise",
        require_absolute_base_pose: bool = True,
        skip_video_loading: bool = False,
        view_read_workers: int = 2,
        source_dataset_name: str | None = None,
        metadata_block_size: int = 128,
        metadata_cache_blocks: int = DEFAULT_SOURCE_METADATA_CACHE_BLOCKS,
        max_retries: int = 10,
        one_anchor_per_row: bool = False,
        row_index_cache_tag: str | None = None,
        dense_caption_probability: float = DEFAULT_DENSE_CAPTION_PROBABILITY,
        action_representation: Literal["state_fk", "native_joint"] = "state_fk",
        use_state: bool = True,
        base_vel_state_scale: float = 0.75,
        local_files: bool = False,
        task_names: Any = None,
        caption_remap: Any = None,
        demos_root: str | None = None,
        task_weights: dict[str, float] | None = None,
        match_lerobot_split: bool = False,
        require_full_episode_source: bool = False,
        keep_ranges_path: str | None = None,
    ) -> None:
        if match_lerobot_split or task_weights is not None or keep_ranges_path is not None:
            raise ValueError("Local bundles already resolve split, task and keep-range selection during export.")
        if action_representation not in {"state_fk", "native_joint"}:
            raise ValueError(f"Unsupported B1K action_representation={action_representation!r}")
        if not np.isfinite(base_vel_state_scale) or base_vel_state_scale <= 0:
            raise ValueError("base_vel_state_scale must be finite and positive.")
        self.action_representation = action_representation
        self.local_files: bool = local_files
        self._task_weights: dict[str, float] | None = (
            None if task_weights is None else normalize_task_weights(task_weights)
        )
        self._keep_ranges_override: dict[str, list[list[int]]] | None = None
        self._keep_ranges_digest: str | None = None
        if keep_ranges_path is not None:
            payload = Path(keep_ranges_path).read_bytes()
            self._keep_ranges_override = json.loads(payload)
            self._keep_ranges_digest = hashlib.sha256(payload).hexdigest()
        self._weighted_windows: Any = None
        self._requested_split: str = split
        self._match_lerobot_split: bool = match_lerobot_split
        self._require_full_episode_source: bool = require_full_episode_source
        self._lerobot_episode_ids: set[int] | None = None
        self._lerobot_episode_digest: str | None = None
        if self._task_weights is not None:
            if task_names is not None and set(self._task_weights) - set(task_names):
                raise ValueError("Positive task weights include tasks outside task_names.")
            task_names = list(self._task_weights)
        self.emit_state: bool = use_state
        self.base_vel_state_scale: float = base_vel_state_scale
        # Official prompts (task_name -> task) from ``{demos_root}/meta/tasks.jsonl``, read
        # here rather than at module import: this module is imported by the whole experiment
        # registry (posttrain/config/config.py::_register_experiment_catalogs), so a
        # module-level read would make every unrelated job depend on the b1k dataset mount.
        # Mirrors behavior1k_lerobot_dataset.py, which loads its prompts in __init__ too.
        # Skipped entirely when both derived values are supplied at launch.
        official_prompts = (
            _load_official_prompts(demos_root) if demos_root and (task_names is None or caption_remap is None) else {}
        )
        if official_prompts:
            log.info(f"{type(self).__name__}: loaded {len(official_prompts)} official prompts from {demos_root}")
        # Filter the dataset to these snake ``task_name``s (== the cache canonical_caption);
        # defaults to every task in the demos root's meta/tasks.jsonl.
        if task_names is None and official_prompts:
            task_names = sorted(official_prompts)
        self._task_names: frozenset[str] | None = (
            None if task_names is None else frozenset(str(t).strip() for t in task_names)
        )
        # Map the snake ``task_name`` the cache stores back to the natural-language
        # ``source_caption`` so the model is conditioned on the source_caption prompt.
        self._caption_remap: dict[str, str] = dict(caption_remap) if caption_remap else dict(official_prompts)
        if viewpoint not in {"ego_view", "wrist_view", "concat_view"}:
            raise ValueError(f"Unsupported B1K viewpoint={viewpoint!r}.")
        resolved_source_dataset_name = self._resolve_source_dataset_name(lance_dataset_spec, source_dataset_name)
        selected_source_datasets = (
            lance_dataset_spec.source_datasets
            if lance_dataset_spec.combine_source_datasets
            else (resolved_source_dataset_name,)
        )
        unsupported_sources = [
            name for name in selected_source_datasets if get_action_binary_spec_name(name) not in _SOURCE_SPEC_NAMES
        ]
        if unsupported_sources:
            raise ValueError(f"LanceBehavior1KDataset received non-B1K ACT2 sources: {unsupported_sources!r}.")

        self._excluded_episode_ids: frozenset[int] = frozenset()
        # The benchmark manifest belongs only to the production subtask population.
        # Legacy and combined-source specs remain unfiltered.
        if excluded_episode_manifest_path is not None and selected_source_datasets == (
            BEHAVIOR1K_BENCHMARK_SOURCE_DATASET,
        ):
            self._excluded_episode_ids = load_behavior1k_holdout_episode_indices(excluded_episode_manifest_path)

        # The legacy ``behavior_1k_demos`` spec carries no ``base_pose`` field, so absolute
        # base-pose replay is unavailable; requiring it would raise on every sampled item.
        # Fail fast at construction rather than per-item during training.
        if (
            action_representation == "state_fk"
            and require_absolute_base_pose
            and any(get_action_binary_spec_name(name) == "behavior_1k_demos" for name in selected_source_datasets)
        ):
            raise ValueError(
                "require_absolute_base_pose=True is incompatible with the legacy "
                "'behavior_1k_demos' spec (no base_pose field); set require_absolute_base_pose=False "
                "or use the 'behavior_1k_demos_subtask' spec."
            )

        self.urdf_path: Path = Path(urdf_path)
        self.mode: str = mode
        self.viewpoint: Viewpoint = viewpoint
        self.pose_convention: PoseConvention = pose_convention
        self.require_absolute_base_pose: bool = require_absolute_base_pose
        self.embodiment_type: str = _EMBODIMENT_TYPE
        self.action_spec: ActionSpec = _JOINT_ACTION_SPEC if action_representation == "native_joint" else _ACTION_SPEC
        view_names = {
            "ego_view": (_HEAD_VIEW,),
            "wrist_view": (_LEFT_WRIST_VIEW, _RIGHT_WRIST_VIEW),
            "concat_view": _VIEW_NAMES,
        }[viewpoint]

        super().__init__(
            lance_dataset_spec,
            creds_path,
            view_names=view_names,
            chunk_length=chunk_length,
            fps=fps,
            split="full" if match_lerobot_split else split,
            val_ratio=val_ratio,
            seed=seed,
            skip_video_loading=skip_video_loading,
            view_read_workers=view_read_workers,
            source_dataset_name=resolved_source_dataset_name,
            metadata_block_size=metadata_block_size,
            metadata_cache_blocks=metadata_cache_blocks,
            max_retries=max_retries,
            one_anchor_per_row=one_anchor_per_row,
            row_index_cache_tag=row_index_cache_tag,
            dense_caption_probability=0.0 if action_representation == "native_joint" else dense_caption_probability,
        )

    @staticmethod
    def _field(window: MaterializedActionWindow, name: str, width: int) -> np.ndarray:
        """Return one finite ACT2 field shaped ``[T+1,width]``."""

        value = window.per_frame_fields.get(name)
        if value is None:
            raise ValueError(f"B1K ACT2 payload is missing required field {name!r}.")
        array = np.asarray(value, dtype=np.float32)  # [T+1,D]
        if array.ndim != 2 or array.shape[1] != width or array.shape[0] < 2:
            raise ValueError(f"Expected B1K field {name!r} [T+1,{width}], got {array.shape}.")
        if not np.isfinite(array).all():
            raise InvalidActionSampleError(f"B1K ACT2 field {name!r} contains NaN or Inf.")
        return np.ascontiguousarray(array)  # [T+1,D]

    def _build_video(self, window: MaterializedActionWindow, expected_frames: int) -> torch.Tensor:
        """Build B1K's head, wrist, or head-plus-wrists video layout."""

        if self.viewpoint == "ego_view":
            return self._video_view_tchw(window, _HEAD_VIEW, expected_frames).permute(1, 0, 2, 3).contiguous()
        left = self._video_view_tchw(window, _LEFT_WRIST_VIEW, expected_frames)  # [T+1,C,H,W]
        right = self._video_view_tchw(window, _RIGHT_WRIST_VIEW, expected_frames)  # [T+1,C,H,W]
        if self.viewpoint == "wrist_view":
            video_tchw = torch.cat((left, right), dim=-1)  # [T+1,C,H,2W]
        else:
            head = self._video_view_tchw(window, _HEAD_VIEW, expected_frames)  # [T+1,C,H,W]
            video_tchw = compose_multiview(head, left, right)  # [T+1,C,3H/2,W]
        return video_tchw.permute(1, 0, 2, 3).contiguous()  # [C,T+1,H',W']

    def _build_native_joint_item(
        self,
        global_index: int,
        aux_seed: int,
        source_rows: dict[str, dict[str, Any]],
        window: MaterializedActionWindow,
    ) -> dict[str, Any]:
        del global_index, source_rows
        state = self._field(window, "state", 61)  # [T+1, 61]
        action23 = self._field(window, "action", 23)  # [T+1, 23] native R1Pro command

        # Predicted actions = first T native commands (anchored at window frame 0),
        # the Lance analog of LeRobot's action[-chunk:] at the same anchor.
        action = torch.from_numpy(action23[:-1].copy())  # [T, 23]
        if self.emit_state:
            initial_state = state61_to_action23(
                torch.from_numpy(state[0].copy()).float(),
                base_vel_state_scale=self.base_vel_state_scale,
            )  # [23]
            action = torch.cat([initial_state.unsqueeze(0), action], dim=0)  # [T+1, 23]

        window_fps = self._window_fps(window)
        if self.skip_video_loading:
            return {"action": action}

        # window.caption is the snake task_name (cache canonical_caption); map it to the
        # natural-language source_caption so the prompt matches the LeRobot "official" run.
        raw_caption = require_action_caption(window.caption, dataset_name="BEHAVIOR-1K")
        caption = self._caption_remap.get(str(raw_caption).strip(), raw_caption)
        result: dict[str, Any] = {
            "ai_caption": caption,
            "video": self._build_video(window, int(state.shape[0])),  # [C, T+1, H, W]
            "action": action,
            "conditioning_fps": torch.tensor(window_fps, dtype=torch.float32),
            "mode": resolve_lance_action_mode(self.mode, aux_seed),
            "domain_id": torch.tensor(get_domain_id(self.embodiment_type), dtype=torch.long),
            "viewpoint": self.viewpoint,
        }
        if self.viewpoint == "concat_view":
            result["additional_view_description"] = (
                "The top row is from the robot's head camera. The bottom row contains two "
                "horizontally concatenated wrist-camera views (left, right)."
            )
        return result

    def _build_training_item(
        self,
        global_index: int,
        aux_seed: int,
        source_rows: dict[str, dict[str, Any]],
        window: MaterializedActionWindow,
    ) -> dict[str, Any]:
        """Convert one B1K ACT2 window into the shared trainer contract."""

        if self.action_representation == "native_joint":
            return self._build_native_joint_item(global_index, aux_seed, source_rows, window)
        del global_index, source_rows
        state = self._field(window, "state", 61)  # [T+1,61]

        base_pose_value = window.per_frame_fields.get("base_pose")
        base_poses: np.ndarray | None = None
        base_pose_source: str | None = None
        if base_pose_value is not None:
            base_pose_values = np.asarray(base_pose_value, dtype=np.float32)  # [T+1,16]
            if base_pose_values.shape != (state.shape[0], 16):
                raise ValueError(f"B1K base_pose must have shape [{state.shape[0]},16], got {base_pose_values.shape}.")
            base_poses = base_pose_values.reshape(state.shape[0], 4, 4)  # [T+1,4,4]
            base_pose_source = "behavior1k_rawdata_omnigibson_world_base_pose"
        elif self.require_absolute_base_pose:
            raise InvalidActionSampleError("B1K ACT2 sample has no absolute raw-replay base-pose trajectory.")

        # The window index already subsampled this window to the resolved target
        # rate, so consecutive state rows are 1/window_fps apart -- not
        # 1/source_fps. Base-velocity integration, the idle-frame thresholds and
        # the conditioning signal all describe the returned rows, so they share
        # this one rate.
        window_fps = self._window_fps(window)
        action, action_extras = build_behavior1k_state_fk_action(
            state,
            urdf_path=self.urdf_path,
            fps=window_fps,
            pose_convention=self.pose_convention,
            base_poses=base_poses,
            base_pose_source=base_pose_source,
        )  # [T,29]
        if self.skip_video_loading:
            return {
                "action": action,
                "state": torch.from_numpy(state.copy()),  # [T+1,61]
                **action_extras,
            }

        # Idle-frame captioning is only consumed on the video path below, so skip the
        # FK-derived computation entirely when video loading is disabled.
        idle_frames = compute_framewise_idle_frames(
            action,
            _ACTION_SPEC,
            fps=window_fps,
            pose_convention=self.pose_convention,
        )
        caption = require_action_caption(window.caption, dataset_name="BEHAVIOR-1K")
        result: dict[str, Any] = {
            "ai_caption": caption,
            "video": self._build_video(window, int(state.shape[0])),
            "action": action,
            "state": torch.from_numpy(state.copy()),  # [T+1,61]
            "conditioning_fps": torch.tensor(window_fps, dtype=torch.float32),  # []
            "mode": resolve_lance_action_mode(self.mode, aux_seed),
            "domain_id": torch.tensor(get_domain_id(self.embodiment_type), dtype=torch.long),  # []
            "viewpoint": self.viewpoint,
            **action_extras,
        }
        if idle_frames is not None:
            result["idle_frames"] = idle_frames
        if self.viewpoint == "concat_view":
            result["additional_view_description"] = (
                "The top row is from the robot's head camera. The bottom row contains two "
                "horizontally concatenated wrist-camera views (left, right)."
            )
        return result

    @property
    def action_dim(self) -> int:
        """Return the shared B1K/AgiBot SE(3) action width."""

        return _JOINT_ACTION_DIM if self.action_representation == "native_joint" else _ACTION_DIM

    def get_action_normalizer(self, _sample: dict[str, Any] | None = None) -> None:
        """Return no normalizer until B1K state-derived norm stats are approved."""

        return None

    def get_unified_action_mapping(self) -> UnifiedActionMapping:
        """Map B1K's native ``[head9,right9,right_open,left9,left_open]`` layout."""

        if self.action_representation == "native_joint":
            raise NotImplementedError("Native joint commands require action_schema='native'.")
        return humanoid_pose_mapping()
