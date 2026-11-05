# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cosmos_framework.data.generator.action.action_binary import get_action_binary_spec_name
from cosmos_framework.data.generator.action.lance.abc_gripper_manifest import (
    ABC_GRIPPER_MANIFEST_PATH,
    ABCGripperBinding,
    ABCGripperManifest,
    load_abc_gripper_manifest,
)
from cosmos_framework.data.generator.action.lance.lance_action_dataset import (
    ACTION_NORMALIZERS_DIR,
    DEFAULT_DENSE_CAPTION_PROBABILITY,
    BaseLanceActionDataset,
    MaterializedActionWindow,
    TargetFps,
    require_action_caption,
    resolve_lance_action_mode,
)
from cosmos_framework.data.generator.action.utils.action_processing import (
    ActionNormalizationMethod,
    ActionNormalizer,
    load_action_normalizer,
)
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.utils.image_augmentation import apply_synchronized_image_augmentation
from cosmos_framework.data.generator.action.utils.pose_utils import PoseConvention, compute_framewise_idle_frames
from cosmos_framework.data.generator.action.utils.unified_action_schema import (
    UnifiedActionMapping,
    dual_arm_pose_mapping,
)
from cosmos_framework.data.generator.action.yam_dataset_config import (
    ABC_CAMERA_FEATURES,
    MOLMOACT2_CAMERA_FEATURES,
    XDOF_CAMERA_FEATURES_480,
    YAM_TO_OPENCV,
)
from cosmos_framework.data.generator.action.yam_semantics import (
    YAMKinematics,
    build_yam_action,
    build_yam_action_spec,
    build_yam_image_augmentor,
    compose_yam_concat_view,
    yam_normalizer_filename,
)
from cosmos_framework.data.generator.lance.lance_action_source import LanceActionDatasetSpec

_CONCAT_VIEW = "concat_view"


_CONCAT_VIEW_DESCRIPTION = (
    "The top row shows the overhead/top camera. The bottom row contains the left camera "
    "on the left and the right camera on the right."
)


_YAM_ARM_CAMERAS: tuple[str, ...] = ("top", "left", "right")


class LanceYAMDataset(BaseLanceActionDataset):
    """Reconstruct i2rt YAM samples from ACT2 joints via MuJoCo FK."""

    # XDoF and ABC crank grippers use this physical fully-open width. MolmoAct2
    # and the other classified ABC subsets override it with 0.096 m.
    fully_open_gripper_width_m: float = 0.071

    def __init__(
        self,
        lance_dataset_spec: LanceActionDatasetSpec,
        creds_path: str | Path,
        chunk_length: int = 16,
        fps: TargetFps = 15,
        mode: str = "joint",
        split: str = "train",
        val_ratio: float = 0.01,
        seed: int = 42,
        action_space: str = "ee_pose",
        use_state: bool = False,
        pose_convention: PoseConvention = "backward_anchored",
        viewpoint: str = _CONCAT_VIEW,
        use_image_augmentation: bool = False,
        action_normalization: ActionNormalizationMethod | None = None,
        apply_forward_clamp: bool = False,
        normalizer_filename: str | None = None,
        action_stats_path: str | Path | None = None,
        skip_video_loading: bool = False,
        view_read_workers: int = 3,
        source_dataset_name: str | None = None,
        retention_quality_score_threshold: float | None = 0.0,
        abc_gripper_manifest_path: str | Path | None = None,
        dense_caption_probability: float = DEFAULT_DENSE_CAPTION_PROBABILITY,
    ) -> None:
        resolved_source_dataset_name = self._resolve_source_dataset_name(lance_dataset_spec, source_dataset_name)
        # A combined spec resolves to the spec name, which is not an ACT2 dataset key,
        # so read the family off the populations. Requiring exactly one also guards the
        # cache: mixing ACT2 widths would silently re-stride every frame.
        populations = lance_dataset_spec.source_datasets or (resolved_source_dataset_name,)
        source_families = sorted({get_action_binary_spec_name(name) for name in populations})
        if len(source_families) != 1:
            raise ValueError(
                f"Lance Action spec {lance_dataset_spec.name!r} mixes ACT2 source families "
                f"{source_families}; one spec must hold exactly one fixed-stride layout."
            )
        source_family = source_families[0]

        if source_family == "abc_130k":
            dataset_label = "ABC-130k YAM"
            embodiment_type = "abc_yam"
            arm_joint_slices = (slice(0, 6), slice(7, 13))
            gripper_slices = (slice(6, 7), slice(13, 14))
            camera_views = dict(ABC_CAMERA_FEATURES)
        elif source_family == "molmoact2_bimanual_yam":
            dataset_label = "MolmoAct2 YAM"
            embodiment_type = "molmoact2_yam"
            arm_joint_slices = (slice(0, 6), slice(7, 13))
            gripper_slices = (slice(6, 7), slice(13, 14))
            camera_views = dict(MOLMOACT2_CAMERA_FEATURES)
        elif source_family in ("xdof_yam_v5", "xdof_yam_v5_annotated"):
            # The annotated drop is 64 wide where v5 is 46, but dims 0..45 carry the
            # same schema, so one slice set serves both. XDOF also stores a 320p stream
            # per camera; train on the 480p one.
            dataset_label = "XDOF YAM (annotated)" if source_family.endswith("_annotated") else "XDOF YAM"
            embodiment_type = "xdof_yam"
            arm_joint_slices = (slice(34, 40), slice(40, 46))
            gripper_slices = (slice(32, 33), slice(33, 34))
            camera_views = dict(XDOF_CAMERA_FEATURES_480)
        else:
            raise ValueError(f"Unsupported YAM source_dataset_name={resolved_source_dataset_name!r}.")

        if abc_gripper_manifest_path is not None and source_family != "abc_130k":
            raise ValueError("abc_gripper_manifest_path is only valid for the ABC-130k source family.")
        self._source_family: str = source_family
        self._abc_gripper_manifest: ABCGripperManifest | None = (
            load_abc_gripper_manifest(
                ABC_GRIPPER_MANIFEST_PATH if abc_gripper_manifest_path is None else abc_gripper_manifest_path
            )
            if source_family == "abc_130k"
            else None
        )
        self.fully_open_gripper_width_m = 0.096 if source_family == "molmoact2_bimanual_yam" else 0.071

        if action_space not in ("ee_pose", "joint_pose"):
            raise ValueError(f"Unsupported action_space={action_space!r}")
        # build_yam_action enforces these too, but only once a worker materializes a
        # sample. Repeat them here so a bad config dies at construction rather than
        # deep inside a dataloader worker mid-training.
        if action_space == "joint_pose" and pose_convention != "absolute":
            raise ValueError(f"joint_pose currently supports only pose_convention='absolute', got {pose_convention!r}")
        if viewpoint != _CONCAT_VIEW:
            raise ValueError(f"LanceYAMDataset currently supports viewpoint='concat_view', got {viewpoint!r}")
        self._dataset_label: str = dataset_label
        self._embodiment_type: str = embodiment_type
        self._arm_joint_slices: tuple[slice, slice] = arm_joint_slices
        self._gripper_slices: tuple[slice, slice] = gripper_slices
        self._camera_views: dict[str, str] = camera_views
        self.mode: str = mode
        self.viewpoint: str = viewpoint
        self.action_space: str = action_space
        self.use_state: bool = use_state
        self.use_image_augmentation: bool = use_image_augmentation
        self._image_augmentor: Callable[[torch.Tensor], torch.Tensor] | None = None
        self.pose_convention: PoseConvention = pose_convention
        self.action_spec = build_yam_action_spec(action_space)
        # EE-site rotation -> OpenCV (R_opencv = R_native @ _to_opencv), applied to
        # every FK pose in ``_build_action``. Held on the instance and passed explicitly
        # rather than left to build_yam_action's default, mirroring YamActionDataset:
        # both paths must show the same matrix at the call site, and a test that
        # substitutes one has somewhere to substitute it.
        self._to_opencv: np.ndarray = YAM_TO_OPENCV
        self._kinematics: YAMKinematics | None = None
        self._kinematics_pid: int | None = None
        super().__init__(
            lance_dataset_spec,
            creds_path,
            view_names=tuple(camera_views[name] for name in _YAM_ARM_CAMERAS),
            chunk_length=chunk_length,
            fps=fps,
            split=split,
            val_ratio=val_ratio,
            seed=seed,
            skip_video_loading=skip_video_loading,
            view_read_workers=view_read_workers,
            source_dataset_name=resolved_source_dataset_name,
            retention_quality_score_threshold=retention_quality_score_threshold,
            dense_caption_probability=dense_caption_probability,
        )
        # Shared with the LeRobot path so a rename cannot leave the two loaders
        # normalizing with different stats.
        default_normalizer = yam_normalizer_filename(action_space, str(self.pose_convention))
        stats_path = (
            Path(action_stats_path)
            if action_stats_path is not None
            else ACTION_NORMALIZERS_DIR / (normalizer_filename or default_normalizer)
        )
        self.action_normalizer: ActionNormalizer | None = load_action_normalizer(
            action_normalization,
            stats_path=stats_path,
            apply_forward_clamp=apply_forward_clamp,
        )

    def _abc_gripper_binding(self, source_rows: dict[str, dict[str, Any]]) -> ABCGripperBinding | None:
        """Resolve and cross-check one ABC episode binding across all views."""
        manifest = self._abc_gripper_manifest
        if manifest is None:
            return None
        action_spans = self.lance_dataset_spec.action_spans
        episode_keys = {
            (str(source_row[action_spans.source_dataset]), int(source_row[action_spans.episode_index]))
            for source_row in source_rows.values()
        }
        if len(episode_keys) != 1:
            raise ValueError(f"ABC source views disagree on physical episode identity: {sorted(episode_keys)!r}.")
        source_dataset, episode_index = next(iter(episode_keys))
        binding = manifest.binding(source_dataset, episode_index)
        if binding is None:
            raise ValueError(
                f"ABC episode {(source_dataset, episode_index)!r} is not classified by the gripper manifest."
            )
        return binding

    @staticmethod
    def _attach_abc_gripper_binding(
        result: dict[str, Any],
        binding: ABCGripperBinding | None,
    ) -> dict[str, Any]:
        """Attach per-episode ABC gripper semantics to a trainer-facing sample."""
        if binding is not None:
            result["gripper_subset"] = binding.subset
            result["gripper_type"] = binding.gripper_type
            result["fully_open_gripper_width_m"] = binding.fully_open_width_m
        return result

    # -- process-bound MuJoCo state ------------------------------------------
    def _get_kinematics(self) -> YAMKinematics:
        """Lazily create MuJoCo state in the worker that materializes the sample."""
        current_pid = os.getpid()
        if self._kinematics is None or self._kinematics_pid != current_pid:
            # Never reuse MuJoCo model/data handles inherited from a parent process.
            self._kinematics = None
            self._kinematics_pid = None
            self._kinematics = YAMKinematics()
            self._kinematics_pid = current_pid
        kinematics = self._kinematics
        assert kinematics is not None
        return kinematics

    # -- action construction -------------------------------------------------
    def _build_action(
        self,
        state: np.ndarray,  # [T+1,S]
        raw_action: np.ndarray,  # [T+1,S]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Bind this dataset's facts to the shared YAM action construction.

        A named seam rather than an inline call: the shared semantics guarantee
        the math, but the binding below -- which slices, which convention, which
        chunk length -- is this loader's own, and is what the cross-path
        equivalence test exercises.

        ``use_state`` mirrors the joint-pose posttrain recipe, which prepends the
        frame-0 proprio as a conditioning row -- without it a Lance joint run
        would emit [T,14] where those checkpoints trained on [T+1,14].
        """
        return build_yam_action(
            state,
            raw_action,
            arm_joint_slices=self._arm_joint_slices,
            gripper_slices=self._gripper_slices,
            chunk_length=self.chunk_length,
            action_space=self.action_space,
            pose_convention=self.pose_convention,
            ee_site_poses=self._get_kinematics().ee_site_poses,
            to_opencv=self._to_opencv,
            use_state=self.use_state,
            dataset_name=self._dataset_label,
        )

    # -- trainer contract ----------------------------------------------------
    def _build_training_item(
        self,
        global_index: int,
        aux_seed: int,
        source_rows: dict[str, dict[str, Any]],
        window: MaterializedActionWindow,
    ) -> dict[str, Any]:
        """Convert one materialized window into the shared YAM trainer contract."""
        del global_index
        gripper_binding = self._abc_gripper_binding(source_rows)
        state_values = window.per_frame_fields.get("state")
        action_values = window.per_frame_fields.get("action")
        if state_values is None or action_values is None:
            raise ValueError("Action binary is missing required fields 'state' and 'action'.")
        state = np.asarray(state_values, dtype=np.float32)  # [T+1,S]
        raw_action = np.asarray(action_values, dtype=np.float32)  # [T+1,S]
        expected_frames = state.shape[0] if state.ndim else 0
        # Both arrays are checked, not just state: this path always requests one T+1
        # window, but ``build_yam_action`` deliberately tail-slices a T or T+1 action
        # for LeRobot compatibility, so a short/ragged ACT2 read would be absorbed
        # silently rather than raising. Matches the boundary check LanceDROIDDataset does.
        if state.ndim != 2 or expected_frames < 2 or raw_action.ndim != 2 or raw_action.shape[0] != expected_frames:
            raise ValueError(
                "Expected 2D YAM state/action arrays with T+1>=2 matching frames; "
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
            return self._attach_abc_gripper_binding(result, gripper_binding)

        top = self._video_view_tchw(window, self._camera_views["top"], expected_frames)  # [T+1,C,H,W]
        left = self._video_view_tchw(window, self._camera_views["left"], expected_frames)
        right = self._video_view_tchw(window, self._camera_views["right"], expected_frames)
        if self.use_image_augmentation:
            if self._image_augmentor is None:
                _, _, h, w = top.shape
                self._image_augmentor = build_yam_image_augmentor(h, w)
            # Seed from the sampler's aux_seed, never worker-global RNG: this rank's
            # sample stream must replay identically after a checkpoint resume.
            # torchvision v2 transforms take no Generator, so fork the global RNG;
            # devices=[] keeps CUDA out of it since workers augment on CPU.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(aux_seed)
                top, left, right = apply_synchronized_image_augmentation([top, left, right], self._image_augmentor)
        video = compose_yam_concat_view(top, left, right).permute(1, 0, 2, 3).contiguous()  # [C,T+1,H_out,W]
        result = {
            "ai_caption": require_action_caption(window.caption, dataset_name=self._dataset_label),
            "video": video,
            "action": action,
            "conditioning_fps": torch.tensor(self._window_fps(window), dtype=torch.float32),  # []
            "mode": resolve_lance_action_mode(self.mode, aux_seed),
            "domain_id": torch.tensor(get_domain_id(self._embodiment_type), dtype=torch.long),  # []
            "viewpoint": self.viewpoint,
            "additional_view_description": _CONCAT_VIEW_DESCRIPTION,
            **extras,
        }
        if idle_frames is not None:
            result["idle_frames"] = idle_frames
        return self._attach_abc_gripper_binding(result, gripper_binding)

    @property
    def action_dim(self) -> int:
        """Return the unpadded YAM action width."""
        return 14 if self.action_space == "joint_pose" else 20

    def get_action_normalizer(
        self,
        _sample: dict[str, Any] | None = None,
    ) -> ActionNormalizer | None:
        """Return the configured action normalizer for transform-time preprocessing."""
        return self.action_normalizer

    def get_unified_action_mapping(self) -> UnifiedActionMapping:
        """Map the dual-arm EE action into ``unified_v1``."""
        if self.action_space != "ee_pose":
            raise ValueError("unified_v1 is only defined for YAM action_space='ee_pose'")
        return dual_arm_pose_mapping(order="left_right")
