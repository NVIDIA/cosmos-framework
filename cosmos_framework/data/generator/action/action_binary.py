# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Native local-bundle Bimanual semantics; no remote index access."""

from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np

MAGIC: bytes = b"ACT2"


VERSION: int = 2


DEFAULT_ACT2_HEADER_SIZE_BYTES: int = 1024


_HEADER_PREFIX_SIZE: int = 8


MAX_ACT2_HEADER_JSON_BYTES: int = 16 * 1024 * 1024


_DTYPE = np.dtype("<f4")


ACTION_BINARY_CONTRACT_FINGERPRINT_KEY: str = "action_binary_contract_fingerprint"


class ActionBinaryDecodeError(ValueError):
    """Raised when one ACT2 payload violates the self-describing binary contract."""


@dataclass(frozen=True)
class ActionFieldSpec:
    """One named array within a frame or clip record."""

    name: str
    shape: tuple[int, ...]

    @property
    def value_count(self) -> int:
        return prod(self.shape)


@dataclass(frozen=True)
class ActionBinarySpec:
    """Reusable fixed-stride action layout.

    ``header_size_bytes`` is part of the on-disk ACT2 contract. Once a dataset's
    sidecars have been written, changing it requires rewriting those sidecars.
    """

    per_frame_fields: tuple[ActionFieldSpec, ...]
    per_clip_fields: tuple[ActionFieldSpec, ...] = ()
    dtype: np.dtype = _DTYPE
    header_size_bytes: int = DEFAULT_ACT2_HEADER_SIZE_BYTES

    def __post_init__(self) -> None:
        if self.header_size_bytes < _HEADER_PREFIX_SIZE:
            raise ValueError(
                f"ACT2 header_size_bytes must be at least {_HEADER_PREFIX_SIZE}, got {self.header_size_bytes}."
            )
        max_header_size = _HEADER_PREFIX_SIZE + MAX_ACT2_HEADER_JSON_BYTES
        if self.header_size_bytes > max_header_size:
            raise ValueError(f"ACT2 header_size_bytes must be at most {max_header_size}, got {self.header_size_bytes}.")

    @property
    def per_frame_value_count(self) -> int:
        return sum(field.value_count for field in self.per_frame_fields)

    @property
    def frame_record_size_bytes(self) -> int:
        return self.per_frame_value_count * self.dtype.itemsize

    @property
    def per_clip_value_count(self) -> int:
        return sum(field.value_count for field in self.per_clip_fields)

    @property
    def per_clip_size_bytes(self) -> int:
        return self.per_clip_value_count * self.dtype.itemsize


ACTION_BINARY_SPECS: dict[str, ActionBinarySpec] = {
    "libero": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(7,)),
            ActionFieldSpec(name="state", shape=(8,)),
        ),
    ),
    "libero_90": ActionBinarySpec(
        # libero_90 additionally carries decomposed proprioception under
        # observation.states.*. Keep the shared flat EE/gripper state and retain
        # the joint vector separately; sibling LIBERO roots do not expose it.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(7,)),
            ActionFieldSpec(name="state", shape=(8,)),
            ActionFieldSpec(name="state_joint_position", shape=(7,)),
        ),
    ),
    "embodiment_a": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="hand_left_cam", shape=(63,)),
            ActionFieldSpec(name="hand_right_cam", shape=(63,)),
            ActionFieldSpec(name="hand_left_cam_rotation", shape=(84,)),
            ActionFieldSpec(name="hand_right_cam_rotation", shape=(84,)),
            ActionFieldSpec(name="camera_position", shape=(3,)),
            ActionFieldSpec(name="camera_rotation", shape=(4,)),
        ),
        per_clip_fields=(ActionFieldSpec(name="intrinsics", shape=(8,)),),
    ),
    "av": ActionBinarySpec(
        # Existing AV clips carry one 81-frame ego trajectory and one camera-frame
        # trajectory. Routes and the fixed ego-to-camera transform are clip-level.
        per_frame_fields=(
            ActionFieldSpec(name="trajectory_ego", shape=(7,)),
            ActionFieldSpec(name="trajectory_camera", shape=(7,)),
        ),
        per_clip_fields=(
            ActionFieldSpec(name="route_ego", shape=(20, 3)),
            ActionFieldSpec(name="route_camera", shape=(20, 3)),
            ActionFieldSpec(name="t_ego_camera", shape=(4, 4)),
        ),
    ),
    # Absolute camera-to-world pose per video frame; rotation is quaternion xyzw.
    "camera_pose": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="camera_position", shape=(3,)),
            ActionFieldSpec(name="camera_rotation", shape=(4,)),
        ),
    ),
    # VITRA ego4d, from .../cosmos3_action_datasets/VITRA/processed/ego4d/<res>/.
    # Immutable legacy contract used by the existing production ACT2 bins. It was
    # registered from each resolution's meta/info.json feature set, which does not
    # describe the per-episode camera_intrinsics metadata column. Keep this layout
    # unchanged so those bins retain their zero-byte per-clip tail.
    #
    # This REPLACES the earlier egoexo4d spec, which was a different drop and a
    # different layout:
    #   * rotations here are 84 = 21 joints x 4 QUATERNIONS -- embodiment_a's convention --
    #     where egoexo4d shipped 189 = 21 x 9 flattened 3x3 matrices;
    #   * this drop HAS camera_position (3) and camera_rotation (4); egoexo4d had
    #     no camera columns at all.
    # Encoding this data with the old spec would have mis-strided every frame.
    #
    # Still not the embodiment_a spec: VITRA also carries world-frame hands. World-frame
    # poses are kept because the cut is where they would otherwise be lost.
    "vitra_ego4d": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="hand_left_cam", shape=(63,)),
            ActionFieldSpec(name="hand_right_cam", shape=(63,)),
            ActionFieldSpec(name="hand_left_cam_rotation", shape=(84,)),
            ActionFieldSpec(name="hand_right_cam_rotation", shape=(84,)),
            ActionFieldSpec(name="hand_left_world", shape=(63,)),
            ActionFieldSpec(name="hand_right_world", shape=(63,)),
            ActionFieldSpec(name="hand_left_world_rotation", shape=(84,)),
            ActionFieldSpec(name="hand_right_world_rotation", shape=(84,)),
            ActionFieldSpec(name="camera_position", shape=(3,)),
            ActionFieldSpec(name="camera_rotation", shape=(4,)),
        ),
    ),
    # Additive corrected contract for new VITRA bins. Episode metadata carries
    # camera_intrinsics as 8 floats; the legacy spec above remains immutable.
    "vitra_ego4d_v2": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="hand_left_cam", shape=(63,)),
            ActionFieldSpec(name="hand_right_cam", shape=(63,)),
            ActionFieldSpec(name="hand_left_cam_rotation", shape=(84,)),
            ActionFieldSpec(name="hand_right_cam_rotation", shape=(84,)),
            ActionFieldSpec(name="hand_left_world", shape=(63,)),
            ActionFieldSpec(name="hand_right_world", shape=(63,)),
            ActionFieldSpec(name="hand_left_world_rotation", shape=(84,)),
            ActionFieldSpec(name="hand_right_world_rotation", shape=(84,)),
            ActionFieldSpec(name="camera_position", shape=(3,)),
            ActionFieldSpec(name="camera_rotation", shape=(4,)),
        ),
        per_clip_fields=(ActionFieldSpec(name="intrinsics", shape=(8,)),),
    ),
    "droid_lerobot": ActionBinarySpec(
        # Legacy flat DROID LeRobot drops, e.g. droid_lerobot_20260115_no_noops,
        # expose only action[7] and observation.state[7]. Keep this family spec
        # narrow so old flat artifacts and range readers are not re-strided.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(7,)),
            ActionFieldSpec(name="state", shape=(7,)),
        ),
    ),
    "droid_lerobot_decomposed": ActionBinarySpec(
        # DROID+ 20260412 success/failure exposes decomposed raw LeRobot columns.
        # The legacy action/state fields are cartesian xyz/euler + scalar gripper.
        # Extra dynamics are retained as grouped fields so the default 1024-byte
        # ACT2 header still fits:
        #   action_velocity = action.cartesian_velocity + action.gripper_velocity
        #   *_joint_* map directly from DROID joint columns
        #   state_*torque* map directly from DROID measured/computed torque columns
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(7,)),
            ActionFieldSpec(name="state", shape=(7,)),
            ActionFieldSpec(name="action_velocity", shape=(7,)),
            ActionFieldSpec(name="action_joint_position", shape=(7,)),
            ActionFieldSpec(name="action_joint_velocity", shape=(7,)),
            ActionFieldSpec(name="state_joint_position", shape=(7,)),
            ActionFieldSpec(name="state_joint_velocity", shape=(7,)),
            ActionFieldSpec(name="state_joint_torques_computed", shape=(7,)),
            ActionFieldSpec(name="state_motor_torques_measured", shape=(7,)),
        ),
    ),
    "robomind_franka": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(8,)),
            ActionFieldSpec(name="state_joint_position", shape=(8,)),
            ActionFieldSpec(name="state_end_effector", shape=(6,)),
        ),
    ),
    "robomind_franka_dual": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(16,)),
            ActionFieldSpec(name="state_joint_position", shape=(16,)),
            ActionFieldSpec(name="state_end_effector", shape=(12,)),
        ),
    ),
    "robomind_ur": ActionBinarySpec(
        # Raw RoboMIND UR joint targets: 6 arm joints + scalar gripper.
        # Do NOT encode observation.states.end_effector for UR: the dataset loader
        # derives reliable EE poses with FK because the raw EE column is frozen or
        # incorrect in most UR episodes.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(7,)),
            ActionFieldSpec(name="state_joint_position", shape=(7,)),
        ),
    ),
    # ---- Flat action/state datasets -------------------------------------------------
    # Every spec below was read off the dataset's own first data parquet (footer only,
    # via the remote schema probe), never inferred from a family name. Where two roots
    # share a family prefix but differ in width they get separate specs, because a
    # mismatched stride does not fail loudly -- it silently reinterprets bytes.
    "bridge": ActionBinarySpec(
        # bridge_raw / bridge_raw_clean, and the openx container which descends to
        # bridge_raw. xyz + rpy + gripper on both sides.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(7,)),
            ActionFieldSpec(name="state", shape=(7,)),
        ),
    ),
    "bridge_orig": ActionBinarySpec(
        # bridge_orig_lerobot_20260111 is NOT bridge_raw: state is 8, not 7.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(7,)),
            ActionFieldSpec(name="state", shape=(8,)),
        ),
    ),
    "fractal": ActionBinarySpec(
        # Fractal/RT-1 stores xyz + rpy + gripper actions and xyz + xyzw
        # quaternion + gripper state. The trainer derives pose actions from
        # state and keeps only the raw scalar gripper action.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(7,)),
            ActionFieldSpec(name="state", shape=(8,)),
        ),
    ),
    "abc_130k": ActionBinarySpec(
        # ABC-130k, roots {train,val} x {realsense,zedx}; all four carry 14/14.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(14,)),
            ActionFieldSpec(name="state", shape=(14,)),
        ),
    ),
    "molmoact2_bimanual_yam": ActionBinarySpec(
        # MolmoAct2-BimanualYAM-Dataset carries flat action/state vectors, both
        # 14-wide, at the dataset root.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(14,)),
            ActionFieldSpec(name="state", shape=(14,)),
        ),
    ),
    "molmoact2_linearbot_yam": ActionBinarySpec(
        # Five MolmoAct2 EvalRollout LinearBot roots add base.y to both the
        # command and state vectors. Keep their 15/15 physical layout distinct
        # from the 55 reviewed 14/14 bimanual roots.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(15,)),
            ActionFieldSpec(name="state", shape=(15,)),
        ),
    ),
    "robotwin": ActionBinarySpec(
        # RoboTwin 2.0 aloha_agilex (robotwin_20260809), LeRobot v3.0. Verified
        # against the root's own data/chunk-000/file-000.parquet: ``action`` and
        # ``observation.state`` are both list<float> of width 14, sharing one
        # layout -- [left joints(6), left gripper, right joints(6), right gripper]
        # -- per meta/info.json feature names.
        #
        # Raw joints only, deliberately. The 20D EE action
        # ([pos(3), rot6d(6), grip(1)] x 2) is MuJoCo FK on the ARX5 arm to
        # ``link6``, and that frame chain has already been revised three times
        # (see the ee_pose frame-chain fixes in git history). Baking FK output
        # into the immutable ACT2 payload would force a full re-cut of 27,500
        # episodes on the next convention change, so FK stays at train time in
        # ``robotwin_semantics.py`` and one cut serves both joint_pose and
        # ee_pose.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(14,)),
            ActionFieldSpec(name="state", shape=(14,)),
        ),
    ),
    "x_humanoid": ActionBinarySpec(
        # X-Humanoid Agilex/Ark mobile roots are 14/14. Other X-Humanoid
        # embodiments are different physical layouts and get their own specs
        # below; do not treat the family name as one binary contract.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(14,)),
            ActionFieldSpec(name="state", shape=(14,)),
        ),
    ),
    "x_humanoid_a16_s16": ActionBinarySpec(
        # X-Humanoid Franka-sim, Tianyi, Tianyi-mobile, and most non-sim
        # Tienkung task roots carry flat 16/16 action/state vectors.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(16,)),
            ActionFieldSpec(name="state", shape=(16,)),
        ),
    ),
    "x_humanoid_tienkung": ActionBinarySpec(
        # X-Humanoid RoboMIND2.0-Tienkung release_door_handle carries flat
        # 26/26 action/state vectors; other non-sim Tienkung task roots probed
        # in this batch are 16/16 and use x_humanoid_a16_s16.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(26,)),
            ActionFieldSpec(name="state", shape=(26,)),
        ),
    ),
    "x_humanoid_a18_s18": ActionBinarySpec(
        # X-Humanoid RoboMIND2.0-Franka-Part-{1,3} carry flat 18/18
        # action/state vectors, not the 14/14 mobile layout.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(18,)),
            ActionFieldSpec(name="state", shape=(18,)),
        ),
    ),
    "x_humanoid_tienkung_sim": ActionBinarySpec(
        # X-Humanoid RoboMIND2.0-Tienkung-sim is asymmetric: action 26,
        # observation.state 38.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(26,)),
            ActionFieldSpec(name="state", shape=(38,)),
        ),
    ),
    "x_humanoid_ur5_dex": ActionBinarySpec(
        # X-Humanoid RoboMIND2.0-UR5-Dex carries flat 36/36 action/state vectors.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(36,)),
            ActionFieldSpec(name="state", shape=(36,)),
        ),
    ),
    "robodojo": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(14,)),
            ActionFieldSpec(name="state", shape=(14,)),
        ),
    ),
    "xdof_yam_v5": ActionBinarySpec(
        # XDOF_YAM_v5 and the 20260625 re-drop; both 46/46.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(46,)),
            ActionFieldSpec(name="state", shape=(46,)),
        ),
    ),
    "xdof_handpose_intrinsics16": ActionBinarySpec(
        # h7m4 is the source drop label for this XDOF hand-pose delivery. It
        # carries hand/camera pose fields instead of flat 46-wide action/state,
        # plus a 16-float per-episode camera_intrinsics metadata vector that is
        # preserved opaquely here. Keep this separate so existing 46/46 XDOF
        # data is unchanged.
        per_frame_fields=(
            ActionFieldSpec(name="hand_left_cam", shape=(63,)),
            ActionFieldSpec(name="hand_right_cam", shape=(63,)),
            ActionFieldSpec(name="hand_left_cam_rotation", shape=(84,)),
            ActionFieldSpec(name="hand_right_cam_rotation", shape=(84,)),
            ActionFieldSpec(name="camera_position", shape=(3,)),
            ActionFieldSpec(name="camera_rotation", shape=(4,)),
            ActionFieldSpec(name="hand_mask", shape=(1,)),
        ),
        per_clip_fields=(ActionFieldSpec(name="intrinsics", shape=(16,)),),
    ),
    "xdof_handpose_intrinsics8": ActionBinarySpec(
        # XDOF ITW carries the same hand/camera pose fields as the h7m4 drop,
        # but camera_intrinsics is the embodiment_a-style 8-float clip field
        # [fx, fy, cx, cy, k1, k2, p1, p2].
        per_frame_fields=(
            ActionFieldSpec(name="hand_left_cam", shape=(63,)),
            ActionFieldSpec(name="hand_right_cam", shape=(63,)),
            ActionFieldSpec(name="hand_left_cam_rotation", shape=(84,)),
            ActionFieldSpec(name="hand_right_cam_rotation", shape=(84,)),
            ActionFieldSpec(name="camera_position", shape=(3,)),
            ActionFieldSpec(name="camera_rotation", shape=(4,)),
            ActionFieldSpec(name="hand_mask", shape=(1,)),
        ),
        per_clip_fields=(ActionFieldSpec(name="intrinsics", shape=(8,)),),
    ),
    "xdof_yam_v5_annotated": ActionBinarySpec(
        # The annotated 20260429 drop is 64/64, not 46/46 -- a separate spec on purpose.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(64,)),
            ActionFieldSpec(name="state", shape=(64,)),
        ),
    ),
    "pusht": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(2,)),
            ActionFieldSpec(name="state", shape=(2,)),
        ),
    ),
    "robocasa": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(12,)),
            ActionFieldSpec(name="state", shape=(16,)),
        ),
    ),
    "human_and_robot": ActionBinarySpec(
        # HumanAndRobot_lerobot exposes both a 8-wide observation.state and a separate
        # 7-wide observation.state.eef; both are kept, since the cut is where anything
        # not encoded is lost for good.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(7,)),
            ActionFieldSpec(name="state", shape=(8,)),
            ActionFieldSpec(name="state_end_effector", shape=(7,)),
        ),
    ),
    "umi_lerobot": ActionBinarySpec(
        # UMI has no action/state columns at all: it carries a scalar gripper width
        # and a 7-vector camera trajectory (xyz + wxyz quaternion).
        per_frame_fields=(
            ActionFieldSpec(name="right_gripper_width_m", shape=(1,)),
            ActionFieldSpec(name="right_main_camera_trajectory_xyz_wxyz", shape=(7,)),
        ),
    ),
    "fastumi_dual_arm": ActionBinarySpec(
        # fastumi_100k_dual_arm has one scalar gripper width and one 7-vector
        # camera trajectory per arm. All 19 task roots probed with this layout.
        per_frame_fields=(
            ActionFieldSpec(name="left_gripper_width_m", shape=(1,)),
            ActionFieldSpec(name="left_main_camera_trajectory_xyz_wxyz", shape=(7,)),
            ActionFieldSpec(name="right_gripper_width_m", shape=(1,)),
            ActionFieldSpec(name="right_main_camera_trajectory_xyz_wxyz", shape=(7,)),
        ),
    ),
    "hf_so100_single_arm": ActionBinarySpec(
        # so_family_qualified_20260810 / so100-single-arm-midtrain.
        # Representative shard probe: action len 6, observation.state len 6.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(6,)),
            ActionFieldSpec(name="state", shape=(6,)),
        ),
    ),
    "hf_so101_single_arm": ActionBinarySpec(
        # so_family_qualified_20260810 / so101-single-arm-midtrain.
        # Representative shard probe: action len 6, observation.state len 6.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(6,)),
            ActionFieldSpec(name="state", shape=(6,)),
        ),
    ),
    "hf_so101_bimanual": ActionBinarySpec(
        # so_family_qualified_20260810 / so101-bimanual-midtrain.
        # Representative shard probe: action len 12, observation.state len 12.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(12,)),
            ActionFieldSpec(name="state", shape=(12,)),
        ),
    ),
    "geniesim3": ActionBinarySpec(
        # GenieSim3.0_filtered_20260810 is nested task/embodiment/split roots.
        # Representative full split probe: action len 40, observation.state len 186.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(40,)),
            ActionFieldSpec(name="state", shape=(186,)),
        ),
    ),
    "maniparena_sim": ActionBinarySpec(
        # maniparena_dataset_20260727/sim tasks are flat 28/28 action/state.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(28,)),
            ActionFieldSpec(name="state", shape=(28,)),
        ),
    ),
    "maniparena_real": ActionBinarySpec(
        # maniparena_dataset_20260727/real/execution_reasoning tasks are flat
        # 56/56 action/state. The real parent also contains mobile_manipulation
        # 62/62 tasks, so the parent must not be planned as one source_dataset.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(56,)),
            ActionFieldSpec(name="state", shape=(56,)),
        ),
    ),
    "maniparena_real_mobile": ActionBinarySpec(
        # maniparena_dataset_20260727/real/mobile_manipulation tasks are flat
        # 62/62 action/state.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(62,)),
            ActionFieldSpec(name="state", shape=(62,)),
        ),
    ),
    # ---- BEHAVIOR ------------------------------------------------------------------
    # The legacy B1K demo and challenge layouts are kept separate because their
    # state widths differ. The production subtask layout is separate from the
    # legacy demo layout because it adds the raw-replay base-pose field.
    "behavior_1k_demos": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(23,)),
            ActionFieldSpec(name="state", shape=(61,)),
        ),
    ),
    "behavior_1k_demos_subtask": ActionBinarySpec(
        # The production subtask population keeps the native command stream and
        # measured proprioception, then carries the exact absolute base pose
        # extracted from raw replay. No SE(3) action is stored: the loader
        # derives the 29-D FK action on the fly from measured state and this
        # pose trajectory, which is why this layout is named for the population
        # rather than for the pose representation it feeds.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(23,)),
            ActionFieldSpec(name="state", shape=(61,)),
            ActionFieldSpec(name="base_pose", shape=(16,)),
        ),
    ),
    "behavior_challenge": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(23,)),
            ActionFieldSpec(name="state", shape=(256,)),
        ),
    ),
    # ---- AgiBot --------------------------------------------------------------------
    # "AgiBot" is three unrelated layouts, not one family. Alpha and Beta share a
    # 16-column decomposed schema; GEAR and 2026 each collapse to a single flat
    # action/state pair of a different width again.
    "agibot_world": ActionBinarySpec(
        # AgiBotWorld-Alpha / -Beta. The source splits these into 16 columns
        # (actions.* and observation.states.*), but 16 fields do not fit in the
        # default 1024-byte ACT2 header: 16 entries costs ~1280, ~1055 even with
        # names cut to the bone. Group joints by purpose instead of requiring this
        # existing dataset to use a larger on-disk header.
        #
        # Packing order is the contract -- decoding depends on it, so it is spelled out
        # rather than left to the field name:
        #   action_joint        actions.joint.position                            14
        #   action_end_effector actions.{effector.position 2,
        #                        end.position 2x3, end.orientation 2x4,
        #                        head.position 2, waist.position 2,
        #                        robot.velocity 2}                                22
        #   state_joint         observation.states.joint.position                 14
        #   state_joint_current observation.states.joint.current_value            14
        #   state_end_effector  observation.states.{effector.position 2,
        #                        end.position 2x3, end.orientation 2x4,
        #                        head.position 2, waist.position 2}              20
        #   state_robot         observation.states.robot.{position 3, orientation 4} 7
        per_frame_fields=(
            ActionFieldSpec(name="action_joint", shape=(14,)),
            ActionFieldSpec(name="action_end_effector", shape=(22,)),
            ActionFieldSpec(name="state_joint", shape=(14,)),
            ActionFieldSpec(name="state_joint_current", shape=(14,)),
            ActionFieldSpec(name="state_end_effector", shape=(20,)),
            ActionFieldSpec(name="state_robot", shape=(7,)),
        ),
    ),
    "embodiment_c": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(22,)),
            ActionFieldSpec(name="state", shape=(32,)),
        ),
    ),
    "embodiment_c_dexhand": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(32,)),
            ActionFieldSpec(name="state", shape=(42,)),
        ),
    ),
    "embodiment_c_ext": ActionBinarySpec(
        # Embodiment C custom-ext uses the same flat column names as standard GEAR,
        # but the source layout is wider: action=[36], observation.state=[94].
        # Keep it separate so the custom root can be planned without corrupting
        # the standard 22/32 GEAR clips.
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(36,)),
            ActionFieldSpec(name="state", shape=(94,)),
        ),
    ),
    # AgiBotWorld2026_20260727 is a mixed parent: the planner found six flat
    # action/state layouts under it, sometimes even mixed within one task dir.
    # Keep layout widths in the spec names and plan homogeneous shard roots with
    # canonical source_dataset names such as
    # agibot_2026_a40_s169__AgiBotWorld2026_20260727_ImitationLearning_...
    "agibot_2026_a40_s169": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(40,)),
            ActionFieldSpec(name="state", shape=(169,)),
        ),
    ),
    "agibot_2026_a40_s162": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(40,)),
            ActionFieldSpec(name="state", shape=(162,)),
        ),
    ),
    "agibot_2026_a44_s169": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(44,)),
            ActionFieldSpec(name="state", shape=(169,)),
        ),
    ),
    "agibot_2026_a44_s162": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(44,)),
            ActionFieldSpec(name="state", shape=(162,)),
        ),
    ),
    "agibot_2026_a44_s193": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(44,)),
            ActionFieldSpec(name="state", shape=(193,)),
        ),
    ),
    "agibot_2026_a44_s216": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="action", shape=(44,)),
            ActionFieldSpec(name="state", shape=(216,)),
        ),
    ),
    # ---- WebHumanAction ------------------------------------------------------------
    # embodiment_a-shaped hands (63-dim positions, 84-dim quaternion rotations -- embodiment_a's
    # convention, not VITRA's 189-dim matrices) plus a head and a full soma body.
    # Immutable legacy contract used by the existing production ACT2 bins. Keep its
    # zero-byte per-clip tail unchanged; the additive corrected contract below
    # retains the source's episode-level camera_intrinsics.
    "web_human_action": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="hand_left_cam", shape=(63,)),
            ActionFieldSpec(name="hand_right_cam", shape=(63,)),
            ActionFieldSpec(name="hand_left_cam_rotation", shape=(84,)),
            ActionFieldSpec(name="hand_right_cam_rotation", shape=(84,)),
            ActionFieldSpec(name="head_cam", shape=(3,)),
            ActionFieldSpec(name="head_cam_rotation", shape=(4,)),
            ActionFieldSpec(name="soma_body_cam", shape=(234,)),
            ActionFieldSpec(name="soma_body_cam_rotation", shape=(312,)),
            ActionFieldSpec(name="camera_position", shape=(3,)),
            ActionFieldSpec(name="camera_rotation", shape=(4,)),
        ),
    ),
    # Additive corrected contract for new WebHumanAction bins. Episode metadata
    # carries camera_intrinsics as 8 floats; the legacy spec above remains immutable.
    "web_human_action_v2": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="hand_left_cam", shape=(63,)),
            ActionFieldSpec(name="hand_right_cam", shape=(63,)),
            ActionFieldSpec(name="hand_left_cam_rotation", shape=(84,)),
            ActionFieldSpec(name="hand_right_cam_rotation", shape=(84,)),
            ActionFieldSpec(name="head_cam", shape=(3,)),
            ActionFieldSpec(name="head_cam_rotation", shape=(4,)),
            ActionFieldSpec(name="soma_body_cam", shape=(234,)),
            ActionFieldSpec(name="soma_body_cam_rotation", shape=(312,)),
            ActionFieldSpec(name="camera_position", shape=(3,)),
            ActionFieldSpec(name="camera_rotation", shape=(4,)),
        ),
        per_clip_fields=(ActionFieldSpec(name="intrinsics", shape=(8,)),),
    ),
    "worldengine_walle": ActionBinarySpec(
        # WALLE_week_01_10k/batch_00/v0 shard_000 data parquet probe (2026-08-24):
        # embodiment_a-shaped camera-frame hands and camera pose, but no per-clip
        # camera_intrinsics and no VITRA world-frame hand poses. The source also
        # carries a bool state_mask[2], encoded into ACT2 as a float32 mask field
        # to preserve the training contract.
        per_frame_fields=(
            ActionFieldSpec(name="hand_left_cam", shape=(63,)),
            ActionFieldSpec(name="hand_right_cam", shape=(63,)),
            ActionFieldSpec(name="hand_left_cam_rotation", shape=(84,)),
            ActionFieldSpec(name="hand_right_cam_rotation", shape=(84,)),
            ActionFieldSpec(name="camera_position", shape=(3,)),
            ActionFieldSpec(name="camera_rotation", shape=(4,)),
            ActionFieldSpec(name="state_mask", shape=(2,)),
        ),
    ),
    # embodiment_b nvidia-20260828 drop: .../lerobot_v30/embodiment_b/nvidia-20260828/, 902
    # per-episode-group LeRobot shard roots. Its 20 per-frame fields encode to a
    # 1556-byte JSON header, over the default 1024-byte budget, so this spec uses
    # a larger header_size_bytes instead of packing fields into wider groups.
    "embodiment_b_20260828": ActionBinarySpec(
        per_frame_fields=(
            ActionFieldSpec(name="state_arm_end_position", shape=(14,)),
            ActionFieldSpec(name="state_effector_position", shape=(2,)),
            ActionFieldSpec(name="state_head_position", shape=(7,)),
            ActionFieldSpec(name="state_head_effort", shape=(7,)),
            ActionFieldSpec(name="state_head_velocity", shape=(7,)),
            ActionFieldSpec(name="state_arm_current_value", shape=(14,)),
            ActionFieldSpec(name="state_arm_position", shape=(14,)),
            ActionFieldSpec(name="state_arm_velocity", shape=(14,)),
            ActionFieldSpec(name="state_arm_effort", shape=(14,)),
            ActionFieldSpec(name="state_waist_position", shape=(7,)),
            ActionFieldSpec(name="state_waist_velocity", shape=(7,)),
            ActionFieldSpec(name="state_waist_effort", shape=(7,)),
            ActionFieldSpec(name="state_base_position", shape=(7,)),
            ActionFieldSpec(name="state_base_position_drift", shape=(7,)),
            ActionFieldSpec(name="action_arm_position", shape=(14,)),
            ActionFieldSpec(name="action_end_position", shape=(14,)),
            ActionFieldSpec(name="action_effector_position", shape=(2,)),
            ActionFieldSpec(name="action_head_position", shape=(7,)),
            ActionFieldSpec(name="action_waist_position", shape=(7,)),
            ActionFieldSpec(name="action_base_velocity", shape=(6,)),
        ),
        header_size_bytes=2048,
    ),
}


ACTION_BINARY_SPEC_BY_DATASET: dict[str, str] = {
    "av": "av",
    "libero_10": "libero",
    "libero_90": "libero_90",
    "libero__libero_90": "libero_90",
    # The other three LIBERO_LeRobot_v3 variants. Verified against their own data
    # parquets (chunk-000/file-000) rather than assumed from the family name:
    # each carries action len 7 and observation.state len 8, identical to
    # libero_10, so they share the "libero" spec.
    "libero_goal": "libero",
    "libero_object": "libero",
    "libero_spatial": "libero",
    "camera_estimation_v2_prod": "camera_pose",
    "camera_estimation_v2_drone_ego_prod": "camera_pose",
    "feb_08_500hr_lerobot_no_bframes": "embodiment_a",
    "feb_15_1500hr_lerobot_no_bframes": "embodiment_a",
    "feb_23_1000hr_lerobot_no_bframes": "embodiment_a",
    "mar_02_1000hr_lerobot": "embodiment_a",
    "mar_09_4000hr_lerobot": "embodiment_a",
    "mar_16_7000hr_lerobot": "embodiment_a",
    "mar_30_9000hr_lerobot": "embodiment_a",
    "apr_03_lerobot": "embodiment_a",
    "apr_06_10000hr_lerobot": "embodiment_a",
    # 46,000 h mono/no-wrist embodiment_a. Verified against shard_00 data + episodes parquet:
    # the six per-frame columns match the embodiment_a spec exactly (84-dim quaternion
    # rotations, camera_position 3, camera_rotation 4) and camera_intrinsics is len 8
    # for the per-clip tail. Single view, observation.images.main.
    #
    # NOTE its root ends in /v0/, so full-ingest inference produces the useless name
    # "v0". Pass --source-dataset explicitly with this key for this drop.
    "20260509_46000h_everyday_mono_no_wrist_lerobot": "embodiment_a",
    "mecka__20260509_46000h_everyday_mono_no_wrist_lerobot": "embodiment_a",
    # VITRA ego4d. Roots are .../VITRA/processed/ego4d/<resolution>/, each a flat
    # single-shard layout, so the folder leaf is the RESOLUTION ("1440x1080") --
    # far too generic to register. Pass --source-dataset
    # vitra_ego4d explicitly; the three resolutions then share one spec and one
    # spans table, and their shard_ids differ naturally (the flat layout names a
    # shard after its own directory), so their clip uuids do not collide.
    #
    # The old "egoexo4d" key is GONE, not renamed: that was a different drop with
    # 189-dim matrix rotations and no camera columns. Leaving it registered would
    # let a stale plan cut real data with the wrong stride.
    "vitra_ego4d": "vitra_ego4d",
    "VITRA_ego4d": "vitra_ego4d",
    "vitra_ego4d_v2": "vitra_ego4d_v2",
    "vitra_ego4d_v2__ego4d": "vitra_ego4d_v2",
    "droid_lerobot": "droid_lerobot",
    "droid_lerobot_20260115_no_noops": "droid_lerobot",
    # DROID+ 20260412 is decomposed and was already cut with 252-byte frames.
    # Register exact aliases so those artifacts decode with the decomposed
    # contract, while arbitrary droid_lerobot__* canonical names still fall back
    # to the legacy flat spec only when they intentionally use that prefix.
    "droid_plus_lerobot_640x360_20260412": "droid_lerobot_decomposed",
    "droid_plus_lerobot_640x360_20260412_success": "droid_lerobot_decomposed",
    "droid_plus_lerobot_640x360_20260412_failure": "droid_lerobot_decomposed",
    "droid_lerobot__droid_plus_lerobot_640x360_20260412_success": "droid_lerobot_decomposed",
    "droid_lerobot__droid_plus_lerobot_640x360_20260412_failure": "droid_lerobot_decomposed",
    "droid_lerobot_decomposed__droid_plus_lerobot_640x360_20260412_success": "droid_lerobot_decomposed",
    "droid_lerobot_decomposed__droid_plus_lerobot_640x360_20260412_failure": "droid_lerobot_decomposed",
    "robomind-franka": "robomind_franka",
    "robomind_franka": "robomind_franka",
    "robomind-franka-dual": "robomind_franka_dual",
    "robomind_franka_dual": "robomind_franka_dual",
    "robomind-ur": "robomind_ur",
    "robomind_ur": "robomind_ur",
    # ---- Datasets whose LeRobot root is nested per task or per episode --------------
    # For everything in this block the folder leaf is a per-task or per-episode dir
    # (task_327, xdof.a1_2025-11-05_05-50-45_USA, behavior1k-task0000, ...), not the
    # drop name, because the dir holding data/ is several levels below the drop. There
    # are 35-472 such leaves per drop, so registering them individually is not an
    # option: pass --source-dataset explicitly for these, exactly as the 46,000 h embodiment_a
    # note above requires. The names here are the values to pass.
    "AgiBotWorld-Alpha_20260319": "agibot_world",
    "AgiBotWorld-Beta_20260102": "agibot_world",
    # Released Bimanual local bundles retain the source-era canonical names.
    # Their immutable ACT2 widths are the Framework's embodiment-c contracts.
    "agibot_gear__AgiBotWorld-GEAR_20260208_agibot-custom": "embodiment_c",
    "agibot_gear__AgiBotWorld-GEAR_20260208_agibot-offshelf": "embodiment_c",
    "agibot_gear_ext__AgiBotWorld-GEAR_20260208_agibot-custom-ext": "embodiment_c_ext",
    "EmbodimentCWorld_20260208": "embodiment_c",
    "EmbodimentCWorld_20260208_dexhand": "embodiment_c_dexhand",
    "EmbodimentCWorld_20260208_ext": "embodiment_c_ext",
    "embodiment_c_gripper_ext": "embodiment_c_ext",
    "Behavior-challenge": "behavior_challenge",
    "maniparena_sim": "maniparena_sim",
    "maniparena_real": "maniparena_real",
    "maniparena_real_mobile": "maniparena_real_mobile",
    "maniparena_sim__maniparena_dataset_20260727_sim": "maniparena_sim",
    "maniparena_real__maniparena_dataset_20260727_real_execution_reasoning": "maniparena_real",
    "maniparena_real_mobile__maniparena_dataset_20260727_real_mobile_manipulation": "maniparena_real_mobile",
    "XDOF_YAM_v5": "xdof_yam_v5",
    "XDOF_YAM_v5_20260625": "xdof_yam_v5",
    "XDOF_YAM_v5_20260625_handpose": "xdof_handpose_intrinsics16",
    "XDOF_YAM_v5_20260625_handpose_intrinsics8": "xdof_handpose_intrinsics8",
    "XDOF_YAM_v5_annotated_20260429": "xdof_yam_v5_annotated",
    "WebHumanAction": "web_human_action",
    # The drop name, usable as --source-dataset instead of the family name. Note it
    # is never INFERRED: shard discovery is one level deep, so --input-dir has to
    # point at <drop>/{body,hands}/<res>/<fps>/ whose children are shard_NN, and
    # that infers the fps dir. Both keys exist so either spelling an operator
    # reaches for resolves; a future drop gets its own key beside this one.
    "260522_100k_rawVideos": "web_human_action",
    "web_human_action_v2": "web_human_action_v2",
    "web_human_action_v2__260522_100k_rawVideos": "web_human_action_v2",
    "worldengine_walle_week_01_10k": "worldengine_walle",
    "robocasa": "robocasa",
    "umi_lerobot": "umi_lerobot",
    # ---- Datasets whose root basename is already the drop name ----------------------
    "bridge_raw": "bridge",
    "bridge_raw_clean": "bridge",
    "bridge_orig_lerobot_20260111": "bridge_orig",
    "fractal20220817_data": "fractal",
    "fractal__fractal20220817_data": "fractal",
    "ABC-130k_lerobot_20260725": "abc_130k",
    "MolmoAct2-BimanualYAM-Dataset": "molmoact2_bimanual_yam",
    "molmoact2_bimanual_yam__MolmoAct2-BimanualYAM-Dataset": "molmoact2_bimanual_yam",
    "x_humanoid__X-Humanoid": "x_humanoid",
    "x_humanoid_tienkung__X-Humanoid_RoboMIND2.0-Tienkung_release_door_handle": "x_humanoid_tienkung",
    # The 20260907 simulation recovery preserves the 14D joint action/state
    # contract. Register only the current RoboDojo production populations.
    # Real sources keep separate identities but share the same 14D joint schema.
    "robodojo__robodojo_20260907_sim_arx_x5": "robodojo",
    "robodojo__robodojo_20260907_real_arx_x5": "robodojo",
    "robodojo__robodojo_20260907_real_piper": "robodojo",
    "robodojo__robodojo_20260907_real_piper_x": "robodojo",
    "pusht__pusht_image_20260102": "pusht",
    "behavior_1k_demos__behavior-1k-2026-challenge-demos": "behavior_1k_demos",
    # The canonical subtask population retains the replay-derived base_pose
    # field in its ACT2 payload.
    "behavior_1k_demos_subtask__behavior-1k-2026-challenge-demos": "behavior_1k_demos_subtask",
    "human_and_robot__HumanAndRobot_lerobot": "human_and_robot",
    # Legacy root aliases kept so older cut plans and local no-override probes
    # remain decodable. New reprocessing plans should use the spec-prefixed keys.
    "X-Humanoid": "x_humanoid",
    "pusht_image_20260102": "pusht",
    "behavior-1k-2026-challenge-demos": "behavior_1k_demos",
    "HumanAndRobot_lerobot": "human_and_robot",
    "fastumi_100k_dual_arm": "fastumi_dual_arm",
    "fastumi_dual_arm__fastumi_100k_dual_arm": "fastumi_dual_arm",
    "fastumi_100k_single_arm": "umi_lerobot",
    # Transitional UMI-family aliases: existing plans infer the leaf under
    # umi_lerobot/<leaf>, while new plans should use the spec-prefixed form below.
    "data_scaling_law": "umi_lerobot",
    "data_scaling_laws": "umi_lerobot",
    "exumi": "umi_lerobot",
    "umi_on_legs": "umi_lerobot",
    "maniwav": "umi_lerobot",
    "umi": "umi_lerobot",
    "vitamin": "umi_lerobot",
    "fastumi_single_arm": "umi_lerobot",
    "mvumi": "umi_lerobot",
    "touch_in_the_wild": "umi_lerobot",
    "umi_lerobot__data_scaling_law": "umi_lerobot",
    "umi_lerobot__data_scaling_laws": "umi_lerobot",
    "umi_lerobot__exumi": "umi_lerobot",
    "umi_lerobot__umi_on_legs": "umi_lerobot",
    "umi_lerobot__maniwav": "umi_lerobot",
    "umi_lerobot__umi": "umi_lerobot",
    "umi_lerobot__vitamin": "umi_lerobot",
    "umi_lerobot__fastumi_single_arm": "umi_lerobot",
    "umi_lerobot__fastumi_100k_single_arm": "umi_lerobot",
    "umi_lerobot__mvumi": "umi_lerobot",
    "umi_lerobot__touch_in_the_wild": "umi_lerobot",
    # NOTE the DROID `failure` split (.../droid_plus_lerobot_640x360_20260412/failure/)
    # carries columns identical to the success split, so it shares the spec -- but it
    # is deliberately NOT registered under the inferred name "failure". That string is
    # generic enough that any other drop with a failure/ split would silently inherit
    # DROID's stride and decode garbage. Pass the explicit failure alias above instead.
    "embodiment_b_20260828__embodiment_b_20260828": "embodiment_b_20260828",
}


MIXED_SOURCE_DATASET_PARENTS: frozenset[str] = frozenset(
    {
        "maniparena_real__maniparena_dataset_20260727_real",
    }
)


def _legacy_x_humanoid_spec_name(source_dataset: str) -> str | None:
    """Resolve already-built X-Humanoid plan names to the probed layout spec.

    The first production reprocess manifest used the family prefix
    ``x_humanoid__`` for every X-Humanoid root. The actual data is mixed by
    embodiment, so fresh manifests should use the concrete spec name as the
    prefix. This compatibility shim lets those already-written plan tables cut
    correctly without registering 189 one-off aliases.
    """
    if not source_dataset.startswith("x_humanoid__X-Humanoid_"):
        return None
    stable_id = source_dataset.removeprefix("x_humanoid__X-Humanoid_")
    if stable_id.startswith(("RoboMIND2.0-Agilex-mobile_", "RoboMIND2.0-Ark-mobile_")):
        return "x_humanoid"
    if stable_id.startswith(("RoboMIND2.0-Franka-Part-1_", "RoboMIND2.0-Franka-Part-3_")):
        return "x_humanoid_a18_s18"
    if stable_id.startswith(
        (
            "RoboMIND2.0-Franka-sim_",
            "RoboMIND2.0-Tianyi-mobile_",
            "RoboMIND2.0-Tianyi_RoboMIND2.0-Tianyi_",
        )
    ):
        return "x_humanoid_a16_s16"
    if stable_id in {
        "RoboMIND2.0-Tienkung_release_door_handle",
        "RoboMIND2.0-Tienkung_flip_black_knob_to_close_circuit",
    }:
        return "x_humanoid_tienkung"
    if stable_id.startswith("RoboMIND2.0-Tienkung_"):
        return "x_humanoid_a16_s16"
    if stable_id.startswith("RoboMIND2.0-Tienkung-sim_RoboMIND2.0-Tienkung-sim_"):
        return "x_humanoid_tienkung_sim"
    if stable_id.startswith("RoboMIND2.0-UR5-Dex_"):
        return "x_humanoid_ur5_dex"
    return None


def _resolve_action_binary_spec(source_dataset: str) -> tuple[str, ActionBinarySpec]:
    if source_dataset in MIXED_SOURCE_DATASET_PARENTS:
        raise ValueError(f"No action-binary spec registered for source dataset {source_dataset!r}.")
    if source_dataset in ACTION_BINARY_SPEC_BY_DATASET:
        spec_name = ACTION_BINARY_SPEC_BY_DATASET[source_dataset]
        try:
            return spec_name, ACTION_BINARY_SPECS[spec_name]
        except KeyError as exc:
            raise ValueError(
                f"Source dataset {source_dataset!r} is registered to unknown action-binary spec {spec_name!r}."
            ) from exc
    spec_name = _legacy_x_humanoid_spec_name(source_dataset)
    if spec_name is not None:
        return spec_name, ACTION_BINARY_SPECS[spec_name]
    if source_dataset.startswith("x_humanoid__X-Humanoid_"):
        # These legacy manifest names used the family spec prefix for mixed
        # embodiments. Unknown embodiments must not fall through to the generic
        # canonical-prefix resolver, which would silently decode with 14/14.
        raise ValueError(f"No action-binary spec registered for source dataset {source_dataset!r}.")
    if "__" in source_dataset:
        # Canonical reprocessing names are <spec_name>__<stable_source_id>.
        # The raw leaf still stays unregistered (for example "v0", "real",
        # "failure"), but the explicit manifest name resolves without adding
        # hundreds of brittle one-off aliases.
        spec_name, _stable_id = source_dataset.split("__", 1)
        if spec_name in ACTION_BINARY_SPECS:
            return spec_name, ACTION_BINARY_SPECS[spec_name]
    raise ValueError(f"No action-binary spec registered for source dataset {source_dataset!r}.")


def get_action_binary_spec_name(source_dataset: str) -> str:
    """Return the ACT2 schema name for a dataset alias or production population."""
    return _resolve_action_binary_spec(source_dataset)[0]


def get_action_binary_spec(source_dataset: str) -> ActionBinarySpec:
    """Return the fixed-stride contract for *source_dataset*."""
    return _resolve_action_binary_spec(source_dataset)[1]


def validate_action_bin_consumer_contract(
    header: Mapping[str, Any],
    *,
    source_dataset: str,
    context: str = "action binary",
    episode_id: str | None = None,
) -> None:
    """Require a decoded ACT2 header to match the registered training contract.

    Header decoding establishes that the payload is internally self-consistent;
    this validator supplies the second half of the check by comparing that
    self-description with the code-level source-dataset registry.
    """
    spec_name = get_action_binary_spec_name(source_dataset)
    spec = get_action_binary_spec(source_dataset)
    expected_scalars = {
        "spec_name": spec_name,
        "layout": "frame_major_with_per_clip_tail",
        "dtype": spec.dtype.str,
        "frame_data_offset": spec.header_size_bytes,
        "frame_record_size_bytes": spec.frame_record_size_bytes,
        "per_clip_size_bytes": spec.per_clip_size_bytes,
    }
    identity = "" if episode_id is None else f"episode_id={episode_id!r} "
    for field_name, expected in expected_scalars.items():
        observed = header.get(field_name)
        if observed != expected:
            raise ValueError(
                f"{context} {field_name} is incompatible with source_dataset={source_dataset!r}; "
                f"{identity}observed={observed!r} expected={expected!r}"
            )

    for field_group, fields in (
        ("per_frame_fields", spec.per_frame_fields),
        ("per_clip_fields", spec.per_clip_fields),
    ):
        expected_fields: list[dict[str, Any]] = []
        byte_offset = 0
        for field in fields:
            expected_fields.append(
                {
                    "name": field.name,
                    "shape": list(field.shape),
                    "byte_offset": byte_offset,
                }
            )
            byte_offset += field.value_count * spec.dtype.itemsize
        if header.get(field_group) != expected_fields:
            suffix = "" if episode_id is None else f"; episode_id={episode_id!r}"
            raise ValueError(f"{context} {field_group} is incompatible with source_dataset={source_dataset!r}{suffix}")


def _fields_for_contract_fingerprint(fields: tuple[ActionFieldSpec, ...]) -> list[dict[str, Any]]:
    """Return the stable JSON form for one ACT2 field sequence."""
    return [{"name": field.name, "shape": list(field.shape)} for field in fields]


def action_binary_contract_fingerprint(source_datasets: tuple[str, ...]) -> str:
    """Return a stable hash of the code-level ACT2 contract for source datasets."""
    contracts: list[dict[str, Any]] = []
    for source_dataset in source_datasets:
        spec_name, spec = _resolve_action_binary_spec(source_dataset)
        contracts.append(
            {
                "source_dataset": source_dataset,
                "spec_name": spec_name,
                "format_version": VERSION,
                "header_size": spec.header_size_bytes,
                "dtype": spec.dtype.str,
                "frame_record_size_bytes": spec.frame_record_size_bytes,
                "per_clip_size_bytes": spec.per_clip_size_bytes,
                "per_frame_fields": _fields_for_contract_fingerprint(spec.per_frame_fields),
                "per_clip_fields": _fields_for_contract_fingerprint(spec.per_clip_fields),
            }
        )

    payload = {"format": "ACT2", "contracts": contracts}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _header_for_spec(
    spec_name: str,
    spec: ActionBinarySpec,
    source_dataset: str,
    num_frames: int,
) -> bytes:
    per_frame_fields: list[dict[str, Any]] = []
    byte_offset = 0
    for field in spec.per_frame_fields:
        per_frame_fields.append(
            {
                "name": field.name,
                "shape": list(field.shape),
                "byte_offset": byte_offset,
            }
        )
        byte_offset += field.value_count * spec.dtype.itemsize

    per_clip_fields: list[dict[str, Any]] = []
    byte_offset = 0
    for field in spec.per_clip_fields:
        per_clip_fields.append(
            {
                "name": field.name,
                "shape": list(field.shape),
                "byte_offset": byte_offset,
            }
        )
        byte_offset += field.value_count * spec.dtype.itemsize

    header_size_bytes = spec.header_size_bytes
    per_clip_offset = header_size_bytes + num_frames * spec.frame_record_size_bytes
    header = {
        "format_version": VERSION,
        "source_dataset": source_dataset,
        "spec_name": spec_name,
        "num_frames": num_frames,
        "frame_data_offset": header_size_bytes,
        "layout": "frame_major_with_per_clip_tail",
        "dtype": spec.dtype.str,
        "frame_record_size_bytes": spec.frame_record_size_bytes,
        "per_frame_fields": per_frame_fields,
        "per_clip_offset": per_clip_offset,
        "per_clip_size_bytes": spec.per_clip_size_bytes,
        "per_clip_fields": per_clip_fields,
    }
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(header_bytes) > header_size_bytes - _HEADER_PREFIX_SIZE:
        raise ValueError(f"Action-binary header is too large: {len(header_bytes)} bytes.")

    prefix = MAGIC + len(header_bytes).to_bytes(4, "little", signed=False)
    return prefix + header_bytes + bytes(header_size_bytes - len(prefix) - len(header_bytes))


def write_action_bin(
    path: str,
    action_data: Mapping[str, Any],
    source_dataset: str,
) -> None:
    """Serialize *action_data* to *path* using the dataset's ACT2 contract."""
    with open(path, "wb") as file:
        file.write(encode_action_bin(action_data, source_dataset))


def encode_action_bin(action_data: Mapping[str, Any], source_dataset: str) -> bytes:
    """Return an ACT2 payload encoded with the contract for *source_dataset*."""
    spec_name, spec = _resolve_action_binary_spec(source_dataset)
    expected_names = {field.name for field in (*spec.per_frame_fields, *spec.per_clip_fields)}
    if set(action_data) != expected_names:
        raise ValueError(
            f"Action fields for {source_dataset!r} must be {sorted(expected_names)}, got {sorted(action_data)}."
        )

    columns: list[np.ndarray] = []
    num_frames: int | None = None
    for field in spec.per_frame_fields:
        array = np.asarray(action_data[field.name], dtype=spec.dtype)
        if array.ndim != len(field.shape) + 1 or array.shape[1:] != field.shape:
            raise ValueError(f"Action field {field.name!r} must have shape (frames, {field.shape}), got {array.shape}.")
        if num_frames is None:
            num_frames = array.shape[0]
        elif array.shape[0] != num_frames:
            raise ValueError(f"Action field {field.name!r} has {array.shape[0]} frames; expected {num_frames}.")
        columns.append(array.reshape(array.shape[0], field.value_count))

    if num_frames is None:
        num_frames = 0
    frame_data = np.concatenate(columns, axis=1)

    per_clip_columns: list[np.ndarray] = []
    for field in spec.per_clip_fields:
        array = np.asarray(action_data[field.name], dtype=spec.dtype)
        if array.shape != field.shape:
            raise ValueError(f"Per-clip action field {field.name!r} must have shape {field.shape}, got {array.shape}.")
        per_clip_columns.append(array.reshape(field.value_count))
    per_clip_data = np.concatenate(per_clip_columns) if per_clip_columns else np.empty((0,), dtype=spec.dtype)

    return (
        _header_for_spec(spec_name, spec, source_dataset, num_frames)
        + np.ascontiguousarray(frame_data, dtype=spec.dtype).tobytes()
        + np.ascontiguousarray(per_clip_data, dtype=spec.dtype).tobytes()
    )


def read_action_bin(path: str) -> dict[str, np.ndarray]:
    """Load a complete ACT2 file from *path*."""
    with open(path, "rb") as file:
        return decode_action_bin(file.read())


def decode_action_bin_header(data: bytes) -> dict[str, Any]:
    """Decode and validate the ACT2 header from *data*.

    ``data`` may be either the complete payload or the header bytes through the
    header's ``frame_data_offset``. This keeps range readers and ingestion checks
    from trusting a caller-provided frame count before consulting the binary's own
    header.
    """
    if len(data) < _HEADER_PREFIX_SIZE or data[:4] != MAGIC:
        raise ActionBinaryDecodeError(f"Not an ACT2 action-binary payload (magic={data[:4]!r}).")

    header_length = int.from_bytes(data[4:8], "little", signed=False)
    if header_length > MAX_ACT2_HEADER_JSON_BYTES:
        raise ActionBinaryDecodeError(
            f"ACT2 header JSON declares {header_length} byte(s), exceeding maximum {MAX_ACT2_HEADER_JSON_BYTES}."
        )
    if len(data) < _HEADER_PREFIX_SIZE + header_length:
        raise ActionBinaryDecodeError(
            f"Truncated action-binary header: JSON declares {header_length} byte(s), "
            f"but only {max(0, len(data) - _HEADER_PREFIX_SIZE)} byte(s) are available."
        )
    try:
        header = json.loads(data[8 : 8 + header_length].decode("utf-8"))
    except (RecursionError, ValueError) as exc:
        raise ActionBinaryDecodeError(f"Invalid action-binary header JSON: {exc}") from exc
    if not isinstance(header, Mapping):
        raise ActionBinaryDecodeError("Invalid action-binary header: expected a JSON object.")
    _validate_action_bin_header(header, header_length=header_length)
    frame_data_offset = int(header["frame_data_offset"])
    if len(data) < frame_data_offset:
        raise ActionBinaryDecodeError(
            f"Truncated action-binary header: expected at least {frame_data_offset} byte(s), got {len(data)}."
        )
    return dict(header)


def _validate_action_bin_fields(
    raw_fields: Any,
    *,
    field_group: str,
    record_size_bytes: int,
    dtype: np.dtype,
) -> None:
    """Validate field descriptors before NumPy receives offsets and shapes."""
    if not isinstance(raw_fields, list):
        raise ActionBinaryDecodeError(f"ACT2 {field_group} must be a list.")
    names: set[str] = set()
    occupied_ranges: list[tuple[int, int]] = []
    for index, raw_field in enumerate(raw_fields):
        if not isinstance(raw_field, Mapping):
            raise ActionBinaryDecodeError(f"ACT2 {field_group}[{index}] must be an object.")
        try:
            name = str(raw_field["name"])
            raw_shape = raw_field["shape"]
            byte_offset = int(raw_field["byte_offset"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ActionBinaryDecodeError(f"Invalid ACT2 {field_group}[{index}]: {exc}") from exc
        if not name or name in names:
            raise ActionBinaryDecodeError(f"ACT2 {field_group}[{index}] has an empty or duplicate name {name!r}.")
        if not isinstance(raw_shape, list) or not raw_shape:
            raise ActionBinaryDecodeError(f"ACT2 {field_group}[{index}] shape must be a non-empty list.")
        try:
            shape = tuple(int(size) for size in raw_shape)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ActionBinaryDecodeError(f"Invalid ACT2 {field_group}[{index}] shape: {exc}") from exc
        if any(size <= 0 for size in shape):
            raise ActionBinaryDecodeError(f"ACT2 {field_group}[{index}] shape dimensions must be positive.")
        if byte_offset < 0 or byte_offset % dtype.itemsize != 0:
            raise ActionBinaryDecodeError(f"ACT2 {field_group}[{index}] byte offset is invalid: {byte_offset}.")
        byte_end = byte_offset + prod(shape) * dtype.itemsize
        if byte_end > record_size_bytes:
            raise ActionBinaryDecodeError(
                f"ACT2 {field_group}[{index}] ends at byte {byte_end}; record size is {record_size_bytes}."
            )
        if any(
            byte_offset < existing_end and byte_end > existing_start for existing_start, existing_end in occupied_ranges
        ):
            raise ActionBinaryDecodeError(f"ACT2 {field_group}[{index}] overlaps another field.")
        names.add(name)
        occupied_ranges.append((byte_offset, byte_end))


def _validate_action_bin_header(header: Mapping[str, Any], *, header_length: int) -> None:
    try:
        format_version = int(header["format_version"])
        frame_data_offset = int(header["frame_data_offset"])
        dtype = np.dtype(header["dtype"])
        num_frames = int(header["num_frames"])
        frame_record_size_bytes = int(header["frame_record_size_bytes"])
        per_clip_offset = int(header["per_clip_offset"])
        per_clip_size_bytes = int(header["per_clip_size_bytes"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ActionBinaryDecodeError(f"Invalid action-binary header: {exc}") from exc

    if format_version != VERSION:
        raise ActionBinaryDecodeError(f"Unsupported action-binary version: {format_version}.")
    if frame_data_offset < _HEADER_PREFIX_SIZE:
        raise ActionBinaryDecodeError(
            f"ACT2 frame-data offset must be at least {_HEADER_PREFIX_SIZE}, got {frame_data_offset}."
        )
    if header_length > frame_data_offset - _HEADER_PREFIX_SIZE:
        raise ActionBinaryDecodeError(
            f"ACT2 header JSON has {header_length} byte(s), but frame-data offset is {frame_data_offset}."
        )
    if dtype.itemsize <= 0 or dtype.hasobject or dtype.kind not in "biufc":
        raise ActionBinaryDecodeError(f"ACT2 dtype must be a fixed-width numeric dtype, got {dtype}.")
    if num_frames < 0:
        raise ActionBinaryDecodeError(f"ACT2 num_frames must be non-negative, got {num_frames}.")
    if frame_record_size_bytes <= 0:
        raise ActionBinaryDecodeError(f"ACT2 frame record size must be positive, got {frame_record_size_bytes}.")
    if frame_record_size_bytes % dtype.itemsize != 0:
        raise ActionBinaryDecodeError(
            f"ACT2 frame record size {frame_record_size_bytes} is not divisible by dtype itemsize {dtype.itemsize}."
        )

    expected_per_clip_offset = frame_data_offset + num_frames * frame_record_size_bytes
    if per_clip_offset != expected_per_clip_offset:
        raise ActionBinaryDecodeError(
            f"ACT2 per-clip offset must be {expected_per_clip_offset}, got {per_clip_offset}."
        )
    if per_clip_size_bytes < 0:
        raise ActionBinaryDecodeError(f"ACT2 per-clip size must be non-negative, got {per_clip_size_bytes}.")
    if per_clip_size_bytes % dtype.itemsize != 0:
        raise ActionBinaryDecodeError(
            f"ACT2 per-clip size {per_clip_size_bytes} is not divisible by dtype itemsize {dtype.itemsize}."
        )
    _validate_action_bin_fields(
        header.get("per_frame_fields"),
        field_group="per_frame_fields",
        record_size_bytes=frame_record_size_bytes,
        dtype=dtype,
    )
    _validate_action_bin_fields(
        header.get("per_clip_fields"),
        field_group="per_clip_fields",
        record_size_bytes=per_clip_size_bytes,
        dtype=dtype,
    )


def assert_action_bin_frame_count(
    data: bytes,
    expected_num_frames: int,
    *,
    expected_source_dataset: str | None = None,
    payload_size: int | None = None,
    context: str = "action binary",
) -> dict[str, Any]:
    """Assert that an ACT2 header matches an expected frame count.

    When *payload_size* is supplied, the complete file size is also checked
    against the header-derived body size, catching truncated or overlong
    sidecars without reading the full payload.
    """
    header = decode_action_bin_header(data)
    actual_num_frames = int(header["num_frames"])
    if actual_num_frames != expected_num_frames:
        raise ValueError(f"{context} declares {actual_num_frames} frame(s); expected nb_frames={expected_num_frames}.")

    if expected_source_dataset:
        expected_frame_data_offset = get_action_binary_spec(expected_source_dataset).header_size_bytes
        actual_frame_data_offset = int(header["frame_data_offset"])
        if actual_frame_data_offset != expected_frame_data_offset:
            raise ValueError(
                f"{context} frame_data_offset is {actual_frame_data_offset}; expected {expected_frame_data_offset}."
            )
        actual_source_dataset = str(header.get("source_dataset") or "")
        if actual_source_dataset != expected_source_dataset:
            raise ValueError(
                f"{context} source_dataset is {actual_source_dataset!r}; expected {expected_source_dataset!r}."
            )

    if payload_size is not None:
        expected_size = int(header["per_clip_offset"]) + int(header["per_clip_size_bytes"])
        if payload_size != expected_size:
            raise ValueError(f"{context} payload has {payload_size} byte(s); expected {expected_size}.")

    return header


def decode_action_frames(data: bytes, source_dataset: str) -> dict[str, np.ndarray]:
    """Decode complete ACT2 frame records using the code-level dataset contract."""
    spec = get_action_binary_spec(source_dataset)
    if len(data) % spec.frame_record_size_bytes != 0:
        raise ValueError(
            f"Action frame data has {len(data)} bytes; expected a multiple of {spec.frame_record_size_bytes}."
        )

    num_frames = len(data) // spec.frame_record_size_bytes
    records = np.frombuffer(data, dtype=spec.dtype).reshape(num_frames, spec.per_frame_value_count)
    result: dict[str, np.ndarray] = {}
    offset = 0
    for field in spec.per_frame_fields:
        field_values = records[:, offset : offset + field.value_count]
        result[field.name] = field_values.reshape(num_frames, *field.shape).copy()
        offset += field.value_count
    return result


def decode_action_per_clip(data: bytes, source_dataset: str) -> dict[str, np.ndarray]:
    """Decode the ACT2 per-clip tail using the code-level dataset contract."""
    spec = get_action_binary_spec(source_dataset)
    if len(data) != spec.per_clip_size_bytes:
        raise ValueError(f"Per-clip action data has {len(data)} bytes; expected {spec.per_clip_size_bytes}.")

    values = np.frombuffer(data, dtype=spec.dtype)
    result: dict[str, np.ndarray] = {}
    offset = 0
    for field in spec.per_clip_fields:
        field_values = values[offset : offset + field.value_count]
        result[field.name] = field_values.reshape(field.shape).copy()
        offset += field.value_count
    return result


def decode_action_bin(data: bytes) -> dict[str, np.ndarray]:
    """Decode a complete, self-describing ACT2 payload."""
    header = decode_action_bin_header(data)

    dtype = np.dtype(header["dtype"])
    num_frames = int(header["num_frames"])
    frame_record_size_bytes = int(header["frame_record_size_bytes"])
    per_clip_offset = int(header["per_clip_offset"])
    per_clip_size_bytes = int(header["per_clip_size_bytes"])
    expected_size = per_clip_offset + per_clip_size_bytes
    if len(data) != expected_size:
        raise ActionBinaryDecodeError(f"Action-binary payload has {len(data)} bytes; expected {expected_size}.")

    values_per_record = frame_record_size_bytes // dtype.itemsize
    records = np.frombuffer(
        data,
        dtype=dtype,
        count=num_frames * values_per_record,
        offset=int(header["frame_data_offset"]),
    ).reshape(num_frames, values_per_record)
    result: dict[str, np.ndarray] = {}
    for field in header["per_frame_fields"]:
        name = str(field["name"])
        shape = tuple(int(size) for size in field["shape"])
        value_offset = int(field["byte_offset"]) // dtype.itemsize
        value_count = prod(shape)
        field_values = records[:, value_offset : value_offset + value_count]
        result[name] = field_values.reshape(num_frames, *shape).copy()

    per_clip_values = np.frombuffer(data, dtype=dtype, offset=per_clip_offset)
    for field in header["per_clip_fields"]:
        name = str(field["name"])
        shape = tuple(int(size) for size in field["shape"])
        value_offset = int(field["byte_offset"]) // dtype.itemsize
        value_count = prod(shape)
        field_values = per_clip_values[value_offset : value_offset + value_count]
        result[name] = field_values.reshape(shape).copy()
    return result


def slice_action_bin(data: bytes, start_frame: int, end_frame_exclusive: int) -> bytes:
    """Return a byte-exact half-open ACT2 frame slice.

    Frame records and the per-clip tail are copied verbatim. Only the
    self-describing header is rewritten with the sliced frame count and new
    per-clip offset. This deliberately avoids resolving the payload through the
    mutable code-level dataset registry or round-tripping values through NumPy.
    """
    start = operator.index(start_frame)
    end = operator.index(end_frame_exclusive)
    header = decode_action_bin_header(data)
    num_frames = int(header["num_frames"])
    if start < 0 or end <= start or end > num_frames:
        raise IndexError(f"ACT2 slice [{start}, {end}) must be a non-empty interval within [0, {num_frames}).")

    frame_data_offset = int(header["frame_data_offset"])
    frame_record_size_bytes = int(header["frame_record_size_bytes"])
    per_clip_offset = int(header["per_clip_offset"])
    per_clip_size_bytes = int(header["per_clip_size_bytes"])
    expected_size = per_clip_offset + per_clip_size_bytes
    if len(data) != expected_size:
        raise ActionBinaryDecodeError(f"Action-binary payload has {len(data)} bytes; expected {expected_size}.")

    sliced_header = dict(header)
    sliced_num_frames = end - start
    sliced_header["num_frames"] = sliced_num_frames
    sliced_header["per_clip_offset"] = frame_data_offset + sliced_num_frames * frame_record_size_bytes
    header_bytes = json.dumps(sliced_header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(header_bytes) > frame_data_offset - _HEADER_PREFIX_SIZE:
        raise ActionBinaryDecodeError(
            f"Sliced action-binary header is too large: {len(header_bytes)} bytes for "
            f"frame_data_offset={frame_data_offset}."
        )
    prefix = MAGIC + len(header_bytes).to_bytes(4, "little", signed=False)
    encoded_header = prefix + header_bytes + bytes(frame_data_offset - len(prefix) - len(header_bytes))

    frame_start = frame_data_offset + start * frame_record_size_bytes
    frame_end = frame_data_offset + end * frame_record_size_bytes
    return encoded_header + data[frame_start:frame_end] + data[per_clip_offset:expected_size]
