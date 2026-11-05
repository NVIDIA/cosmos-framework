# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T

from cosmos_framework.data.generator.action.robot_assets import get_action_robot_asset_path
from cosmos_framework.data.generator.action.utils.action_spec import (
    ActionSpec,
    Gripper,
    Joint,
    Pos,
    Rot,
    build_action_spec,
)
from cosmos_framework.data.generator.action.utils.pose_utils import PoseConvention, pose_abs_to_abs9d, pose_abs_to_rel
from cosmos_framework.data.generator.action.yam_dataset_config import YAM_TO_OPENCV

YAM_ARM_JOINT_COUNT: int = 6


YAM_ACTION_DIM_EE_POSE: int = 20


YAM_ACTION_DIM_JOINT_POSE: int = 14


_YAM_MJCF: str = "i2rt_yam.xml"


_YAM_EE_SITE: str = "tcp_site"


_YAM_ARM_JOINT_NAMES: tuple[str, ...] = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


EeSitePoseFn = Callable[[torch.Tensor], np.ndarray]


YAM_IMAGE_CROP_RATIO: float = 0.95


def build_yam_image_augmentor(image_height: int, image_width: int) -> T.Compose:
    """Build the YAM augmentation recipe: the DROID one without hue.

    Hue is dropped deliberately. It forces an RGB->HSV->RGB round trip that
    measured 874 ms of the 967 ms total for one sample (3 views x 17 frames at
    640x480, one worker thread) -- 76% of the whole augmentor, and the reason
    augmentation was switched off for YAM at all. Without it the recipe is ~4x
    cheaper; brightness/contrast/saturation still cover exposure and colour
    intensity, so only robustness to white-balance cast is given up.

    Deliberately NOT pre-downscaling before the crop: measured at 1.07x, because
    YAM's native 640x480 per view already tiles to about the 480-tier target
    (640x640). It would only pay on a source far above the training resolution.

    Kept here rather than as a flag on ``build_droid_image_augmentor`` so DROID
    and LIBERO keep sole ownership of their own recipe.
    """
    return T.Compose(
        [
            T.RandomCrop((int(image_height * YAM_IMAGE_CROP_RATIO), int(image_width * YAM_IMAGE_CROP_RATIO))),
            T.Resize((image_height, image_width), antialias=True),
            T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
        ]
    )


_DEFAULT_POS_TOL = 1e-4  # m


_DEFAULT_ROT_TOL = 1e-4  # rad


_DEFAULT_MAX_ITERS = 100


_DEFAULT_DAMPING = 1e-3


_MAX_JOINT_STEP = 0.2  # rad


@dataclass(frozen=True)
class IKResult:
    """One arm's IK solution and how well it actually hit the target."""

    qpos: np.ndarray  # [6] joint angles, clamped to the model's joint ranges
    converged: bool
    pos_error: float  # m
    rot_error: float  # rad
    iterations: int


def _rotation_log(rotation: np.ndarray) -> np.ndarray:
    """Log map of a rotation matrix -> rotation vector [3] (axis * angle)."""
    cos_theta = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-9:
        return np.zeros(3)
    if theta > np.pi - 1e-6:
        # Near pi the skew part vanishes, so the axis has to come from the symmetric part:
        # (R - cos(theta) I) / (1 - cos(theta)) == outer(axis, axis) there. The diagonal
        # gives each |axis_i|, but the SIGNS ONLY LIVE IN THE OFF-DIAGONALS -- taking a
        # square root per component would force every entry nonnegative and silently
        # return a different rotation (a pi turn about (1,-1,0) would come back as
        # (1,1,0)). So read the row of the largest component, which fixes the relative
        # signs, and normalize.
        outer = (rotation - cos_theta * np.eye(3)) / (1.0 - cos_theta)
        dominant = int(np.argmax(np.diag(outer)))
        axis = outer[dominant] / np.sqrt(max(outer[dominant, dominant], 1e-12))
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        # Only the GLOBAL sign is genuinely ambiguous at exactly pi. Just below pi the
        # skew part is small but still signed, so use it to stay continuous with the
        # branch below; at exactly pi it is zero and either sign is the same rotation.
        skew_near_pi = np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        )
        if float(axis @ skew_near_pi) < 0.0:
            axis = -axis
        return axis * theta
    skew = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    return skew * (theta / (2.0 * np.sin(theta)))


class YAMKinematics:
    """i2rt YAM MuJoCo kinematics, both directions: joints <-> ``tcp_site`` pose.

    Both arms share one single-arm model: each arm's joints are solved in that
    arm's own base frame, which is exactly the frame the 20D action is expressed
    in. MuJoCo model/data handles are process-bound, so callers that fork or
    spawn workers must build this lazily in the process that runs it rather than
    inheriting an instance.

    ``ee_site_poses`` is the forward direction, and it is what DEFINES the 20D
    ``ee_pose`` action -- training uses nothing else. ``solve``/``solve_trajectory``
    are its inverse, needed only when serving an ``ee_pose`` checkpoint to a robot
    that takes joint targets, which in practice is every one of them (the ManiSkill
    YAM exposes only ``pd_joint_pos``). They live in one class deliberately: an
    inverse taken against a different model or site returns joints whose FK is not
    the pose that was asked for, so keeping the pair together makes "FK and IK
    agree" a statement about one model rather than two. Same reasoning puts
    ``pose_abs_to_rel`` and ``pose_rel_to_abs`` in one module.

    WHY A HAND-ROLLED SOLVER. MuJoCo ships no kinematics IK. ``mj_inverse`` is
    inverse *dynamics*; ``mujoco.minimize.least_squares`` is a general nonlinear
    least-squares optimizer, and driving it still means supplying the pose residual
    and the site Jacobian -- most of the work here -- after which, measured on this
    model, it was ~10x slower per solve and did not reach tolerance on any of 60
    targets that damped least squares solved. The canonical ready-made option,
    ``dm_control.utils.inverse_kinematics.qpos_from_site_pose``, would pull in
    dm_control for one function and is not installed. What is here instead is ~40
    lines of damped least squares on ``mj_jacSite``, and in the regime it actually
    runs in -- warm-started from the previous solution, targets ~0.02 rad apart --
    it converges on 200/200 sampled targets in 2.8 iterations mean.
    """

    def __init__(self, ee_site: str = _YAM_EE_SITE) -> None:
        """``ee_site`` is which site the ee_pose action is expressed at, and it is a property
        of the DATASET the checkpoint trained on -- see the class docstring."""
        import mujoco

        mjcf_path = str(get_action_robot_asset_path(_YAM_MJCF))
        self.model: Any = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data: Any = mujoco.MjData(self.model)
        self.ee_site: str = ee_site
        self.ee_site_id: int = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        if self.ee_site_id < 0:
            raise RuntimeError(f"YAM MJCF is missing site {ee_site!r}.")

        qpos_addr: list[int] = []
        for joint_name in _YAM_ARM_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"YAM MJCF is missing joint {joint_name!r}.")
            qpos_addr.append(int(self.model.jnt_qposadr[joint_id]))
        self.arm_qpos_addr: np.ndarray = np.asarray(qpos_addr, dtype=np.int64)  # [6]

        # The single-arm model's base body. FK reports poses in world; the action is in
        # the arm base frame. The base is static, so resolve the transform once.
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "arm")
        if base_id < 0:
            raise RuntimeError("YAM MJCF is missing body 'arm'.")
        self._base_body_id = int(base_id)

        self.data.qpos[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        base_world = np.eye(4)
        base_world[:3, 3] = self.data.xpos[self._base_body_id]
        base_world[:3, :3] = self.data.xmat[self._base_body_id].reshape(3, 3)
        self._base_world = base_world  # [4,4] arm-base -> world
        self._world_base = np.linalg.inv(base_world)  # [4,4] world -> arm-base

        # Arm joint ids, resolved once: ``YAMKinematics`` stores qpos addresses, while
        # joint limits are indexed by joint id and the Jacobian by DOF address.
        jnt_ids = [int(np.argmax(self.model.jnt_qposadr == addr)) for addr in self.arm_qpos_addr]
        self._arm_dof_addr = self.model.jnt_dofadr[jnt_ids]  # [6], columns of mj_jacSite

        # Joint limits, so a solution is never returned outside the arm's real range.
        limited = self.model.jnt_limited[jnt_ids].astype(bool)  # [6]
        ranges = self.model.jnt_range[jnt_ids]  # [6,2]
        self._q_lower = np.where(limited, ranges[:, 0], -np.inf).astype(np.float64)
        self._q_upper = np.where(limited, ranges[:, 1], np.inf).astype(np.float64)

        # Scratch buffers for mj_jacSite, reused across the many solves in a chunk.
        self._jac_pos = np.zeros((3, self.model.nv))
        self._jac_rot = np.zeros((3, self.model.nv))

    def ee_site_poses(self, arm_joints: torch.Tensor | np.ndarray) -> np.ndarray:  # [T,6] -> [T,4,4]
        """Run MuJoCo FK for one arm's 6 joints and return ``_YAM_EE_SITE`` poses."""
        import mujoco

        arm_q = (
            arm_joints.detach().cpu().numpy().astype(np.float32)
            if isinstance(arm_joints, torch.Tensor)
            else np.asarray(arm_joints, dtype=np.float32)
        )  # [T,6]
        if arm_q.ndim != 2 or arm_q.shape[-1] != YAM_ARM_JOINT_COUNT:
            raise ValueError(f"YAM FK expects [T,{YAM_ARM_JOINT_COUNT}] arm joints, got {tuple(arm_q.shape)}.")

        poses = np.tile(np.eye(4, dtype=np.float32), (len(arm_q), 1, 1))  # [T,4,4]
        for t, q in enumerate(arm_q):
            self.data.qpos[:] = 0.0  # [nq]
            self.data.qpos[self.arm_qpos_addr] = q  # [6]
            mujoco.mj_forward(self.model, self.data)
            poses[t, :3, 3] = self.data.site_xpos[self.ee_site_id]  # [3]
            poses[t, :3, :3] = self.data.site_xmat[self.ee_site_id].reshape(3, 3)  # [3,3]
        return poses

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """``(lower, upper)`` joint ranges, ``[6]`` each. Unlimited joints are +-inf."""
        return self._q_lower.copy(), self._q_upper.copy()

    def fk(self, arm_joints: np.ndarray) -> np.ndarray:
        """One arm's 6 joints -> ``tcp_site`` pose ``[4,4]`` in the arm base frame.

        ``YAMKinematics.ee_site_poses`` returns world poses. The dataset only ever uses
        a single-arm model whose base sits at the world origin, so the two coincide
        there; this rebases explicitly so the module does not silently depend on that.
        """
        pose_world = self.ee_site_poses(np.asarray(arm_joints, dtype=np.float32).reshape(1, 6))[0]
        return self._world_base @ pose_world

    def solve(
        self,
        target_pose: np.ndarray,  # [4,4] tcp_site in the arm base frame
        q_init: np.ndarray,  # [6] warm start, normally the measured joints
        *,
        pos_tol: float = _DEFAULT_POS_TOL,
        rot_tol: float = _DEFAULT_ROT_TOL,
        max_iters: int = _DEFAULT_MAX_ITERS,
        damping: float = _DEFAULT_DAMPING,
    ) -> IKResult:
        """Solve for the joints that put the EE site at ``target_pose``.

        Returns the best iterate found. A non-converged result is still the closest
        reachable pose the solver reached, which for an unreachable target is the
        useful thing to command -- callers decide whether to trust it by checking
        ``converged`` and the two error terms.

        ``max_iters`` bounds EVALUATIONS, not applied steps, and the result is always
        an iterate whose error was actually measured -- so ``pos_error``/``rot_error``
        never describe a pose the solver only predicted. The consequence is that
        ``max_iters=1`` returns the seed: it evaluates, steps, and stops before
        measuring the step. Anything from 2 up behaves as expected, and the default
        of 100 is far above the 3 or so this needs when warm-started.
        """
        import mujoco

        target = np.asarray(target_pose, dtype=np.float64).reshape(4, 4)
        target_world = self._base_world @ target
        target_pos = target_world[:3, 3]
        target_rot = target_world[:3, :3]

        q = np.clip(np.asarray(q_init, dtype=np.float64).reshape(6), self._q_lower, self._q_upper)

        jac_pos, jac_rot, dof_addr = self._jac_pos, self._jac_rot, self._arm_dof_addr
        best = None
        iterations = 0
        for iterations in range(1, max_iters + 1):
            self.data.qpos[:] = 0.0
            self.data.qpos[self.arm_qpos_addr] = q
            mujoco.mj_forward(self.model, self.data)

            cur_pos = self.data.site_xpos[self.ee_site_id].copy()
            cur_rot = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()

            err_pos = target_pos - cur_pos
            err_rot = _rotation_log(cur_rot.T @ target_rot)
            err_rot = cur_rot @ err_rot  # body-frame residual -> world, to match the Jacobian

            pos_error = float(np.linalg.norm(err_pos))
            rot_error = float(np.linalg.norm(err_rot))
            if best is None or (pos_error + rot_error) < (best[1] + best[2]):
                best = (q.copy(), pos_error, rot_error)
            if pos_error < pos_tol and rot_error < rot_tol:
                return IKResult(q.copy(), True, pos_error, rot_error, iterations)

            mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.ee_site_id)
            jac = np.vstack([jac_pos[:, dof_addr], jac_rot[:, dof_addr]])  # [6,6]
            err = np.concatenate([err_pos, err_rot])  # [6]

            # Damped least squares: (J^T J + lambda^2 I)^-1 J^T e, in the 6x6 form.
            dq = np.linalg.solve(jac.T @ jac + (damping**2) * np.eye(6), jac.T @ err)
            step = float(np.max(np.abs(dq)))
            if step > _MAX_JOINT_STEP:
                dq *= _MAX_JOINT_STEP / step
            q = np.clip(q + dq, self._q_lower, self._q_upper)

        assert best is not None
        q_best, pos_error, rot_error = best
        return IKResult(q_best, pos_error < pos_tol and rot_error < rot_tol, pos_error, rot_error, iterations)

    def solve_trajectory(
        self,
        target_poses: np.ndarray,  # [T,4,4] in the arm base frame
        q_init: np.ndarray,  # [6] measured joints at the start of the chunk
        **kwargs,
    ) -> tuple[np.ndarray, list[IKResult]]:
        """Solve a chunk of targets, each warm-started from the previous solution.

        Chaining the seed is what keeps the chunk on one IK branch: solving each target
        independently from the measured joints lets neighbouring targets land in
        different elbow/wrist configurations, which is a discontinuity the arm would
        try to track as a jump.
        """
        poses = np.asarray(target_poses, dtype=np.float64).reshape(-1, 4, 4)
        q = np.asarray(q_init, dtype=np.float64).reshape(6)
        solutions = np.empty((len(poses), 6), dtype=np.float32)
        results: list[IKResult] = []
        for t, pose in enumerate(poses):
            result = self.solve(pose, q, **kwargs)
            solutions[t] = result.qpos
            results.append(result)
            q = result.qpos
        return solutions, results


def build_yam_action_spec(action_space: str) -> ActionSpec:
    """Declare the YAM action layout so both storage paths score idle frames alike."""
    if action_space == "joint_pose":
        return build_action_spec(
            Joint(n=YAM_ARM_JOINT_COUNT, prefix="left"),
            Gripper(prefix="left"),
            Joint(n=YAM_ARM_JOINT_COUNT, prefix="right"),
            Gripper(prefix="right"),
        )
    return build_action_spec(
        Pos(prefix="left"),
        Rot("rot6d", prefix="left"),
        Gripper(prefix="left"),
        Pos(prefix="right"),
        Rot("rot6d", prefix="right"),
        Gripper(prefix="right"),
    )


def yam_normalizer_filename(action_space: str, pose_convention: str) -> str:
    """Return the shared normalizer filename used by all YAM loaders.

    ABC / MolmoAct2 / XDOF are the same robot and share one set of stats, so the bundled
    names are not per embodiment -- unlike the ``<embodiment_type>_...`` default in
    :meth:`BaseActionLeRobotDataset._normalizer_filename`, which the three YAM classes
    cannot use because they must keep distinct ``_embodiment_type`` values for domain
    routing. Lives here so the LeRobot and Lance paths normalize with the same file.
    """
    if action_space == "joint_pose":
        return f"yam_joint_pose_{pose_convention}.json"
    return f"yam_ee_{pose_convention}_rot6d.json"


def _as_tensor(values: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Accept either storage layer's per-frame array as a float tensor."""
    if isinstance(values, torch.Tensor):
        return values
    return torch.from_numpy(np.asarray(values, dtype=np.float32))


def build_yam_action(
    state: torch.Tensor | np.ndarray,  # [T+1,S]
    raw_action: torch.Tensor | np.ndarray,  # [T,S] or [T+1,S]
    *,
    arm_joint_slices: tuple[slice, slice],
    gripper_slices: tuple[slice, slice],
    chunk_length: int,
    action_space: str,
    pose_convention: str,
    ee_site_poses: EeSitePoseFn,
    to_opencv: np.ndarray = YAM_TO_OPENCV,
    use_state: bool = False,
    dataset_name: str = "YAM",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build the shared i2rt YAM dual-arm action contract.

    Arm joints always come from ``state``; grippers always come from ``raw_action``.
    Both are tail-sliced to ``chunk_length`` so callers may pass either ``T`` or
    ``T+1`` gripper rows.

    Returns:
        ``(action, extras)``. For ``action_space="ee_pose"`` the action is
        ``[T,20]`` (or ``[T+1,20]`` with ``use_state``, whose extra leading row is the
        frame-0 absolute pose) and ``extras`` carries
        ``initial_pose_left``/``initial_pose_right`` for the viewer. For
        ``action_space="joint_pose"`` the action is ``[T,14]`` (or ``[T+1,14]`` with
        ``use_state``) and ``extras`` is empty.
    """
    if action_space not in ("ee_pose", "joint_pose"):
        raise ValueError(f"Unsupported action_space={action_space!r}")
    if action_space == "joint_pose" and pose_convention != "absolute":
        raise ValueError(f"joint_pose currently supports only pose_convention='absolute', got {pose_convention!r}")
    # ee_pose + use_state is allowed under every convention. The conditioning row is the
    # frame-0 ABSOLUTE pose even when the action rows are relative: under a relative
    # convention frame 0 is the anchor, so a row in the action space would be the identity
    # and carry no proprio at all. Mixing the two spaces in one tensor is deliberate --
    # "where the arm is" plus "how it should move" -- see the note at the prepend.

    state_t = _as_tensor(state)
    action_t = _as_tensor(raw_action)
    left_slice, right_slice = arm_joint_slices
    left_grip_slice, right_grip_slice = gripper_slices
    # Slice stops differ per dataset (XDOF joints end at 46, grippers at 34), so
    # derive each minimum width separately.
    min_state = max(left_slice.stop, right_slice.stop)
    min_action = max(left_grip_slice.stop, right_grip_slice.stop)
    if state_t.shape[-1] < min_state or action_t.shape[-1] < min_action:
        raise ValueError(
            f"{dataset_name} needs state >= {min_state} and action >= {min_action} dims, "
            f"got state={state_t.shape[-1]} and action={action_t.shape[-1]}."
        )
    left_gripper = action_t[:, left_grip_slice].numpy().astype(np.float32)  # [T,1]
    right_gripper = action_t[:, right_grip_slice].numpy().astype(np.float32)  # [T,1]

    if action_space == "joint_pose":
        # Absolute 14D joints [Lj(6), Lg(1), Rj(6), Rg(1)] straight from the
        # state (drop frame 0, like ee_pose) -- no FK.
        left_joints = state_t[:, left_slice].numpy().astype(np.float32)[-chunk_length:]  # [T,6]
        right_joints = state_t[:, right_slice].numpy().astype(np.float32)[-chunk_length:]  # [T,6]
        action = np.concatenate(  # [T,14]
            [left_joints, left_gripper[-chunk_length:], right_joints, right_gripper[-chunk_length:]],
            axis=-1,
        ).astype(np.float32)
        if use_state:
            # Prepend the frame-0 proprio as a conditioning row (grippers from the
            # *state* column), so the sequence planner marks row 0 as a given/condition
            # token. This makes the action [T+1,14] while the action *steps* stay T;
            # it says nothing about the video length, which the caller sets separately
            # via video_subsample_factor (the shipped recipe pairs 33 action rows with
            # 17 video frames).
            initial_state = np.concatenate(  # [1,14]
                [
                    state_t[0, left_slice].numpy(),
                    state_t[0, left_grip_slice].numpy(),
                    state_t[0, right_slice].numpy(),
                    state_t[0, right_grip_slice].numpy(),
                ],
                axis=-1,
            ).astype(np.float32)[None]
            action = np.concatenate([initial_state, action], axis=0)  # [T+1,14]
        return torch.from_numpy(action).float(), {}  # [T(+1),14]

    # -- ee_pose: MuJoCo FK to the end-effector site + OpenCV correction --
    left_abs = ee_site_poses(state_t[:, left_slice])  # [T+1,4,4]
    right_abs = ee_site_poses(state_t[:, right_slice])  # [T+1,4,4]
    # OpenCV-correct the rotation on the full [T+1] trajectory before the rel/abs
    # conversion AND before ``initial_pose_*`` is captured, so the action and the
    # viewer's initial pose share one frame (mirrors DROID). Position unchanged.
    rot_to_opencv = to_opencv.astype(left_abs.dtype)  # [3,3]
    left_abs[:, :3, :3] = left_abs[:, :3, :3] @ rot_to_opencv  # [T+1,3,3]
    right_abs[:, :3, :3] = right_abs[:, :3, :3] @ rot_to_opencv  # [T+1,3,3]

    if pose_convention == "absolute":
        left_pose = pose_abs_to_abs9d(left_abs)[-chunk_length:]  # [T,9]
        right_pose = pose_abs_to_abs9d(right_abs)[-chunk_length:]  # [T,9]
    else:  # backward_anchored / backward_framewise
        relative_convention = cast(PoseConvention, pose_convention)
        left_pose = pose_abs_to_rel(left_abs, rotation_format="rot6d", pose_convention=relative_convention)[
            -chunk_length:
        ]  # [T,9]
        right_pose = pose_abs_to_rel(right_abs, rotation_format="rot6d", pose_convention=relative_convention)[
            -chunk_length:
        ]  # [T,9]

    action = np.concatenate(  # [T,20]
        [left_pose, left_gripper[-chunk_length:], right_pose, right_gripper[-chunk_length:]],
        axis=-1,
    ).astype(np.float32)
    if use_state:
        # Frame 0 prepended as a conditioning row, mirroring the joint_pose branch, so the
        # sequence planner marks row 0 a given token: action becomes [T+1,20] while the
        # action *steps* stay T. Grippers come from the *state* column, as for joint_pose.
        #
        # The row is ABSOLUTE regardless of pose_convention. Under backward_anchored /
        # backward_framewise the action rows are relative to frame 0, so an in-space row
        # would be the identity for every sample; absolute is the only encoding that says
        # where the arm actually starts.
        #
        # Consequence to know: one normalizer is applied to the whole tensor downstream, so
        # under a relative convention row 0 is scaled by delta statistics. asinh is
        # monotonic, so the row stays informative rather than saturating flat, but it is
        # compressed relative to the action rows. Separate stats for row 0 would need the
        # normalizer to become row-aware, which it is not today.
        #
        # pose_abs_to_abs9d drops its leading frame to stay step-aligned with the relative
        # conversion, so frame 0 is fed twice to get frame 0 back out.
        initial_pose = np.concatenate(  # [1,20]
            [
                pose_abs_to_abs9d(left_abs[[0, 0]])[0],  # [9]
                state_t[0, left_grip_slice].numpy(),  # [1]
                pose_abs_to_abs9d(right_abs[[0, 0]])[0],  # [9]
                state_t[0, right_grip_slice].numpy(),  # [1]
            ],
            axis=-1,
        ).astype(np.float32)[None]
        action = np.concatenate([initial_pose, action], axis=0)  # [T+1,20]
    extras = {
        "initial_pose_left": torch.from_numpy(left_abs[0].copy()).float(),  # [4,4]
        "initial_pose_right": torch.from_numpy(right_abs[0].copy()).float(),  # [4,4]
    }
    return torch.from_numpy(action).float(), extras  # [T,20] or [T+1,20] with use_state


def compose_yam_concat_view(
    top: torch.Tensor,  # [T,C,H,W]
    left: torch.Tensor,  # [T,C,H,W]
    right: torch.Tensor,  # [T,C,H,W]
) -> torch.Tensor:  # [T,C,3H/2,W]
    """Tile the three YAM cameras into the shared ``concat_view`` layout.

    Layout (per frame)::

        ┌──────────────┐
        │    top       │   (H, W)
        ├───────┬──────┤
        │ left  │ right│   (H/2, W/2) each
        └───────┴──────┘

    Left and right exterior cameras are downscaled by 2x so that they tile to
    the same width as the top view. The output height is 3H/2.
    """
    _, _, h_top, w_top = top.shape
    half_h, half_w = h_top // 2, w_top // 2
    left = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)
    right = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)
    bottom = torch.cat([left, right], dim=-1)
    return torch.cat([top, bottom], dim=-2)
