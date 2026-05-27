#!/usr/bin/env python
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software and related materials and are subject
# to, without limitation, any intellectual property or other proprietary rights that
# may be applicable to such software and related materials, and any terms and
# conditions to which such software and related materials are subject.
"""Overlay GT and predicted action trajectories from eval outputs.

Loads ground truth (batch_data.safetensors or output.safetensors) alongside
predicted (output.safetensors) action tensors directory and renders both
trajectories overlaid in the same viser scene.

Color scheme: GT trajectories in green, pred trajectories in red. The
robot mesh is driven only by the pred trajectory.

Usage — browse an eval dir:
    uv run python cosmos_framework/data/vfm/action/urdf_visualizer/compare_viewer.py \\
        --eval-dir /mnt/cosmos-eval/<job_dir> --share

Usage — single pair:
    uv run python cosmos_framework/data/vfm/action/urdf_visualizer/compare_viewer.py \\
        --gt  ground_truth/bridge/inverse_dynamics/0/batch_data.safetensors \\
        --pred outputs/bridge/inverse_dynamics/0/output.safetensors \\
        --dataset bridge

Dependencies: viser safetensors numpy cv2 torch (mujoco pin for robot meshes)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cosmos_framework.utils import log
from cosmos_framework.data.vfm.action.urdf_visualizer.unified_action import (
    ActionFormat,
    SceneState,
    build_scene_state,
    to_unified,
)
from cosmos_framework.data.vfm.action.urdf_visualizer.unified_renderer import UnifiedRenderer
from cosmos_framework.data.vfm.action.urdf_visualizer.viewer import (
    DatasetEntry,
    _build_datasets,
    _load_symbol,
)

# ── Minimal dataset entries ────────────────────────────


def _build_minimal_entries() -> dict[str, DatasetEntry]:
    """Hardcoded subset of the registry

    Source ``to_opencv`` matrices from each dataset module
    """
    # All matrices inlined to avoid importing dataset modules that pull in lerobot,
    # which is not available on CPU Lepton jobs.
    _BRIDGE_TO_OPENCV = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)
    _DROID_TO_OPENCV = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    _GOOGLE_ROBOT_TO_OPENCV = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
    _ROBOMIND_FRANKA_TO_OPENCV = np.array(
        [[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    AGIBOT_GEAR_GRIPPER_TO_OPENCV_BY_WRIST = {
        "left_wrist": np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        "right_wrist": np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    }

    franka_to_opencv = _ROBOMIND_FRANKA_TO_OPENCV[:3, :3]
    eye3 = np.eye(3, dtype=np.float32)

    def E(**kw: Any) -> DatasetEntry:
        return DatasetEntry(action_format=kw.pop("action_format", ActionFormat.SINGLE_ARM_10D), **kw)

    return {
        "bridge": E(name="bridge", robot_name="widowx", max_finger_width=0.06, fps=5, to_opencv=_BRIDGE_TO_OPENCV),
        "droid": E(name="droid", robot_name="franka_panda", max_finger_width=0.08, fps=15, to_opencv=_DROID_TO_OPENCV),
        "fractal": E(
            name="fractal",
            robot_name="google_robot",
            max_finger_width=0.05,
            fps=3,
            to_opencv=_GOOGLE_ROBOT_TO_OPENCV,
            camera_fov_deg=69.0,
            camera_aspect=320 / 256,
        ),
        "robomind_franka": E(
            name="robomind_franka", robot_name="franka_panda", max_finger_width=0.08, fps=10, to_opencv=franka_to_opencv
        ),
        "robomind_franka_dual": E(
            name="robomind_franka_dual",
            robot_name="franka_panda",
            max_finger_width=0.08,
            fps=10,
            action_format=ActionFormat.DUAL_ARM_20D,
            to_opencv=franka_to_opencv,
            dual_base_left=np.array(
                [[1, 0, 0, 0.0], [0, 1, 0, 0.3], [0, 0, 1, 0.0], [0, 0, 0, 1.0]],
                dtype=np.float32,
            ),
            dual_base_right=np.array(
                [[1, 0, 0, 0.0], [0, 1, 0, -0.3], [0, 0, 1, 0.0], [0, 0, 0, 1.0]],
                dtype=np.float32,
            ),
        ),
        "robomind_ur": E(name="robomind_ur", robot_name="ur5e", max_finger_width=0.085, fps=10, to_opencv=eye3),
        "umi": E(name="umi", robot_name="", max_finger_width=0.08, fps=10),
        "hand_pose": E(
            name="hand_pose", robot_name="", max_finger_width=0.0, fps=15, action_format=ActionFormat.UNIFIED_57D
        ),
        "hwb_egoverse": E(
            name="hwb_egoverse", robot_name="", max_finger_width=0.0, fps=15, action_format=ActionFormat.UNIFIED_57D
        ),
        "av": E(name="av", robot_name="", max_finger_width=0.0, fps=10, action_format=ActionFormat.EGO_9D),
        "camera": E(
            name="camera",
            robot_name="",
            max_finger_width=0.0,
            fps=10,
            action_format=ActionFormat.EGO_9D,
            camera_fov_deg=69.0,
            camera_aspect=640 / 480,
        ),
        "embodiment_c_gripper": E(
            name="embodiment_c_gripper",
            robot_name="embodiment_c",
            max_finger_width=0.12,
            fps=10,
            camera_fov_deg=69.0,
            camera_aspect=640 / 480,
            robot_embodiment_type="embodiment_c_gripper",
            to_opencv=AGIBOT_GEAR_GRIPPER_TO_OPENCV_BY_WRIST,
        ),
        "embodiment_c_gripper_ext": E(
            name="embodiment_c_gripper_ext",
            robot_name="embodiment_c",
            max_finger_width=0.12,
            fps=10,
            camera_fov_deg=69.0,
            camera_aspect=640 / 480,
            robot_embodiment_type="embodiment_c_gripper_ext",
            to_opencv=AGIBOT_GEAR_GRIPPER_TO_OPENCV_BY_WRIST,
        ),
        "agibotworld_beta": E(
            name="agibotworld_beta",
            robot_name="embodiment_c",
            max_finger_width=0.12,
            fps=10,
            camera_fov_deg=69.0,
            camera_aspect=640 / 480,
            robot_embodiment_type="embodiment_c_gripper",
            to_opencv=AGIBOT_GEAR_GRIPPER_TO_OPENCV_BY_WRIST,
        ),
    }


# ── Color palettes ────────────────────────────────────────────────────────────

PALETTE_GT = {
    "ego": (39, 174, 96),  # green (matches right/left for ego-only datasets)
    "ego_top": (39, 174, 96),
    "right": (39, 174, 96),  # green
    "left": (46, 204, 113),  # light green
}

PALETTE_PRED = {
    "ego": (231, 76, 60),  # red (matches right/left for ego-only datasets)
    "ego_top": (192, 57, 43),
    "right": (231, 76, 60),  # red
    "left": (192, 57, 43),  # dark red
}

# ── Safetensors I/O ───────────────────────────────────────────────────────────


def _load_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Load all tensors from a safetensors file as float32 numpy arrays."""
    from safetensors.numpy import load_file

    return {k: np.asarray(v, dtype=np.float32) for k, v in load_file(str(path)).items()}


def _extract_action(data: dict[str, np.ndarray], target_dim: int | None = None) -> np.ndarray:
    """Return (T, D) action array from a safetensors dict.

    If ``target_dim`` is given, truncate the trailing dim to that size — model
    predictions are saved at the full untruncated dim while GT is already
    truncated to ``raw_action_dim``.
    """
    if "action" not in data:
        raise KeyError(f"No 'action' key found. Available: {sorted(data)}")
    a = data["action"]
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    a = a.astype(np.float32)
    if target_dim is not None and a.shape[-1] > target_dim:
        a = a[..., :target_dim]
    return a


def _extract_pose(data: dict[str, np.ndarray], key: str) -> np.ndarray | None:
    """Extract a (4,4) pose matrix from a safetensors dict, or None."""
    v = data.get(key)
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 3 and v.shape[0] == 1:
        v = v[0]
    if v.shape == (4, 4):
        return v
    return None


# ── Action denormalization (inverse of training-time normalization) ──────────

_NORMALIZER_DIR = Path(__file__).resolve().parents[1] / "normalizers"

# Dataset name → (normalizer JSON filename, stats key, method).
# Both eval GT and pred are saved AFTER normalization, so the viewer must
# invert it to get raw body-frame deltas back.
_DATASET_NORMALIZER: dict[str, tuple[str, str, str]] = {
    "bridge": ("bridge_orig_lerobot_backward_framewise_rot6d.json", "global", "quantile"),
    "droid": ("droid_lerobot_backward_framewise_rot6d.json", "global", "quantile"),
    "fractal": ("fractal_backward_framewise_rot6d.json", "global", "quantile"),
    "robomind_franka": ("robomind-franka_backward_framewise_rot6d.json", "global", "quantile"),
    "robomind_franka_dual": ("robomind-franka-dual_backward_framewise_rot6d.json", "global", "quantile"),
    "robomind_ur": ("robomind-ur_backward_framewise_rot6d.json", "global", "quantile"),
    "embodiment_c_gripper": ("embodiment_c_gripper_backward_framewise_rot6d.json", "global", "quantile"),
    "embodiment_c_gripper_ext": ("embodiment_c_gripper_backward_framewise_rot6d.json", "global", "quantile"),
    "hand_pose": ("hand_pose_backward_framewise_rot6d.json", "global", "quantile"),
    "hwb_egoverse": ("hand_pose_backward_framewise_rot6d.json", "global", "quantile"),
}


def _load_norm_stats(dataset_name: str | None) -> tuple[dict[str, np.ndarray], str] | None:
    """Return (stats, method) for ``dataset_name`` or None if unknown."""
    if dataset_name is None:
        return None
    cfg = _DATASET_NORMALIZER.get(dataset_name)
    if cfg is None:
        return None
    fname, key, method = cfg
    path = _NORMALIZER_DIR / fname
    if not path.exists():
        log.warning(f"Normalizer JSON not found: {path}")
        return None
    with path.open() as f:
        raw = json.load(f)
    block = raw.get(key, raw)
    stat_keys = {"mean", "std", "min", "max", "q01", "q99"}
    stats = {k: np.asarray(v, dtype=np.float32) for k, v in block.items() if k in stat_keys}
    return stats, method


def _denormalize_action(action: np.ndarray, stats: dict[str, np.ndarray], method: str) -> np.ndarray:
    """Inverse of training normalization. Returns raw body-frame action."""
    D = action.shape[-1]
    if method == "quantile":
        q01, q99 = stats["q01"][:D], stats["q99"][:D]
        denom = np.maximum(q99 - q01, 1e-8)
        return (action + 1.0) / 2.0 * denom + q01
    if method == "minmax":
        lo, hi = stats["min"][:D], stats["max"][:D]
        denom = np.maximum(hi - lo, 1e-8)
        return (action + 1.0) / 2.0 * denom + lo
    if method == "meanstd":
        mean, std = stats["mean"][:D], np.maximum(stats["std"][:D], 1e-8)
        return action * std + mean
    raise ValueError(f"Unknown normalization method: {method!r}")


# ── Auto action-format inference ──────────────────────────────────────────────

_DIM_TO_FORMAT: dict[int, ActionFormat] = {
    9: ActionFormat.EGO_9D,
    10: ActionFormat.SINGLE_ARM_10D,
    20: ActionFormat.DUAL_ARM_20D,
    57: ActionFormat.UNIFIED_57D,
}


def _infer_action_format(action: np.ndarray) -> ActionFormat:
    dim = int(action.shape[-1])
    fmt = _DIM_TO_FORMAT.get(dim)
    if fmt is None:
        raise ValueError(
            f"Cannot infer action format from trailing dim {dim}. "
            f"Known: {sorted(_DIM_TO_FORMAT)}. "
            "Pass --action-format explicitly."
        )
    return fmt


# ── State building ────────────────────────────────────────────────────────────


def _build_state_from_action(
    action: np.ndarray,
    gt_data: dict[str, np.ndarray],
    entry: DatasetEntry | None,
    action_format: ActionFormat,
) -> SceneState:
    """Convert a raw action array to a render-ready SceneState.

    Uses initial pose and pose convention from the dataset entry (or GT data).
    The same initial pose is applied to both GT and pred states so trajectories
    start at the same anchor point.
    """
    initial_pose = _extract_pose(gt_data, "initial_pose")
    if initial_pose is None:
        initial_pose = np.eye(4, dtype=np.float32)
    initial_pose_right = _extract_pose(gt_data, "initial_pose_right")
    initial_pose_left = _extract_pose(gt_data, "initial_pose_left")

    pose_convention = entry.pose_convention if entry else "backward_framewise"
    right_base_pose = entry.dual_base_right if entry else None
    left_base_pose = entry.dual_base_left if entry else None

    if entry and entry.to_unified_fn:
        import inspect as _inspect

        converter = _load_symbol(entry.to_unified_fn)
        params = _inspect.signature(converter).parameters
        embodiment_type = entry.robot_embodiment_type or str(entry.dataset_kwargs.get("embodiment_type", ""))
        if "embodiment_type" in params:
            unified = converter({"action": action}, embodiment_type=embodiment_type)
        elif "kind" in params:
            unified = converter(action, kind="gripper")
        else:
            unified = converter(action)
    else:
        unified = to_unified(action, action_format=action_format)

    uses_dual = action_format is ActionFormat.DUAL_ARM_20D
    if uses_dual and initial_pose_left is None:
        initial_pose_left = initial_pose

    return build_scene_state(
        unified,
        initial_pose=initial_pose,
        initial_pose_right=initial_pose_right,
        initial_pose_left=initial_pose_left,
        right_base_pose=right_base_pose,
        left_base_pose=left_base_pose,
        pose_convention=pose_convention,
    )


# ── Sample discovery ──────────────────────────────────────────────────────────


@dataclass
class SamplePair:
    gt_path: Path
    pred_path: Path
    label: str
    dataset_key: str | None = None  # canonical key ("bridge" from "bridge_20260416")
    gt_video_path: Path | None = None
    pred_video_path: Path | None = None


def _dataset_key_from_versioned(versioned_name: str) -> str:
    """Strip the ``_YYYYMMDD`` suffix from a versioned dataset name.

    e.g. ``"bridge_20260416"`` → ``"bridge"``,
         ``"robomind_franka_dual_20260414"`` → ``"robomind_franka_dual"``.
    Falls back to the input if no version suffix is found.
    """
    return re.sub(r"_\d{8}$", "", versioned_name)


def _load_mp4_frames(path: Path) -> np.ndarray | None:
    """Decode an mp4 to ``(T, H, W, 3)`` uint8 RGB. Returns None on failure."""
    if path is None or not path.exists():
        return None
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        log.warning("cv2 not available — video panels disabled")
        return None
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        log.warning(f"No frames decoded from {path}")
        return None
    return np.stack(frames, axis=0).astype(np.uint8)


_GT_FILENAMES = ("batch_data.safetensors", "output.safetensors")


def _find_gt_file(gt_batch_dir: Path) -> Path | None:
    """Return the first GT safetensors file present in ``gt_batch_dir``."""
    for fname in _GT_FILENAMES:
        candidate = gt_batch_dir / fname
        if candidate.exists():
            return candidate
    return None


def _discover_pairs(eval_dir: Path) -> list[SamplePair]:
    """Walk an eval job directory and collect (gt, pred) safetensors pairs.

    Expected layout (Lepton/orchestrated eval output):
        eval_dir/
        ├── outputs/{ds}/{mode}/{batch_id}/[{sample?}/]output.safetensors
        └── ground_truth/{ds}/{mode}/{batch_id}/{batch_data,output}.safetensors
    """
    outputs_root = eval_dir / "outputs"
    gt_root = eval_dir / "ground_truth"

    if not outputs_root.exists():
        raise FileNotFoundError(f"outputs/ dir not found under {eval_dir}")
    if not gt_root.exists():
        raise FileNotFoundError(f"ground_truth/ dir not found under {eval_dir}")

    pairs: list[SamplePair] = []
    for pred_file in sorted(outputs_root.rglob("output.safetensors")):
        rel = pred_file.relative_to(outputs_root)
        parts = rel.parts  # (ds, mode, batch_id, [seed], output.safetensors)
        if len(parts) < 4:
            continue
        ds, mode, batch_id = parts[0], parts[1], parts[2]
        seed = parts[3] if len(parts) == 5 else None
        gt_file = _find_gt_file(gt_root / ds / mode / batch_id)
        if gt_file is None:
            log.warning(f"No GT found for {pred_file.relative_to(eval_dir)}")
            continue
        label = f"{ds}/{mode}/{batch_id}" + (f"/{seed}" if seed else "")

        # Optional video files: pred = vision.mp4 next to the pred safetensors;
        # GT = first *.mp4 in the GT batch dir (typically vision_gt.mp4).
        pred_video = pred_file.parent / "vision.mp4"
        gt_videos = sorted((gt_root / ds / mode / batch_id).glob("*.mp4"))
        pairs.append(
            SamplePair(
                gt_path=gt_file,
                pred_path=pred_file,
                label=label,
                dataset_key=_dataset_key_from_versioned(ds),
                gt_video_path=gt_videos[0] if gt_videos else None,
                pred_video_path=pred_video if pred_video.exists() else None,
            )
        )

    return pairs


# ── Viewer session ────────────────────────────────────────────────────────────


def _collect_scene_points_both(gt_state: SceneState, pred_state: SceneState) -> np.ndarray:
    """Collect all trajectory points from both states for camera framing."""
    points: list[np.ndarray] = []
    for state in (gt_state, pred_state):
        for poses in (state.ego_poses, state.right_poses, state.left_poses):
            if poses is not None and len(poses) > 0:
                points.append(poses[:, :3, 3].astype(np.float32))
    return np.concatenate(points, axis=0) if points else np.zeros((1, 3), dtype=np.float32)


def _format_compare_text(gt_state: SceneState, pred_state: SceneState, t: int) -> str:
    """Show per-step GT vs pred action values and per-component L2 error."""
    if t == 0:
        return "*t=0: anchor pose (identity)*"

    gt_raw = gt_state.action_raw
    pred_raw = pred_state.action_raw
    if gt_raw is None or pred_raw is None:
        return ""
    idx = t - 1
    if idx >= len(gt_raw) or idx >= len(pred_raw):
        return ""

    g = gt_raw[idx]
    p = pred_raw[idx]
    mask = gt_state.mask
    min_d = min(len(g), len(p))
    diff = p[:min_d] - g[:min_d]
    mse_total = float(np.mean(diff**2))

    lines = [
        f"step {idx} → {t}",
        f"MSE (all dims): {mse_total:.6f}",
        "═" * 32,
    ]

    def _fmt3(v: np.ndarray) -> str:
        return "  ".join(f"{x:+.4f}" for x in v[:3])

    def _mse3(a: np.ndarray, b: np.ndarray) -> str:
        return f"err={np.mean((b - a) ** 2):.5f}"

    if mask.ego and min_d >= 9:
        lines += [
            f"Ego pos  GT {_fmt3(g[0:3])}",
            f"         Pr {_fmt3(p[0:3])}  {_mse3(g[0:3], p[0:3])}",
        ]
    if mask.right_wrist and min_d >= 18:
        lines += [
            "",
            f"R wrist  GT {_fmt3(g[9:12])}",
            f"         Pr {_fmt3(p[9:12])}  {_mse3(g[9:12], p[9:12])}",
        ]
        if gt_state.gripper_right is not None and t < len(gt_state.gripper_right):
            gr_gt = float(gt_state.gripper_right[t])
            gr_pr = float(pred_state.gripper_right[t]) if pred_state.gripper_right is not None else float("nan")
            lines.append(f"  grip   GT {gr_gt:+.4f}  Pr {gr_pr:+.4f}")
    if mask.left_wrist and min_d >= 42:
        lines += [
            "",
            f"L wrist  GT {_fmt3(g[33:36])}",
            f"         Pr {_fmt3(p[33:36])}  {_mse3(g[33:36], p[33:36])}",
        ]
        if gt_state.gripper_left is not None and t < len(gt_state.gripper_left):
            gl_gt = float(gt_state.gripper_left[t])
            gl_pr = float(pred_state.gripper_left[t]) if pred_state.gripper_left is not None else float("nan")
            lines.append(f"  grip   GT {gl_gt:+.4f}  Pr {gl_pr:+.4f}")

    return "```\n" + "\n".join(lines) + "\n```"


# ── Main viewer ───────────────────────────────────────────────────────────────


def _dummy_entry(action_format: ActionFormat) -> DatasetEntry:
    """Stub entry for samples whose dataset isn't in the registry.

    With ``robot_name=""`` the renderer skips the URDF mesh + IK; trajectories
    and EE frames still render correctly.
    """
    return DatasetEntry(
        name="unknown",
        robot_name="",
        max_finger_width=0.05,
        fps=10,
        action_format=action_format,
    )


def launch_compare_viewer(
    pairs: list[SamplePair],
    entry: DatasetEntry | None,
    action_format_override: ActionFormat | None,
    port: int = 8014,
    share: bool = False,
    denorm_gt: bool = False,
    denorm_pred: bool = False,
    datasets: dict[str, DatasetEntry] | None = None,
    on_share_url: Callable[[str], None] | None = None,
    idle_timeout_s: int | None = None,
) -> None:
    """Launch the interactive compare viewer.

    Args:
        on_share_url: Called once with the public share URL when share=True.
        idle_timeout_s: Exit after this many seconds with no connected clients
            (only starts counting after the first client has connected and left).
    """
    import threading as _threading

    import viser

    server = viser.ViserServer(host="0.0.0.0", port=port)

    @dataclass
    class ViewerSession:
        time_slider: Any
        speed_slider: Any
        load_lock: Any = field(default_factory=_threading.Lock)
        gt_state: SceneState | None = None
        pred_state: SceneState | None = None
        playing: bool = False
        last_frame_time: float = 0.0

    sessions: dict[int, ViewerSession] = {}
    sessions_lock = _threading.Lock()
    idle_state = {"had_connection": False, "last_activity": _time.time()}

    @server.on_client_connect
    def _(client) -> None:
        client.scene.reset()
        client.scene.set_up_direction("+z")
        client.gui.reset()

        gt_renderer = UnifiedRenderer(client, name_prefix="/gt", palette=PALETTE_GT)
        pred_renderer = UnifiedRenderer(client, name_prefix="/pred", palette=PALETTE_PRED)

        with client.gui.add_folder("Sample"):
            sample_labels = [p.label for p in pairs]
            sample_dropdown = client.gui.add_dropdown("Sample", options=sample_labels, initial_value=sample_labels[0])
            status_text = client.gui.add_markdown("*Ready*")

        with client.gui.add_folder("Display", expand_by_default=False):
            show_robot = client.gui.add_checkbox("Show robot mesh", initial_value=True)
            show_frames = client.gui.add_checkbox("Show wrist frames", initial_value=True)
            show_traj = client.gui.add_checkbox("Show trajectory", initial_value=True)
            show_fingertips = client.gui.add_checkbox("Show fingertips", initial_value=True)
            show_gt = client.gui.add_checkbox("Show GT", initial_value=True)
            show_pred = client.gui.add_checkbox("Show Pred", initial_value=True)
            axis_scale = client.gui.add_slider("Axis scale", min=0.1, max=20.0, step=0.1, initial_value=1.0)

        with client.gui.add_folder("Playback"):
            time_slider = client.gui.add_slider("Time", min=0, max=1, step=1, initial_value=0)
            play_button = client.gui.add_button("▶ Play")
            speed_slider = client.gui.add_slider("Speed (fps)", min=1, max=30, step=1, initial_value=3)

        with client.gui.add_folder("Camera", expand_by_default=True):
            image_panel = client.gui.add_image(np.zeros((64, 64, 3), dtype=np.uint8))
        gt_renderer.set_video_panel(image_panel)

        with client.gui.add_folder("GT vs Pred", expand_by_default=True):
            compare_text = client.gui.add_markdown("*No sample loaded*")

        with client.gui.add_folder("Legend", expand_by_default=False):
            client.gui.add_markdown("**GT** trajectories 🟢 green | **Pred** trajectories 🔴 red")

        show = {
            "frames": True,
            "traj": True,
            "fingertips": True,
            "ego": True,
            "robot": True,
            "robot_frame_filters": {},
            "gt": True,
            "pred": True,
        }
        session = ViewerSession(time_slider=time_slider, speed_slider=speed_slider)

        def _effective_show(renderer_key: str) -> dict:
            """Build show dict with ego/frames/traj gated by per-renderer toggle.

            Only the pred renderer draws the robot mesh — GT contributes only
            its trajectory + frames so we have one mesh that follows pred.
            """
            s = dict(show)
            if not show.get(renderer_key, True):
                s["frames"] = False
                s["traj"] = False
                s["fingertips"] = False
                s["ego"] = False
                s["robot"] = False
            if renderer_key == "gt":
                s["robot"] = False  # always hide GT mesh; pred mesh is the only one
            return s

        def _update_both(t: int) -> None:
            if session.gt_state is not None:
                gt_renderer.update(t, _effective_show("gt"))
            if session.pred_state is not None:
                pred_renderer.update(t, _effective_show("pred"))
            if session.gt_state is not None and session.pred_state is not None:
                compare_text.content = _format_compare_text(session.gt_state, session.pred_state, t)

        def do_load() -> None:
            label = sample_dropdown.value
            idx = sample_labels.index(label)
            pair = pairs[idx]
            status_text.content = f"⏳ Loading {label}..."
            try:
                gt_data = _load_safetensors(pair.gt_path)
                pred_data = _load_safetensors(pair.pred_path)

                gt_action = _extract_action(gt_data)
                # Pred is saved at the full model action dim; truncate to GT's dim.
                pred_action = _extract_action(pred_data, target_dim=int(gt_action.shape[-1]))

                # Resolve the dataset entry: explicit --dataset override beats
                # the per-pair auto-detection from the eval directory layout.
                # Unknown datasets fall back to a dummy entry that renders
                # trajectories + frames without a robot mesh.
                pair_entry = entry
                if pair_entry is None and datasets is not None and pair.dataset_key is not None:
                    pair_entry = datasets.get(pair.dataset_key)

                # Denormalization is per-side: depends on the eval's training
                # config (action_normalization). Toggle via --denorm-gt /
                # --denorm-pred CLI flags.
                ds_key = pair_entry.name if pair_entry else pair.dataset_key
                norm = _load_norm_stats(ds_key) if ds_key else None
                if norm is not None and (denorm_gt or denorm_pred):
                    stats, method = norm
                    if denorm_gt:
                        gt_action = _denormalize_action(gt_action, stats, method)
                    if denorm_pred:
                        pred_action = _denormalize_action(pred_action, stats, method)

                has_custom_converter = pair_entry is not None and pair_entry.to_unified_fn is not None
                if action_format_override:
                    fmt = action_format_override
                elif has_custom_converter:
                    fmt = ActionFormat.SINGLE_ARM_10D  # dummy — custom converter owns raw-format handling
                else:
                    fmt = _infer_action_format(gt_action)

                gt_state = _build_state_from_action(gt_action, gt_data, pair_entry, fmt)
                pred_state = _build_state_from_action(pred_action, gt_data, pair_entry, fmt)

                # Load GT camera video
                gt_state.video = _load_mp4_frames(pair.gt_video_path)

                session.gt_state = gt_state
                session.pred_state = pred_state

                _entry = pair_entry or _dummy_entry(fmt)
                fov_deg = _entry.camera_fov_deg
                gt_renderer.load(gt_state, _entry, to_opencv=_entry.to_opencv)
                pred_renderer.load(pred_state, _entry, to_opencv=_entry.to_opencv)

                T = max(gt_state.T, pred_state.T)
                time_slider.max = max(T, 1)
                time_slider.value = 0

                all_pts = _collect_scene_points_both(gt_state, pred_state)
                _reset_camera_to_trajectory_both(client, all_pts, gt_state, fov_deg)

                gt_renderer.update_axis_scale(axis_scale.value)
                pred_renderer.update_axis_scale(axis_scale.value)
                _update_both(0)

                status_text.content = f"✅ {label} | GT T={gt_state.T} | Pred T={pred_state.T} | format={fmt.value}"
                log.info(f"Loaded pair: {label}, format={fmt.value}")

            except Exception as e:
                status_text.content = f"❌ {e}"
                log.error(f"Load failed: {e}")
                import traceback

                traceback.print_exc()

        def _load_threaded() -> None:
            if not session.load_lock.acquire(blocking=False):
                return

            def _run() -> None:
                try:
                    do_load()
                finally:
                    session.load_lock.release()

            _threading.Thread(target=_run, daemon=True).start()

        @sample_dropdown.on_update
        def _(_) -> None:
            _load_threaded()

        @time_slider.on_update
        def _(_) -> None:
            _update_both(time_slider.value)

        @play_button.on_click
        def _(_) -> None:
            session.playing = not session.playing
            session.last_frame_time = _time.time()
            play_button.label = "⏸ Pause" if session.playing else "▶ Play"

        @show_robot.on_update
        def _(_) -> None:
            show["robot"] = show_robot.value
            _update_both(time_slider.value)

        @show_frames.on_update
        def _(_) -> None:
            show["frames"] = show_frames.value
            _update_both(time_slider.value)

        @show_traj.on_update
        def _(_) -> None:
            show["traj"] = show_traj.value
            _update_both(time_slider.value)

        @show_fingertips.on_update
        def _(_) -> None:
            show["fingertips"] = show_fingertips.value
            _update_both(time_slider.value)

        @show_gt.on_update
        def _(_) -> None:
            show["gt"] = show_gt.value
            _update_both(time_slider.value)

        @show_pred.on_update
        def _(_) -> None:
            show["pred"] = show_pred.value
            _update_both(time_slider.value)

        @axis_scale.on_update
        def _(_) -> None:
            gt_renderer.update_axis_scale(axis_scale.value)
            pred_renderer.update_axis_scale(axis_scale.value)

        with sessions_lock:
            sessions[client.client_id] = session
            idle_state["had_connection"] = True
            idle_state["last_activity"] = _time.time()
        _load_threaded()

    @server.on_client_disconnect
    def _(client) -> None:
        with sessions_lock:
            sessions.pop(client.client_id, None)
            idle_state["last_activity"] = _time.time()

    log.info(f"✅ Compare viewer ready at http://0.0.0.0:{port}")
    if share:
        share_url = server.request_share_url()
        if share_url:
            log.info(f"🌐 Share URL: {share_url}")
            if on_share_url is not None:
                on_share_url(share_url)

    try:
        while True:
            now = _time.time()
            with sessions_lock:
                active = list(sessions.values())
                if idle_timeout_s is not None:
                    _had = idle_state["had_connection"]
                    _idle = now - idle_state["last_activity"]
                    _empty = len(sessions) == 0
            if idle_timeout_s is not None and _had and _empty and _idle > idle_timeout_s:
                log.info(f"No active connections for {idle_timeout_s}s — shutting down.")
                break
            for session in active:
                if not session.playing:
                    continue
                if session.gt_state is None and session.pred_state is None:
                    continue
                frame_period = 1.0 / max(float(session.speed_slider.value), 1.0)
                if now - session.last_frame_time < frame_period:
                    continue
                T = max(
                    session.gt_state.T if session.gt_state else 0,
                    session.pred_state.T if session.pred_state else 0,
                )
                t = (session.time_slider.value + 1) % max(T, 1)
                session.time_slider.value = t
                session.last_frame_time = now
            _time.sleep(0.02)
    except KeyboardInterrupt:
        log.info("Shutting down.")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _reset_camera_to_trajectory_both(
    client: Any,
    all_points: np.ndarray,
    gt_state: SceneState,
    camera_fov_deg: float,
) -> None:
    """Frame the viewport around all trajectory points from both states."""
    from cosmos_framework.data.vfm.action.urdf_visualizer.viewer import (
        _get_observation_forward_direction,
        _get_observation_up_direction,
    )

    center = all_points.mean(axis=0)
    extent = all_points - center[None, :]
    radius = float(np.linalg.norm(extent, axis=1).max()) if len(all_points) > 0 else 0.0
    radius = max(radius, 0.15)

    fov_rad = float(np.deg2rad(camera_fov_deg))
    fit_distance = radius / max(np.tan(fov_rad / 2.0), 0.35)

    view_forward = _get_observation_forward_direction(gt_state)
    if view_forward is None:
        view_dir = np.array([1.0, -1.0, 0.7], dtype=np.float32)
        view_dir /= np.linalg.norm(view_dir)
        camera_position = center + view_dir * max(fit_distance * 1.35, 0.5)
        view_forward = center - camera_position
        view_forward /= np.linalg.norm(view_forward)
    else:
        camera_position = center - view_forward * max(fit_distance * 1.35, 0.5)

    view_forward = center - camera_position
    view_forward /= np.linalg.norm(view_forward)
    up_direction = _get_observation_up_direction(gt_state, view_forward)

    camera = client.camera
    deadline = _time.time() + 1.0
    while getattr(camera._state, "update_timestamp", 0.0) == 0.0 and _time.time() < deadline:
        _time.sleep(0.01)
    if getattr(camera._state, "update_timestamp", 0.0) == 0.0:
        return

    camera.fov = fov_rad
    camera.up_direction = tuple(up_direction.tolist())
    camera.position = tuple(camera_position.tolist())
    camera.look_at = tuple(center.tolist())
    client.flush()


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GT vs predicted action trajectories")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--eval-dir",
        type=Path,
        help="Eval job directory containing outputs/ and ground_truth/ subdirs",
    )
    group.add_argument(
        "--gt",
        type=Path,
        help="Path to ground-truth batch_data.safetensors",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        help="Path to predicted output.safetensors (required with --gt)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Optional dataset key (e.g. bridge, droid). When omitted, auto-detected"
            " per sample from the eval-dir layout (ground_truth/<ds_versioned>/...)."
            " Pass to pin a single registry entry for all samples."
        ),
    )
    parser.add_argument(
        "--action-format",
        choices=[fmt.value for fmt in ActionFormat],
        default=None,
        help="Override the raw action format (default: inferred from tensor shape)",
    )
    parser.add_argument("--port", type=int, default=8014)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--denorm-gt",
        action="store_true",
        help="Denormalize GT action (use when training experiment had action_normalization!=None).",
    )
    parser.add_argument(
        "--denorm-pred",
        action="store_true",
        help="Denormalize pred action (use when training experiment had action_normalization!=None).",
    )
    args = parser.parse_args()

    if args.gt is not None and args.pred is None:
        parser.error("--pred is required when --gt is specified")

    # ── Build pairs list ──
    if args.eval_dir is not None:
        pairs = _discover_pairs(args.eval_dir)
        if not pairs:
            log.error(f"No (GT, pred) pairs found under {args.eval_dir}")
            raise SystemExit(1)
        log.info(f"Found {len(pairs)} sample pairs in {args.eval_dir}")
    else:
        gt_path: Path = args.gt
        pred_path: Path = args.pred
        if not gt_path.exists():
            parser.error(f"GT file not found: {gt_path}")
        if not pred_path.exists():
            parser.error(f"Pred file not found: {pred_path}")
        label = gt_path.parent.name
        pairs = [SamplePair(gt_path=gt_path, pred_path=pred_path, label=label)]

    # ── Build dataset registry (used for per-pair auto-detection from eval-dir
    # layout) and optionally pin to a single entry via --dataset. Fall back to
    # the hardcoded minimal entries when the full registry can't be built
    try:
        datasets = _build_datasets()
    except Exception as e:
        log.warning(f"Could not build full dataset registry ({e}); using hardcoded minimal entries.")
        datasets = _build_minimal_entries()

    entry: DatasetEntry | None = None
    if args.dataset:
        entry = datasets.get(args.dataset)
        if entry is None:
            log.warning(
                f"Dataset '{args.dataset}' not found. Known: {sorted(datasets)}. Running without robot metadata."
            )

    action_format_override = ActionFormat(args.action_format) if args.action_format else None

    launch_compare_viewer(
        pairs=pairs,
        entry=entry,
        action_format_override=action_format_override,
        port=args.port,
        share=args.share,
        denorm_gt=args.denorm_gt,
        denorm_pred=args.denorm_pred,
        datasets=datasets,
    )


if __name__ == "__main__":
    main()
