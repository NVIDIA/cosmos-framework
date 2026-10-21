# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Camera-trajectory conditioning frontend for autoregressive inference.

Builds the model-visible batch for camera-pose conditioned generation, where the
per-frame action is derived from a camera trajectory rather than read from an
action file.

The trajectory maths itself lives in
``cosmos_framework.data.generator.action.utils.camera_trajectory``, which is the
same module the native runtime uses. Keeping the primitives shared -- rather than
transcribed -- is what makes camera conditioning reproduce across runtimes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cosmos_framework.data.generator.action.utils.action_processing import (
    ActionProcessingRecord,
    make_batched_action_processing_fields,
    make_camera_global_asinh_normalizer,
    make_pose_action_scale_normalizer,
    pad_action_to_max_dim,
)
from cosmos_framework.data.generator.action.utils.camera_trajectory import (
    CameraTrainParams,
    load_custom_pose_jsons,
    parse_camera_string,
    trajectory_to_action,
)
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.utils.pose_utils import PoseConvention
from cosmos_framework.data.generator.action.utils.transforms import (
    find_closest_target_size,
    reflection_pad_to_target,
)
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.inference.args import CameraActionNormalization
from cosmos_framework.inference.vision import crop_media_to_content, read_media_frames
from cosmos_framework.utils.generator.data_utils import get_vision_data_resolution

_FORWARD_DYNAMICS_MODE = "forward_dynamics"


def _is_pose_json_spec(camera_trajectory: str) -> bool:
    """Return True if ``camera_trajectory`` is a path to a JSON pose file."""
    spec = camera_trajectory.strip()
    return bool(spec) and spec.endswith(".json")


def _resolve_camera_trajectory_poses(camera_trajectory: str, target_frames: int) -> np.ndarray:
    """Resolve a camera trajectory spec into absolute ``(target_frames, 4, 4)`` c2w poses.

    ``camera_trajectory`` is either a command string parsed by
    ``parse_camera_string`` (e.g. ``"w-300"``) or a path to a JSON file decoding
    to a ``(T, 4, 4)`` cam2world (OpenCV) trajectory anchored at ``T_0 = I``.

    The parsed trajectory is padded (repeating the last pose) or truncated to
    exactly ``target_frames`` so the returned poses match the frames fed to the
    model.
    """
    if _is_pose_json_spec(camera_trajectory):
        # One trajectory per inference sample: load the single (T,4,4) c2w JSON.
        poses_abs = load_custom_pose_jsons(camera_trajectory.strip())[0]
    else:
        # Multi-trajectory ('|'-separated) batching is an eval-only feature;
        # inference processes one trajectory per sample.
        segment = camera_trajectory.split("|", 1)[0].strip()
        if not segment:
            raise ValueError("camera_trajectory is empty")
        poses_abs = parse_camera_string(segment)

    t_have = poses_abs.shape[0]
    if t_have < target_frames:
        pad = np.tile(poses_abs[-1:], (target_frames - t_have, 1, 1))
        poses_abs = np.concatenate([poses_abs, pad], axis=0)
    elif t_have > target_frames:
        poses_abs = poses_abs[:target_frames]
    return poses_abs


def _camera_trajectory_to_action(
    poses_abs: np.ndarray,
    target_frames: int,
    pose_convention: PoseConvention,
) -> torch.Tensor:
    """Convert absolute ``(target_frames, 4, 4)`` c2w poses into a per-frame action tensor.

    Returns a ``[target_frames - 1, raw_action_dim]`` float tensor whose
    convention matches the training-time camera dataset settings.
    """
    if pose_convention == "absolute":
        raise ValueError("camera trajectory inference requires a relative pose convention")

    action_np = trajectory_to_action(
        poses_abs,
        target_frames,
        CameraTrainParams(
            rotation_format="rot6d",
            pose_convention=pose_convention,
            translation_scale=1.0,
            rotation_scale=1.0,
            num_frames=target_frames,
        ),
    )
    return torch.from_numpy(action_np).float()


def build_camera_batch(
    *,
    video: torch.Tensor,
    action: torch.Tensor,
    action_processing_record: ActionProcessingRecord,
    prompt: str,
    domain_name: str,
    action_chunk_size: int,
    fps: int,
    resolution: str | None,
    content_crop_box: tuple[int, int, int, int] | None = None,
    input_video_key: str,
    batch_size: int,
    device: torch.device | str,
    condition_first_frame: bool,
    duration_template: str | None = None,
    resolution_template: str | None = None,
) -> dict:
    """Assemble the forward-dynamics inference batch dict for camera conditioning."""
    video = crop_media_to_content(video, content_crop_box)
    target_frames = action_chunk_size + 1
    _, num_frames, h, w = video.shape

    if num_frames < target_frames:
        pad = video[:, -1:].repeat(1, target_frames - num_frames, 1, 1)
        video = torch.cat([video, pad], dim=1)
    elif num_frames > target_frames:
        video = video[:, :target_frames]

    if resolution is None:
        resolution = get_vision_data_resolution((h, w))

    target_w, target_h = find_closest_target_size(h, w, resolution)
    pad_dict: dict[str, Any] = {"video": video}
    reflection_pad_to_target(pad_dict, ["video"], keep_aspect_ratio=True, target_w=target_w, target_h=target_h)
    video_padded = pad_dict["video"]
    padded_image_size = pad_dict["image_size"]

    # FDM SequencePlan: all actions are clean conditioning; vision conditioning
    # depends on whether frame 0 is anchored (I2V) or denoised (T2V).
    # action_start_frame_offset=1 because action_length == video_length - 1
    # (action a{i} guides v{i} -> v{i+1}).
    sequence_plan = SequencePlan(
        has_text=True,
        has_vision=True,
        has_action=True,
        condition_frame_indexes_vision=[0] if condition_first_frame else [],
        condition_frame_indexes_action=list(range(action_chunk_size)),
        action_start_frame_offset=1,
    )

    duration_seconds = int(target_frames / fps) if fps > 0 else 0
    ai_caption = prompt.strip()
    if duration_template:
        ai_caption += duration_template.format(duration=duration_seconds, fps=fps)
    if resolution_template:
        ai_caption += resolution_template.format(height=target_h, width=target_w)

    return {
        input_video_key: [[video_padded]] * batch_size,
        "action": [[action]] * batch_size,
        **make_batched_action_processing_fields(action_processing_record, batch_size),
        "mode": [_FORWARD_DYNAMICS_MODE] * batch_size,
        "ai_caption": [ai_caption] * batch_size,
        "prompt": [prompt] * batch_size,
        "conditioning_fps": [torch.tensor(fps, dtype=torch.long)] * batch_size,
        "image_size": padded_image_size.unsqueeze(0).to(device=device),
        "domain_id": [torch.tensor(get_domain_id(domain_name), dtype=torch.long)] * batch_size,
        "sequence_plan": [sequence_plan] * batch_size,
    }


def get_camera_sample_data(
    *,
    input_video_key: str,
    batch_size: int,
    prompt: str,
    vision_path: Path | None,
    vision_size: tuple[int, int],
    camera_trajectory: str,
    domain_name: str,
    resolution: str,
    content_crop_box: tuple[int, int, int, int] | None = None,
    action_chunk_size: int,
    max_action_dim: int,
    camera_translation_scale: float,
    camera_action_normalization: CameraActionNormalization,
    fps: int,
    device: torch.device | str,
    duration_template: str | None = None,
    resolution_template: str | None = None,
    output_dir: Path | None = None,
    camera_pose_convention: PoseConvention = "backward_framewise",
) -> dict:
    """Build the inference batch for AR generation with camera-trajectory conditioning.

    When ``output_dir`` is provided, the resolved absolute ``(T, 4, 4)`` c2w
    trajectory that feeds the model is written to
    ``output_dir/camera_trajectory.json``.
    """
    target_frames = action_chunk_size + 1
    if vision_path is not None:
        frames, _ = read_media_frames(Path(vision_path), max_frames=target_frames)  # [C,T,H,W]
    else:
        target_w, target_h = vision_size
        frames = torch.zeros(3, target_frames, target_h, target_w, dtype=torch.uint8)  # [C,T,H,W]

    poses_c2w = _resolve_camera_trajectory_poses(camera_trajectory, target_frames=target_frames)  # [T,4,4]
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pose_file = output_dir / "camera_trajectory.json"
        pose_file.write_text(json.dumps(poses_c2w.tolist()))

    raw = _camera_trajectory_to_action(
        poses_c2w,
        target_frames=target_frames,
        pose_convention=camera_pose_convention,
    )  # [T,D_raw]
    raw_action_dim = raw.shape[-1]
    if camera_action_normalization == "global_asinh":
        if camera_translation_scale != 1.0:
            raise ValueError("global_asinh camera actions require camera_translation_scale=1.0")
        action_normalizer = make_camera_global_asinh_normalizer(pose_convention=camera_pose_convention)
    else:
        action_normalizer = make_pose_action_scale_normalizer(
            int(raw_action_dim),
            translation_scale=camera_translation_scale,
            rotation_scale=1.0,
        )
    action = action_normalizer.normalize_action(raw)  # [T,D_raw]
    action = pad_action_to_max_dim(action, max_action_dim)  # [T,D_model]
    action_processing_record = ActionProcessingRecord(
        raw_action_dim=int(raw_action_dim),
        action_normalizer=action_normalizer,
    )

    return build_camera_batch(
        video=frames,
        action=action,
        action_processing_record=action_processing_record,
        prompt=prompt,
        domain_name=domain_name,
        action_chunk_size=action_chunk_size,
        fps=fps,
        resolution=resolution,
        content_crop_box=content_crop_box,
        input_video_key=input_video_key,
        batch_size=batch_size,
        device=device,
        condition_first_frame=vision_path is not None,
        duration_template=duration_template,
        resolution_template=resolution_template,
    )
