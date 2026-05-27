# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from cosmos_framework.inference.vision import read_media_frames
from cosmos_framework.data.vfm.action.domain_utils import get_domain_id
from cosmos_framework.data.vfm.action.transforms import (
    find_closest_target_size,
    pad_action_to_max_dim,
    reflection_pad_to_target,
)
from cosmos_framework.data.vfm.sequence_packing import SequencePlan
from cosmos_framework.utils.vfm.data_utils import get_vision_data_resolution

_FORWARD_DYNAMICS_MODE = "forward_dynamics"


def _camera_trajectory_to_action(camera_trajectory: str, target_frames: int) -> torch.Tensor:
    """Parse a camera trajectory string into a per-frame relative action tensor.

    Returns a ``[target_frames - 1, raw_action_dim]`` float tensor whose
    convention matches the training-time CameraDatasetSharded settings.
    """
    from projects.cosmos3.vfm.evaluation.action.eval_camera import (
        CameraTrainParams,
        _trajectory_to_action,
        parse_camera_string,
    )

    # Multi-trajectory ('|'-separated) batching is an eval-only feature; inference
    # processes one trajectory per sample.
    segment = camera_trajectory.split("|", 1)[0].strip()
    if not segment:
        raise ValueError("camera_trajectory is empty")

    poses_abs = parse_camera_string(segment)
    action_np = _trajectory_to_action(
        poses_abs,
        target_frames,
        CameraTrainParams(
            rotation_format="rot6d",
            pose_convention="backward_framewise",
            translation_scale=10.0,
            rotation_scale=1.0,
            num_frames=target_frames,
        ),
    )
    return torch.from_numpy(action_np).float()


def _build_camera_batch(
    video: torch.Tensor,
    action: torch.Tensor,
    raw_action_dim: int,
    prompt: str,
    domain_name: str,
    action_chunk_size: int,
    fps: int,
    resolution: str | None,
    input_video_key: str,
    batch_size: int,
    device: torch.device | str,
    condition_first_frame: bool,
    duration_template: str | None = None,
    resolution_template: str | None = None,
) -> dict:
    """Assemble the FDM inference batch dict.

    Mirrors ``cosmos3.action.build_action_batch`` with ``mode`` hardcoded to
    ``"forward_dynamics"`` so this module doesn't need to import ``ActionMode``.
    """
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

    duration_seconds = int(num_frames / fps) if fps > 0 else 0
    ai_caption = prompt.strip()
    if duration_template:
        ai_caption += duration_template.format(duration=duration_seconds, fps=fps)
    if resolution_template:
        ai_caption += resolution_template.format(height=target_h, width=target_w)

    return {
        input_video_key: [[video_padded]] * batch_size,
        "action": [[action]] * batch_size,
        "raw_action_dim": [torch.tensor(raw_action_dim, dtype=torch.long)] * batch_size,
        "mode": [_FORWARD_DYNAMICS_MODE] * batch_size,
        "ai_caption": [ai_caption] * batch_size,
        "prompt": [prompt] * batch_size,
        "conditioning_fps": [torch.tensor(fps, dtype=torch.long)] * batch_size,
        "image_size": padded_image_size.unsqueeze(0).to(device=device),
        "domain_id": [torch.tensor(get_domain_id(domain_name), dtype=torch.long)] * batch_size,
        "sequence_plan": [sequence_plan] * batch_size,
    }


def get_camera_sample_data(
    input_video_key: str,
    batch_size: int,
    prompt: str,
    vision_path: Path | None,
    vision_size: tuple[int, int],
    camera_trajectory: str,
    domain_name: str,
    resolution: str,
    action_chunk_size: int,
    max_action_dim: int,
    fps: int,
    device: torch.device | str,
    duration_template: str | None = None,
    resolution_template: str | None = None,
) -> dict:
    """Build the inference batch for AR generation with camera-trajectory conditioning."""
    if vision_path is not None:
        frames, _ = read_media_frames(Path(vision_path), max_frames=action_chunk_size + 1)
    else:
        target_w, target_h = vision_size
        frames = torch.zeros(3, action_chunk_size + 1, target_h, target_w, dtype=torch.uint8)

    raw = _camera_trajectory_to_action(camera_trajectory, target_frames=action_chunk_size + 1)
    action = pad_action_to_max_dim(raw, max_action_dim)
    raw_action_dim = raw.shape[-1]

    return _build_camera_batch(
        video=frames,
        action=action,
        raw_action_dim=raw_action_dim,
        prompt=prompt,
        domain_name=domain_name,
        action_chunk_size=action_chunk_size,
        fps=fps,
        resolution=resolution,
        input_video_key=input_video_key,
        batch_size=batch_size,
        device=device,
        condition_first_frame=vision_path is not None,
        duration_template=duration_template,
        resolution_template=resolution_template,
    )
