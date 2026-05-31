# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from cosmos_framework.data.vfm.action.domain_utils import EMBODIMENT_TO_RAW_ACTION_DIM, get_domain_id
from cosmos_framework.data.vfm.action.json_formatter import ActionPromptJsonFormatter
from cosmos_framework.data.vfm.action.transforms import (
    build_sequence_plan_from_mode,
    find_closest_target_size,
    pad_action_to_max_dim,
    reflection_pad_to_target,
)
from cosmos_framework.inference.args import ModelMode
from cosmos_framework.inference.vision import read_media_frames
from cosmos_framework.utils.vfm.data_utils import get_vision_data_resolution


def _load_actions(
    action_path: Path | str | None,
    model_mode: ModelMode,
    action_chunk_size: int,
    max_action_dim: int,
    raw_action_dim: int | None,
) -> torch.Tensor:
    """Load actions from JSON (or zeros for policy mode and inverse dynamics mode). Returns padded action tensor."""
    match model_mode:
        case ModelMode.FORWARD_DYNAMICS:
            assert action_path is not None, "action_path is required for forward_dynamics mode"
            p = Path(str(action_path))
            raw = torch.tensor(json.loads(p.read_text()), dtype=torch.float32)
            raw_dim = raw.shape[-1]
            assert raw_dim == raw_action_dim, (
                f"Raw action dimension from file ({raw_dim}) does not match expected dimension ({raw_action_dim})"
            )
            return pad_action_to_max_dim(raw, max_action_dim)
        case ModelMode.POLICY | ModelMode.INVERSE_DYNAMICS:
            assert raw_action_dim is not None, "raw_action_dim is required for policy and inverse_dynamics modes"
            return torch.zeros(action_chunk_size, max_action_dim, dtype=torch.float32)
        case _:
            raise ValueError(f"Unsupported action model_mode: {model_mode}")


def _format_prompt(
    prompt: str,
    view_point: str,
    video: torch.Tensor,
    action: torch.Tensor,
    fps: torch.Tensor,
    image_size: torch.Tensor,
) -> str:
    """Helper function to build the action prompt with optional duration and resolution info."""
    data_dict = {
        "viewpoint": view_point,
        "ai_caption": prompt.strip(),
        "video": video,
        "action": action,
        "conditioning_fps": fps,
        "image_size": image_size,
    }
    prompt_json_formatter = ActionPromptJsonFormatter()
    ai_caption = prompt_json_formatter(data_dict)[prompt_json_formatter.caption_key]
    if isinstance(ai_caption, dict):
        ai_caption = json.dumps(ai_caption)
    return ai_caption


def build_action_batch(
    *,
    video: torch.Tensor,
    action: torch.Tensor,
    raw_action_dim: int,
    prompt: str,
    view_point: str,
    domain_name: str,
    model_mode: ModelMode,
    action_chunk_size: int,
    fps: int,
    resolution: str | None = None,
    input_video_key: str,
    batch_size: int = 1,
    device: Any = "cuda",
) -> dict:
    """Build an Action data batch from pre-loaded video and action tensors."""
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

    sequence_plan = build_sequence_plan_from_mode(
        mode=model_mode.value,
        video_length=target_frames,
        action_length=action_chunk_size,
        has_text=True,
    )

    ai_caption = _format_prompt(
        prompt=prompt,
        view_point=view_point,
        video=video_padded,
        action=action,
        fps=torch.tensor(fps, dtype=torch.long),
        image_size=padded_image_size,
    )

    return {
        input_video_key: [[video_padded]] * batch_size,
        "action": [[action]] * batch_size,
        "raw_action_dim": [torch.tensor(raw_action_dim, dtype=torch.long)] * batch_size,
        "mode": [model_mode.value] * batch_size,
        "ai_caption": [ai_caption] * batch_size,
        "prompt": [prompt] * batch_size,
        "conditioning_fps": [torch.tensor(fps, dtype=torch.long)] * batch_size,
        "image_size": padded_image_size.unsqueeze(0).to(device=device),
        "domain_id": [torch.tensor(get_domain_id(domain_name), dtype=torch.long)] * batch_size,
        "sequence_plan": [sequence_plan] * batch_size,
    }


def get_action_sample_data(
    model_config: Any,
    *,
    batch_size: int,
    prompt: str,
    vision_path: Path,
    model_mode: ModelMode,
    action_path: Path | None,
    domain_name: str,
    view_point: str = "ego_view",
    resolution: str,
    action_chunk_size: int,
    max_action_dim: int,
    fps: int,
    device: Any,
) -> dict:
    """Load observation image/video + optional actions and build an Action inference batch."""
    domain_name = domain_name.lower().strip()
    if domain_name not in EMBODIMENT_TO_RAW_ACTION_DIM:
        raise ValueError(
            f"invalid domain_name {domain_name!r}; expected one of {sorted(EMBODIMENT_TO_RAW_ACTION_DIM.keys())}"
        )

    raw_action_dim = EMBODIMENT_TO_RAW_ACTION_DIM[domain_name]
    frames, _ = read_media_frames(Path(vision_path), max_frames=action_chunk_size + 1)
    assert action_path is not None or raw_action_dim is not None, (
        "Either action_path or raw_action_dim must be provided"
    )
    action = _load_actions(action_path, model_mode, action_chunk_size, max_action_dim, raw_action_dim)

    return build_action_batch(
        video=frames,
        action=action,
        raw_action_dim=raw_action_dim,
        prompt=prompt,
        view_point=view_point,
        domain_name=domain_name,
        model_mode=model_mode,
        action_chunk_size=action_chunk_size,
        fps=fps,
        resolution=resolution,
        input_video_key=model_config.input_video_key,
        batch_size=batch_size,
        device=device,
    )
