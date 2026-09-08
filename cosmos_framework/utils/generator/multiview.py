# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import torch


def safe_multiview_camera_name(camera_key: str, view_index: int) -> str:
    safe_camera_key = "".join(ch if ch.isalnum() else "_" for ch in camera_key).strip("_")
    return f"view{view_index:02d}_{safe_camera_key or 'camera'}"


def split_multiview_tensor_by_view(
    tensor: torch.Tensor,
    sample_n_views: int,
    num_video_frames_per_view: int,
) -> torch.Tensor | None:  # tensor: [B,C,V*F,H,W] or [C,V*F,H,W], returns [V,C,F,H,W] or None
    if tensor.dim() == 5:
        if tensor.shape[0] != 1:
            return None
        view_tensor = tensor[0]  # [C,V*F,H,W]
    elif tensor.dim() == 4:
        view_tensor = tensor  # [C,V*F,H,W]
    else:
        return None

    expected_num_frames = sample_n_views * num_video_frames_per_view
    if view_tensor.shape[1] != expected_num_frames:
        return None

    view_tensor = view_tensor.reshape(
        view_tensor.shape[0],
        sample_n_views,
        num_video_frames_per_view,
        view_tensor.shape[2],
        view_tensor.shape[3],
    )  # [C,V,F,H,W]
    return view_tensor.permute(1, 0, 2, 3, 4).contiguous()  # [V,C,F,H,W]


def require_multiview_tensor_by_view(
    tensor: torch.Tensor,
    sample_n_views: int,
    num_video_frames_per_view: int,
) -> torch.Tensor:  # tensor: [B,C,V*F,H,W] or [C,V*F,H,W], returns [V,C,F,H,W]
    video_by_view = split_multiview_tensor_by_view(
        tensor,
        sample_n_views,
        num_video_frames_per_view,
    )  # [V,C,F,H,W] or None
    if video_by_view is None:
        raise ValueError(
            "Expected multiview tensor shape [B,C,V*F,H,W] with B=1 or [C,V*F,H,W], "
            f"got shape={tuple(tensor.shape)}, sample_n_views={sample_n_views}, "
            f"num_video_frames_per_view={num_video_frames_per_view}."
        )
    return video_by_view


def split_multiview_video_by_view(
    video: torch.Tensor,
    *,
    sample_n_views: int,
    num_video_frames_per_view: int,
) -> list[torch.Tensor]:  # video: [B,C,V*F,H,W] or [C,V*F,H,W], returns list[[C,F,H,W]]
    video_by_view = require_multiview_tensor_by_view(
        video,
        sample_n_views,
        num_video_frames_per_view,
    )  # [V,C,F,H,W]
    return [video_by_view[view_idx].contiguous() for view_idx in range(sample_n_views)]  # list[[C,F,H,W]]


def decode_multiview_latent_per_view(
    decode: Callable[[torch.Tensor], torch.Tensor],
    latent: torch.Tensor,
    sample_n_views: int,
    num_video_frames_per_view: int,
) -> torch.Tensor:  # latent: [B,C,V*T_latent,H,W] or [C,V*T_latent,H,W], returns same rank with T=V*F
    """Decode camera-major latent clips independently and concatenate their pixels."""
    if latent.ndim not in (4, 5):
        raise ValueError(
            f"Multiview latents must have shape [B,C,T,H,W] or [C,T,H,W], got shape {tuple(latent.shape)}."
        )

    temporal_dim = latent.ndim - 3
    num_latent_frames = int(latent.shape[temporal_dim])
    if num_latent_frames % sample_n_views != 0:
        raise ValueError(
            "Multiview latent length must be divisible by sample_n_views: "
            f"got T={num_latent_frames}, sample_n_views={sample_n_views}."
        )

    latent_frames_per_view = num_latent_frames // sample_n_views
    decoded_views: list[torch.Tensor] = []
    for view_idx in range(sample_n_views):
        view_latent = latent.narrow(  # [B,C,T_latent,H,W] or [C,T_latent,H,W]
            temporal_dim,
            view_idx * latent_frames_per_view,
            latent_frames_per_view,
        )
        decoded_view = decode(view_latent)  # [B,C,F,H_pixel,W_pixel] or [C,F,H_pixel,W_pixel]
        if decoded_view.ndim != latent.ndim:
            raise ValueError(
                "Decoded multiview tensors must preserve the latent rank: "
                f"got latent shape {tuple(view_latent.shape)} and decoded shape {tuple(decoded_view.shape)}."
            )
        if decoded_view.shape[temporal_dim] != num_video_frames_per_view:
            raise ValueError(
                "Decoded camera clip length must match num_video_frames_per_view: "
                f"got T={decoded_view.shape[temporal_dim]}, expected {num_video_frames_per_view}."
            )
        decoded_views.append(decoded_view)

    return torch.cat(decoded_views, dim=temporal_dim)  # [B,C,V*F,H_pixel,W_pixel] or [C,V*F,H_pixel,W_pixel]


def load_multiview_media_pixels(
    path: Path,
    *,
    target_h: int,
    target_w: int,
    max_frames: int,
    keep: str = "first",
    video_loader: Callable[..., torch.Tensor],
    image_loader: Callable[..., torch.Tensor],
    video_extensions: frozenset[str],
    image_extensions: frozenset[str],
) -> torch.Tensor:  # returns [3,T,H,W]
    suffix = path.suffix.lower()
    if suffix in video_extensions:
        frames_normalized = video_loader(
            path,
            target_h=target_h,
            target_w=target_w,
            max_frames=max_frames,
            keep=cast(Literal["first", "last"], keep),
        )  # [3,T,H,W]
        frames_uint8 = ((frames_normalized + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)  # [3,T,H,W]
        return frames_uint8
    if suffix not in image_extensions:
        raise ValueError(f"Unsupported multiview media extension: {path.suffix!r} ({path}).")
    image_uint8 = image_loader(path, target_h=target_h, target_w=target_w)  # [3,H,W]
    return image_uint8.unsqueeze(1)  # [3,1,H,W]


def pad_multiview_view_video(
    frames: torch.Tensor | None,
    *,
    num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:  # frames: [3,T,H,W] or None, returns [3,F,H,W]
    video = torch.full((3, num_frames, height, width), 128, dtype=torch.uint8)  # [3,F,H,W]
    if frames is None:
        return video
    if (
        frames.dim() != 4
        or int(frames.shape[0]) != 3
        or int(frames.shape[2]) != height
        or int(frames.shape[3]) != width
    ):
        raise ValueError(
            "Expected multiview frames shape [3,T,H,W] matching the requested output size, "
            f"got {tuple(frames.shape)} for H={height}, W={width}."
        )
    t_fill = min(int(frames.shape[1]), num_frames)
    if t_fill <= 0:
        return video
    video[:, :t_fill] = frames[:, :t_fill]  # [3,t_fill,H,W]
    if t_fill < num_frames:
        video[:, t_fill:] = video[:, t_fill - 1 : t_fill].expand(
            -1,
            num_frames - t_fill,
            -1,
            -1,
        )  # [3,F-t_fill,H,W]
    return video


def build_camera_major_video(view_videos: list[torch.Tensor], *, device: Any) -> torch.Tensor:  # returns [1,3,V*F,H,W]
    if not view_videos:
        raise ValueError("Multiview inference requires at least one camera view.")
    camera_major_video = torch.cat(view_videos, dim=1).unsqueeze(0).to(device=device)  # [1,3,V*F,H,W]
    return camera_major_video


def normalize_multiview_control_weights(weights: list[float]) -> list[float]:
    """Return normalized per-control weights for the multiview transfer batch."""
    if not weights:
        raise ValueError("Multiview transfer inference requires an explicit transfer hint such as wsm={}.")
    if any(weight < 0 for weight in weights):
        raise ValueError(f"control_weights must all be non-negative, got {weights}.")
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError(f"control_weights must have a positive sum, got {weights}.")
    return [weight / total_weight for weight in weights]


def slice_multiview_view_frames(
    frames: torch.Tensor,
    *,
    start_frame: int,
    end_frame: int,
) -> torch.Tensor:  # frames: [3,T,H,W], returns [3,T_slice,H,W]
    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError(f"Invalid multiview frame slice [{start_frame}, {end_frame}).")
    total_frames = int(frames.shape[1])
    if start_frame >= total_frames:
        raise ValueError(
            f"Caption frame_range starts at frame {start_frame}, but the multiview media has only {total_frames} frames."
        )
    clipped_end_frame = min(end_frame, total_frames)
    return frames[:, start_frame:clipped_end_frame]  # [3,T_slice,H,W]


def generated_multiview_condition_frames(
    view_videos: list[torch.Tensor],
    *,
    num_condition_frames: int,
) -> list[torch.Tensor | None]:  # view_videos: list[[3,F,H,W]], returns list[[3,F_cond,H,W]]
    """Convert decoded generated pixels in [0, 1] into uint8 feedback condition frames."""
    if num_condition_frames <= 0:
        return [None for _ in view_videos]
    condition_frames: list[torch.Tensor | None] = []
    for view_video in view_videos:
        tail = view_video[:, -num_condition_frames:]  # [3,F_cond,H,W]
        condition = (tail * 255.0).round().clamp(0, 255).to(torch.uint8)  # [3,F_cond,H,W], [0,1] -> uint8
        condition_frames.append(condition)
    return condition_frames
