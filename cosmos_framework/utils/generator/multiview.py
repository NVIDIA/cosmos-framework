# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from collections.abc import Callable

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
