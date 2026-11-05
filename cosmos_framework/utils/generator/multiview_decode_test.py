# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""The frame arithmetic that makes a single decode call wrong for more than one camera."""

import pytest
import torch

from cosmos_framework.utils.generator.multiview import (
    decode_multiview_latent_per_view,
    split_multiview_video_by_view,
)

# Wan 2.2 4x16x16: get_pixel_num_frames(T) = (T - 1) * 4 + 1. The -3 is dropped once per CALL,
# which is the whole reason a camera-major latent cannot be decoded in one go.
_TEMPORAL_FACTOR = 4


def _fake_decode(latent: torch.Tensor) -> torch.Tensor:
    """A decode with the real temporal contract and nothing else."""
    temporal_dim = latent.ndim - 3
    num_latent_frames = latent.shape[temporal_dim]
    num_pixel_frames = (num_latent_frames - 1) * _TEMPORAL_FACTOR + 1
    shape = list(latent.shape)
    shape[temporal_dim] = num_pixel_frames
    return torch.zeros(shape, dtype=latent.dtype)


@pytest.mark.parametrize("sample_n_views", [2, 11])
def test_per_view_decode_yields_the_length_the_split_requires(sample_n_views: int) -> None:
    latent_frames_per_view = 75
    frames_per_view = (latent_frames_per_view - 1) * _TEMPORAL_FACTOR + 1  # 297

    latent = torch.zeros(1, 3, sample_n_views * latent_frames_per_view, 8, 8)
    decoded = decode_multiview_latent_per_view(_fake_decode, latent, sample_n_views, frames_per_view)

    assert decoded.shape[2] == sample_n_views * frames_per_view
    views = split_multiview_video_by_view(
        decoded[0], sample_n_views=sample_n_views, num_video_frames_per_view=frames_per_view
    )
    assert len(views) == sample_n_views
    assert all(view.shape[1] == frames_per_view for view in views)


def test_a_single_decode_call_is_short_by_three_frames_per_extra_camera() -> None:
    """Why the fix is needed, stated as arithmetic rather than as a crash.

    One call over the whole camera-major latent trims the VAE's factor_t - 1 frames once, not
    once per camera, so it returns 3 * (V - 1) frames more than the split accepts.
    """
    sample_n_views, latent_frames_per_view = 11, 75
    frames_per_view = (latent_frames_per_view - 1) * _TEMPORAL_FACTOR + 1

    latent = torch.zeros(1, 3, sample_n_views * latent_frames_per_view, 8, 8)
    one_call = _fake_decode(latent).shape[2]
    required = sample_n_views * frames_per_view

    assert one_call == 3297
    assert required == 3267
    assert one_call - required == _TEMPORAL_FACTOR - 1 == 3 or one_call - required == 3 * (sample_n_views - 1)

    with pytest.raises(ValueError):
        split_multiview_video_by_view(
            _fake_decode(latent)[0],
            sample_n_views=sample_n_views,
            num_video_frames_per_view=frames_per_view,
        )


def test_the_two_agree_for_a_single_camera() -> None:
    """Which is why this never surfaced: every exercised path so far had V=1."""
    latent_frames_per_view = 75
    frames_per_view = (latent_frames_per_view - 1) * _TEMPORAL_FACTOR + 1

    latent = torch.zeros(1, 3, latent_frames_per_view, 8, 8)
    assert _fake_decode(latent).shape[2] == 1 * frames_per_view
