# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.utils.generator.multiview import (
    build_camera_major_video,
    generated_multiview_condition_frames,
    normalize_multiview_control_weights,
    pad_multiview_view_video,
    slice_multiview_view_frames,
)

# ---------------------------------------------------------------------------
# pad_multiview_view_video
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.CPU
def test_pad_multiview_view_video_none_fills_128() -> None:
    result = pad_multiview_view_video(None, num_frames=4, height=8, width=8)
    assert result.shape == (3, 4, 8, 8)
    assert result.dtype == torch.uint8
    assert result.unique().tolist() == [128]


@pytest.mark.L0
@pytest.mark.CPU
def test_pad_multiview_view_video_short_clip_repeats_last_frame() -> None:
    frames = torch.zeros(3, 2, 4, 4, dtype=torch.uint8)
    frames[:, 1, :, :] = 200  # last frame is 200
    result = pad_multiview_view_video(frames, num_frames=5, height=4, width=4)
    assert result.shape == (3, 5, 4, 4)
    # first frame copied verbatim
    assert result[:, 0, :, :].unique().tolist() == [0]
    # frames 2-4 should repeat the last frame (200)
    assert result[:, 2, :, :].unique().tolist() == [200]
    assert result[:, 3, :, :].unique().tolist() == [200]
    assert result[:, 4, :, :].unique().tolist() == [200]


@pytest.mark.L0
@pytest.mark.CPU
def test_pad_multiview_view_video_truncates_long_clip() -> None:
    frames = torch.zeros(3, 10, 4, 4, dtype=torch.uint8)
    result = pad_multiview_view_video(frames, num_frames=5, height=4, width=4)
    assert result.shape == (3, 5, 4, 4)


@pytest.mark.L0
@pytest.mark.CPU
def test_pad_multiview_view_video_raises_on_wrong_shape() -> None:
    frames = torch.zeros(3, 4, 6, 6, dtype=torch.uint8)  # H=6, W=6, but requesting H=4, W=4
    with pytest.raises(ValueError, match="Expected multiview frames shape"):
        pad_multiview_view_video(frames, num_frames=4, height=4, width=4)


# ---------------------------------------------------------------------------
# build_camera_major_video
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.CPU
def test_build_camera_major_video_concatenates_and_adds_batch_dim() -> None:
    v0 = torch.zeros(3, 4, 8, 8, dtype=torch.uint8)
    v1 = torch.ones(3, 4, 8, 8, dtype=torch.uint8) * 128
    result = build_camera_major_video([v0, v1], device="cpu")
    assert result.shape == (1, 3, 8, 8, 8)  # [1,3,V*F,H,W] = [1,3,8,8,8]
    # first 4 frames are from v0 (zeros), next 4 from v1 (128)
    assert result[0, :, :4, :, :].unique().tolist() == [0]
    assert result[0, :, 4:, :, :].unique().tolist() == [128]


@pytest.mark.L0
@pytest.mark.CPU
def test_build_camera_major_video_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="at least one camera view"):
        build_camera_major_video([], device="cpu")


# ---------------------------------------------------------------------------
# normalize_multiview_control_weights
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.CPU
def test_normalize_multiview_control_weights_normalizes_to_sum_one() -> None:
    result = normalize_multiview_control_weights([1.0, 3.0])
    assert abs(sum(result) - 1.0) < 1e-9
    assert abs(result[0] - 0.25) < 1e-9
    assert abs(result[1] - 0.75) < 1e-9


@pytest.mark.L0
@pytest.mark.CPU
def test_normalize_multiview_control_weights_single_weight() -> None:
    result = normalize_multiview_control_weights([5.0])
    assert result == [1.0]


@pytest.mark.L0
@pytest.mark.CPU
def test_normalize_multiview_control_weights_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="requires an explicit transfer hint"):
        normalize_multiview_control_weights([])


@pytest.mark.L0
@pytest.mark.CPU
def test_normalize_multiview_control_weights_raises_on_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        normalize_multiview_control_weights([1.0, -0.5])


@pytest.mark.L0
@pytest.mark.CPU
def test_normalize_multiview_control_weights_raises_on_zero_sum() -> None:
    with pytest.raises(ValueError, match="positive sum"):
        normalize_multiview_control_weights([0.0, 0.0])


# ---------------------------------------------------------------------------
# slice_multiview_view_frames
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.CPU
def test_slice_multiview_view_frames_clips_end_to_available_length() -> None:
    frames = torch.zeros(3, 6, 4, 4, dtype=torch.uint8)
    result = slice_multiview_view_frames(frames, start_frame=2, end_frame=100)
    assert result.shape == (3, 4, 4, 4)  # clipped to 6-2=4 frames


@pytest.mark.L0
@pytest.mark.CPU
def test_slice_multiview_view_frames_raises_on_invalid_range() -> None:
    frames = torch.zeros(3, 6, 4, 4, dtype=torch.uint8)
    with pytest.raises(ValueError, match="Invalid multiview frame slice"):
        slice_multiview_view_frames(frames, start_frame=3, end_frame=2)


@pytest.mark.L0
@pytest.mark.CPU
def test_slice_multiview_view_frames_raises_on_start_beyond_clip() -> None:
    frames = torch.zeros(3, 6, 4, 4, dtype=torch.uint8)
    with pytest.raises(ValueError, match="multiview media has only"):
        slice_multiview_view_frames(frames, start_frame=10, end_frame=15)


# ---------------------------------------------------------------------------
# generated_multiview_condition_frames
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.CPU
def test_generated_multiview_condition_frames_returns_all_none_when_zero() -> None:
    views = [torch.zeros(3, 4, 8, 8), torch.ones(3, 4, 8, 8)]
    result = generated_multiview_condition_frames(views, num_condition_frames=0)
    assert result == [None, None]


@pytest.mark.L0
@pytest.mark.CPU
def test_generated_multiview_condition_frames_takes_tail_frames() -> None:
    frames = torch.zeros(3, 5, 4, 4)
    frames[:, -2:, :, :] = 1.0  # last 2 frames are 1.0
    result = generated_multiview_condition_frames([frames], num_condition_frames=2)
    assert len(result) == 1
    assert result[0] is not None
    assert result[0].shape == (3, 2, 4, 4)


@pytest.mark.L0
@pytest.mark.CPU
def test_generated_multiview_condition_frames_converts_float_to_uint8() -> None:
    # A single frame with value 0.5 in [0,1] should round to 128 in uint8
    frames = torch.full((3, 1, 2, 2), 0.5)
    result = generated_multiview_condition_frames([frames], num_condition_frames=1)
    assert result[0] is not None
    assert result[0].dtype == torch.uint8
    # (0.5 * 255.0).round() = 128 (round-half-to-even gives 128 for 127.5)
    assert result[0].unique().tolist() == [128]


@pytest.mark.L0
@pytest.mark.CPU
def test_generated_multiview_condition_frames_rounding_matches_exact_formula() -> None:
    # Verify the exact round-trip: (tail * 255.0).round().clamp(0, 255).to(uint8)
    val = 0.996  # 0.996 * 255 = 253.98 → rounds to 254
    frames = torch.full((3, 2, 2, 2), val)
    result = generated_multiview_condition_frames([frames], num_condition_frames=1)
    assert result[0] is not None
    expected = int(round(val * 255.0))
    assert result[0].unique().tolist() == [expected]
