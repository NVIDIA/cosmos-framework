# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

from cosmos_framework.data.generator.augmentors.interleaved_video_parsing import (
    VideoTransferAlignedChunkedFramesParsing,
)


def _make_parser(min_num_frames: int = 17) -> VideoTransferAlignedChunkedFramesParsing:
    return VideoTransferAlignedChunkedFramesParsing(
        input_keys=["metas", "video"],
        args={
            "max_num_frames": 601,
            "min_num_frames": min_num_frames,
            "teacher_forcing_frames_per_chunk": 4,
            "max_stride": 3,
            "min_stride": 1,
        },
    )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "chunk_end,stride",
    [
        (4, 1),  # TF alignment leaves one source frame and one latent frame.
        (16, 1),  # TF alignment leaves one source frame, below the TF minimum.
        (48, 3),  # Striding leaves 16 candidates, which TF alignment reduces to one frame.
    ],
)
def test_chunked_parser_rejects_frame_plan_below_post_stride_minimum(
    monkeypatch: pytest.MonkeyPatch, chunk_end: int, stride: int
) -> None:
    parser = _make_parser()
    monkeypatch.setattr(parser, "_sample_stride_with_bias", lambda max_stride, min_stride: stride)

    frame_indices, sampled_stride = parser._sample_frame_indices_for_chunk(
        decoder_len=100,
        chunk_start=0,
        chunk_end=chunk_end,
    )

    assert frame_indices == []
    assert sampled_stride == stride


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "chunk_end,stride",
    [
        (17, 1),
        (49, 3),
    ],
)
def test_chunked_parser_accepts_frame_plan_at_post_stride_minimum(
    monkeypatch: pytest.MonkeyPatch, chunk_end: int, stride: int
) -> None:
    parser = _make_parser()
    monkeypatch.setattr(parser, "_sample_stride_with_bias", lambda max_stride, min_stride: stride)

    frame_indices, sampled_stride = parser._sample_frame_indices_for_chunk(
        decoder_len=100,
        chunk_start=0,
        chunk_end=chunk_end,
    )

    assert len(frame_indices) == 17
    assert sampled_stride == stride


@pytest.mark.L0
@pytest.mark.CPU
def test_chunked_parser_rejects_invalid_minimum() -> None:
    with pytest.raises(AssertionError, match="min_num_frames must be >= 1"):
        _make_parser(min_num_frames=0)


@pytest.mark.L0
@pytest.mark.CPU
def test_chunked_parser_aligns_recipe_maximum_to_teacher_forcing_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = _make_parser()
    monkeypatch.setattr(parser, "_sample_stride_with_bias", lambda max_stride, min_stride: 1)

    frame_indices, sampled_stride = parser._sample_frame_indices_for_chunk(
        decoder_len=601,
        chunk_start=0,
        chunk_end=601,
    )

    assert len(frame_indices) == 593
    assert sampled_stride == 1
    num_latent_frames = 1 + (len(frame_indices) - 1) // 4
    assert (num_latent_frames - 1) % parser.teacher_forcing_frames_per_chunk == 0
