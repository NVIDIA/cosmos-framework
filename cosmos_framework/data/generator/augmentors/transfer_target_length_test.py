# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


import json

import pytest
import torch

import cosmos_framework.data.generator.augmentors.text_tokenizer as text_tokenizer
from cosmos_framework.data.generator.augmentors.duration_fps_text_timestamps import DurationFPSTextTimeStamps
from cosmos_framework.data.generator.augmentors.interleaved_video_parsing import (
    VideoTransferAlignedSelectedControlParsing,
)
from cosmos_framework.data.generator.augmentors.text_transforms_for_video import (
    TextTransformForVideoTransferChunkedFrames,
)


def _make_target_parser(target_num_frames: int = 81) -> VideoTransferAlignedSelectedControlParsing:
    return VideoTransferAlignedSelectedControlParsing(
        input_keys=["metas", "video"],
        args={
            "max_num_frames": 201,
            "min_num_frames": 1,
            "target_num_frames": target_num_frames,
            "teacher_forcing_frames_per_chunk": 1,
            "min_stride": 1,
            "max_stride": 3,
            "min_fps": 5.0,
            "max_fps": 60.0,
            "seek_mode": "exact",
        },
    )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("target_num_frames,source_num_frames", [(81, 200), (137, 400), (201, 400)])
def test_target_parser_retains_entire_caption_span(target_num_frames: int, source_num_frames: int) -> None:
    parser = _make_target_parser(target_num_frames)
    indices, stride = parser._sample_frame_indices_for_chunk(500, 20, 20 + source_num_frames)

    assert len(indices) == len(set(indices)) == target_num_frames
    assert indices[0] == 20
    assert indices[-1] == 20 + source_num_frames - 1
    assert indices == sorted(indices)
    assert stride == pytest.approx((source_num_frames - 1) / (target_num_frames - 1))


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "start,end,min_stride",
    [(-1, 199, None), (400, 600, None), (0, 80, None), (0, 300, None), (0, 200, 3)],
)
def test_target_parser_rejects_partial_caption_windows_or_invalid_stride(
    start: int, end: int, min_stride: int | None
) -> None:
    indices, _ = _make_target_parser()._sample_frame_indices_for_chunk(500, start, end, min_stride)
    assert indices == []


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("target_num_frames", [0, 8, 82, 205])
def test_target_parser_rejects_invalid_target_lengths(target_num_frames: int) -> None:
    with pytest.raises(ValueError, match="target_num_frames"):
        _make_target_parser(target_num_frames)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("latent_frames", range(1, 52))
def test_every_latent_bucket_accepts_its_complete_caption_window(latent_frames: int) -> None:
    target_num_frames = 1 + 4 * (latent_frames - 1)
    parser = _make_target_parser(target_num_frames)
    indices, stride = parser._sample_frame_indices_for_chunk(240, 10, 10 + target_num_frames)
    assert indices == list(range(10, 10 + target_num_frames))
    assert stride == 1.0

    caption_transform = TextTransformForVideoTransferChunkedFrames(
        input_keys=["metas"],
        args={"caption_config": {"captions": 1.0}, "target_num_frames": target_num_frames, "min_num_frames": 1},
    )
    assert caption_transform._supports_target_length(
        10, 10 + target_num_frames, {}, {"nb_frames": 240, "framerate": 30.0}
    )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("source_num_frames", [2, 3, 4, 81, 201])
def test_one_frame_bucket_rejects_multi_frame_caption_windows(source_num_frames: int) -> None:
    parser = _make_target_parser(1)
    assert parser._sample_frame_indices_for_chunk(240, 10, 10 + source_num_frames)[0] == []
    caption_transform = TextTransformForVideoTransferChunkedFrames(
        input_keys=["metas"],
        args={"caption_config": {"captions": 1.0}, "target_num_frames": 1, "min_num_frames": 1},
    )
    assert not caption_transform._supports_target_length(
        10, 10 + source_num_frames, {}, {"nb_frames": 240, "framerate": 30.0}
    )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("target_num_frames", [1, 5])
def test_short_bucket_selects_only_its_complete_caption(target_num_frames: int) -> None:
    narrative = {"description": "A red door.", "duration": "8s", "fps": 24.0}
    transform = TextTransformForVideoTransferChunkedFrames(
        input_keys=["metas"],
        args={"caption_config": {"captions": 1.0}, "target_num_frames": target_num_frames, "min_num_frames": 1},
    )
    sample = {
        "metas": {
            "framerate": 30.0,
            "nb_frames": 201,
            "captions": {
                "caption_structured": json.dumps(
                    {
                        "long": {"start_frame": 0, "end_frame": 201, "caption": json.dumps({"description": "Wrong"})},
                        "short": {
                            "start_frame": 7,
                            "end_frame": 7 + target_num_frames,
                            "caption": json.dumps(narrative),
                        },
                    }
                )
            },
        },
    }
    transformed = transform(sample)
    assert transformed is not None
    assert transformed["sampled_chunk_key"] == "short"
    assert (transformed["chunk_start_frame"], transformed["chunk_end_frame"]) == (7, 7 + target_num_frames)
    assert json.loads(transformed["ai_caption"]) == {"description": narrative["description"]}


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("control_num_frames", [299, 300])
def test_target_length_selects_matching_caption_and_rebuilds_metadata(
    monkeypatch: pytest.MonkeyPatch, control_num_frames: int
) -> None:
    narrative = {
        "description": "The door opens, a person enters, and the door closes.",
        "duration": "8s",
        "fps": 24.0,
    }
    caption_transform = TextTransformForVideoTransferChunkedFrames(
        input_keys=["metas"],
        args={
            "caption_config": {"captions": 1.0},
            "target_num_frames": 81,
            "min_stride": 1,
            "max_stride": 3,
            "min_fps": 5.0,
            "max_fps": 60.0,
        },
    )
    sample = {
        "metas": {
            "framerate": 30.0,
            "nb_frames": 500,
            "width": 12,
            "height": 8,
            "captions": {
                "caption_structured": json.dumps(
                    {
                        "short": {"start_frame": 0, "end_frame": 9, "caption": json.dumps({"description": "Wrong"})},
                        "complete": {"start_frame": 100, "end_frame": 300, "caption": json.dumps(narrative)},
                    }
                )
            },
        },
        "video": b"rgb",
        "persisted_control": b"control",
        "_persisted_control_meta": {"framerate": 30.0, "nb_frames": 500, "width": 12, "height": 8},
        "_selected_control_modality": "seg",
        "__url__": "url",
        "__key__": "key",
    }
    transformed = caption_transform(sample)
    assert transformed is not None
    assert transformed["sampled_chunk_key"] == "complete"
    assert json.loads(transformed["ai_caption"]) == {"description": narrative["description"]}

    parser = _make_target_parser()
    monkeypatch.setattr(parser, "_validate_and_probe", lambda *_args: True)
    monkeypatch.setattr(parser, "_probe_video_len", lambda payload: 500 if payload == b"rgb" else control_num_frames)
    calls: list[list[int]] = []

    def fake_decode(
        _payload: bytes,
        indices: list[int],
        _transforms: object = None,
        _output_dtype: torch.dtype = torch.uint8,
    ) -> torch.Tensor:
        calls.append(list(indices))
        return torch.zeros(3, len(indices), 8, 12, dtype=torch.uint8)  # [C,T,H,W]

    monkeypatch.setattr(parser, "_decode_frames_at", fake_decode)
    parsed = parser(transformed)
    if control_num_frames < 300:
        assert parsed is None
        assert calls == []
        return

    assert parsed is not None
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert (calls[0][0], calls[0][-1]) == (100, 299)
    assert parsed["video"]["num_frames"] == 81
    effective_fps = 30.0 * 80 / 199
    assert parsed["video"]["conditioning_fps"] == pytest.approx(effective_fps)
    metadata_sample = {
        "ai_caption": parsed["ai_caption"],
        "video": parsed["video"]["video"],  # [C,T,H,W]
        "conditioning_fps": parsed["video"]["conditioning_fps"],
    }
    metadata_transform = DurationFPSTextTimeStamps(args={"fractional_duration": True, "skip_on_error": False})
    assert metadata_transform(metadata_sample) is metadata_sample
    expected_caption = json.dumps({"description": narrative["description"]}) + (
        f". The video is {81 / effective_fps:.1f} seconds long and is of {effective_fps:.0f} FPS."
    )
    assert metadata_sample["ai_caption"] == expected_caption
    tokenized_captions: list[str] = []

    class CaptionRecorder:
        def tokenize_text(self, caption: str, system_prompt: str) -> list[int]:
            tokenized_captions.append(caption)
            assert "control signals" in system_prompt
            return [11, 12, 13]

    monkeypatch.setattr(text_tokenizer, "lazy_instantiate", lambda _config: CaptionRecorder())
    tokenizer = text_tokenizer.TextTokenizerTransformForTransfer(
        input_keys=["ai_caption"], output_keys=["text_token_ids"], args={"tokenizer_config": {}}
    )
    assert tokenizer(metadata_sample) is metadata_sample
    assert tokenized_captions == [expected_caption]
    assert metadata_sample["text_token_ids"].tolist() == [11, 12, 13]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("fps,min_stride", [(5.0, 1), (30.0, 3)])
def test_target_caption_rejects_unsupported_effective_fps_and_source_stride(fps: float, min_stride: int) -> None:
    transform = TextTransformForVideoTransferChunkedFrames(
        input_keys=["metas"],
        args={"caption_config": {"captions": 1.0}, "target_num_frames": 81, "min_fps": 5.0, "max_stride": 3},
    )
    assert not transform._supports_target_length(
        0, 200, {"min_stride": min_stride}, {"nb_frames": 200, "framerate": fps}
    )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("target_num_frames", [None, 81])
def test_source_duration_fields_are_removed_only_for_exact_target_captions(target_num_frames: int | None) -> None:
    original_caption = {"description": "The person leaves.", "duration": "8s", "fps": 24.0}
    transform = TextTransformForVideoTransferChunkedFrames(
        input_keys=["metas"], args={"caption_config": {"captions": 1.0}, "target_num_frames": target_num_frames}
    )
    sample = {
        "metas": {
            "framerate": 30.0,
            "nb_frames": 200,
            "captions": {
                "caption_structured": json.dumps(
                    {"complete": {"start_frame": 0, "end_frame": 200, "caption": json.dumps(original_caption)}}
                )
            },
        }
    }
    transformed = transform(sample)
    assert transformed is not None
    expected_caption = (
        original_caption if target_num_frames is None else {"description": original_caption["description"]}
    )
    assert transformed["ai_caption"] == json.dumps(expected_caption)
