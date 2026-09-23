# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Training labels and paired cropped audio share the video source clock."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from cosmos_framework.data.generator.augmentors.reasoner.timestamp import overlay_text
from cosmos_framework.data.generator.augmentors.reasoner.tokenize_data import TokenizeData
from cosmos_framework.data.generator.augmentors.reasoner.tokenize_data_test import (
    _FakeAudioProcessor,
    _FakeVLMProcessor,
)
from cosmos_framework.data.generator.processors.audio_utils import (
    AUDIO_END_TOKEN,
    AUDIO_PAD_TOKEN,
    AUDIO_START_TOKEN,
    get_audio_segment_token_lengths,
)
from cosmos_framework.data.generator.processors.base import maybe_parse_video_content
from cosmos_framework.utils.generator.video_source_metadata import calculate_video_timestamps

pytestmark = [pytest.mark.L1, pytest.mark.CPU]


@pytest.mark.parametrize("temporal_patch_size,expected", [(1, [1.0, 1.3, 1.7]), (2, [1.2, 1.2, 1.7])])
def test_overlay_uses_temporal_patch_size_not_spatial_merge(temporal_patch_size: int, expected: list[float]) -> None:
    frames = [Image.new("RGB", (64, 64))] * 3
    processor = SimpleNamespace(
        name="/local/edge", temporal_patch_size=temporal_patch_size, merge_size=4, USES_SOURCE_VIDEO_TIMESTAMPS=True
    )
    result, times = overlay_text(
        frames,
        4.0,
        processor=processor,
        video_metadata={"fps": 30.0, "total_num_frames": 100, "frames_indices": [30, 40, 50]},
    )
    assert result is frames
    assert times == expected


def test_cropped_audio_uses_same_origin_and_token_partition() -> None:
    processor = _FakeVLMProcessor()
    tokenize = TokenizeData(
        processor=processor,
        sound_und=True,
        audio_processor=_FakeAudioProcessor(token_lengths=(10,), timestamp_stride=0.11),
    )
    metadata = {"fps": 10.0, "total_num_frames": 100, "frames_indices": [50, 52, 56, 58]}
    data = {
        "__key__": "crop",
        "__url__": SimpleNamespace(root="test", path="crop"),
        "media": {
            "video": {
                "videos": [Image.new("RGB", (32, 32))] * 4,
                "fps": 4.0,
                "video_metadata": metadata,
                "audio": np.zeros(16000, dtype=np.float32),
                "audio_start_seconds": 5.0,
            }
        },
        "conversation": [
            {"role": "user", "content": [{"type": "video", "video": "video"}, {"type": "audio", "audio": "video"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ],
    }
    assert tokenize(data) is not None
    content = processor.last_conversation[0]["content"]
    assert content[0]["video_metadata"] == metadata
    assert (
        content[1]["text"]
        == f"{AUDIO_START_TOKEN}<5.1 seconds>{AUDIO_PAD_TOKEN * 4}<5.7 seconds>{AUDIO_PAD_TOKEN * 6}{AUDIO_END_TOKEN}"
    )


@pytest.mark.parametrize(
    "metadata,mode",
    [
        (None, "qwen_index"),
        ({"fps": 30.0, "total_num_frames": 100, "frames_indices": [0]}, "qwen_index"),
        ({"fps": 30.0, "total_num_frames": 100, "frames_indices": [0, 1, 2, 3]}, "legacy_fps"),
    ],
)
def test_tokenize_rejects_invalid_explicit_source_metadata(metadata: object, mode: str) -> None:
    processor = _FakeVLMProcessor()
    data = {
        "__key__": "invalid",
        "__url__": SimpleNamespace(root="test", path="invalid"),
        "media": {"video": {"videos": [Image.new("RGB", (32, 32))] * 4, "fps": 4.0, "video_metadata": metadata}},
        "conversation": [{"role": "user", "content": [{"type": "video", "video": "video"}]}],
    }
    with pytest.raises(ValueError, match="video_metadata"):
        TokenizeData(processor=processor, video_timestamp_mode=mode)(data)


def test_repeated_source_frames_partition_audio_without_losing_tokens() -> None:
    assert get_audio_segment_token_lengths(5, [1.0, 1.0, 1.2], audio_token_timestamps=[1.0, 1.05, 1.1, 1.15, 1.2]) == [
        1,
        2,
        2,
    ]
    with pytest.raises(ValueError, match="nondecreasing"):
        get_audio_segment_token_lengths(5, [1.2, 1.0])


def _multi_video_audio_sample(content_order: list[tuple[str, str]]) -> dict[str, Any]:
    media = {}
    for key, start, value in (("video_a", 50, 2.0), ("video_b", 200, 7.0)):
        media[key] = {
            "videos": [Image.new("RGB", (32, 32))] * 4,
            "fps": 4.0,
            "video_metadata": {
                "fps": 10.0,
                "total_num_frames": 300,
                "frames_indices": [start + index for index in (0, 2, 6, 8)],
            },
            "audio": np.full(320, value, dtype=np.float32),
            "audio_start_seconds": start / 10.0,
        }
    return {
        "__key__": "multi-video-crops",
        "__url__": SimpleNamespace(root="test", path="multi-video-crops"),
        "media": media,
        "conversation": [
            {"role": "user", "content": [{"type": kind, kind: key} for kind, key in content_order]},
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ],
    }


def test_multi_video_audio_uses_its_own_source_clock() -> None:
    processor = _FakeVLMProcessor()
    data = _multi_video_audio_sample(
        [("video", "video_a"), ("video", "video_b"), ("audio", "video_a"), ("audio", "video_b")]
    )
    output = TokenizeData(
        processor=processor,
        sound_und=True,
        audio_processor=_FakeAudioProcessor(token_lengths=(10, 10), timestamp_stride=0.11),
    )(data)
    assert output is not None
    content = processor.last_conversation[0]["content"]
    for item, start in zip(content[2:], (5, 20), strict=True):
        assert item["text"] == (
            f"{AUDIO_START_TOKEN}<{start}.1 seconds>{AUDIO_PAD_TOKEN * 4}"
            f"<{start}.7 seconds>{AUDIO_PAD_TOKEN * 6}{AUDIO_END_TOKEN}"
        )
    assert output["audio_features"][:, 0, 0].tolist() == [2.0, 7.0]


@pytest.mark.parametrize("audio_layout", ["separate_with_timestamps", "separate_no_timestamps", "interleaved_av"])
def test_audio_cannot_borrow_another_video_clock_before_its_own_video(audio_layout: str) -> None:
    processor = _FakeVLMProcessor()
    data = _multi_video_audio_sample([("video", "video_b"), ("audio", "video_a"), ("video", "video_a")])
    output = TokenizeData(
        processor=processor,
        sound_und=True,
        audio_layout=audio_layout,
        audio_processor=_FakeAudioProcessor(token_lengths=(10,)),
    )(data)
    assert output is None
    assert processor.last_conversation is None


def test_interleaved_audio_cannot_attach_to_an_unrelated_adjacent_video() -> None:
    processor = _FakeVLMProcessor()
    data = _multi_video_audio_sample([("video", "video_a"), ("video", "video_b"), ("audio", "video_a")])
    output = TokenizeData(
        processor=processor,
        sound_und=True,
        audio_layout="interleaved_av",
        audio_processor=_FakeAudioProcessor(token_lengths=(10,)),
    )(data)
    assert output is None
    assert processor.last_conversation is None


def test_interleaved_cropped_pairs_keep_each_clock_and_audio_feature_order() -> None:
    processor = _FakeVLMProcessor()
    data = _multi_video_audio_sample(
        [("video", "video_a"), ("audio", "video_a"), ("video", "video_b"), ("audio", "video_b")]
    )
    output = TokenizeData(
        processor=processor,
        sound_und=True,
        audio_layout="interleaved_av",
        audio_processor=_FakeAudioProcessor(token_lengths=(10, 10), timestamp_stride=0.11),
    )(data)
    assert output is not None
    tokens = processor.tokenizer.vocabulary
    chunk = [tokens[key] for key in ("<timestamp>", "<|vision_start|>", "<video>", "<|vision_end|>")]
    expected = []
    for count in (4, 6, 4, 6):
        expected.extend(chunk)
        expected.extend([tokens[AUDIO_START_TOKEN], *([tokens[AUDIO_PAD_TOKEN]] * count), tokens[AUDIO_END_TOKEN]])
    assert output["input_ids"].tolist() == [*expected, tokens["answer"]]
    assert output["audio_features"][:, 0, 0].tolist() == [2.0, 7.0]


@pytest.mark.parametrize("mode", ["qwen_index", "legacy_fps"])
def test_framewise_crop_repeats_pixels_and_uses_the_same_label_clock(mode: str) -> None:
    processor = _FakeVLMProcessor(name="Qwen3-test")
    frames = [Image.new("RGB", (32, 32), color=(value, 0, 0)) for value in (0, 100, 200)]
    video_media = {
        "videos": frames,
        "fps": 2.0,
        "source_frames_indices": [0, 10, 20],
        "source_fps": 30.0,
        "source_total_num_frames": 70,
    }
    metadata = {"fps": 30.0, "total_num_frames": 100, "frames_indices": [30, 40, 50]}
    if mode == "qwen_index":
        video_media["video_metadata"] = metadata
    data = {
        "__key__": "framewise-crop",
        "__url__": SimpleNamespace(root="test", path="framewise-crop"),
        "media": {"video": video_media},
        "conversation": [{"role": "user", "content": [{"type": "video", "video": "video"}]}],
    }
    assert TokenizeData(processor=processor, video_temporal_mode="framewise", video_timestamp_mode=mode)(data)
    content = processor.last_conversation[0]["content"][0]
    assert len(content["video"]) == 6
    assert [image.getpixel((0, 0))[0] for image in content["video"]] == [0, 0, 100, 100, 200, 200]
    _, fps, totals, indices = maybe_parse_video_content(processor.last_conversation)
    expected_indices = [30, 30, 40, 40, 50, 50] if mode == "qwen_index" else [0, 0, 10, 10, 20, 20]
    assert indices == [expected_indices]
    assert totals == ([100] if mode == "qwen_index" else [70])
    assert metadata["frames_indices"] == [30, 40, 50]
    _, label_times = overlay_text(
        frames,
        video_media["fps"],
        processor=processor,
        source_frames_indices=video_media["source_frames_indices"],
        source_fps=video_media["source_fps"],
        video_metadata=video_media.get("video_metadata"),
    )
    rendered = [float(f"{time:.1f}") for time in calculate_video_timestamps(indices[0], fps[0], 2)]
    assert label_times == rendered == ([1.0, 1.3, 1.7] if mode == "qwen_index" else [0.0, 0.3, 0.7])
