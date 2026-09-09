# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import io
import json
import pickle

import pytest
import torch
from PIL import Image

from cosmos_framework.data.generator.augmentors.video_editing_transform import (
    PairedVideoEditingToTrainingFormat,
    SelectAddRemoveVideoEditingCaption,
    aligned_frame_indices,
    parse_video_editing_conversation,
    parse_video_editing_conversation_with_references,
    valid_video_frame_count,
)
from cosmos_framework.utils.generator.torchcodec_video import VideoMetadata


def _conversation(instruction: str = "Remove the car", reference_keys: tuple[str, ...] = ()) -> str:
    return json.dumps(
        [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": "video_0.mp4"},
                        *[{"type": "image", "image": key} for key in reference_keys],
                        {"type": "text", "text": instruction},
                    ],
                },
                {"role": "assistant", "content": [{"type": "video", "video": "video_1.mp4"}]},
            ]
        ]
    )


def test_parse_video_editing_conversation() -> None:
    assert parse_video_editing_conversation(_conversation()) == (
        "video_0.mp4",
        "video_1.mp4",
        "Remove the car",
    )


def _caption_augmentation() -> dict[str, dict[str, str]]:
    return {
        "add": {
            "original": "Add original",
            "short": "Add short",
            "medium": "Add medium",
            "long": "Add long",
        },
        "remove": {
            "original": "Remove original",
            "short": "Remove short",
            "medium": "Remove medium",
            "long": "Remove long",
        },
    }


def test_add_caption_selection_reverses_source_and_target_videos() -> None:
    transform = SelectAddRemoveVideoEditingCaption(
        add_probability=1.0,
        caption_variant_probabilities={"original": 0.0, "short": 0.0, "medium": 0.0, "long": 1.0},
    )
    sample = {
        "__key__": "add-row",
        "media": {"source_video": b"object-present", "edited_video": b"object-removed"},
        "caption_augmentation": json.dumps(_caption_augmentation()),
    }

    result = transform(sample)

    assert result is sample
    assert parse_video_editing_conversation(result["texts"]) == ("edited_video", "source_video", "Add long")
    assert result["selected_edit_direction"] == "add"
    assert result["selected_caption_variant"] == "long"


def test_remove_caption_selection_preserves_source_and_target_videos() -> None:
    transform = SelectAddRemoveVideoEditingCaption(
        add_probability=0.0,
        caption_variant_probabilities={"original": 1.0, "short": 0.0, "medium": 0.0, "long": 0.0},
    )
    sample = {
        "__key__": "remove-row",
        "media": {"source_video": b"object-present", "edited_video": b"object-removed"},
        "caption_augmentation": json.dumps(_caption_augmentation()).encode(),
    }

    result = transform(sample)

    assert result is sample
    assert parse_video_editing_conversation(result["texts"]) == (
        "source_video",
        "edited_video",
        "Remove original",
    )
    assert result["selected_edit_direction"] == "remove"
    assert result["selected_caption_variant"] == "original"


@pytest.mark.parametrize(
    "probabilities",
    (
        {"original": 0.1, "short": 0.1, "medium": 0.1},
        {"original": 0.1, "short": 0.1, "medium": 0.1, "long": 0.6},
        {"original": -0.1, "short": 0.1, "medium": 0.1, "long": 0.9},
    ),
)
def test_caption_selection_rejects_invalid_probabilities(probabilities: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="probabilit"):
        SelectAddRemoveVideoEditingCaption(caption_variant_probabilities=probabilities)


def test_caption_selection_rejects_incomplete_payload() -> None:
    transform = SelectAddRemoveVideoEditingCaption()

    assert transform({"__key__": "bad-row", "caption_augmentation": '{"add": {}}'}) is None


def _jpeg_bytes(width: int = 40, height: int = 20) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(stream, format="JPEG")
    return stream.getvalue()


def test_parse_video_editing_conversation_with_reference_images() -> None:
    payload = _conversation(
        "change the white lab coat to a blue denim jacket",
        reference_keys=("image_0.jpg", "image_1.png"),
    )
    assert parse_video_editing_conversation_with_references(payload) == (
        "video_0.mp4",
        "video_1.mp4",
        ["image_0.jpg", "image_1.png"],
        "change the white lab coat to a blue denim jacket",
    )
    # Keep the legacy parser compatible with callers that do not need references.
    assert parse_video_editing_conversation(payload) == (
        "video_0.mp4",
        "video_1.mp4",
        "change the white lab coat to a blue denim jacket",
    )


def test_transform_inserts_reference_image_before_target(monkeypatch: pytest.MonkeyPatch) -> None:
    transform = PairedVideoEditingToTrainingFormat(
        max_pixels=32 * 32,
        padding_divisor=16,
        min_reference_image_side=16,
    )
    source = torch.zeros((3, 5, 16, 32))
    target = torch.ones((3, 5, 16, 32))
    metadata = VideoMetadata(num_frames=5, average_fps=16.0, height=16, width=32)
    monkeypatch.setattr(transform, "_decode_pair", lambda *_: (source, target, metadata, 32, 16))
    media = {
        "video_0.mp4": b"source",
        "image_0.jpg": _jpeg_bytes(),
        "video_1.mp4": b"target",
    }

    result = transform(
        {
            "__key__": "goku-bigfacing-reference_swap-799d0f920fa52ad6ef5a0bf8",
            "texts": _conversation(
                "change the white lab coat to a blue denim jacket",
                reference_keys=("image_0.jpg",),
            ),
            "media": pickle.dumps(media),
        }
    )

    assert result is not None
    assert len(result["video"]) == 3
    assert result["video"][0] is source
    assert result["video"][1].shape[0:2] == (3, 1)
    assert result["video"][2] is target
    assert len(result["image_size"]) == 3
    assert result["sequence_plan"].share_vision_temporal_positions is False
    assert result["sequence_plan"].vision_temporal_position_groups == [0, None, 0]


def test_transform_rejects_small_reference_image(monkeypatch: pytest.MonkeyPatch) -> None:
    transform = PairedVideoEditingToTrainingFormat(min_reference_image_side=128)
    source = torch.zeros((3, 5, 256, 256))
    target = torch.ones((3, 5, 256, 256))
    metadata = VideoMetadata(num_frames=5, average_fps=16.0, height=256, width=256)
    monkeypatch.setattr(transform, "_decode_pair", lambda *_: (source, target, metadata, 256, 256))

    result = transform(
        {
            "__key__": "small-reference",
            "texts": _conversation("replace the object", reference_keys=("image_0.jpg",)),
            "media": pickle.dumps(
                {
                    "video_0.mp4": b"source",
                    "image_0.jpg": _jpeg_bytes(width=64, height=256),
                    "video_1.mp4": b"target",
                }
            ),
        }
    )

    assert result is None


@pytest.mark.parametrize(("available", "maximum", "expected"), [(100, 93, 93), (92, 93, 89), (4, 93, 1)])
def test_valid_video_frame_count(available: int, maximum: int, expected: int) -> None:
    assert valid_video_frame_count(available, maximum) == expected


def test_aligned_frame_indices_uses_shared_timeline() -> None:
    metadata = VideoMetadata(num_frames=120, average_fps=24.0, height=64, width=96)
    assert aligned_frame_indices(metadata, target_fps=16.0, num_frames=5) == [0, 2, 3, 4, 6]


def test_pair_validation_allows_one_target_frame_of_duration_rounding() -> None:
    transform = PairedVideoEditingToTrainingFormat(target_fps=16.0)
    source = VideoMetadata(num_frames=160, average_fps=16.0, height=64, width=96)
    within_one_frame = VideoMetadata(num_frames=159, average_fps=16.0, height=64, width=96)

    common_duration, _, _ = transform._validate_pair(source, within_one_frame)

    assert common_duration == pytest.approx(159 / 16)


def test_pair_validation_rejects_duration_difference_over_one_target_frame() -> None:
    transform = PairedVideoEditingToTrainingFormat(target_fps=16.0)
    source = VideoMetadata(num_frames=160, average_fps=16.0, height=64, width=96)
    misaligned = VideoMetadata(num_frames=158, average_fps=16.0, height=64, width=96)

    with pytest.raises(ValueError, match="duration_mismatch"):
        transform._validate_pair(source, misaligned)


def test_parse_rejects_empty_instruction() -> None:
    with pytest.raises(ValueError, match="instruction is empty"):
        parse_video_editing_conversation(_conversation(""))
