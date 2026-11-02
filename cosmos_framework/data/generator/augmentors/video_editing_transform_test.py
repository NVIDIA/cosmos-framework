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

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


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


@pytest.mark.parametrize(
    ("source_frames", "target_frames", "fps", "allowed"),
    [
        (80, 77, 16.0, True),
        (77, 77, 16.0, True),
        (81, 77, 16.0, False),
        (77, 80, 16.0, False),
        (80, 77, 15.0, False),
        (78, 75, 16.0, False),
    ],
)
def test_bernini_source_tail_is_bounded_and_native_rate(
    source_frames: int, target_frames: int, fps: float, allowed: bool
) -> None:
    transform = PairedVideoEditingToTrainingFormat(max_num_frames=77, bernini_77f_source_sampling=True)
    source = VideoMetadata(num_frames=source_frames, average_fps=fps, height=64, width=96)
    target = VideoMetadata(num_frames=target_frames, average_fps=fps, height=64, width=96)
    if allowed:
        duration, _, _ = transform._validate_pair(source, target)
        assert duration == pytest.approx(77 / 16)
        assert aligned_frame_indices(source, 16.0, 77) == list(range(77))
        assert aligned_frame_indices(target, 16.0, 77) == list(range(77))
    else:
        with pytest.raises(ValueError, match="bernini_77f_pair"):
            transform._validate_pair(source, target)


def test_source_tail_remains_strict_without_explicit_opt_in() -> None:
    source = VideoMetadata(num_frames=80, average_fps=16.0, height=64, width=96)
    target = VideoMetadata(num_frames=77, average_fps=16.0, height=64, width=96)
    with pytest.raises(ValueError, match="duration_mismatch"):
        PairedVideoEditingToTrainingFormat(max_num_frames=77)._validate_pair(source, target)


@pytest.mark.parametrize(("maximum", "fps"), [(93, 16.0), (77, 15.0)])
def test_bernini_sampling_requires_its_native_target(maximum: int, fps: float) -> None:
    with pytest.raises(ValueError, match="Bernini source sampling"):
        PairedVideoEditingToTrainingFormat(max_num_frames=maximum, target_fps=fps, bernini_77f_source_sampling=True)


def test_bernini_indices_match_generation_before_truncation() -> None:
    indices = PairedVideoEditingToTrainingFormat._bernini_source_frame_indices(80)
    assert len(indices) == 77
    assert indices == torch.linspace(0, 79, 81).round().long().tolist()[:77]
    assert indices != list(range(77))
    assert indices[0] == 0
    assert indices[-1] == 75
    assert len(set(indices)) == 76
    assert PairedVideoEditingToTrainingFormat._bernini_source_frame_indices(77) == list(range(77))


@pytest.mark.parametrize("fps", [16.0, 16.0000005, 15.9999995])
@pytest.mark.parametrize("source_frames", [77, 80])
def test_bernini_decode_keeps_both_videos_at_77_frames_despite_fps_rounding(
    monkeypatch: pytest.MonkeyPatch, fps: float, source_frames: int
) -> None:
    from cosmos_framework.data.generator.augmentors import video_editing_transform as module

    transform = PairedVideoEditingToTrainingFormat(max_num_frames=77, bernini_77f_source_sampling=True)
    source_metadata = VideoMetadata(num_frames=source_frames, average_fps=fps, height=64, width=96)
    target_metadata = VideoMetadata(num_frames=77, average_fps=fps, height=64, width=96)
    monkeypatch.setattr(
        module, "probe_video", lambda payload, **_: source_metadata if payload == b"source" else target_metadata
    )
    selected: dict[bytes, list[int]] = {}

    def decode(payload: bytes, indices: list[int], **kwargs: object) -> tuple[torch.Tensor, VideoMetadata]:
        selected[payload] = indices
        return torch.zeros((len(indices), 3, 64, 96), dtype=torch.uint8), target_metadata

    monkeypatch.setattr(module, "decode_frames_tchw_uint8", decode)
    source, target, _, _, _ = transform._decode_pair(b"source", b"target")
    assert source.shape[1] == target.shape[1] == 77
    assert selected[b"target"] == list(range(77))
    assert selected[b"source"] == transform._bernini_source_frame_indices(source_frames)
