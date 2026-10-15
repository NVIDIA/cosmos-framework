# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from collections.abc import Iterator
from unittest.mock import Mock

import pytest
import torch

from cosmos_framework.data.generator.augmentors import text_tokenizer, text_transforms_for_video
from cosmos_framework.data.generator.augmentors.duration_fps_text_timestamps import DurationFPSTextTimeStamps
from cosmos_framework.data.generator.augmentors.resolution_text_info import ResolutionTextInfo
from cosmos_framework.data.generator.augmentors.text_transforms_for_video import (
    CaptionSchemaError,
    TextTransformForVideoTransferChunkedFrames,
    TextTransformForVideoTransferFullFrames,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

_TRANSFORMS = [TextTransformForVideoTransferFullFrames, TextTransformForVideoTransferChunkedFrames]
_DENSE_CAPTION = "A robot opens the drawer, picks up the cup, and closes the drawer."


def _sample(payload: object) -> dict:
    return {
        "metas": {"nb_frames": 600, "framerate": 30.0},
        "captions": {
            "caption_structured": json.dumps(
                {
                    "chunk_200_400": {
                        "start_frame": 200,
                        "end_frame": 400,
                        "caption": json.dumps(payload),
                    }
                }
            )
        },
    }


def _make_transform(
    transform_class: type[TextTransformForVideoTransferFullFrames],
    probability: object = 0.07,
) -> TextTransformForVideoTransferFullFrames:
    return transform_class(
        input_keys=["metas"],
        args={"caption_config": {"captions": {"ratio": 1.0, "dense_caption_probability": probability}}},
    )


@pytest.mark.parametrize("transform_class", _TRANSFORMS)
@pytest.mark.parametrize("draw,uses_dense", [(0.0, True), (0.069999, True), (0.07, False), (0.999999, False)])
def test_caption_mixture_preserves_selected_chunk(
    monkeypatch: pytest.MonkeyPatch,
    transform_class: type[TextTransformForVideoTransferFullFrames],
    draw: float,
    uses_dense: bool,
) -> None:
    structured = {"subjects": [{"description": "A robot."}], "temporal_caption": _DENSE_CAPTION}
    monkeypatch.setattr(text_transforms_for_video.random, "random", lambda: draw)

    result = _make_transform(transform_class)(_sample(structured))

    assert result is not None
    assert result["ai_caption"] == (_DENSE_CAPTION if uses_dense else json.dumps(structured))
    assert result["sampled_caption_style"] == "captions"
    if transform_class is TextTransformForVideoTransferChunkedFrames:
        assert result["sampled_chunk_key"] == "chunk_200_400"
        assert (result["chunk_start_frame"], result["chunk_end_frame"]) == (200, 400)


@pytest.mark.parametrize("transform_class", _TRANSFORMS)
@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan"), float("inf"), float("-inf"), None, "bad"])
def test_caption_mixture_rejects_invalid_probability(
    transform_class: type[TextTransformForVideoTransferFullFrames], probability: object
) -> None:
    with pytest.raises(ValueError, match="dense_caption_probability"):
        _make_transform(transform_class, probability)


@pytest.mark.parametrize("transform_class", _TRANSFORMS)
@pytest.mark.parametrize(
    "payload", [{}, {"temporal_caption": ""}, {"temporal_caption": "  "}, {"temporal_caption": 5}, []]
)
def test_caption_mixture_validates_both_formats_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
    transform_class: type[TextTransformForVideoTransferFullFrames],
    payload: object,
) -> None:
    def unexpected_draw() -> float:
        pytest.fail("Malformed captions must be rejected before sampling their format.")

    monkeypatch.setattr(text_transforms_for_video.random, "random", unexpected_draw)

    transform = _make_transform(transform_class)
    with pytest.raises(CaptionSchemaError):
        transform._format_caption(payload, {"dense_caption_probability": 0.07})
    assert transform(_sample(payload)) is None
    assert transform._caption_rejected == {"schema_error": 1}


@pytest.mark.parametrize("transform_class", _TRANSFORMS)
@pytest.mark.parametrize("caption_in_metadata", [False, True])
def test_caption_schema_diagnostics_are_bounded_and_separate_from_decode_errors(
    monkeypatch: pytest.MonkeyPatch,
    transform_class: type[TextTransformForVideoTransferFullFrames],
    caption_in_metadata: bool,
) -> None:
    warning = Mock()
    monkeypatch.setattr(text_transforms_for_video.log, "warning", warning)
    transform = _make_transform(transform_class)
    invalid_sample = _sample({"temporal_caption": " "})
    invalid_sample.update({"__url__": "s3://test/captions.tar", "__key__": "bad-caption"})
    if caption_in_metadata:
        invalid_sample["metas"]["captions"] = invalid_sample.pop("captions")

    for _ in range(100):
        assert transform(invalid_sample) is None

    assert warning.call_count == 4
    for call, count in zip(warning.call_args_list, [1, 2, 3, 100], strict=True):
        message = call.args[0]
        assert f"{transform_class.__name__}: schema_error:" in message
        assert "nonempty temporal_caption string" in message
        assert "caption_source='captions', chunk='chunk_200_400'" in message
        assert "url: s3://test/captions.tar, key: bad-caption" in message
        assert "worker-local transform counts (pid=" in message
        assert f"attempted={count}, yielded=0, rejected={{'schema_error': {count}}}" in message
        assert call.kwargs == {"rank0_only": False}

    assert transform(_sample({"temporal_caption": _DENSE_CAPTION})) is not None
    malformed_json = _sample({})
    chunks = json.loads(malformed_json["captions"]["caption_structured"])
    chunks["chunk_200_400"]["caption"] = "invalid JSON"
    malformed_json["captions"]["caption_structured"] = json.dumps(chunks)
    assert transform(malformed_json) is None
    assert warning.call_count == 5
    decode_message = warning.call_args.args[0]
    assert f"{transform_class.__name__}: decode_error: Failed to decode caption_structured" in decode_message
    assert "caption_source='captions', chunk='chunk_200_400'" in decode_message
    assert "attempted=102, yielded=1, rejected={'schema_error': 100, 'decode_error': 1}" in decode_message
    assert transform._caption_attempted == 102
    assert transform._caption_yielded == 1
    assert transform._caption_rejected == {"schema_error": 100, "decode_error": 1}


@pytest.mark.parametrize("transform_class", _TRANSFORMS)
def test_caption_diagnostics_reset_counts_in_a_new_worker(
    monkeypatch: pytest.MonkeyPatch,
    transform_class: type[TextTransformForVideoTransferFullFrames],
) -> None:
    warning = Mock()
    monkeypatch.setattr(text_transforms_for_video.log, "warning", warning)
    transform = _make_transform(transform_class)
    assert transform(_sample({"temporal_caption": _DENSE_CAPTION})) is not None
    assert transform(_sample({})) is None
    warning.reset_mock()
    # Simulate inheriting a previously used transform in a forked worker.
    transform._caption_stats_pid = -1

    assert transform(_sample({})) is None

    warning.assert_called_once()
    assert "attempted=1, yielded=0, rejected={'schema_error': 1}" in warning.call_args.args[0]


@pytest.mark.parametrize("transform_class", _TRANSFORMS)
@pytest.mark.parametrize("payload", [{"description": "A robot."}, "Legacy scalar caption"])
def test_caption_mixture_default_preserves_structured_behavior(
    monkeypatch: pytest.MonkeyPatch,
    transform_class: type[TextTransformForVideoTransferFullFrames],
    payload: object,
) -> None:
    def unexpected_draw() -> float:
        pytest.fail("Existing structured-only recipes must not draw a caption-format probability.")

    monkeypatch.setattr(text_transforms_for_video.random, "random", unexpected_draw)
    transform = transform_class(input_keys=["metas"], args={"caption_config": {"captions": 1.0}})

    result = transform(_sample(payload))

    assert result is not None
    assert result["ai_caption"] == json.dumps(payload)


def test_dense_caption_keeps_exact_target_chunk_and_metadata_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(text_transforms_for_video.random, "random", lambda: 0.0)
    transform = TextTransformForVideoTransferChunkedFrames(
        input_keys=["metas"],
        args={
            "caption_config": {"captions": {"ratio": 1.0, "dense_caption_probability": 0.07}},
            "target_num_frames": 81,
        },
    )

    result = transform(_sample({"temporal_caption": _DENSE_CAPTION, "duration": "old", "fps": 1.0}))

    assert result is not None
    assert result["ai_caption"] == _DENSE_CAPTION
    assert (result["chunk_start_frame"], result["chunk_end_frame"]) == (200, 400)


class _RecordingProcessor:
    captions: list[str]

    def __init__(self) -> None:
        self.captions = []

    def tokenize_text(self, caption: str, *, system_prompt: str) -> list[int]:
        self.captions.append(caption)
        return [1, 2]


@pytest.mark.parametrize("format_draw", [0.069999, 0.07])
@pytest.mark.parametrize("dropout_draw,dropped", [(0.099999, True), (0.1, False)])
def test_caption_format_and_cfg_dropout_are_independent(
    monkeypatch: pytest.MonkeyPatch, format_draw: float, dropout_draw: float, dropped: bool
) -> None:
    draws: Iterator[float] = iter([format_draw, dropout_draw])
    monkeypatch.setattr(text_transforms_for_video.random, "random", lambda: next(draws))
    structured = {"temporal_caption": _DENSE_CAPTION}
    result = _make_transform(TextTransformForVideoTransferChunkedFrames)(_sample(structured))
    assert result is not None
    result["video"] = torch.zeros(3, 5, 16, 32)  # [C,T,H,W]
    result["image_size"] = torch.tensor([16, 32, 16, 32])  # [4]
    result["conditioning_fps"] = 5.0
    result = DurationFPSTextTimeStamps(args={})(result)
    assert result is not None
    result = ResolutionTextInfo(args={})(result)
    assert result is not None
    prefix = _DENSE_CAPTION if format_draw < 0.07 else json.dumps(structured)
    formatted_caption = result["ai_caption"]
    assert formatted_caption.startswith(prefix)
    assert "The video is 1.0 seconds long and is of 5 FPS." in formatted_caption
    assert "This video is of 16x32 resolution." in formatted_caption

    processor = _RecordingProcessor()
    monkeypatch.setattr(text_tokenizer, "lazy_instantiate", lambda _config: processor)
    tokenizer = text_tokenizer.TextTokenizerTransformForTransfer(
        input_keys=["ai_caption"],
        output_keys=["text_token_ids"],
        args={"tokenizer_config": object(), "cfg_dropout_rate": 0.1},
    )
    result = tokenizer(result)

    assert result is not None
    assert result["ai_caption"] == ("" if dropped else formatted_caption)
    assert processor.captions == [result["ai_caption"]]
    assert list(draws) == []
