# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

import pytest
import torch

import cosmos_framework.data.generator.augmentors.interleaved_video_parsing as video_parsing
from cosmos_framework.data.imaginaire.webdataset.augmentors.image.normalize import Normalize
from cosmos_framework.data.imaginaire.webdataset.augmentors.image.padding import ReflectionPadding
from cosmos_framework.data.imaginaire.webdataset.augmentors.image.resize import ResizeLargestSideAspectPreserving
from cosmos_framework.data.generator.augmentors.interleaved_video_parsing import (
    VideoTransferAlignedChunkedFramesParsing,
    VideoTransferAlignedSelectedControlParsing,
)
from cosmos_framework.data.generator.augmentors.merge_datadict import DataDictMerger
from cosmos_framework.data.generator.augmentors.text_transforms_for_video import (
    TextTransformForVideoTransferChunkedFrames,
)
from cosmos_framework.data.generator.augmentors.transfer_control_transform import AddSelectedControlFromVideo


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


def _make_selected_parser(max_num_frames: int = 101) -> VideoTransferAlignedSelectedControlParsing:
    return VideoTransferAlignedSelectedControlParsing(
        input_keys=["metas", "video"],
        args={
            "max_num_frames": max_num_frames,
            "max_stride": 1,
            "min_stride": 1,
            "min_fps": 5.0,
            "max_fps": 60.0,
            "seek_mode": "exact",
        },
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_selected_parser_preserves_101_frame_causal_contract() -> None:
    indices, stride = _make_selected_parser()._sample_frame_indices_for_chunk(150, 0, 150)
    assert stride == 1
    assert len(indices) == 101
    assert (len(indices) - 1) % 4 == 0


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("modality,channels", [("depth", 1), ("seg", 3)])
def test_selected_parser_uses_shared_shorter_stream_plan_and_maps_control(
    monkeypatch: pytest.MonkeyPatch, modality: str, channels: int
) -> None:
    parser = _make_selected_parser()
    monkeypatch.setattr(parser, "_validate_and_probe", lambda *_args: True)
    monkeypatch.setattr(parser, "_probe_video_len", lambda payload: 120 if payload == b"rgb" else 99)
    calls: list[tuple[bytes, list[int]]] = []

    def fake_decode(
        payload: bytes,
        indices: list[int],
        transforms: object = None,
        output_dtype: torch.dtype = torch.uint8,
    ) -> torch.Tensor:
        del transforms, output_dtype
        calls.append((payload, list(indices)))
        dtype = torch.float32 if payload == b"control" and modality == "depth" else torch.uint8
        value = 1.0 if dtype == torch.float32 else 255
        return torch.full(  # [C,T,H,W]
            (channels if payload == b"control" else 3, len(indices), 8, 12), value, dtype=dtype
        )

    monkeypatch.setattr(parser, "_decode_frames_at", fake_decode)
    sample = {
        "metas": {"framerate": 30.0, "nb_frames": 120, "width": 12, "height": 8},
        "video": b"rgb",
        "persisted_control": b"control",
        "_persisted_control_meta": {
            "framerate": 30.0,
            "nb_frames": 99,
            "width": 12,
            "height": 8,
        },
        "_selected_control_modality": modality,
        "__url__": "url",
        "__key__": "key",
        "chunk_start_frame": 0,
        "chunk_end_frame": 99,
    }
    parsed = parser(sample)
    assert parsed is not None
    assert calls[0][1] == calls[1][1]
    assert max(calls[0][1]) < 99
    control_key = "depth" if modality == "depth" else "segmentation"
    assert parsed[control_key].shape == (3, 97, 8, 12)
    if modality == "depth":
        assert parsed[control_key].dtype == torch.float32
        assert torch.all(parsed[control_key] == 255.0)


@pytest.mark.L0
@pytest.mark.CPU
def test_selected_parser_rejects_descriptor_geometry_or_fps_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = _make_selected_parser()
    monkeypatch.setattr(parser, "_validate_and_probe", lambda *_args: True)
    sample = {
        "metas": {"framerate": 30.0, "nb_frames": 120, "width": 12, "height": 8},
        "video": b"rgb",
        "persisted_control": b"control",
        "_persisted_control_meta": {"framerate": 29.0, "nb_frames": 120, "width": 13, "height": 8},
        "_selected_control_modality": "depth",
        "__url__": "url",
        "__key__": "key",
    }
    assert parser(sample) is None


@pytest.mark.L0
@pytest.mark.CPU
def test_selected_parser_accepts_fps_drift_within_one_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDecoder:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.metadata = type("Metadata", (), {"width": 12, "height": 8, "average_fps": 30.0})()

        def __len__(self) -> int:
            return 654

    parser = _make_selected_parser()
    payload = b"rgb"
    parser._expected_decode_metadata = [
        (payload, {"width": 12, "height": 8, "framerate": 30.04594180704441, "nb_frames": 654})
    ]
    monkeypatch.setattr(video_parsing, "VideoDecoder", FakeDecoder)
    assert parser._probe_video_len(payload) == 654


def test_selected_parser_rejects_fps_drift_over_one_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDecoder:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.metadata = type("Metadata", (), {"width": 12, "height": 8, "average_fps": 29.0})()

        def __len__(self) -> int:
            return 654

    parser = _make_selected_parser()
    payload = b"rgb"
    parser._expected_decode_metadata = [(payload, {"width": 12, "height": 8, "framerate": 30.0, "nb_frames": 654})]
    monkeypatch.setattr(video_parsing, "VideoDecoder", FakeDecoder)
    with pytest.raises(ValueError, match="Decoded framerate"):
        parser._probe_video_len(payload)


def test_selected_parser_rejects_decoded_header_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDecoder:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.metadata = type("Metadata", (), {"width": 13, "height": 8, "average_fps": 30.0})()

        def __len__(self) -> int:
            return 101

    parser = _make_selected_parser()
    payload = b"rgb"
    parser._expected_decode_metadata = [(payload, {"width": 12, "height": 8, "framerate": 30.0, "nb_frames": 101})]
    monkeypatch.setattr(video_parsing, "VideoDecoder", FakeDecoder)
    with pytest.raises(ValueError, match="Decoded width"):
        parser._probe_video_len(payload)


@pytest.mark.L0
@pytest.mark.CPU
def test_selected_parser_returns_none_on_corrupt_persisted_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = _make_selected_parser()
    monkeypatch.setattr(parser, "_validate_and_probe", lambda *_args: True)
    monkeypatch.setattr(parser, "_probe_video_len", lambda _payload: 101)

    def fake_decode(
        payload: bytes,
        _indices: list[int],
        _transforms: object = None,
        _output_dtype: torch.dtype = torch.uint8,
    ) -> torch.Tensor:
        if payload == b"control":
            raise RuntimeError("corrupt")
        return torch.zeros(3, 101, 8, 12)  # [C,T,H,W]

    monkeypatch.setattr(parser, "_decode_frames_at", fake_decode)
    sample = {
        "metas": {"framerate": 30.0, "nb_frames": 101, "width": 12, "height": 8},
        "video": b"rgb",
        "persisted_control": b"control",
        "_persisted_control_meta": {"framerate": 30.0, "nb_frames": 101, "width": 12, "height": 8},
        "_selected_control_modality": "seg",
        "__url__": "url",
        "__key__": "key",
        "chunk_start_frame": 0,
        "chunk_end_frame": 101,
    }
    assert parser(sample) is None


@pytest.mark.L0
@pytest.mark.CPU
def test_selected_parser_constructs_uint8_rgb_semantic_and_float32_depth_decoders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[bytes, torch.dtype]] = []

    class FakeDecoder:
        def __init__(self, payload: bytes, **kwargs: object) -> None:
            output_dtype = kwargs["output_dtype"]
            assert isinstance(output_dtype, torch.dtype)
            self.output_dtype = output_dtype
            calls.append((payload, output_dtype))

        def get_frames_at(self, indices: list[int]) -> object:
            frames = torch.zeros(len(indices), 3, 2, 2, dtype=self.output_dtype)  # [T,C,H,W]
            return type("FrameBatch", (), {"data": frames})()

    monkeypatch.setattr(video_parsing, "VideoDecoder", FakeDecoder)
    parser = _make_selected_parser()
    depth_payload = bytes(bytearray(b"depth"))
    parser._depth_control_video = depth_payload

    parser._decode_frames_at(b"rgb", [0])  # [C,T,H,W]
    parser._decode_frames_at(b"semantic", [0])  # [C,T,H,W]
    parser._decode_frames_at(depth_payload, [0])  # [C,T,H,W]

    assert calls == [
        (b"rgb", torch.uint8),
        (b"semantic", torch.uint8),
        (depth_payload, torch.float32),
    ]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("chunk_start,chunk_end", [(0, 9), (20, 29)])
def test_chunked_caption_and_selected_parser_decode_matching_shared_window(
    monkeypatch: pytest.MonkeyPatch, chunk_start: int, chunk_end: int
) -> None:
    caption_transform = TextTransformForVideoTransferChunkedFrames(
        input_keys=["metas"],
        args={"caption_config": {"captions": 1.0}, "keep_metas": True, "min_num_frames": 1},
    )
    sample = {
        "metas": {
            "framerate": 30.0,
            "nb_frames": 100,
            "width": 12,
            "height": 8,
            "captions": {
                "caption_structured": json.dumps(
                    {
                        "chosen": {
                            "start_frame": chunk_start,
                            "end_frame": chunk_end,
                            "caption": json.dumps({"description": "matching caption"}),
                        }
                    }
                )
            },
        },
        "video": b"rgb",
        "persisted_control": b"control",
        "_persisted_control_meta": {
            "framerate": 30.0,
            "nb_frames": 100,
            "width": 12,
            "height": 8,
        },
        "_selected_control_modality": "seg",
        "__url__": "url",
        "__key__": "key",
    }
    transformed = caption_transform(sample)
    assert transformed is not None
    assert transformed["chunk_start_frame"] == chunk_start
    assert transformed["chunk_end_frame"] == chunk_end

    parser = _make_selected_parser()
    monkeypatch.setattr(parser, "_validate_and_probe", lambda *_args: True)
    monkeypatch.setattr(parser, "_probe_video_len", lambda _payload: 100)
    calls: list[tuple[bytes, list[int]]] = []

    def fake_decode(
        payload: bytes,
        indices: list[int],
        _transforms: object = None,
        _output_dtype: torch.dtype = torch.uint8,
    ) -> torch.Tensor:
        calls.append((payload, list(indices)))
        return torch.zeros(3, len(indices), 8, 12, dtype=torch.uint8)  # [C,T,H,W]

    monkeypatch.setattr(parser, "_decode_frames_at", fake_decode)
    parsed = parser(transformed)

    assert parsed is not None
    assert calls[0][1] == calls[1][1]
    assert calls[0][1][0] == chunk_start
    assert calls[0][1][-1] < chunk_end
    assert "chunk_start_frame" not in parsed
    assert "chunk_end_frame" not in parsed


@pytest.mark.L0
@pytest.mark.CPU
def test_gray12_depth_preserves_precision_through_selected_control_and_generic_normalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = _make_selected_parser(max_num_frames=5)
    monkeypatch.setattr(parser, "_validate_and_probe", lambda *_args: True)
    monkeypatch.setattr(parser, "_probe_video_len", lambda _payload: 5)
    gray12_levels = torch.tensor([1.0, 2048.0, 4095.0], dtype=torch.float32).div(4095.0)  # [W]

    class FakeDecoder:
        def __init__(self, _payload: bytes, **kwargs: object) -> None:
            output_dtype = kwargs["output_dtype"]
            assert isinstance(output_dtype, torch.dtype)
            self.output_dtype = output_dtype

        def get_frames_at(self, indices: list[int]) -> object:
            if self.output_dtype == torch.float32:
                frames = gray12_levels.view(1, 1, 1, 3).expand(len(indices), 3, 1, 3)  # [T,C,H,W]
            else:
                frames = torch.zeros(len(indices), 3, 1, 3, dtype=torch.uint8)  # [T,C,H,W]
            return type("FrameBatch", (), {"data": frames})()

    monkeypatch.setattr(video_parsing, "VideoDecoder", FakeDecoder)
    sample = {
        "metas": {"framerate": 30.0, "nb_frames": 5, "width": 3, "height": 1},
        "video": b"rgb",
        "persisted_control": bytes(bytearray(b"depth")),
        "_persisted_control_meta": {"framerate": 30.0, "nb_frames": 5, "width": 3, "height": 1},
        "_selected_control_modality": "depth",
        "chunk_start_frame": 0,
        "chunk_end_frame": 5,
        "__url__": "url",
        "__key__": "key",
    }
    parsed = parser(sample)
    assert parsed is not None
    assert parsed["depth"].dtype == torch.float32
    assert torch.allclose(parsed["depth"][0, 0, 0], gray12_levels * 255.0)

    merged = DataDictMerger(input_keys=["video"], output_keys=["video"])(parsed)
    assert merged is not None
    selected = AddSelectedControlFromVideo(input_keys=["video"])(merged)
    assert selected is not None
    selected.pop("__url__")
    selected["aspect_ratio"] = "3,1"
    size_map = {"3,1": (3, 1)}
    selected = ResizeLargestSideAspectPreserving(input_keys=["video", "control_input"], args={"size": size_map})(
        selected
    )
    selected = ReflectionPadding(input_keys=["video", "control_input"], args={"size": size_map})(selected)
    normalized = Normalize(
        input_keys=["video", "control_input"],
        args={"mean": 0.0, "std": 1.0},
    )(selected)
    normalized_depth = normalized["control_input"]  # [C,T,H,W]

    assert torch.allclose(normalized_depth[0, 0, 0], gray12_levels)
    assert normalized_depth[0, 0, 0, 0] > 0
    quantized_midpoint = gray12_levels[1].mul(255.0).round().div(255.0)  # []
    assert not torch.isclose(normalized_depth[0, 0, 0, 1], quantized_midpoint)
