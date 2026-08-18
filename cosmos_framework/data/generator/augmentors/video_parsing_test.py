# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unit tests for video parsing without requiring real TorchCodec media.

Validates the categorical per-video stride sampling used by the FPS-mixing ablation:
coercion of OmegaConf string keys, the realized per-video stride distribution, input
validation, and that omitting stride_config leaves the exp-decay path unchanged. Also
validates that embedded audio is aligned by video and audio PTS to the stride-aware
selected video-frame interval.
"""

import sys
from collections import Counter
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

# TorchCodec is not installed in every unit-test environment. The production module
# imports its decoder classes at module load time, so provide decoder stubs only when
# that optional dependency cannot be imported; individual audio tests patch AudioDecoder.
try:
    from torchcodec.decoders import AudioDecoder as _TorchCodecAudioDecoder  # noqa: F401
except (ImportError, RuntimeError):
    _torchcodec_stub = ModuleType("torchcodec")
    setattr(_torchcodec_stub, "__path__", [])
    _torchcodec_decoders_stub = ModuleType("torchcodec.decoders")
    setattr(_torchcodec_decoders_stub, "AudioDecoder", MagicMock)
    setattr(_torchcodec_decoders_stub, "VideoDecoder", MagicMock)
    sys.modules.setdefault("torchcodec", _torchcodec_stub)
    sys.modules.setdefault("torchcodec.decoders", _torchcodec_decoders_stub)

from cosmos_framework.data.generator.augmentors import video_parsing
from cosmos_framework.data.generator.augmentors.video_parsing import (
    VideoParsingChunkedFrames,
    VideoParsingWithFullFrames,
)

# Minimal args to instantiate the augmentor for stride-sampling only (no video decode).
_BASE_ARGS: dict[str, object] = dict(
    max_stride=2,
    min_stride=2,
    use_dynamic_fps=True,
    min_fps=10.0,
    max_fps=60.0,
    causal_vae=True,
    max_num_frames=1000,
)


def _make(stride_config: object | None = None) -> VideoParsingChunkedFrames:
    args = dict(_BASE_ARGS)
    if stride_config is not None:
        args["stride_config"] = stride_config
    return VideoParsingChunkedFrames(input_keys=["metas", "video"], output_keys=None, args=args)


def _make_full_frames(
    audio_sample_rate: int = 10, *, extract_audio: bool = False, stride: int = 2
) -> VideoParsingWithFullFrames:
    args = dict(_BASE_ARGS)
    args["audio_sample_rate"] = audio_sample_rate
    args["extract_audio"] = extract_audio
    args["min_stride"] = stride
    args["max_stride"] = stride
    return VideoParsingWithFullFrames(input_keys=["metas", "video"], output_keys=None, args=args)


def _audio_core_stub() -> ModuleType:
    core_stub = ModuleType("torchcodec._core")
    setattr(core_stub, "create_from_bytes", MagicMock(return_value=object()))
    setattr(
        core_stub,
        "get_container_metadata",
        MagicMock(return_value=SimpleNamespace(best_audio_stream_index=0)),
    )
    return core_stub


def _audio_decoder_mock(audio: torch.Tensor, sample_rate: int, pts_seconds: float = 0.0) -> MagicMock:  # audio: [C,N]
    decoder = MagicMock()
    decoder.get_all_samples.return_value = SimpleNamespace(
        data=audio,
        sample_rate=sample_rate,
        pts_seconds=pts_seconds,
    )
    return decoder


def _video_decoder_mock(num_frames: int, video_fps: float, pts_offset_seconds: float) -> MagicMock:
    decoder = MagicMock()
    decoder.__len__.return_value = num_frames

    def _get_frames_at(indices: list[int]) -> SimpleNamespace:
        data = torch.zeros((len(indices), 3, 2, 2), dtype=torch.uint8)  # [T,C,H,W]
        pts_seconds = torch.tensor(
            [pts_offset_seconds + index / video_fps for index in indices], dtype=torch.float64
        )  # [T]
        duration_seconds = torch.full((len(indices),), 1 / video_fps, dtype=torch.float64)  # [T]
        return SimpleNamespace(data=data, pts_seconds=pts_seconds, duration_seconds=duration_seconds)

    decoder.get_frames_at.side_effect = _get_frames_at
    return decoder


def _video_data(num_frames: int, video_fps: float = 10.0) -> dict[str, object]:
    return {
        "metas": {
            "width": 2,
            "height": 2,
            "framerate": video_fps,
            "nb_frames": num_frames,
        },
        "video": b"video",
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_stride_config_coercion_omegaconf_str_keys() -> None:
    # OmegaConf serializes int keys as strings; __init__ must coerce back to {int: float}.
    aug = _make(OmegaConf.create({"1": 0.5, "2": 0.5}))
    assert aug.stride_config == {1: 0.5, 2: 0.5}
    assert all(isinstance(k, int) for k in aug.stride_config)
    assert all(isinstance(v, float) for v in aug.stride_config.values())


@pytest.mark.L0
@pytest.mark.CPU
def test_stride_config_none_leaves_exp_decay_unchanged() -> None:
    aug = _make(None)
    assert aug.stride_config is None
    # exp-decay path still yields a valid stride in [min_stride, max_stride].
    s = aug._sample_stride_with_bias(aug.max_stride, aug.min_stride)
    assert aug.min_stride <= s <= aug.max_stride


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "config,expected",
    [
        ({2: 1.0}, {2: 1.0}),  # baseline: always half-FPS
        ({1: 0.5, 2: 0.5}, {1: 0.5, 2: 0.5}),  # mix50: per-video 50/50 native+half
        ({1: 0.3, 2: 0.7}, {1: 0.3, 2: 0.7}),  # generic ratios
    ],
)
def test_stride_config_realized_distribution(config: dict[int, float], expected: dict[int, float]) -> None:
    aug = _make(OmegaConf.create({str(k): v for k, v in config.items()}))
    np.random.seed(0)  # deterministic => flake-free
    n = 8000
    c = Counter(aug._sample_stride_with_bias(aug.max_stride, aug.min_stride) for _ in range(n))
    realized = {k: c[k] / n for k in config}
    for k, p in expected.items():
        assert abs(realized[k] - p) < 0.03, f"stride {k}: realized {realized[k]:.3f} vs expected {p}"
    # only strides from the config are ever sampled
    assert set(c) <= set(config)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "bad",
    [
        {},  # empty mapping
        {0: 1.0},  # stride < 1
        {1: 0.5, 2: 0.4},  # probabilities do not sum to 1
    ],
)
def test_stride_config_validation_rejects_bad_input(bad: dict[int, float]) -> None:
    with pytest.raises(AssertionError):
        _make(OmegaConf.create({str(k): v for k, v in bad.items()}))


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    (
        "frame_indices",
        "frame_stride",
        "frame_end_exclusive",
        "first_frame_pts_seconds",
        "last_frame_pts_seconds",
        "expected_range",
    ),
    [
        ([2, 3, 4], 1, 30, 1.2, 1.4, (1.2, 1.5)),
        ([2, 4, 6], 2, 30, 1.2, 1.6, (1.2, 1.8)),
        ([2, 5, 8], 3, 30, 1.2, 1.8, (1.2, 2.1)),
        ([2, 5, 8], 3, 10, 1.2, 1.8, (1.2, 2.0)),
        ([8], 3, 10, 1.8, 1.8, (1.8, 2.0)),
        ([24, 27], 3, 28, 3.4, 3.7, (3.4, 3.8)),
    ],
)
def test_get_audio_time_range_is_stride_aware_and_boundary_clamped(
    frame_indices: list[int],
    frame_stride: int,
    frame_end_exclusive: int,
    first_frame_pts_seconds: float,
    last_frame_pts_seconds: float,
    expected_range: tuple[float, float],
) -> None:
    audio_time_range = VideoParsingWithFullFrames._get_audio_time_range(
        video_fps=10.0,
        frame_indices=frame_indices,
        frame_stride=frame_stride,
        frame_end_exclusive=frame_end_exclusive,
        first_frame_pts_seconds=first_frame_pts_seconds,
        last_frame_pts_seconds=last_frame_pts_seconds,
    )

    assert audio_time_range == pytest.approx(expected_range)


@pytest.mark.L0
@pytest.mark.CPU
def test_extract_audio_chunk_resamples_to_exact_requested_range() -> None:
    augmentor = _make_full_frames(audio_sample_rate=10)
    decoded_audio = torch.zeros((1, 3), dtype=torch.float32)  # [C,N_orig]
    resampled_audio = torch.arange(6, dtype=torch.float32).reshape(1, 6)  # [C,N_resampled]
    resampled_audio_np = resampled_audio.numpy()  # [C,N_resampled]
    decoder = _audio_decoder_mock(decoded_audio, sample_rate=5, pts_seconds=0.5)
    librosa_stub = ModuleType("librosa")
    resample_mock = MagicMock(return_value=resampled_audio_np)
    setattr(librosa_stub, "resample", resample_mock)

    with (
        patch.dict(sys.modules, {"torchcodec._core": _audio_core_stub(), "librosa": librosa_stub}),
        patch.object(video_parsing, "AudioDecoder", return_value=decoder) as decoder_class,
    ):
        audio_chunk = augmentor._extract_audio_chunk(
            b"video", start_seconds=0.5, stop_seconds=1.1
        )  # [C,N_target] or None

    assert audio_chunk is not None
    torch.testing.assert_close(audio_chunk, resampled_audio)
    decoder_class.assert_called_once_with(b"video")
    resample_mock.assert_called_once()


@pytest.mark.L0
@pytest.mark.CPU
def test_extract_audio_chunk_preserves_late_start_and_early_end_as_silence() -> None:
    augmentor = _make_full_frames(audio_sample_rate=10)
    decoded_audio = torch.arange(1, 5, dtype=torch.float32).reshape(1, 4)  # [C,N_orig]
    decoder = _audio_decoder_mock(decoded_audio, sample_rate=10, pts_seconds=0.4)

    with (
        patch.dict(sys.modules, {"torchcodec._core": _audio_core_stub()}),
        patch.object(video_parsing, "AudioDecoder", return_value=decoder),
    ):
        audio_chunk = augmentor._extract_audio_chunk(
            b"video", start_seconds=0.2, stop_seconds=1.0
        )  # [C,N_target] or None

    expected = torch.tensor([[0, 0, 1, 2, 3, 4, 0, 0]], dtype=torch.float32)  # [C,N_target]
    assert audio_chunk is not None
    torch.testing.assert_close(audio_chunk, expected)


@pytest.mark.L0
@pytest.mark.CPU
def test_extract_audio_chunk_crops_audio_before_requested_pts() -> None:
    augmentor = _make_full_frames(audio_sample_rate=10)
    decoded_audio = torch.arange(1, 11, dtype=torch.float32).reshape(1, 10)  # [C,N_orig]
    decoder = _audio_decoder_mock(decoded_audio, sample_rate=10, pts_seconds=0.0)

    with (
        patch.dict(sys.modules, {"torchcodec._core": _audio_core_stub()}),
        patch.object(video_parsing, "AudioDecoder", return_value=decoder),
    ):
        audio_chunk = augmentor._extract_audio_chunk(
            b"video", start_seconds=0.2, stop_seconds=0.7
        )  # [C,N_target] or None

    expected = torch.arange(3, 8, dtype=torch.float32).reshape(1, 5)  # [C,N_target]
    assert audio_chunk is not None
    torch.testing.assert_close(audio_chunk, expected)


@pytest.mark.L0
@pytest.mark.CPU
def test_extract_audio_chunk_returns_silence_when_ranges_do_not_overlap() -> None:
    augmentor = _make_full_frames(audio_sample_rate=10)
    decoded_audio = torch.arange(1, 6, dtype=torch.float32).reshape(1, 5)  # [C,N_orig]
    decoder = _audio_decoder_mock(decoded_audio, sample_rate=10, pts_seconds=2.0)

    with (
        patch.dict(sys.modules, {"torchcodec._core": _audio_core_stub()}),
        patch.object(video_parsing, "AudioDecoder", return_value=decoder),
    ):
        audio_chunk = augmentor._extract_audio_chunk(
            b"video", start_seconds=0.2, stop_seconds=1.0
        )  # [C,N_target] or None

    expected = torch.zeros((1, 8), dtype=torch.float32)  # [C,N_target]
    assert audio_chunk is not None
    torch.testing.assert_close(audio_chunk, expected)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("start_seconds", "stop_seconds"),
    [(0.0, 0.0), (1.0, 0.0), (float("nan"), 1.0), (0.0, float("nan"))],
)
def test_extract_audio_chunk_rejects_invalid_time_range(start_seconds: float, stop_seconds: float) -> None:
    augmentor = _make_full_frames(audio_sample_rate=10)

    with patch.object(video_parsing, "AudioDecoder") as decoder_class:
        audio_chunk = augmentor._extract_audio_chunk(
            b"video", start_seconds=start_seconds, stop_seconds=stop_seconds
        )  # None

    assert audio_chunk is None
    decoder_class.assert_not_called()


@pytest.mark.L0
@pytest.mark.CPU
def test_full_frames_passes_pts_and_media_bound_to_audio_extraction() -> None:
    video_fps = 10.0
    augmentor = _make_full_frames(audio_sample_rate=10, extract_audio=True, stride=3)
    decoder = _video_decoder_mock(num_frames=13, video_fps=video_fps, pts_offset_seconds=0.25)
    audio_chunk = torch.ones((1, 1), dtype=torch.float32)  # [C,N_audio]

    with (
        patch.object(video_parsing, "VideoDecoder", return_value=decoder),
        patch.object(augmentor, "_extract_audio_chunk", return_value=audio_chunk) as extract_audio_mock,
    ):
        output = augmentor(_video_data(num_frames=13, video_fps=video_fps))

    assert output is not None
    assert decoder.get_frames_at.call_args.args[0] == [0, 3, 6, 9, 12]
    call_kwargs = extract_audio_mock.call_args.kwargs
    assert call_kwargs["video_bytes"] == b"video"
    assert call_kwargs["start_seconds"] == pytest.approx(0.25)
    assert call_kwargs["stop_seconds"] == pytest.approx(1.55)


@pytest.mark.L0
@pytest.mark.CPU
def test_chunked_frames_passes_pts_and_chunk_bound_to_audio_extraction() -> None:
    video_fps = 10.0
    args = dict(_BASE_ARGS)
    args.update(
        audio_sample_rate=10,
        extract_audio=True,
        min_stride=3,
        max_stride=3,
    )
    augmentor = VideoParsingChunkedFrames(input_keys=["metas", "video"], output_keys=None, args=args)
    decoder = _video_decoder_mock(num_frames=20, video_fps=video_fps, pts_offset_seconds=0.25)
    audio_chunk = torch.ones((1, 1), dtype=torch.float32)  # [C,N_audio]
    data_dict = _video_data(num_frames=20, video_fps=video_fps)
    data_dict["chunk_start_frame"] = 2
    data_dict["chunk_end_frame"] = 15

    with (
        patch.object(video_parsing, "VideoDecoder", return_value=decoder),
        patch.object(augmentor, "_extract_audio_chunk", return_value=audio_chunk) as extract_audio_mock,
    ):
        output = augmentor(data_dict)

    assert output is not None
    assert decoder.get_frames_at.call_args.args[0] == [2, 5, 8, 11, 14]
    call_kwargs = extract_audio_mock.call_args.kwargs
    assert call_kwargs["video_bytes"] == b"video"
    assert call_kwargs["start_seconds"] == pytest.approx(0.45)
    assert call_kwargs["stop_seconds"] == pytest.approx(1.75)
