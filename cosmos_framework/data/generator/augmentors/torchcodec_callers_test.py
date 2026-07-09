# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for Cosmos3 dataset call sites that use TorchCodec video helpers."""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.L1, pytest.mark.CPU]


def _import_or_skip(module_name: str) -> Any:
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.skip(f"Optional dependency unavailable while importing {module_name}: {exc.name}")


def test_pkl_qwen_decoder_uses_probe_and_decodes_selected_window(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_or_skip("cosmos_framework.data.generator.augmentors.pkl_to_media")
    video_bytes = b"video-bytes"
    calls: dict[str, object] = {}

    def fake_probe_video(source: object, num_threads: int = 0, **_: object) -> SimpleNamespace:
        calls["probe"] = (source, num_threads)
        return SimpleNamespace(num_frames=10, average_fps=12.0, height=8, width=10)

    def fake_decode_frames_tchw_uint8(
        source: object,
        indices: list[int],
        num_threads: int = 0,
        **_: object,
    ) -> tuple[torch.Tensor, SimpleNamespace]:
        calls["decode"] = (source, indices, num_threads)
        frames = torch.arange(3 * 3 * 8 * 10, dtype=torch.uint8).reshape(3, 3, 8, 10)  # [T,C,H,W]
        return frames, fake_probe_video(source, num_threads=num_threads)

    monkeypatch.setattr(module, "probe_video", fake_probe_video)
    monkeypatch.setattr(module, "decode_frames_tchw_uint8", fake_decode_frames_tchw_uint8)
    monkeypatch.setattr(module, "smart_nframes", lambda *_args, **_kwargs: 3)
    monkeypatch.setattr(module, "smart_resize", lambda height, width, **_kwargs: (height, width))

    result = module._video_decoder_qwen_func(
        key="clip.mp4",
        data=video_bytes,
        min_fps_thres=1,
        max_fps_thres=60,
        target_fps=3.0,
        min_video_token_length=1,
        max_video_token_length=1024,
        num_threads=5,
        start_frame=2,
        end_frame=8,
    )

    assert calls["probe"] == (video_bytes, 5)
    assert calls["decode"] == (video_bytes, [2, 4, 7], 5)
    assert result is not None
    assert tuple(result["videos"].shape) == (3, 3, 8, 10)
    assert result["videos"].dtype == torch.float32
    assert result["fps"] == pytest.approx(6.0)


@pytest.mark.parametrize(
    "module_name",
    [
        "cosmos_framework.data.generator.augmentors.reasoner.bytes_to_media",
        "cosmos_framework.utils.datasets.augmentors.bytes_to_media",
    ],
)
def test_bytes_to_media_uses_probe_durations_for_token_budget(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    module = _import_or_skip(module_name)
    seen_sources: list[tuple[object, int]] = []

    def fake_probe_video(source: object, num_threads: int = 0, **_: object) -> SimpleNamespace:
        seen_sources.append((source, num_threads))
        frame_counts = {b"first": 80, b"second": 40}
        return SimpleNamespace(num_frames=frame_counts[source], average_fps=20.0, height=16, width=16)

    monkeypatch.setattr(module, "probe_video", fake_probe_video)
    augmentor = module.BytesToMedia(
        min_video_token_length=8,
        max_video_token_length=80,
        num_threads=3,
        is_input_pickle_byptes=False,
    )

    durations = augmentor._get_video_durations(
        {
            "video_first.mp4": b"first",
            "control_input_depth": b"second",
            "image.jpg": b"image",
        },
        {},
    )

    assert durations == {"video_first.mp4": 4.0, "control_input_depth": 2.0}
    assert seen_sources == [(b"first", 3), (b"second", 3)]
    weighted_params = augmentor._get_decoder_params(
        video_count=2,
        video_duration=durations["video_first.mp4"],
        total_video_duration=sum(durations.values()),
    )
    assert weighted_params["max_video_token_length"] == 53


@pytest.mark.parametrize(
    "module_name",
    [
        "cosmos_framework.data.generator.augmentors.reasoner.bytes_to_media",
        "cosmos_framework.utils.datasets.augmentors.bytes_to_media",
    ],
)
def test_bytes_to_media_probe_duration_respects_start_end_frame(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    module = _import_or_skip(module_name)
    monkeypatch.setattr(
        module,
        "probe_video",
        lambda *_args, **_kwargs: SimpleNamespace(num_frames=100, average_fps=25.0, height=16, width=16),
    )
    augmentor = module.BytesToMedia(num_threads=2, is_input_pickle_byptes=False)

    duration = augmentor._probe_video_duration_seconds(
        b"video",
        identifier="media['video.mp4']",
        start_frame=7,
        end_frame=32,
    )
    empty_duration = augmentor._probe_video_duration_seconds(
        b"video",
        identifier="media['video.mp4']",
        start_frame=7,
        end_frame=7,
    )

    assert duration == pytest.approx(1.0)
    assert empty_duration is None


def test_multiview_extract_frames_resizes_torchcodec_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_or_skip("cosmos_framework.data.generator.multiview.multiview_dataset")
    calls: list[tuple[object, list[int]]] = []

    def fake_decode_frames_tchw_uint8(
        source: object,
        indices: list[int],
        **_: object,
    ) -> tuple[torch.Tensor, SimpleNamespace]:
        calls.append((source, indices))
        frames = torch.arange(len(indices) * 3 * 4 * 6, dtype=torch.uint8).reshape(len(indices), 3, 4, 6)  # [T,C,H,W]
        return frames, SimpleNamespace(average_fps=29.97)

    monkeypatch.setattr(module, "decode_frames_tchw_uint8", fake_decode_frames_tchw_uint8)

    frames, fps, original_hw = module.ExtractFramesAndCaptions._extract_frames(  # [T,C,H,W]
        b"video",
        [1, 3, 5],
        (8, 12),
    )

    assert calls == [(b"video", [1, 3, 5])]
    assert tuple(frames.shape) == (3, 3, 8, 12)
    assert frames.dtype == torch.uint8
    assert fps == pytest.approx(29.97)
    assert original_hw == (4, 6)


def test_sekai_frame_count_uses_probe_video(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_or_skip("projects.cosmos3.sil.omnidreams.datasets.sekai")
    seen: list[bytes] = []

    def fake_probe_video(source: bytes) -> SimpleNamespace:
        seen.append(source)
        return SimpleNamespace(num_frames=37, average_fps=30.0, height=16, width=16)

    monkeypatch.setattr(module, "probe_video", fake_probe_video)

    assert module._num_available_video_frames(b"sekai-video") == 37
    assert seen == [b"sekai-video"]


def test_sekai_fit_window_clamps_to_available_span() -> None:
    module = _import_or_skip("projects.cosmos3.sil.omnidreams.datasets.sekai")

    assert module._fit_window_to_available_video(
        frame_start=20,
        num_video_frames=8,
        stride=2,
        num_available_frames=30,
    ) == (20, 5)
    assert module._fit_window_to_available_video(
        frame_start=100,
        num_video_frames=8,
        stride=3,
        num_available_frames=10,
    ) == (0, 4)
