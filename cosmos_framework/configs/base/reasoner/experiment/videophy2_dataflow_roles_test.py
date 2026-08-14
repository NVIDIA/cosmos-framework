# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from cosmos_framework.configs.base.reasoner.experiment.videophy2_dataflow_roles import (
    _decode_video_to_pil_frames,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class _FakeFramesTensor:
    def __init__(self, frame_count: int) -> None:
        self._frame_count = frame_count

    def permute(self, *_dims: int) -> _FakeFramesTensor:
        return self

    def contiguous(self) -> _FakeFramesTensor:
        return self

    def cpu(self) -> _FakeFramesTensor:
        return self

    def numpy(self) -> np.ndarray:
        return np.zeros((self._frame_count, 1, 1, 3), dtype=np.uint8)


def test_decode_video_updates_fps_after_frame_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    selected_indices: list[int] = []

    class FakeVideoDecoder:
        metadata = SimpleNamespace(num_frames=900, average_fps=30.0)

        def __init__(self, video_bytes: bytes) -> None:
            assert video_bytes == b"video-bytes"

        def get_frames_at(self, *, indices: list[int]) -> SimpleNamespace:
            selected_indices.extend(indices)
            return SimpleNamespace(data=_FakeFramesTensor(len(indices)))

    decoders_module = ModuleType("torchcodec.decoders")
    decoders_module.VideoDecoder = FakeVideoDecoder  # type: ignore[attr-defined]
    torchcodec_module = ModuleType("torchcodec")
    torchcodec_module.decoders = decoders_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torchcodec", torchcodec_module)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", decoders_module)

    frames, effective_fps = _decode_video_to_pil_frames(b"video-bytes")

    assert len(selected_indices) == 32
    assert len(frames) == 32
    assert effective_fps == pytest.approx(32 / 900 * 30.0)
