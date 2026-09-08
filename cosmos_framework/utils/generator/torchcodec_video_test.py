# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for the streaming all-frames decode helper.

The fixture is a real 15-frame H.264 clip, embedded rather than committed as a binary because the
repo keeps no binary test assets. Tests that need a container whose edit list breaks indexed
decoding patch the fixture's ``elst`` box in place, so the only difference between the working and
the failing case is the edit list itself.
"""

from __future__ import annotations

import base64
import logging
import struct
from pathlib import Path

import pytest
import torch

from cosmos_framework.utils.generator import torchcodec_video
from cosmos_framework.utils.generator.torchcodec_video import (
    decode_all_frames_nhwc_uint8,
    decode_frames_nhwc_uint8,
    probe_video,
)

_LOG_NAME = "cosmos_framework.utils.generator.torchcodec_video"

# testsrc, 32x32, 10 fps, 1.5 s -> 15 frames, keyframe every 5.
_CLIP_B64 = (
    "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAACnxtZGF0AAACqgYF//+m3EXpvebZSLeWLNgg2SPu"
    "73gyNjQgLSBjb3JlIDE2NCByMzEwOCAzMWUxOWY5IC0gSC4yNjQvTVBFRy00IEFWQyBjb2RlYyAtIENvcHlsZWZ0IDIwMDMt"
    "MjAyMyAtIGh0dHA6Ly93d3cudmlkZW9sYW4ub3JnL3gyNjQuaHRtbCAtIG9wdGlvbnM6IGNhYmFjPTEgcmVmPTMgZGVibG9j"
    "az0xOjA6MCBhbmFseXNlPTB4MzoweDExMyBtZT1oZXggc3VibWU9NyBwc3k9MSBwc3lfcmQ9MS4wMDowLjAwIG1peGVkX3Jl"
    "Zj0xIG1lX3JhbmdlPTE2IGNocm9tYV9tZT0xIHRyZWxsaXM9MSA4eDhkY3Q9MSBjcW09MCBkZWFkem9uZT0yMSwxMSBmYXN0"
    "X3Bza2lwPTEgY2hyb21hX3FwX29mZnNldD0tMiB0aHJlYWRzPTEgbG9va2FoZWFkX3RocmVhZHM9MSBzbGljZWRfdGhyZWFk"
    "cz0wIG5yPTAgZGVjaW1hdGU9MSBpbnRlcmxhY2VkPTAgYmx1cmF5X2NvbXBhdD0wIGNvbnN0cmFpbmVkX2ludHJhPTAgYmZy"
    "YW1lcz0zIGJfcHlyYW1pZD0yIGJfYWRhcHQ9MSBiX2JpYXM9MCBkaXJlY3Q9MSB3ZWlnaHRiPTEgb3Blbl9nb3A9MCB3ZWln"
    "aHRwPTIga2V5aW50PTUga2V5aW50X21pbj0xIHNjZW5lY3V0PTQwIGludHJhX3JlZnJlc2g9MCByY19sb29rYWhlYWQ9NSBy"
    "Yz1jcmYgbWJ0cmVlPTEgY3JmPTIzLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEu"
    "NDAgYXE9MToxLjAwAIAAAAGwZYiEAEfwNJeDpT1X/4Rq6LJgoEZ4DbaeZpPs4R2AMIjom84EKp5Z5Lc849BCbBCs+HUHWp4R"
    "EN34VO7uknLC+6Kv6QClKPekhJO12isMCizynQHpd/zcBvTglmm9xk9ND28vvd4ZpUCQeLJNaCQQJhBHXAe12WvGgGGOxDMW"
    "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"
)

# Patching the edit list to this media_time makes TorchCodec's frame count and its decode start
# disagree, which is what breaks indexed decoding. Other values happen not to diverge, so the exact
# value matters and is pinned deliberately.
_BREAKING_MEDIA_TIME = 5000


def _write_clip(path: Path) -> Path:
    path.write_bytes(base64.b64decode("".join(_CLIP_B64)))
    return path


def _with_edit_list(src: Path, dst: Path, media_time: int) -> Path:
    """Rewrite the clip's ``elst`` media_time, changing nothing else."""
    data = bytearray(src.read_bytes())
    box = data.find(b"elst")
    assert box > 0, "fixture is expected to carry an edit list"
    # elst payload: version+flags (4), entry_count (4), then entries of
    # segment_duration (4), media_time (4), media_rate (4).
    struct.pack_into(">i", data, box + 4 + 4 + 4 + 4, media_time)
    dst.write_bytes(bytes(data))
    return dst


@pytest.fixture
def plain_clip(tmp_path: Path) -> Path:
    return _write_clip(tmp_path / "plain.mp4")


@pytest.fixture
def edit_list_clip(tmp_path: Path) -> Path:
    return _with_edit_list(_write_clip(tmp_path / "src.mp4"), tmp_path / "elst.mp4", _BREAKING_MEDIA_TIME)


class _FakeDecoder:
    """Yields ``real_frames`` [C,H,W] tensors while its metadata claims ``declared_frames``.

    Matches the real API: iteration yields bare tensors, not frame objects.
    """

    def __init__(self, real_frames: int, declared_frames: int, fps: float = 10.0):
        self._real_frames = real_frames
        self.metadata = type("_M", (), {"num_frames": declared_frames, "average_fps": fps})()

    def __iter__(self):
        for i in range(self._real_frames):
            yield torch.full((3, 4, 5), i, dtype=torch.uint8)


def test_decode_all_frames_matches_the_indexed_path_on_a_plain_clip(plain_clip: Path) -> None:
    """With nothing unusual in the container, streaming must reproduce indexed decoding exactly."""
    expected, _ = decode_frames_nhwc_uint8(plain_clip, list(range(probe_video(plain_clip).num_frames)))

    frames, metadata = decode_all_frames_nhwc_uint8(plain_clip)

    assert frames.shape == expected.shape
    assert (frames == expected).all(), "streaming decode changed the pixels of an ordinary video"
    assert metadata.num_frames == frames.shape[0] == 15
    assert metadata.average_fps == pytest.approx(10.0)


def test_indexed_decode_breaks_on_an_edit_list_clip(edit_list_clip: Path) -> None:
    """Documents the upstream behaviour this module works around.

    TorchCodec 0.10.0 counts frames on the full media timeline but starts decoding at a mis-scaled
    edit-list offset, so ``range(num_frames)`` runs past the end. If this test ever fails, TorchCodec
    has changed and the workaround in ``decode_all_frames_nhwc_uint8`` should be re-examined.
    """
    count = probe_video(edit_list_clip).num_frames

    with pytest.raises(RuntimeError, match="no more frames"):
        decode_frames_nhwc_uint8(edit_list_clip, list(range(count)))


def test_decode_all_frames_survives_an_edit_list_clip(edit_list_clip: Path) -> None:
    """The point of the module: the clip above decodes instead of raising."""
    frames, metadata = decode_all_frames_nhwc_uint8(edit_list_clip)

    assert frames.shape[0] > 0
    assert metadata.num_frames == frames.shape[0]
    assert frames.dtype.name == "uint8"


def test_decode_all_frames_returns_nhwc_uint8(plain_clip: Path) -> None:
    frames, metadata = decode_all_frames_nhwc_uint8(plain_clip)

    assert frames.shape == (15, 32, 32, 3), "layout must stay [T,H,W,C] for read_video callers"
    assert frames.dtype.name == "uint8"
    assert (metadata.height, metadata.width) == (32, 32)


def test_decode_all_frames_grows_when_the_count_under_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    """The buffer is sized from metadata; a stream longer than that must not lose frames."""
    monkeypatch.setattr(
        torchcodec_video, "_build_decoder", lambda source, **kwargs: _FakeDecoder(real_frames=10, declared_frames=2)
    )

    frames, metadata = decode_all_frames_nhwc_uint8("fake.mp4")

    assert frames.shape == (10, 4, 5, 3)
    assert metadata.num_frames == 10
    for i in range(10):
        assert frames[i, 0, 0, 0] == i, "a grown buffer must keep frames in order"


def test_decode_all_frames_warns_when_the_count_disagrees(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        torchcodec_video, "_build_decoder", lambda source, **kwargs: _FakeDecoder(real_frames=8, declared_frames=12)
    )

    with caplog.at_level(logging.WARNING, logger=_LOG_NAME):
        decode_all_frames_nhwc_uint8("fake.mp4")

    assert "indexed scan: 12 frames" in caplog.text
    assert "decoded: 8" in caplog.text


def test_decode_all_frames_is_quiet_when_the_count_agrees(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        torchcodec_video, "_build_decoder", lambda source, **kwargs: _FakeDecoder(real_frames=6, declared_frames=6)
    )

    with caplog.at_level(logging.WARNING, logger=_LOG_NAME):
        decode_all_frames_nhwc_uint8("fake.mp4")

    assert caplog.text == ""


def test_decode_all_frames_rejects_an_empty_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        torchcodec_video, "_build_decoder", lambda source, **kwargs: _FakeDecoder(real_frames=0, declared_frames=5)
    )

    with pytest.raises(ValueError, match="zero frames"):
        decode_all_frames_nhwc_uint8("fake.mp4")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
