# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Timestamp clocks for native temporal patches and independent source frames."""

from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw, ImageFont

from cosmos_framework.data.generator.augmentors.reasoner.timestamp import overlay_text
from cosmos_framework.data.generator.augmentors.reasoner.timestamp_without_augment_message import (
    TimeStampWithoutAugmentMessage,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_timestamp_clock_preserves_native_and_framewise_values() -> None:
    images = [Image.new("RGB", (64, 64)) for _ in range(4)]
    processor = SimpleNamespace(name="Qwen/Qwen3.5-27B", merge_size=2)
    native, timestamps = overlay_text(images, 2.0, processor=processor)
    assert native is images
    assert timestamps == [0.2, 0.2, 1.2, 1.2]
    framewise, timestamps = overlay_text(
        images,
        2.0,
        processor=processor,
        source_frames_indices=[0, 20, 39, 59],
        source_fps=30.0,
    )
    assert framewise is images
    assert timestamps == [0.0, 0.7, 1.3, 2.0]


def test_framewise_overlay_renders_the_same_source_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    images = [Image.new("RGB", (64, 64)) for _ in range(3)]
    font = ImageFont.load_default()
    monkeypatch.setattr(ImageFont, "truetype", lambda *args, **kwargs: font)
    drawn: list[str] = []
    original_text = ImageDraw.ImageDraw.text

    def capture_text(self: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, **kwargs: object) -> None:
        drawn.append(text)
        original_text(self, xy, text, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture_text)
    processed, timestamps = overlay_text(
        images,
        2.0,
        source_frames_indices=[0, 10, 35],
        source_fps=30.0,
    )
    assert timestamps == [0.0, 0.3, 1.2]
    assert drawn == ["0.00s", "0.30s", "1.20s"]
    assert all(image.size == (64, 92) for image in processed)


@pytest.mark.parametrize(
    "indices,source_fps,match",
    [
        ([0], None, "both"),
        (None, 30.0, "both"),
        ([0, 1], 30.0, "one source index"),
        ([0], 0.0, "positive finite"),
        ([0], float("nan"), "positive finite"),
    ],
)
def test_timestamp_rejects_inconsistent_source_metadata(
    indices: list[int] | None,
    source_fps: float | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        overlay_text([Image.new("RGB", (64, 64))], 2.0, source_frames_indices=indices, source_fps=source_fps)


def test_overlay_only_augmentor_uses_framewise_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    images = [Image.new("RGB", (64, 64)) for _ in range(4)]
    observed: list[list[float]] = []

    def capture_overlay(*args: object, **kwargs: object) -> tuple[list[Image.Image], list[float]]:
        output, timestamps = overlay_text(*args, **kwargs)
        observed.append(timestamps)
        return output, timestamps

    monkeypatch.setattr(
        "cosmos_framework.data.generator.augmentors.reasoner.timestamp_without_augment_message.overlay_text",
        capture_overlay,
    )
    conversation = [{"role": "assistant", "content": "Keep the original target."}]
    sample = {
        "__url__": SimpleNamespace(root="temporal-test"),
        "media": {
            "video": {"videos": images, "fps": 2.0, "source_frames_indices": [0, 20, 39, 59], "source_fps": 30.0}
        },
        "conversation": conversation,
    }
    augmentor = TimeStampWithoutAugmentMessage(
        urls_needs_timestamp=["temporal-test"],
        processor=SimpleNamespace(name="Qwen/Qwen3.5-27B", merge_size=2),
    )
    assert augmentor(sample) is sample
    assert sample["conversation"] is conversation
    assert observed == [[0.0, 0.7, 1.3, 2.0]]
