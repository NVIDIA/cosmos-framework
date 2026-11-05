# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Preserve categorical controls in the legacy and streaming Transfer paths."""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from cosmos_framework.inference import transfer, transfer_streaming
from cosmos_framework.inference.args import TransferHintKey

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "hint", [TransferHintKey.SEG, TransferHintKey.DEPTH, TransferHintKey.EDGE, TransferHintKey.BLUR]
)
@pytest.mark.parametrize("source_hw", [(480, 832), (240, 416), (720, 1280), (480, 640)])
def test_precomputed_control_resize_preserves_palette_or_continuous_interpolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_hw: tuple[int, int],
    hint: TransferHintKey,
    streaming: bool,
) -> None:
    """Segmentation retains labels; other controls keep their legacy pixel values."""
    height, width = source_hw
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:, : width // 2, 0] = 255
    pixels[:, width // 2 :, 1] = 255
    pixels[height // 2 :, :, 2] = 255
    path = tmp_path / "control.png"
    Image.fromarray(pixels).save(path)
    control = SimpleNamespace(control_path=path)
    if streaming:
        # This test loads a PNG; video decoder version validation is covered separately.
        monkeypatch.setattr(transfer_streaming, "_require_causal_transfer_pyav", lambda: None)
        sample = SimpleNamespace(
            transfer_resize_mode="center_crop",
            resolution="480",
            aspect_ratio="16,9",
            max_frames=1,
            fps=30,
            vision_path=None,
            transfer_hints={hint: control},
        )
        actual = transfer_streaming._load_streaming_transfer_media(sample).control_frames
    else:
        actual = transfer.load_transfer_control_frames(
            hint_key=hint,
            transfer=control,
            resolution="480",
            aspect_ratio="16,9",
            max_frames=1,
        )

    assert actual.shape == (3, 1, 480, 832)
    assert actual.dtype == torch.uint8
    source = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    if source_hw == (480, 832):
        assert torch.equal(actual, source.permute(1, 0, 2, 3))
    if hint == TransferHintKey.SEG:
        palette = {(255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255)}
        colors = np.unique(actual.permute(1, 2, 3, 0).reshape(-1, 3).numpy(), axis=0)
        assert {tuple(color) for color in colors} == palette
    else:
        # Freeze the previous continuous-pixel resize/crop, including nonmatching aspects.
        scale = max(480 / height, 832 / width)
        expected = TF.resize(source, [math.ceil(scale * height), math.ceil(scale * width)])
        expected = TF.center_crop(expected, [480, 832]).permute(1, 0, 2, 3)
        assert torch.equal(actual, expected)
