# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.augmentors.transfer_control_input.control_input import AddControlInputBlur

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _test_frames() -> torch.Tensor:
    """Deterministic 3-frame clip with smooth gradients, sharp rectangles and fine noise texture."""
    rng = np.random.default_rng(0)
    height, width = 240, 416
    y, x = np.mgrid[0:height, 0:width]
    image = (127 + 60 * np.sin(x / 37.0) + 40 * np.cos(y / 23.0)).astype(np.float32)
    for _ in range(20):
        x0, y0 = rng.integers(0, width - 80), rng.integers(0, height - 60)
        image[y0 : y0 + rng.integers(10, 60), x0 : x0 + rng.integers(10, 80)] = rng.integers(0, 255)
    image = np.clip(image + rng.normal(0, 25, image.shape), 0, 255).astype(np.uint8)
    return torch.from_numpy(np.repeat(image[None, None], 3, axis=0).repeat(3, axis=1))  # [3,T,H,W]


def _blur_error(frames: torch.Tensor, preset: str) -> float:
    augmentor = AddControlInputBlur(
        input_keys=["video"], output_keys=["control_input_blur"], use_random=False, downup_preset=preset
    )
    control = augmentor({"video": frames})["control_input_blur"]  # [3,T,H,W]
    return (control.float() - frames.float()).abs().mean().item()


def test_none_preset_returns_input_unchanged() -> None:
    frames = _test_frames()
    assert _blur_error(frames, "none") == 0.0


def test_presets_blur_in_named_order() -> None:
    """The full chain very_low < low < medium < high < very_high, pinned at the test clip's size.

    The order depends on resolution (see the comment on blur_downup_preset_values), so this
    checks it at one fixed size rather than claiming it for every input.
    """
    frames = _test_frames()
    names = ("very_low", "low", "medium", "high", "very_high")
    errors = [_blur_error(frames, preset) for preset in names]
    report = " ".join(f"{name}={error:.3f}" for name, error in zip(names, errors, strict=True))
    assert all(a < b for a, b in zip(errors, errors[1:])), report
