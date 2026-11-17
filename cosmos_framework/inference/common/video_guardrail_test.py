# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""The video guardrail post-processing runs without cuDNN autotuning and leaves the global setting as it was."""

import numpy as np
import pytest
import torch

from cosmos_framework.inference.common.inference import _run_video_guardrail

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class _Runner:
    """A guardrail runner that records the cuDNN autotune flag its post-processing ran under."""

    def __init__(self, safe: bool = True, fail: bool = False) -> None:
        self.safe = safe
        self.fail = fail
        self.seen: list[bool] = []
        self.safety_seen: list[bool] = []

    def run_safety_check(self, frames: np.ndarray) -> tuple[bool, str]:
        self.safety_seen.append(torch.backends.cudnn.benchmark)
        return self.safe, "" if self.safe else "unsafe"

    def postprocess(self, frames: np.ndarray) -> np.ndarray:
        self.seen.append(torch.backends.cudnn.benchmark)
        if self.fail:
            raise RuntimeError("detector failed")
        return frames


@pytest.mark.parametrize("autotune", [True, False])
def test_postprocess_runs_without_autotune_and_restores_flag(monkeypatch: pytest.MonkeyPatch, autotune: bool) -> None:
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", autotune)
    runner = _Runner()
    video = torch.rand(3, 2, 4, 4)

    out, _ = _run_video_guardrail(runner, video)

    assert runner.seen == [False]
    assert torch.backends.cudnn.benchmark is autotune
    assert out is not None and out.shape == video.shape


def test_flag_restored_when_postprocess_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    runner = _Runner(fail=True)
    with pytest.raises(RuntimeError, match="detector failed"):
        _run_video_guardrail(runner, torch.rand(3, 1, 4, 4))
    assert runner.seen == [False]
    assert torch.backends.cudnn.benchmark is True


def test_blocked_video_skips_postprocess_and_keeps_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    runner = _Runner(safe=False)

    out, message = _run_video_guardrail(runner, torch.rand(3, 1, 4, 4))

    assert out is None and message == "unsafe"
    assert runner.seen == []
    assert torch.backends.cudnn.benchmark is True


@pytest.mark.parametrize("autotune", [True, False])
def test_safety_check_sees_the_original_flag(monkeypatch: pytest.MonkeyPatch, autotune: bool) -> None:
    """Only post-processing runs with autotuning off; the safety check keeps the process setting."""
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", autotune)
    runner = _Runner()

    _run_video_guardrail(runner, torch.rand(3, 1, 4, 4))

    assert runner.safety_seen == [autotune]
    assert runner.seen == [False]
