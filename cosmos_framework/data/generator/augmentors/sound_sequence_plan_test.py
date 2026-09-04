# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Any

import pytest
import torch

from cosmos_framework.data.generator.augmentors import sound_sequence_plan as sound_sequence_plan_module
from cosmos_framework.data.generator.augmentors.sound_sequence_plan import SoundSequencePlanBuilder
from cosmos_framework.data.generator.sequence_packing import SequencePlan

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _data_dict(vision_indexes: list[int]) -> dict[str, Any]:
    return {
        "video": torch.zeros(3, 9, 4, 4),
        "sound": torch.zeros(1, 48000),
        "audio_sample_rate": 48000,
        "conditioning_fps": 24.0,
        "sequence_plan": SequencePlan(
            has_text=True,
            has_vision=True,
            condition_frame_indexes_vision=vision_indexes,
        ),
    }


def _builder(*, mode: str, sound_prefix_conditioning_prob: float = 1.0) -> SoundSequencePlanBuilder:
    return SoundSequencePlanBuilder(
        input_keys=[],
        args={
            "mode": mode,
            "temporal_compression_factor": 4,
            "sound_latent_fps": 25.0,
            "sound_prefix_conditioning_prob": sound_prefix_conditioning_prob,
        },
    )


def _fail_if_rng_consumed(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail() -> float:
        raise AssertionError("random.random must not be consumed")

    monkeypatch.setattr(sound_sequence_plan_module.random, "random", _fail)


def test_t2vs_preserves_generated_sound_plan_for_v2v_sample() -> None:
    data = _data_dict([0, 1])

    result = _builder(mode="t2vs")(data)

    assert result is data
    assert data["sequence_plan"].condition_frame_indexes_vision == [0, 1]
    assert data["sequence_plan"].has_sound is True
    assert data["sequence_plan"].condition_frame_indexes_sound == []


def test_vs2vs_adds_matching_sound_prefix_for_v2v_sample() -> None:
    data = _data_dict([0, 1])

    result = _builder(mode="vs2vs", sound_prefix_conditioning_prob=1.0)(data)

    assert result is data
    assert data["sequence_plan"].condition_frame_indexes_vision == [0, 1]
    assert data["sequence_plan"].has_sound is True
    assert data["sequence_plan"].condition_frame_indexes_sound == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(("draw", "expected_prefix"), [(0.49, [0, 1, 2, 3, 4]), (0.5, [])])
def test_vs2vs_draw_boundary_respects_configured_probability(
    monkeypatch: pytest.MonkeyPatch, draw: float, expected_prefix: list[int]
) -> None:
    data = _data_dict([0, 1])
    sound = data["sound"]
    monkeypatch.setattr(sound_sequence_plan_module.random, "random", lambda: draw)

    result = _builder(mode="vs2vs", sound_prefix_conditioning_prob=0.5)(data)

    assert result is data
    assert data["sound"] is sound
    assert data["sequence_plan"].has_sound is True
    assert data["sequence_plan"].condition_frame_indexes_sound == expected_prefix


def test_vs2vs_disables_sound_when_sample_has_no_sound(monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_if_rng_consumed(monkeypatch)
    data = _data_dict([0, 1])
    data["sound"] = None

    result = _builder(mode="vs2vs")(data)

    assert result is data
    assert data["sequence_plan"].condition_frame_indexes_vision == [0, 1]
    assert data["sequence_plan"].has_sound is False
    assert data["sequence_plan"].condition_frame_indexes_sound == []


@pytest.mark.parametrize("vision_indexes", [[], [0]])
def test_vs2vs_keeps_joint_sound_generation_for_short_vision_prefix(
    vision_indexes: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_if_rng_consumed(monkeypatch)
    data = _data_dict(vision_indexes)

    result = _builder(mode="vs2vs")(data)

    assert result is data
    assert data["sequence_plan"].condition_frame_indexes_vision == vision_indexes
    assert data["sequence_plan"].has_sound is True
    assert data["sequence_plan"].condition_frame_indexes_sound == []


def test_vs2vs_skips_sample_when_audio_too_short_for_prefix() -> None:
    data = _data_dict([0, 1])
    # 9600 samples @ 48 kHz -> 5 sound latents at 25 Hz; the 2-latent vision
    # prefix maps to 5 sound latents, leaving no future latent to predict.
    data["sound"] = torch.zeros(1, 5 * 1920)
    sound = data["sound"]

    result = _builder(mode="vs2vs", sound_prefix_conditioning_prob=1.0)(data)

    assert result is None
    assert data["sound"] is sound
    assert data["sequence_plan"].condition_frame_indexes_vision == [0, 1]
    assert data["sequence_plan"].has_sound is False
    assert data["sequence_plan"].condition_frame_indexes_sound == []


def test_vs2vs_requires_upstream_vision_plan() -> None:
    data = _data_dict([0, 1])
    del data["sequence_plan"]

    with pytest.raises(ValueError, match="upstream vision sequence plan"):
        _builder(mode="vs2vs")(data)


@pytest.mark.parametrize("vision_indexes", [[2, 3], [0, 2]])
def test_vs2vs_rejects_non_contiguous_vision_prefix(vision_indexes: list[int], monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_if_rng_consumed(monkeypatch)
    data = _data_dict(vision_indexes)

    with pytest.raises(ValueError, match="contiguous vision prefix"):
        _builder(mode="vs2vs")(data)
