# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest


def _make_v2v_sample_args(**overrides: Any) -> SimpleNamespace:
    """v2v ``OmniSampleArgs`` stand-in for ``get_sample_data`` tests."""
    from cosmos_framework.inference.args import ModelMode, NegativeMetadataMode

    defaults = dict(
        action_path=None,
        aspect_ratio="16,9",
        autoregressive=False,
        camera_trajectory=None,
        condition_frame_indexes_vision=[0, 1],
        condition_video_keep=None,
        condition_vision_mode="video",
        duration_template=None,
        enable_sound=False,
        fps=24,
        inverse_duration_template=None,
        inverse_resolution_template=None,
        model_mode=ModelMode.VIDEO2VIDEO,
        negative_metadata_mode=NegativeMetadataMode.NONE,
        negative_prompt=None,
        num_frames=125,
        num_outputs=1,
        prompt="prompt",
        resolution_template=None,
        transfer_hints={},
        vision_path="conditioning.mp4",
        vision_size=(32, 16),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.parametrize(
    ("condition_video_keep", "expected_loader_keep"),
    [
        ("last", "last"),
        ("first", "first"),
        (None, "first"),  # default
    ],
)
def test_video_conditioning_plumbs_keep_and_pixel_frame_count(
    monkeypatch: pytest.MonkeyPatch,
    condition_video_keep: str | None,
    expected_loader_keep: str,
) -> None:
    """v2v: tokenizer derives pixel-frame count from latent count, ``keep`` passes through to the loader."""
    torch = pytest.importorskip("torch")

    from cosmos_framework.inference import inference

    class Tokenizer:
        calls: list[int]

        def __init__(self) -> None:
            self.calls = []

        def get_pixel_num_frames(self, num_latent_frames: int) -> int:
            self.calls.append(num_latent_frames)
            return 5

    tokenizer = Tokenizer()
    model = SimpleNamespace(
        input_image_key="image",
        input_video_key="video",
        input_caption_key="caption",
        tokenizer_vision_gen=tokenizer,
    )
    sample_args = _make_v2v_sample_args(condition_video_keep=condition_video_keep)
    conditioning_frames = torch.zeros(3, 5, 16, 32)
    sequence_plan = ["sequence-plan"]
    load_conditioning_video_mock = Mock(return_value=conditioning_frames)
    build_conditioned_video_batch_mock = Mock(
        return_value={
            "video": [torch.zeros(1, 3, 125, 16, 32)],
            "sequence_plan": sequence_plan,
        }
    )
    monkeypatch.setattr(inference, "load_conditioning_video", load_conditioning_video_mock)
    monkeypatch.setattr(inference, "build_conditioned_video_batch", build_conditioned_video_batch_mock)

    out = inference.get_sample_data(sample_args, model, device="cpu")

    assert tokenizer.calls == [2]  # max([0, 1]) + 1
    load_conditioning_video_mock.assert_called_once_with(
        Path("conditioning.mp4"),
        target_h=16,
        target_w=32,
        max_frames=5,
        keep=expected_loader_keep,
    )
    build_conditioned_video_batch_mock.assert_called_once()
    build_args, build_kwargs = build_conditioned_video_batch_mock.call_args
    assert build_args == (conditioning_frames,)
    assert build_kwargs == {
        "condition_frames_vision": [0, 1],
        "w": 32,
        "h": 16,
        "num_frames": 125,
        "fps": 24,
        "batch_size": 1,
    }
    assert out["sequence_plan"] is sequence_plan


def test_json_prompt_metadata_for_single_frame_omits_temporal_fields() -> None:
    from cosmos_framework.inference.inference import _format_json_prompt_with_template

    prompt = _format_json_prompt_with_template(
        {"subjects": [], "duration": "8s", "fps": 24.0},
        fps=24,
        num_frames=1,
        aspect_ratio="1,1",
        h=1024,
        w=1024,
        include_temporal_metadata=False,
    )

    assert prompt == '{"subjects": [], "resolution": {"H": 1024, "W": 1024}, "aspect_ratio": "1,1"}'
    parsed = json.loads(prompt)
    assert parsed["resolution"] == {"H": 1024, "W": 1024}
    assert parsed["aspect_ratio"] == "1,1"
    assert "duration" not in parsed
    assert "fps" not in parsed


def test_json_prompt_metadata_for_video_keeps_temporal_fields() -> None:
    from cosmos_framework.inference.inference import _format_json_prompt_with_template

    prompt = _format_json_prompt_with_template(
        {"subjects": []},
        fps=24,
        num_frames=189,
        aspect_ratio="16,9",
        h=720,
        w=1280,
        include_temporal_metadata=True,
    )

    assert prompt == (
        '{"subjects": [], "duration": "7s", "fps": 24.0, "resolution": {"H": 720, "W": 1280}, "aspect_ratio": "16,9"}'
    )
    assert json.loads(prompt) == {
        "subjects": [],
        "duration": "7s",
        "fps": 24.0,
        "resolution": {"H": 720, "W": 1280},
        "aspect_ratio": "16,9",
    }
