# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Keyword arguments the Qwen3-VL wrapper forwards to the HuggingFace processor."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from cosmos_framework.data.generator.processors import base
from cosmos_framework.data.generator.processors.qwen3vl_processor import VIDEO_BUDGET_RESIZED_KEY, Qwen3VLProcessor


class _RecordingHFProcessor:
    """Stand-in for the HF Qwen3-VL processor (Qwen3-VL geometry) that records how it is called."""

    image_token = "<|image_pad|>"
    video_token = "<|video_pad|>"

    def __init__(self) -> None:
        self.image_processor = SimpleNamespace(size={"shortest_edge": 65536, "longest_edge": 16777216})
        self.video_processor = SimpleNamespace(patch_size=16, temporal_patch_size=2, merge_size=2)
        self.tokenizer = SimpleNamespace(
            convert_tokens_to_ids={"<|image_pad|>": 1, "<|video_pad|>": 2}.get, eos_token_id=3, pad_token_id=0
        )
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, torch.Tensor]:
        self.calls.append(kwargs)
        input_ids = torch.zeros(1, 4, dtype=torch.long)  # [1,N]
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}  # [1,N] each


@pytest.fixture
def processor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Qwen3VLProcessor:
    hf_processor = _RecordingHFProcessor()
    monkeypatch.setattr(base.AutoProcessor, "from_pretrained", lambda *args, **kwargs: hf_processor)
    return Qwen3VLProcessor(name=str(tmp_path))


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "budget_resized",
    [
        True,  # sized by the reasoner decoder: an HF resize would re-cap the token budget
        False,  # frames from any other producer keep the HF resize
    ],
)
def test_hf_video_resize_is_skipped_only_for_decoder_sized_frames(
    processor: Qwen3VLProcessor, budget_resized: bool
) -> None:
    frames = [Image.new("RGB", (96, 64)) for _ in range(4)]
    video = {"type": "video", "video": frames, "fps": 2.0, VIDEO_BUDGET_RESIZED_KEY: budget_resized}

    processor.apply_chat_template([{"role": "user", "content": [video, {"type": "text", "text": "Describe."}]}])

    kwargs = processor.processor.calls[-1]
    videos_kwargs = kwargs.get("processor_kwargs", kwargs)["videos_kwargs"]  # transformers>=5 nests them
    assert videos_kwargs.get("do_resize", True) is not budget_resized


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "other_video",
    [
        [Image.new("RGB", (96, 64)) for _ in range(4)],  # frames from another producer
        "clip.mp4",  # a video HF loads itself
    ],
)
def test_videos_mixing_decoder_sized_and_other_videos_are_rejected(
    processor: Qwen3VLProcessor, other_video: list[Image.Image] | str
) -> None:
    """HF takes one resize setting per call: it would re-cap the decoder-sized video or leave the other unresized."""
    frames = [Image.new("RGB", (96, 64)) for _ in range(4)]
    sized = {"type": "video", "video": frames, "fps": 2.0, VIDEO_BUDGET_RESIZED_KEY: True}
    other = {"type": "video", "video": other_video, "fps": 2.0}

    with pytest.raises(ValueError, match="sized by the reasoner decoder"):
        processor.apply_chat_template([{"role": "user", "content": [sized, other, {"type": "text", "text": "?"}]}])

    assert processor.processor.calls == []


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("add_vision_id", [False, True])
def test_add_vision_id_reaches_the_chat_template_only_when_set(
    processor: Qwen3VLProcessor, add_vision_id: bool
) -> None:
    messages = [{"role": "user", "content": [{"type": "text", "text": "Compare."}]}]

    processor.apply_chat_template(messages, add_vision_id=add_vision_id)

    kwargs = processor.processor.calls[-1]
    assert kwargs.get("add_vision_id", False) is add_vision_id
    assert "add_vision_id" not in kwargs.get("processor_kwargs", {})  # a template variable, not a processor kwarg
