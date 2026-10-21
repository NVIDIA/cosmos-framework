# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Offline SFT packing with real local HF tokenization and media processors."""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("transformers", minversion="5.8.1")

from cosmos_framework.posttrain.data.packer.base import DataPacker
from cosmos_framework.posttrain.data.packer.sft import _MEDIA, QwenSFTDataPacker, SFTPolicyInput, build_sft_packer
from cosmos_framework.posttrain.schema import ObsBundle, SFTTarget, Transition
from cosmos_framework.posttrain.tests.helpers.sft_packer import (
    local_processor,
    local_tokenizer,
    official_qwen3_vl_chat_template,
    official_qwen3_vl_tokenizer_chat_template,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture
def text_packer() -> QwenSFTDataPacker:
    return QwenSFTDataPacker(local_tokenizer(), max_length=4096)


@pytest.fixture
def vlm_packer() -> QwenSFTDataPacker:
    tokenizer = local_tokenizer()
    return QwenSFTDataPacker(tokenizer, processor=local_processor(tokenizer), max_length=4096)


def _supervised(packer: QwenSFTDataPacker, sample: SFTPolicyInput) -> str:
    return packer.tokenizer.decode([token for token in sample.labels if token != -100])


def _image() -> Image.Image:
    return Image.fromarray(np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3))


def _video() -> np.ndarray:
    return np.arange(4 * 8 * 8 * 3, dtype=np.uint8).reshape(4, 8, 8, 3)


def test_multiturn_unshifted_labels_reasoning_tools(text_packer: QwenSFTDataPacker) -> None:
    target = SFTTarget(
        messages=[
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Compute a value."},
            {
                "role": "assistant",
                "content": "Checking.",
                "reasoning_content": "Use arithmetic.",
                "tool_calls": [{"function": {"name": "calculate", "arguments": {"x": 2}}}],
            },
            {"role": "tool", "content": "Tool result 1"},
            {"role": "tool", "content": "Tool result 2"},
            {"role": "assistant", "content": "The answer is two."},
            {"role": "user", "content": "Explain again."},
            {"role": "assistant", "content": "Deux. 二. 🦉"},
        ],
        tools=[{"type": "function", "function": {"name": "calculate"}}],
    )
    sample = text_packer.sft_process_sample(target)
    supervised = _supervised(text_packer, sample)
    assert "<think>\nUse arithmetic.\n</think>\n\nChecking." in supervised
    assert "<tool_call>" in supervised and "calculate" in supervised
    assert "The answer is two.<|im_end|>Deux. 二. 🦉<|im_end|>" in supervised
    assert supervised.count("<|im_end|>") == 3
    assert all(text not in supervised for text in ["system", "assistant", "Be precise", "Tool result", "Explain again"])
    assert all(label == -100 or label == token for token, label in zip(sample.input_ids, sample.labels, strict=True))
    transition = Transition(ObsBundle(), targets=target)
    assert transition.targets is target and transition.reward is None
    assert isinstance(target.messages[0]["content"], str), "processing must not mutate the source"


@pytest.mark.parametrize("packer_name", ["text_packer", "vlm_packer"])
def test_multiturn_inline_reasoning_is_preserved(request: pytest.FixtureRequest, packer_name: str) -> None:
    packer = request.getfixturevalue(packer_name)
    messages = [
        {"role": "user", "content": "First question"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<think>First reasoning"},
                {"type": "text", "text": "</think>First answer"},
            ],
        },
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "<think>Second reasoning</think>Second answer"},
    ]
    before = json.dumps(messages)
    result = _supervised(packer, packer.sft_process_sample(SFTTarget(messages)))
    assert result == (
        "<think>First reasoning</think>First answer<|im_end|><think>Second reasoning</think>Second answer<|im_end|>"
    )
    assert json.dumps(messages) == before


@pytest.mark.parametrize("packer_name", ["text_packer", "vlm_packer"])
def test_template_cannot_drop_earlier_inline_reasoning(request: pytest.FixtureRequest, packer_name: str) -> None:
    packer = request.getfixturevalue(packer_name)
    renderer = packer.processor if packer.processor is not None else packer.tokenizer
    # Reproduce Qwen3's removal of earlier inline reasoning without downloading a template.
    renderer.chat_template = (
        "{% for m in messages %}"
        "{% set text = m.content if m.content is string else m.content | map(attribute='text') | join %}"
        "<|im_start|>{{m.role}}\n"
        "{{ text.split('</think>')[-1] if m.role == 'assistant' and not loop.last else text }}"
        "<|im_end|>\n{% endfor %}"
    )
    target = SFTTarget(
        [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "<think>First reasoning</think>First answer"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "<think>Second reasoning</think>Second answer"},
        ]
    )
    with pytest.raises(ValueError, match="reasoning/tool calls were dropped"):
        packer.sft_process_sample(target)


def test_tool_only_assistant_and_implicit_system(text_packer: QwenSFTDataPacker) -> None:
    target = SFTTarget(
        [
            {"role": "user", "content": "Use a tool."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"function": {"name": "lookup", "arguments": {"query": "value"}}}],
            },
            {"role": "tool", "content": "found"},
            {"role": "assistant", "content": "done"},
        ],
        tools=[{"function": {"name": "lookup"}}],
    )
    result = _supervised(text_packer, text_packer.sft_process_sample(target))
    assert result.count("<|im_end|>") == 2
    assert "lookup" in result and "done" in result and "found" not in result


@pytest.mark.parametrize("kinds", [("image",), ("video",), ("image", "video", "image")])
def test_media_expansion_precedes_assistant_mask(vlm_packer: QwenSFTDataPacker, kinds: tuple[str, ...]) -> None:
    parts = [{"type": kind, kind: _image() if kind == "image" else _video()} for kind in kinds]
    target = SFTTarget(
        [
            {"role": "user", "content": [{"type": "text", "text": "Describe."}, *parts]},
            {"role": "assistant", "content": "First answer.", "reasoning_content": "Inspect all views."},
            {"role": "user", "content": "And now?"},
            {"role": "assistant", "content": "Second answer."},
        ]
    )
    sample = vlm_packer.sft_process_sample(target)
    supervised = _supervised(vlm_packer, sample)
    assert supervised == "<think>\nInspect all views.\n</think>\n\nFirst answer.<|im_end|>Second answer.<|im_end|>"
    assert sample.mm_kwargs
    for kind in set(kinds):
        marker = vlm_packer.tokenizer.convert_tokens_to_ids(f"<|{kind}_pad|>")
        assert sample.input_ids.count(marker) > kinds.count(kind), "fixture must exercise expansion, not single markers"
        assert all(
            label == -100 for token, label in zip(sample.input_ids, sample.labels, strict=True) if token == marker
        )
        assert sample.mm_kwargs[f"{kind}_grid_thw"].shape[0] == kinds.count(kind)
    assert all(value.device.type == "cpu" for value in sample.mm_kwargs.values())


@pytest.mark.parametrize(
    "expansion",
    [
        # Transformers 5.8.1: the placeholder is replaced by timestamped frame groups.
        "<0.0 seconds><|vision_start|><|video_pad|><|video_pad|><|vision_end|>"
        "<0.1 seconds><|vision_start|><|video_pad|><|vision_end|>",
        # Transformers 5.13: the template's outer markers stay around the same groups.
        "<|vision_start|><0.0 seconds><|vision_start|><|video_pad|><|video_pad|><|vision_end|>"
        "<0.1 seconds><|vision_start|><|video_pad|><|vision_end|><|vision_end|>",
        "<|vision_start|><|image_pad|><|image_pad|><|vision_end|>",
    ],
)
def test_media_regex_consumes_every_known_expansion_grammar(expansion: str) -> None:
    assert _MEDIA.sub("", f"Watch.{expansion}Seen.") == "Watch.Seen."
    assert [match.group(0) for match in _MEDIA.finditer(expansion)] == [expansion] or expansion.startswith("<0.0")


def test_assistant_media_and_timestamps_are_ignored(vlm_packer: QwenSFTDataPacker) -> None:
    target = SFTTarget(
        [
            {"role": "user", "content": "Return media."},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Here:"},
                    {"type": "image", "image": _image()},
                    {"type": "video", "video": _video()},
                    {"type": "text", "text": "Done."},
                ],
            },
        ]
    )
    sample = vlm_packer.sft_process_sample(target)
    assert _supervised(vlm_packer, sample) == "Here:Done.<|im_end|>"


def test_timestamp_text_before_image_remains_supervised(vlm_packer: QwenSFTDataPacker) -> None:
    text = "Timestamp:\n<3.0 seconds>"
    target = SFTTarget(
        [
            {"role": "user", "content": "Return an image at this time."},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": text}, {"type": "image", "image": _image()}],
            },
        ]
    )
    sample = vlm_packer.sft_process_sample(target)
    assert _supervised(vlm_packer, sample) == text + "<|im_end|>"


def test_collate_attention_mask_is_int64_and_feeds_the_native_eager_mask(text_packer: QwenSFTDataPacker) -> None:
    """The collated ``attention_mask`` follows ``main``'s contract, ``torch.int64`` 0/1
    (``tests/test_model_qwen3_vl.py::test_native_causal_mask_accepts_sft_padding_masks`` pins it), and the native eager
    path accepts it because ``model/qwen3_vl.py::_qwen3_vl_causal_mask`` casts a 2-D mask to bool before ``eager_mask``
    (``torch.where``). On 05389b0e6d this branch emitted ``torch.bool`` for that path instead; after the rebase of fix
    round 1 ``main``'s test refused it (job 877999: its six parametrizations failed on both heads and passed on main)."""
    from types import SimpleNamespace

    from cosmos_framework.posttrain.model.qwen3_vl import _qwen3_vl_causal_mask

    samples = [
        text_packer.sft_process_sample(
            SFTTarget([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Short"}])
        ),
        text_packer.sft_process_sample(
            SFTTarget(
                [{"role": "user", "content": "Hello there"}, {"role": "assistant", "content": "A longer answer."}]
            )
        ),
    ]
    batch = text_packer.sft_collate_fn(samples, text_packer.sft_compute_max_len(samples))
    assert batch["attention_mask"].dtype == torch.int64
    assert batch["attention_mask"].sum().item() == sum(len(sample.input_ids) for sample in samples)
    length = batch["input_ids"].shape[1]
    mask = _qwen3_vl_causal_mask(
        SimpleNamespace(_attn_implementation="eager"),
        torch.zeros(len(samples), length, 8, dtype=torch.bfloat16),
        batch["attention_mask"],
        torch.arange(length),
        None,
    )
    assert isinstance(mask, torch.Tensor) and tuple(mask.shape) == (len(samples), 1, length, length)


def test_collate_mixed_media_preserves_patch_order_and_padding(vlm_packer: QwenSFTDataPacker) -> None:
    targets = [
        SFTTarget([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Short"}]),
        SFTTarget(
            [
                {"role": "user", "content": [{"type": "image", "image": _image()}]},
                {"role": "assistant", "content": "A longer image answer."},
            ]
        ),
        SFTTarget(
            [
                {"role": "user", "content": [{"type": "video", "video": _video()}]},
                {"role": "assistant", "content": "Video answer."},
            ]
        ),
        SFTTarget(
            [
                {"role": "user", "content": [{"type": "image", "image": _image()}]},
                {"role": "assistant", "content": "Another image answer."},
            ]
        ),
    ]
    samples = [vlm_packer.sft_process_sample(target) for target in targets]
    length = vlm_packer.sft_compute_max_len(samples)
    batch = vlm_packer.sft_collate_fn(samples, length)
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape == (len(samples), length)
    assert batch["input_ids"].dtype == batch["labels"].dtype == torch.long
    for i, sample in enumerate(samples):
        count = len(sample.input_ids)
        assert batch["input_ids"][i, :count].tolist() == sample.input_ids
        assert batch["labels"][i, :count].tolist() == sample.labels
        assert batch["attention_mask"][i].sum().item() == count
        assert batch["labels"][i, count:].eq(-100).all()
        assert batch["input_ids"][i, count:].eq(vlm_packer.pad_token_id).all()
    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        expected = torch.cat(
            [sample.mm_kwargs[key] for sample in samples if sample.mm_kwargs and key in sample.mm_kwargs]
        )
        torch.testing.assert_close(batch[key], expected)
    assert all(value.device.type == "cpu" for value in batch.values())


def test_mm_token_types_are_padded_not_concatenated(vlm_packer: QwenSFTDataPacker) -> None:
    a = SFTPolicyInput([1, 2], [-100, 2], {"mm_token_type_ids": torch.tensor([[0, 1]])})
    b = SFTPolicyInput([3], [3])
    batch = vlm_packer.sft_collate_fn([a, b], 3, ignore_label_id=-7)
    assert batch["mm_token_type_ids"].tolist() == [[0, 1, 0], [0, 0, 0]]
    assert batch["labels"].tolist() == [[-7, 2, -7], [3, -7, -7]]


def test_media_kwargs_reach_real_processors(vlm_packer: QwenSFTDataPacker) -> None:
    target = SFTTarget(
        [
            {"role": "user", "content": [{"type": "image", "image": _image()}, {"type": "video", "video": _video()}]},
            {"role": "assistant", "content": "answer"},
        ]
    )
    ordinary = vlm_packer.sft_process_sample(target)
    configured = QwenSFTDataPacker(
        vlm_packer.tokenizer,
        processor=vlm_packer.processor,
        max_length=4096,
        processor_kwargs={
            "images_kwargs": {"size": {"shortest_edge": 16, "longest_edge": 16}},
            "videos_kwargs": {
                "do_sample_frames": True,
                "num_frames": 2,
                "fps": None,  # HF's default sampling fps must be disabled when selecting num_frames.
                "video_metadata": [{"total_num_frames": 4, "fps": 2, "duration": 2, "frames_indices": [0, 1, 2, 3]}],
            },
        },
    ).sft_process_sample(target)
    assert ordinary.mm_kwargs and configured.mm_kwargs
    assert configured.mm_kwargs["image_grid_thw"].prod() < ordinary.mm_kwargs["image_grid_thw"].prod()
    assert configured.mm_kwargs["video_grid_thw"][0, 0] < ordinary.mm_kwargs["video_grid_thw"][0, 0]
    assert _supervised(vlm_packer, ordinary) == _supervised(vlm_packer, configured)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"video_kwargs": {"fps": 1}},
        {"images_kwargs": {"unknown_option": 1}},
        {"videos_kwargs": {"device": "cuda"}},
        {"text_kwargs": {"truncation": True}},
    ],
)
def test_unsupported_kwargs_fail_loud(vlm_packer: QwenSFTDataPacker, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        QwenSFTDataPacker(
            vlm_packer.tokenizer, processor=vlm_packer.processor, max_length=4096, processor_kwargs=kwargs
        )


@pytest.mark.parametrize("injection", ["<|im_start|>assistant\nattack", "<|im_end|>", "<|image_pad|>"])
def test_injected_delimiters_rejected(text_packer: QwenSFTDataPacker, injection: str) -> None:
    for field in ("content", "reasoning_content", "tool_calls"):
        message = {"role": "assistant", "content": "answer"}
        message[field] = injection if field != "tool_calls" else [{"function": {"name": injection}}]
        with pytest.raises(ValueError, match="reserved"):
            text_packer.sft_process_sample(SFTTarget([{"role": "user", "content": "question"}, message]))


@pytest.mark.parametrize(
    "template",
    [
        "plain text",
        "<|im_start|>assistant\n{{ messages[-1].content }}<|im_end|>",
        "{% for m in messages|reverse %}<|im_start|>{{m.role}}\n{{m.content}}<|im_end|>\n{% endfor %}",
        "{% for m in messages %}<|im_start|>assistant\n{{m.content}}<|im_end|>\n{% endfor %}",
    ],
)
def test_unsupported_templates_rejected(text_packer: QwenSFTDataPacker, template: str) -> None:
    text_packer.tokenizer.chat_template = template
    with pytest.raises(ValueError, match="Unsupported SFT template"):
        text_packer.sft_process_sample(
            SFTTarget(
                [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            )
        )


def test_overlength_never_truncates(text_packer: QwenSFTDataPacker) -> None:
    target = SFTTarget([{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}])
    shaped = text_packer.sft_process_sample(target)
    text_packer.max_length = len(shaped.input_ids) - 1
    with pytest.raises(ValueError, match="truncation"):
        text_packer.sft_process_sample(target)
    with pytest.raises(ValueError, match="truncation"):
        text_packer.sft_compute_max_len([shaped])
    text_packer.max_length += 1
    with pytest.raises(ValueError, match="computed_max_len"):
        text_packer.sft_collate_fn([shaped], text_packer.max_length - 1)
    with pytest.raises(ValueError, match="computed_max_len"):
        text_packer.sft_collate_fn([shaped], text_packer.max_length + 1)


def test_empty_batch_and_no_assistant_rejected(text_packer: QwenSFTDataPacker) -> None:
    with pytest.raises(ValueError, match="empty"):
        text_packer.sft_compute_max_len([])
    with pytest.raises(ValueError, match="assistant"):
        text_packer.sft_process_sample(SFTTarget([{"role": "user", "content": "no answer"}]))


def test_builder_loads_local_text_metadata(tmp_path: Path) -> None:
    tokenizer = local_tokenizer()
    tokenizer.save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    packer = build_sft_packer(str(tmp_path), max_length=4096)
    assert isinstance(packer, DataPacker)
    shaped = packer.sft_process_sample(
        SFTTarget(
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        )
    )
    assert shaped.labels.count(tokenizer.eos_token_id) == 1


def test_builder_loads_local_vlm_metadata(tmp_path: Path) -> None:
    processor = local_processor(local_tokenizer())
    processor.save_pretrained(tmp_path)
    # No AutoConfig/model import is necessary to recognize a new multimodal architecture.
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    packer = build_sft_packer(
        str(tmp_path),
        max_length=4096,
        processor_kwargs={"images_kwargs": {"size": {"shortest_edge": 16, "longest_edge": 16}}},
    )
    result = packer.sft_process_sample(
        SFTTarget(
            [
                {"role": "user", "content": [{"type": "image", "image": _image()}]},
                {"role": "assistant", "content": "answer"},
            ]
        )
    )
    assert result.mm_kwargs and result.mm_kwargs["image_grid_thw"].prod() == 4


@pytest.mark.parametrize("model_type", ["llama", "qwen2_vl", "qwen3_5_moe"])
def test_builder_rejects_unsupported_family(tmp_path: Path, model_type: str) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    with pytest.raises(ValueError, match="only Qwen"):
        build_sft_packer(str(tmp_path), max_length=10)


@pytest.mark.parametrize("model_type", ["qwen3_vl", "qwen3_vl_moe"])
def test_builder_accepts_qwen3_vl_families_with_saved_processor_sidecars(tmp_path: Path, model_type: str) -> None:
    """Qwen3-VL / Qwen3-VL-MoE ship Qwen3VLProcessor; the builder selects the VLM packer from the sidecars."""
    local_processor(local_tokenizer()).save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    packer = build_sft_packer(
        str(tmp_path),
        max_length=4096,
        processor_kwargs={"images_kwargs": {"size": {"shortest_edge": 16, "longest_edge": 16}}},
    )
    assert isinstance(packer, QwenSFTDataPacker) and type(packer.processor).__name__ == "Qwen3VLProcessor"
    result = packer.sft_process_sample(
        SFTTarget(
            [
                {"role": "user", "content": [{"type": "image", "image": _image()}]},
                {"role": "assistant", "content": "answer"},
            ]
        )
    )
    assert result.mm_kwargs and result.mm_kwargs["image_grid_thw"].prod() == 4
    assert _supervised(packer, result) == "answer<|im_end|>"


# --- The official Qwen3-VL template: what the base checkpoint renders, pinned token for token -------

_AGENTIC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "zoom",
            "description": "Zoom into a box of the image",
            "parameters": {"type": "object", "properties": {"box": {"type": "array"}}},
        },
    }
]


def _agentic_messages(image: Image.Image) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "You inspect boards."},
        {
            "role": "user",
            "content": [{"type": "text", "text": "Inspect this component."}, {"type": "image", "image": image}],
        },
        {
            "role": "assistant",
            "content": "I will zoom in.",
            "reasoning_content": "The solder joint needs a closer look.",
            "tool_calls": [{"function": {"name": "zoom", "arguments": {"box": [0, 0, 4, 4]}}}],
        },
        {"role": "tool", "content": "zoomed view returned"},
        {"role": "assistant", "content": "Verdict: OK.", "reasoning_content": "Nothing abnormal in the zoomed view."},
    ]


# One newline after ``</think>`` when the turn has content; the tool call follows the content on its own line.
_AGENTIC_SUPERVISED = (
    "<think>\nThe solder joint needs a closer look.\n</think>\nI will zoom in.\n"
    '<tool_call>\n{"name": "zoom", "arguments": {"box": [0, 0, 4, 4]}}\n</tool_call><|im_end|>'
    "<think>\nNothing abnormal in the zoomed view.\n</think>\nVerdict: OK.<|im_end|>"
)


def _assert_agentic_row(packer: QwenSFTDataPacker, image: Image.Image, expected_image_pads: int) -> SFTPolicyInput:
    sample = packer.sft_process_sample(SFTTarget(_agentic_messages(image), tools=_AGENTIC_TOOLS))
    assert _supervised(packer, sample) == _AGENTIC_SUPERVISED
    assert all(label == -100 or label == token for token, label in zip(sample.input_ids, sample.labels, strict=True))
    rendered = packer.tokenizer.decode(sample.input_ids)
    for observation in (
        "You inspect boards.",
        "# Tools",
        "Inspect this component.",
        "<tool_response>",
        "zoomed view returned",
    ):
        assert observation in rendered and observation not in _supervised(packer, sample)
    pad = packer.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    pads = [index for index, token in enumerate(sample.input_ids) if token == pad]
    assert len(pads) == expected_image_pads and all(sample.labels[index] == -100 for index in pads)
    assert sample.mm_kwargs is not None and int(sample.mm_kwargs["image_grid_thw"].prod()) // 4 == expected_image_pads
    return sample


@pytest.fixture
def official_vlm_packer(tmp_path: Path) -> QwenSFTDataPacker:
    """The builder's own path for a Qwen3-VL-MoE checkpoint: saved sidecars carrying the official template."""
    local_processor(local_tokenizer(), chat_template=official_qwen3_vl_chat_template()).save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_vl_moe"}))
    return build_sft_packer(str(tmp_path), max_length=4096)


def test_official_template_supervises_assistant_bodies_and_masks_media(official_vlm_packer: QwenSFTDataPacker) -> None:
    # The toy geometry (patch 2, merge 2, 256-pixel budget) expands the 8x8 image to 16 patches = 4 merged tokens.
    _assert_agentic_row(official_vlm_packer, _image(), expected_image_pads=4)


def test_official_template_tool_only_turn_renders_no_blank_line(official_vlm_packer: QwenSFTDataPacker) -> None:
    """A tool-only assistant turn (``content`` None or "") renders ``</think>\\n<tool_call>`` with reasoning and
    ``<|im_start|>assistant\\n<tool_call>`` without: the template emits its separating newline only when the
    turn has content, which is how it renders the same history at serving time."""
    for content in (None, ""):
        with_reasoning = [
            *_agentic_messages(_image())[:2],
            {
                "role": "assistant",
                "content": content,
                "reasoning_content": "Look closer.",
                "tool_calls": [{"function": {"name": "zoom", "arguments": {"box": [1, 1, 2, 2]}}}],
            },
        ]
        sample = official_vlm_packer.sft_process_sample(SFTTarget(with_reasoning, tools=_AGENTIC_TOOLS))
        assert _supervised(official_vlm_packer, sample) == (
            '<think>\nLook closer.\n</think>\n<tool_call>\n{"name": "zoom", "arguments": {"box": [1, 1, 2, 2]}}\n</tool_call><|im_end|>'
        )
        without_reasoning = [
            *with_reasoning[:2],
            {k: v for k, v in with_reasoning[2].items() if k != "reasoning_content"},
        ]
        sample = official_vlm_packer.sft_process_sample(SFTTarget(without_reasoning, tools=_AGENTIC_TOOLS))
        assert _supervised(official_vlm_packer, sample) == (
            '<tool_call>\n{"name": "zoom", "arguments": {"box": [1, 1, 2, 2]}}\n</tool_call><|im_end|>'
        )
        assert "<|im_start|>assistant\n<tool_call>" in official_vlm_packer.tokenizer.decode(sample.input_ids)


@pytest.mark.parametrize(
    "content",
    ["", "   ", [{"type": "text", "text": ""}], [{"type": "text", "text": "  "}, {"type": "text", "text": ""}]],
    ids=["empty-string", "whitespace", "empty-text-part", "whitespace-parts"],
)
def test_official_template_tool_call_only_turn_survives_newline_merging_tokenizer(content: Any) -> None:
    """The student's tokenizer merges "\\n\\n" (and "\\n   \\n") into one token. With the official template a
    text-less tool-call turn used to render ``assistant\\n\\n<tool_call>`` because the packer kept an empty or
    whitespace text part, which the template counts as content, so one token straddled the ignored header
    and the supervised body and the row was refused. Every text-less shape now renders no content part."""
    # Constructed directly (not through the builder) so the fixture isolates the straddle from the allow-list.
    tokenizer = local_tokenizer(merge_newlines=True)
    processor = local_processor(tokenizer, chat_template=official_qwen3_vl_chat_template())
    packer = QwenSFTDataPacker(tokenizer, processor=processor, max_length=4096)
    messages = [
        {"role": "user", "content": "Inspect."},
        {
            "role": "assistant",
            "content": content,
            "tool_calls": [{"function": {"name": "zoom", "arguments": {"box": [0, 0, 4, 4]}}}],
        },
        {"role": "tool", "content": "zoomed"},
        {"role": "assistant", "content": "Verdict: OK."},
    ]
    sample = packer.sft_process_sample(SFTTarget(messages, tools=_AGENTIC_TOOLS))
    assert _supervised(packer, sample) == (
        '<tool_call>\n{"name": "zoom", "arguments": {"box": [0, 0, 4, 4]}}\n</tool_call><|im_end|>Verdict: OK.<|im_end|>'
    )
    assert "assistant\n\n<tool_call>" not in packer.tokenizer.decode(sample.input_ids)


def test_official_template_refuses_media_it_does_not_render(official_vlm_packer: QwenSFTDataPacker) -> None:
    """The Qwen3-VL template joins only the text parts of an assistant turn: an assistant image has no placeholder,
    while the processor would still encode it (1 placeholder, 2 grids, half the pad tokens). Refused, not misaligned."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Inspect."}, {"type": "image", "image": _image()}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Here is a crop."}, {"type": "image", "image": _image()}],
        },
    ]
    with pytest.raises(ValueError, match="1 image placeholder\\(s\\) for 2 image part\\(s\\)"):
        official_vlm_packer.sft_process_sample(SFTTarget(messages))


def test_official_template_aligns_tool_turn_images(official_vlm_packer: QwenSFTDataPacker) -> None:
    """A tool reply carrying an image renders its placeholder inside the user block: two images, two grids, all pads -100."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Inspect."}, {"type": "image", "image": _image()}]},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "zoom", "arguments": {"box": [0, 0, 4, 4]}}}],
        },
        {"role": "tool", "content": [{"type": "text", "text": "zoomed"}, {"type": "image", "image": _image()}]},
        {"role": "assistant", "content": "Verdict: OK."},
    ]
    sample = official_vlm_packer.sft_process_sample(SFTTarget(messages, tools=_AGENTIC_TOOLS))
    pad = official_vlm_packer.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    pads = [index for index, token in enumerate(sample.input_ids) if token == pad]
    assert sample.mm_kwargs is not None and sample.mm_kwargs["image_grid_thw"].shape[0] == 2
    assert len(pads) == int(sample.mm_kwargs["image_grid_thw"].prod(dim=-1).sum()) // 4 == 8
    assert all(sample.labels[index] == -100 for index in pads)
    assert _supervised(official_vlm_packer, sample) == (
        '<tool_call>\n{"name": "zoom", "arguments": {"box": [0, 0, 4, 4]}}\n</tool_call><|im_end|>Verdict: OK.<|im_end|>'
    )


def test_official_template_thinking_off_supervises_content_only(official_vlm_packer: QwenSFTDataPacker) -> None:
    sample = official_vlm_packer.sft_process_sample(
        SFTTarget([{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A."}])
    )
    assert _supervised(official_vlm_packer, sample) == "A.<|im_end|>"
    assert "<think>" not in official_vlm_packer.tokenizer.decode(sample.input_ids)


def _spy_render_kwargs(packer: QwenSFTDataPacker, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record the kwargs of every ``apply_chat_template`` call the packer makes on its renderer."""
    renderer = packer.processor if packer.processor is not None else packer.tokenizer
    original = renderer.apply_chat_template
    seen: list[dict[str, Any]] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(renderer, "apply_chat_template", spy)
    return seen


class _Records(logging.Handler):
    """Collect the Transformers processor logger's records; its library logger does not propagate to the root."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def test_official_template_gets_no_undeclared_kwarg_and_renders_byte_identically(
    official_vlm_packer: QwenSFTDataPacker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The official Qwen3-VL template declares no ``preserve_thinking`` variable. Passing it anyway is a no-op that
    Transformers 5 reclassifies as a processor kwarg with a warning on EVERY render (96.6 % of a training job's log
    in the first native-seat run). The packer must pass only declared variables, log nothing, and render the same
    bytes, ids and labels as the kwarg-carrying call did."""
    handler = _Records()
    logger = logging.getLogger("transformers.processing_utils")
    logger.addHandler(handler)
    try:
        messages, tools = _agentic_messages(_image()), _AGENTIC_TOOLS
        legacy_text = official_vlm_packer.processor.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=False, preserve_thinking=True
        )
        # The dead kwarg really warns in this runtime; the assertion below is not vacuous.
        assert any("processor_kwargs" in message for message in handler.messages)
        handler.messages.clear()

        seen = _spy_render_kwargs(official_vlm_packer, monkeypatch)
        sample = _assert_agentic_row(official_vlm_packer, _image(), expected_image_pads=4)
        assert len(seen) == 2  # the rendered text and the boundary probe
        assert all(set(kwargs) == {"tools", "tokenize", "add_generation_prompt"} for kwargs in seen)
        assert handler.messages == []
        assert official_vlm_packer._render(messages, tools) == legacy_text
        assert official_vlm_packer.validation_semantics_identity()["preserve_thinking"] is False

        legacy = QwenSFTDataPacker(
            official_vlm_packer.tokenizer, processor=official_vlm_packer.processor, max_length=4096
        )
        monkeypatch.setattr(
            legacy,
            "_render",
            lambda messages, tools: legacy.processor.apply_chat_template(
                messages, tools=tools, tokenize=False, add_generation_prompt=False, preserve_thinking=True
            ),
        )
        legacy_sample = legacy.sft_process_sample(SFTTarget(messages, tools=tools))
        assert legacy_sample.input_ids == sample.input_ids and legacy_sample.labels == sample.labels
        assert legacy_sample.mm_kwargs is not None and sample.mm_kwargs is not None
        assert set(legacy_sample.mm_kwargs) == set(sample.mm_kwargs)
        for key, value in sample.mm_kwargs.items():
            other = legacy_sample.mm_kwargs[key]
            assert torch.equal(other, value) if isinstance(value, torch.Tensor) else other == value, key
    finally:
        logger.removeHandler(handler)


@pytest.mark.parametrize("packer_name", ["text_packer", "vlm_packer"])
def test_toy_template_that_declares_preserve_thinking_still_receives_it(
    request: pytest.FixtureRequest, packer_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The kwarg is template-driven, not removed: the toy template renders ``reasoning_content`` only when
    ``preserve_thinking`` is true, so a blanket removal would silently drop the reasoning from every toy render
    (the boundary probe would then refuse the template)."""
    packer: QwenSFTDataPacker = request.getfixturevalue(packer_name)
    messages = [
        {"role": "user", "content": "Compute a value."},
        {"role": "assistant", "content": "It is 4.", "reasoning_content": "2 + 2"},
    ]
    seen = _spy_render_kwargs(packer, monkeypatch)
    sample = packer.sft_process_sample(SFTTarget(messages))
    assert seen and all(kwargs["preserve_thinking"] is True for kwargs in seen)
    assert "<think>\n2 + 2\n</think>" in _supervised(packer, sample)
    assert packer.validation_semantics_identity()["preserve_thinking"] is True
    renderer = packer.processor if packer.processor is not None else packer.tokenizer
    without = renderer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    assert "2 + 2" not in without


def test_real_qwen3_vl_sidecars_pack_the_agentic_row() -> None:
    """Opt-in: ``COSMOS_SFT_QWEN3_VL_SIDECARS`` names a checkpoint directory whose text sidecars are read."""
    sidecars = os.environ.get("COSMOS_SFT_QWEN3_VL_SIDECARS")
    if not sidecars:
        pytest.skip("set COSMOS_SFT_QWEN3_VL_SIDECARS=<Qwen3-VL(-MoE) checkpoint dir> to run against real sidecars")
    packer = build_sft_packer(
        sidecars,
        max_length=4096,
        processor_kwargs={"images_kwargs": {"size": {"shortest_edge": 65536, "longest_edge": 65536}}},
    )
    assert type(packer.processor).__name__ == "Qwen3VLProcessor"
    # 8x8 -> 256x256 at the 65536-pixel floor; patch 16 and merge 2 give 64 merged visual tokens.
    sample = _assert_agentic_row(packer, _image(), expected_image_pads=64)
    assert "mm_token_type_ids" in sample.mm_kwargs
    # The real tokenizer merges "\n\n" into one token; a text-less tool-call turn must not straddle the header.
    tool_only = [
        *_agentic_messages(_image())[:2],
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "zoom", "arguments": {"box": [0, 0, 4, 4]}}}],
        },
    ]
    sample = packer.sft_process_sample(SFTTarget(tool_only, tools=_AGENTIC_TOOLS))
    assert (
        _supervised(packer, sample)
        == '<tool_call>\n{"name": "zoom", "arguments": {"box": [0, 0, 4, 4]}}\n</tool_call><|im_end|>'
    )


@pytest.mark.parametrize("field", ["reasoning_content", "tool_calls"])
def test_template_cannot_silently_drop_reasoning_or_calls(text_packer: QwenSFTDataPacker, field: str) -> None:
    text_packer.tokenizer.chat_template = (
        "{% for m in messages %}<|im_start|>{{m.role}}\n{{m.content}}<|im_end|>\n{% endfor %}"
    )
    assistant: dict[str, Any] = {"role": "assistant", "content": "answer"}
    assistant[field] = "reasoning" if field == "reasoning_content" else [{"function": {"name": "lookup"}}]
    with pytest.raises(ValueError, match="reasoning/tool calls were dropped"):
        text_packer.sft_process_sample(SFTTarget([{"role": "user", "content": "question"}, assistant]))


def test_token_straddling_header_and_body_fails_instead_of_losing_body_tokens() -> None:
    packer = QwenSFTDataPacker(local_tokenizer(merge_newlines=True), max_length=4096)
    target = SFTTarget([{"role": "user", "content": "question"}, {"role": "assistant", "content": "\nanswer"}])
    with pytest.raises(ValueError, match="overlaps assistant body"):
        packer.sft_process_sample(target)


def test_assistant_empty_body_still_trains_im_end(text_packer: QwenSFTDataPacker) -> None:
    sample = text_packer.sft_process_sample(
        SFTTarget(
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": ""},
            ]
        )
    )
    assert _supervised(text_packer, sample) == "<|im_end|>"


def test_media_overrun_uses_expanded_not_placeholder_length(vlm_packer: QwenSFTDataPacker) -> None:
    target = SFTTarget(
        [
            {"role": "user", "content": [{"type": "image", "image": _image()}]},
            {"role": "assistant", "content": "answer"},
        ]
    )
    text = vlm_packer.processor.apply_chat_template(target.messages, tokenize=False)
    raw_length = len(vlm_packer.tokenizer(text, add_special_tokens=False)["input_ids"])
    vlm_packer.max_length = raw_length
    with pytest.raises(ValueError, match="exceeds max_length"):
        vlm_packer.sft_process_sample(target)


def test_collate_rejects_wrong_labels_and_unknown_side_inputs(text_packer: QwenSFTDataPacker) -> None:
    with pytest.raises(ValueError, match="identical lengths"):
        text_packer.sft_collate_fn([SFTPolicyInput([1], [])], 1)
    with pytest.raises(ValueError, match="side input"):
        text_packer.sft_collate_fn([SFTPolicyInput([1], [1], {"unknown": torch.ones(1)})], 1)


def test_schema_import_remains_stdlib_only() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from cosmos_framework.posttrain.schema import SFTTarget; "
            "assert not {'torch', 'transformers', 'numpy'} & sys.modules.keys(); assert SFTTarget([]).tools is None; "
            "assert SFTTarget([]).supervise == 'all_assistant'",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_resume_identity_binds_media_budget_and_template(vlm_packer: QwenSFTDataPacker) -> None:
    identity = vlm_packer.validation_semantics_identity()
    configured = QwenSFTDataPacker(
        vlm_packer.tokenizer,
        processor=vlm_packer.processor,
        max_length=2048,
        processor_kwargs={"images_kwargs": {"size": {"shortest_edge": 16, "longest_edge": 64}}},
    )
    assert configured.validation_semantics_identity() != identity
    vlm_packer.processor.chat_template += "\n"
    assert vlm_packer.validation_semantics_identity() != identity


def test_legacy_media_budget_is_not_silently_ignored(vlm_packer: QwenSFTDataPacker) -> None:
    with pytest.raises(ValueError, match="legacy"):
        QwenSFTDataPacker(
            vlm_packer.tokenizer,
            processor=vlm_packer.processor,
            max_length=4096,
            processor_kwargs={"images_kwargs": {"max_pixels": 16}},
        )


def test_think_opener_mask_unsupervises_the_prefilled_opener_only() -> None:
    """Review R125-D-01 knob 7, ruled in reviews R134-01 / R134-02 and corrected by review R138-01: ``mask_think_opener=True``
    unsupervises the ``<think>`` token that opens an assistant body -- the token a prefilling serve supplies (a request
    without tools; NOT the serving path of record, where tools are present, vLLM 0.24 prefills no opener and the student
    must emit it) -- and NOTHING else: the reasoning after it and the closer of an empty block (``\n\n</think>\n\n``, the
    exporter's rendering of a turn without reasoning, or a single-newline ``\n</think>\n``) stay supervised, because the
    student emits them in either case. Checked on
    the base's own chat template (``tests/fixtures/qwen3_vl_chat_template.jinja`` == the base's ``chat_template.json``
    byte for byte) with a tokenizer whose ``\n\n`` is one token like the base's (id 271): 22bfce78f40's Nemo-Chat rule
    masked the single-newline closer (``review_r134/5_p_tok.out``: ``['Ċ', 'Answer', ...]`` left supervised), and the
    template strips a ``<think></think>`` content prefix, so that turn has no opener to mask. The default supervises the
    opener (what the A1 arms trained, and what the path of record needs)."""
    from cosmos_framework.posttrain.tests.helpers import sft_packer as helpers

    tokenizer = helpers.local_tokenizer(merge_newlines=True)
    processor = helpers.local_processor(tokenizer, chat_template=helpers.official_qwen3_vl_chat_template())
    think, think_end = tokenizer.convert_tokens_to_ids("<think>"), tokenizer.convert_tokens_to_ids("</think>")
    newline, two = tokenizer.convert_tokens_to_ids("Ċ"), tokenizer.convert_tokens_to_ids("ĊĊ")
    assert tokenizer.encode("\n\n", add_special_tokens=False) == [two]  # one token, as the base's 271
    target = SFTTarget(
        messages=[
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer one.", "reasoning_content": "Reason one."},
            {"role": "user", "content": "Again"},
            {"role": "assistant", "content": "<think>\n\n</think>\n\nAnswer two."},
            {"role": "user", "content": "Once more"},
            {"role": "assistant", "content": "<think>\n</think>\nAnswer three."},
            {"role": "user", "content": "Last"},
            {"role": "assistant", "content": "<think></think>Answer four."},
        ]
    )
    default = QwenSFTDataPacker(tokenizer, max_length=4096, processor=processor).sft_process_sample(target)
    packer = QwenSFTDataPacker(tokenizer, max_length=4096, processor=processor, mask_think_opener=True)
    masked = packer.sft_process_sample(target)
    assert masked.input_ids == default.input_ids
    openers = [i for i, t in enumerate(masked.input_ids) if t == think]
    closers = [i for i, t in enumerate(masked.input_ids) if t == think_end]
    assert len(openers) == 3 and len(closers) == 3  # turn four's '<think></think>' prefix is stripped by the template
    for pos in openers:
        assert default.labels[pos] == think, "the default supervises the opener"
        assert masked.labels[pos] == -100, "the variant masks it"
    changed = [i for i, (a, b) in enumerate(zip(default.labels, masked.labels, strict=True)) if a != b]
    assert changed == openers, "the opener and only the opener"
    for pos in closers:
        assert masked.labels[pos] == think_end, (
            "every closer stays supervised: after reasoning and after an empty block"
        )
    second, third = openers[1], openers[2]
    assert masked.input_ids[second : second + 4] == [think, two, think_end, two]
    assert masked.labels[second + 1 : second + 4] == [two, think_end, two]
    assert masked.input_ids[third : third + 4] == [think, newline, think_end, newline]
    assert masked.labels[third + 1 : third + 4] == [newline, think_end, newline]  # 22bfce78f40 masked these three
    supervised = _supervised(packer, masked)
    assert "Reason one." in supervised and "</think>" in supervised and "Answer four." in supervised
    identity = QwenSFTDataPacker(tokenizer, max_length=4096, mask_think_opener=True).validation_semantics_identity()
    assert identity["mask_think_opener"] is True
    assert "mask_think_opener" not in QwenSFTDataPacker(tokenizer, max_length=4096).validation_semantics_identity()
    with pytest.raises(ValueError, match="<think>"):
        QwenSFTDataPacker(local_tokenizer_without_think(), max_length=4096, mask_think_opener=True)


def local_tokenizer_without_think() -> Any:
    from cosmos_framework.posttrain.tests.helpers import sft_packer as helpers

    tokenizer = helpers.local_tokenizer()
    vocab = dict(tokenizer.get_vocab())
    # A tokenizer whose vocabulary lacks the thinking tokens: the knob must refuse rather than silently do nothing.
    return type(
        "NoThinkTokenizer",
        (),
        {
            "is_fast": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "get_vocab": lambda self: {k: v for k, v in vocab.items() if k not in ("<think>", "</think>")},
            "encode": lambda self, text, add_special_tokens=False: tokenizer.encode(text, add_special_tokens=False),
            "chat_template": tokenizer.chat_template,
        },
    )()


# --- Row-level supervision: ``SFTTarget.supervise`` (fix round T6: the prefix rows) ----------------------------------

_THREE_TURN_MESSAGES = [
    {"role": "user", "content": "Use a tool."},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"function": {"name": "lookup", "arguments": {"query": "value"}}}],
    },
    {"role": "tool", "content": "found"},
    {"role": "assistant", "content": "done"},
]
_THREE_TURN_TOOLS = [{"function": {"name": "lookup"}}]


def _independent_block_counts(packer: QwenSFTDataPacker, rendered: str) -> list[int]:
    """A row validator's definition of the supervised count per assistant block, computed without labels or offsets:
    each ``<|im_start|>assistant\n...<|im_end|>`` block's body text (media placeholders removed) tokenized on its own,
    plus one for the block's ``<|im_end|>``."""
    from cosmos_framework.posttrain.data.packer.sft import _CHAT

    counts = []
    for block in _CHAT.finditer(rendered):
        if block[1] == "assistant":
            fragments = [fragment for fragment in _MEDIA.split(block[2]) if fragment]
            counts.append(sum(len(packer.tokenizer(f, add_special_tokens=False)["input_ids"]) for f in fragments) + 1)
    return counts


def _loss_tokens_per_block(packer: QwenSFTDataPacker, labels: list[int]) -> list[int]:
    """Supervised label counts split at every supervised ``<|im_end|>`` (a trailing open run is its own entry)."""
    end = packer.tokenizer.convert_tokens_to_ids("<|im_end|>")
    blocks, current = [], 0
    for label in labels:
        if label == -100:
            continue
        current += 1
        if label == end:
            blocks.append(current)
            current = 0
    if current:
        blocks.append(current)
    return blocks


def _loss_count(sample: SFTPolicyInput) -> int:
    return sum(1 for label in sample.labels if label != -100)


def test_supervise_last_assistant_three_turn_row_exact_loss_counts(text_packer: QwenSFTDataPacker) -> None:
    """Fixture for ``SFTTarget.supervise``: a three-turn row (assistant tool call, tool result, assistant answer) under
    both values with the exact loss-token counts. The toy tokenizer has a complete byte alphabet and no merges, so an
    ASCII body costs one token per byte: turn 1 renders ``<tool_call>`` (11) + the 67-char ``tojson`` of the call +
    ``</tool_call>`` (12) = 90 bytes + ``<|im_end|>`` = 91 loss tokens; turn 2 renders ``done`` = 4 + 1 = 5.
    ``all_assistant`` (the default) supervises 96 = [91, 5]; ``last_assistant`` supervises 5 = [5]: the first assistant
    block at -100 like the tool result, the ids unchanged, the supervised positions of the last turn identical, and the
    independent per-block count (a validator's definition) equals the labels' count over exactly the supervised blocks.
    On the parent commit the field does not exist (``TypeError: unexpected keyword argument 'supervise'``)."""
    from cosmos_framework.posttrain.data.packer.sft import SUPERVISE_CONTRACTS

    default = text_packer.sft_process_sample(SFTTarget(_THREE_TURN_MESSAGES, tools=_THREE_TURN_TOOLS))
    explicit = text_packer.sft_process_sample(
        SFTTarget(_THREE_TURN_MESSAGES, tools=_THREE_TURN_TOOLS, supervise="all_assistant")
    )
    last = text_packer.sft_process_sample(
        SFTTarget(_THREE_TURN_MESSAGES, tools=_THREE_TURN_TOOLS, supervise="last_assistant")
    )
    assert explicit.input_ids == default.input_ids and explicit.labels == default.labels
    assert last.input_ids == default.input_ids, "the rule changes labels only; the sequence is the same conversation"
    target = SFTTarget(_THREE_TURN_MESSAGES, tools=_THREE_TURN_TOOLS)
    rendered = text_packer._render(text_packer._conversation(target), _THREE_TURN_TOOLS)
    independent = _independent_block_counts(text_packer, rendered)
    assert independent == [91, 5]
    assert _loss_count(default) == 96 and _loss_tokens_per_block(text_packer, default.labels) == [91, 5] == independent
    assert _loss_count(last) == 5 and _loss_tokens_per_block(text_packer, last.labels) == [5] == independent[-1:]
    assert _supervised(text_packer, default) == (
        '<tool_call>[{"function": {"name": "lookup", "arguments": {"query": "value"}}}]</tool_call><|im_end|>'
        "done<|im_end|>"
    )
    assert _supervised(text_packer, last) == "done<|im_end|>"
    kept = [i for i, label in enumerate(last.labels) if label != -100]
    assert kept == [i for i, label in enumerate(default.labels) if label != -100][-5:], "the same five positions"
    assert all(label == -100 or label == token for token, label in zip(last.input_ids, last.labels, strict=True))
    assert "lookup" in text_packer.tokenizer.decode(last.input_ids), "the history stays in the sequence"
    assert default.masking == explicit.masking == SUPERVISE_CONTRACTS["all_assistant"]
    assert default.masking == "qwen_chatml_all_assistant_bodies_and_end_v2"
    assert last.masking == SUPERVISE_CONTRACTS["last_assistant"] == "qwen_chatml_last_assistant_body_and_end_v1"
    assert text_packer.validation_semantics_identity()["masking"] == "qwen_chatml_all_assistant_bodies_and_end_v2"
    assert SFTPolicyInput([1], [1]).masking == "qwen_chatml_all_assistant_bodies_and_end_v2"


def test_supervise_last_assistant_official_template_keeps_history_at_minus_100(
    official_vlm_packer: QwenSFTDataPacker,
) -> None:
    """The agentic row under the official Qwen3-VL template with ``supervise="last_assistant"`` and a history turn
    carrying no reasoning (review R151-04: the original row, whose history turn carries ``reasoning_content``, is refused,
    asserted first): only the second turn's body (its think block, text and ``<|im_end|>``) is supervised; the first
    turn's text and tool call stay in the ids at -100 beside the system block, the tools block, the image pads and the
    tool response, and the ids equal the ``all_assistant`` packing of the same messages. The ``mask_think_opener``
    interplay is unchanged: the variant unsupervises the LAST turn's opener, the only opener in the row, and nothing
    else. Fix round T6."""
    with_history_reasoning = _agentic_messages(_image())
    with pytest.raises(ValueError, match="history assistant turn 2 carries reasoning \\(reasoning_content\\)"):
        official_vlm_packer.sft_process_sample(
            SFTTarget(with_history_reasoning, tools=_AGENTIC_TOOLS, supervise="last_assistant")
        )
    messages = [dict(message) for message in with_history_reasoning]
    del messages[2]["reasoning_content"]  # the history turn carries content and tool_calls only (row rule R5)
    target = SFTTarget(messages, tools=_AGENTIC_TOOLS, supervise="last_assistant")
    sample = official_vlm_packer.sft_process_sample(target)
    assert _supervised(official_vlm_packer, sample) == (
        "<think>\nNothing abnormal in the zoomed view.\n</think>\nVerdict: OK.<|im_end|>"
    )
    both = official_vlm_packer.sft_process_sample(SFTTarget(messages, tools=_AGENTIC_TOOLS))
    assert sample.input_ids == both.input_ids and _loss_count(sample) < _loss_count(both)
    assert _supervised(official_vlm_packer, both) == (
        'I will zoom in.\n<tool_call>\n{"name": "zoom", "arguments": {"box": [0, 0, 4, 4]}}\n</tool_call><|im_end|>'
        "<think>\nNothing abnormal in the zoomed view.\n</think>\nVerdict: OK.<|im_end|>"
    )
    rendered = official_vlm_packer.tokenizer.decode(sample.input_ids)
    assert "I will zoom in." in rendered and '"name": "zoom"' in rendered and "zoomed view returned" in rendered
    template_text = official_vlm_packer._render(official_vlm_packer._conversation(target), _AGENTIC_TOOLS)
    independent = _independent_block_counts(official_vlm_packer, template_text)
    assert len(independent) == 2 and _loss_tokens_per_block(official_vlm_packer, sample.labels) == independent[-1:]
    assert _loss_tokens_per_block(official_vlm_packer, both.labels) == independent
    pad = official_vlm_packer.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    pads = [index for index, token in enumerate(sample.input_ids) if token == pad]
    assert len(pads) == 4 and all(sample.labels[index] == -100 for index in pads)
    assert sample.masking == "qwen_chatml_last_assistant_body_and_end_v1"
    masked = QwenSFTDataPacker(
        official_vlm_packer.tokenizer, processor=official_vlm_packer.processor, max_length=4096, mask_think_opener=True
    ).sft_process_sample(target)
    think = official_vlm_packer.tokenizer.convert_tokens_to_ids("<think>")
    openers = [i for i, token in enumerate(sample.input_ids) if token == think]
    assert openers == [i for i, token in enumerate(both.input_ids) if token == think] and len(openers) == 1
    assert sample.labels[openers[0]] == think, "the default supervises the opener the student emits"
    changed = [i for i, (a, b) in enumerate(zip(sample.labels, masked.labels, strict=True)) if a != b]
    assert changed == openers, "the variant removes the last turn's opener and nothing else"


def test_supervise_refuses_unknown_values_and_rows_without_an_assistant_turn(text_packer: QwenSFTDataPacker) -> None:
    """Fail closed: the schema refuses any value outside the two at construction, the packer refuses a target mutated
    after construction, a row without an assistant turn is refused under ``last_assistant`` as under the default, and the
    block-index helper a row validator must share with the packer is pinned."""
    from cosmos_framework.posttrain.data.packer.sft import supervised_assistant_blocks

    with pytest.raises(ValueError, match="supervise must be one of"):
        SFTTarget(_THREE_TURN_MESSAGES, supervise="last")
    with pytest.raises(ValueError, match="supervise must be one of"):
        SFTTarget(_THREE_TURN_MESSAGES, supervise=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="assistant"):
        text_packer.sft_process_sample(
            SFTTarget([{"role": "user", "content": "no answer"}], supervise="last_assistant")
        )
    mutated = SFTTarget(_THREE_TURN_MESSAGES, tools=_THREE_TURN_TOOLS)
    mutated.supervise = "every_other"
    with pytest.raises(ValueError, match="supervise must be one of"):
        text_packer.sft_process_sample(mutated)
    assert supervised_assistant_blocks(3, "all_assistant") == [0, 1, 2]
    assert supervised_assistant_blocks(3, "last_assistant") == [2]
    assert supervised_assistant_blocks(1, "last_assistant") == [0] == supervised_assistant_blocks(1, "all_assistant")
    with pytest.raises(ValueError, match="no assistant target"):
        supervised_assistant_blocks(0, "last_assistant")
    with pytest.raises(ValueError, match="supervise must be one of"):
        supervised_assistant_blocks(2, "first_assistant")


def test_last_assistant_row_prefix_is_the_served_prefix_under_both_templates_and_history_reasoning_is_refused() -> None:
    """Review R151-04 (fix round T6). The packer renders rows through the PROCESSOR template (``chat_template.json``); a
    tokenizer-only loader (vLLM with a tools array) serves the TOKENIZER template (``tokenizer_config.json``), which renders
    no ``reasoning_content`` and keeps a ``<think></think>`` content prefix the processor template strips. Both texts are
    repo fixtures saved verbatim from the base of record. A ``last_assistant`` row whose history turns carry no reasoning:
    the row's text begins with the tokenizer template's served prefix (the history rendered with
    ``add_generation_prompt=True``) byte for byte, the packed ids begin with the served prefix's ids at -100, and the two
    templates agree on the whole row when no turn carries reasoning; the student's own last turn keeps its reasoning in
    the row. The same row WITH a history ``reasoning_content``: the processor rendering diverges from the served prefix
    inside the history (a think block the serve never shows) and the packer REFUSES the row, naming the turn; so does a
    history content opening with ``<think></think>``; an ``all_assistant`` row with the same history still packs. On the
    parent commit the row packs silently (no rule) and the tokenizer-template fixture does not exist."""
    tokenizer = local_tokenizer(merge_newlines=True)
    processor = local_processor(tokenizer, chat_template=official_qwen3_vl_chat_template())
    packer = QwenSFTDataPacker(tokenizer, processor=processor, max_length=4096)
    served = local_tokenizer(merge_newlines=True)
    served.chat_template = official_qwen3_vl_tokenizer_chat_template()
    assert served.chat_template != processor.chat_template and len(served.chat_template) == 5292
    history = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Solve the task in TASK.md."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "zoom", "arguments": {"box": [0, 0, 4, 4]}}}],
        },
        {"role": "tool", "content": "zoomed view returned"},
        {
            "role": "assistant",
            "content": "Reading the next crop.",
            "tool_calls": [{"function": {"name": "zoom", "arguments": {"box": [1, 1, 3, 3]}}}],
        },
        {"role": "tool", "content": "second crop returned"},
        {"role": "user", "content": "Attached image(s) from tool result."},
    ]
    emission = {"role": "assistant", "content": "Verdict: OK.", "reasoning_content": "The pad looks fine."}

    def texts(messages: list[dict[str, Any]]) -> tuple[str, str, str]:
        normalized = packer._conversation(SFTTarget(messages, tools=_AGENTIC_TOOLS))
        row = packer._render(normalized, _AGENTIC_TOOLS)
        tokenizer_row = served.apply_chat_template(
            normalized, tools=_AGENTIC_TOOLS, tokenize=False, add_generation_prompt=False
        )
        served_prefix = served.apply_chat_template(
            normalized[:-1], tools=_AGENTIC_TOOLS, tokenize=False, add_generation_prompt=True
        )
        return row, tokenizer_row, served_prefix

    row, tokenizer_row, served_prefix = texts([*history, emission])
    assert served_prefix.endswith("<|im_start|>assistant\n") and "<think>" not in served_prefix
    assert row.startswith(served_prefix) and tokenizer_row.startswith(served_prefix)
    assert row[len(served_prefix) :] == "<think>\nThe pad looks fine.\n</think>\nVerdict: OK.<|im_end|>\n"
    assert tokenizer_row[len(served_prefix) :] == "Verdict: OK.<|im_end|>\n", (
        "the tokenizer template renders no reasoning"
    )
    plain_row, plain_tokenizer_row, plain_prefix = texts([*history, {"role": "assistant", "content": "Verdict: OK."}])
    assert plain_row == plain_tokenizer_row and plain_prefix == served_prefix, "no reasoning anywhere: byte-identical"
    sample = packer.sft_process_sample(
        SFTTarget([*history, emission], tools=_AGENTIC_TOOLS, supervise="last_assistant")
    )
    prefix_ids = tokenizer(served_prefix, add_special_tokens=False)["input_ids"]
    assert sample.input_ids[: len(prefix_ids)] == prefix_ids, "the row's ids begin with the served prefix's ids"
    assert all(label == -100 for label in sample.labels[: len(prefix_ids)])
    assert _supervised(packer, sample) == "<think>\nThe pad looks fine.\n</think>\nVerdict: OK.<|im_end|>"
    assert sample.masking == "qwen_chatml_last_assistant_body_and_end_v1"
    # a history turn WITH reasoning_content: the processor template renders a think block the served prefix never has
    with_reasoning = [*history]
    with_reasoning[4] = {**history[4], "reasoning_content": "Let me read the image."}
    bad_row, bad_tokenizer_row, bad_prefix = texts([*with_reasoning, emission])
    assert not bad_row.startswith(bad_prefix) and bad_tokenizer_row.startswith(bad_prefix)
    assert (
        "<think>\nLet me read the image.\n</think>\n" in bad_row and "Let me read the image." not in bad_tokenizer_row
    )
    assert bad_row.index("Let me read the image.") < bad_row.index("Verdict: OK."), "the divergence is in the history"
    with pytest.raises(ValueError, match="history assistant turn 4 carries reasoning \\(reasoning_content\\)"):
        packer.sft_process_sample(
            SFTTarget([*with_reasoning, emission], tools=_AGENTIC_TOOLS, supervise="last_assistant")
        )
    unchanged = packer.sft_process_sample(SFTTarget([*with_reasoning, emission], tools=_AGENTIC_TOOLS))
    assert "Let me read the image." in _supervised(packer, unchanged), "all_assistant rows are untouched"
    # a history content opening with <think></think>: the processor template strips it, the tokenizer template keeps it
    leading = [*history]
    leading[4] = {**history[4], "content": "<think></think>Reading the next crop."}
    lead_row, lead_tokenizer_row, lead_prefix = texts([*leading, emission])
    assert (
        "<think></think>" in lead_tokenizer_row
        and "<think></think>" not in lead_row
        and not lead_row.startswith(lead_prefix)
    )
    with pytest.raises(
        ValueError, match="history assistant turn 4 carries reasoning \\(content opening with <think>\\)"
    ):
        packer.sft_process_sample(SFTTarget([*leading, emission], tools=_AGENTIC_TOOLS, supervise="last_assistant"))
    # the last turn's own reasoning is never the history's: a one-turn row with reasoning packs under last_assistant
    one = packer.sft_process_sample(
        SFTTarget([history[0], history[1], emission], tools=_AGENTIC_TOOLS, supervise="last_assistant")
    )
    assert _supervised(packer, one) == "<think>\nThe pad looks fine.\n</think>\nVerdict: OK.<|im_end|>"


def test_processor_canonical_template_trims_the_replayed_reasoning_where_the_verbatim_template_pads_it(
    tmp_path: Path,
) -> None:
    """Row rule R1 of record (fix round T6; review round 147c): the unified export ships the processor template with ONE
    change, ``reasoning_content`` rendered ``| trim``. A replayed reasoning string that arrives with a newline at both ends
    (dsh's form) renders ``<think>\\n\\nLet me read TASK.md.\\n\\n</think>\\n`` under the verbatim processor template (the
    PADDED form) and ``<think>\\nLet me read TASK.md.\\n</think>\\n`` under the canonical one (the student's own emission
    form): the same token count, the ``\\n\\n`` tokens become ``\\n`` tokens -- the base's 271 -> 198 and 382 -> 624 at
    positions 7248 / 7254 / 8016 / 8025 of the captured request, 8,969 ids either way, pinned in the lab's fix record. A
    trimmed string (row rule R2) renders identically under both, so a row packed through the base's own template and a
    serve through the canonical one agree. Fix round T6b: the same two forms WITHOUT tools, and through the processor
    template file of a unified scratch export (``chat_template.json``, which a processor loader reads), which carries the
    canonical text too. On the parent of T6 ``canonicalize_processor_template`` does not exist; on T6 itself the export's
    ``chat_template.json`` is not written."""
    from cosmos_framework.posttrain.checkpoint.conversion.hf import (
        CANONICAL_REASONING_LINE,
        PROCESSOR_REASONING_LINE,
        canonicalize_processor_template,
    )

    verbatim_text = official_qwen3_vl_chat_template()
    canonical_text = canonicalize_processor_template(verbatim_text)
    assert verbatim_text.count(PROCESSOR_REASONING_LINE) == 1 and canonical_text.count(CANONICAL_REASONING_LINE) == 1
    assert canonical_text != verbatim_text and len(canonical_text) == len(verbatim_text) + len(" | trim") + 2
    tokenizer = local_tokenizer(merge_newlines=True)
    verbatim = local_processor(tokenizer, chat_template=verbatim_text)
    canonical = local_processor(tokenizer, chat_template=canonical_text)
    replayed = [
        {"role": "user", "content": "Solve the task in TASK.md."},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "\nLet me read TASK.md.\n",
            "tool_calls": [{"function": {"name": "zoom", "arguments": {"box": [0, 0, 4, 4]}}}],
        },
        {"role": "tool", "content": "# Task"},
    ]
    padded = verbatim.apply_chat_template(replayed, tools=_AGENTIC_TOOLS, tokenize=False, add_generation_prompt=True)
    stripped = canonical.apply_chat_template(replayed, tools=_AGENTIC_TOOLS, tokenize=False, add_generation_prompt=True)
    assert "<think>\n\nLet me read TASK.md.\n\n</think>\n<tool_call>" in padded
    assert "<think>\nLet me read TASK.md.\n</think>\n<tool_call>" in stripped
    padded_ids = tokenizer(padded, add_special_tokens=False)["input_ids"]
    stripped_ids = tokenizer(stripped, add_special_tokens=False)["input_ids"]
    two, one = tokenizer.convert_tokens_to_ids("ĊĊ"), tokenizer.convert_tokens_to_ids("Ċ")
    assert len(padded_ids) == len(stripped_ids), "the same count, as 8,969 = 8,969 on the captured request"
    differing = [i for i, (a, b) in enumerate(zip(padded_ids, stripped_ids, strict=True)) if a != b]
    assert len(differing) == 2 and all(padded_ids[i] == two and stripped_ids[i] == one for i in differing)
    trimmed = [
        {**message, "reasoning_content": message["reasoning_content"].strip()}
        if "reasoning_content" in message
        else message
        for message in replayed
    ]
    assert (
        verbatim.apply_chat_template(trimmed, tools=_AGENTIC_TOOLS, tokenize=False, add_generation_prompt=True)
        == stripped
    )
    assert (
        canonical.apply_chat_template(trimmed, tools=_AGENTIC_TOOLS, tokenize=False, add_generation_prompt=True)
        == stripped
    )
    packer = QwenSFTDataPacker(tokenizer, processor=canonical, max_length=4096)
    row = [*replayed, {"role": "assistant", "content": "Done.", "reasoning_content": "\nAll read.\n"}]
    sample = packer.sft_process_sample(SFTTarget(row, tools=_AGENTIC_TOOLS))
    assert _supervised(packer, sample).endswith("<think>\nAll read.\n</think>\nDone.<|im_end|>")
    # no tools (a processor loader's path): the same padded vs stripped forms
    padded_no_tools = verbatim.apply_chat_template(replayed, tokenize=False, add_generation_prompt=True)
    stripped_no_tools = canonical.apply_chat_template(replayed, tokenize=False, add_generation_prompt=True)
    assert "<think>\n\nLet me read TASK.md.\n\n</think>\n<tool_call>" in padded_no_tools
    assert "<think>\nLet me read TASK.md.\n</think>\n<tool_call>" in stripped_no_tools
    assert "# Tools" not in stripped_no_tools and padded_no_tools != padded
    # and through the processor template FILE of a unified scratch export of a fake base carrying the official template
    from cosmos_framework.posttrain.checkpoint.conversion.hf import write_unified_chat_template

    base_dir, export_dir = tmp_path / "base", tmp_path / "export"
    base_dir.mkdir()
    export_dir.mkdir()
    (base_dir / "chat_template.json").write_text(json.dumps({"chat_template": verbatim_text}))
    (base_dir / "tokenizer_config.json").write_text(json.dumps({"chat_template": "TOKENIZER-TEMPLATE"}))
    record = write_unified_chat_template(str(export_dir), str(base_dir))
    exported_processor_text = json.loads((export_dir / "chat_template.json").read_text())["chat_template"]
    assert exported_processor_text == canonical_text and record["files"] == [
        "tokenizer_config.json",
        "chat_template.json",
    ]
    assert json.loads((export_dir / "tokenizer_config.json").read_text())["chat_template"] == canonical_text
    exported = local_processor(tokenizer, chat_template=exported_processor_text)
    assert exported.apply_chat_template(replayed, tokenize=False, add_generation_prompt=True) == stripped_no_tools
    assert exported.apply_chat_template(replayed, tools=_AGENTIC_TOOLS, tokenize=False, add_generation_prompt=True) == (
        stripped
    )
    # review R149-02 (fix round T6d), the two row conditions behind "exact under any serve of a unified export" (pins):
    # (a) tool-call arguments travel as the PARSED OBJECT: the template renders a dict through ``tojson`` (the serve
    #     re-serialises the parsed arguments the same way); a JSON STRING is emitted verbatim, so the model's compact bytes
    #     differ from the served prefix at the first argument
    served_prefix = stripped  # the canonical template over the parsed-object row = what the serve renders
    as_string = [dict(message) for message in replayed]
    as_string[1] = {**replayed[1], "tool_calls": [{"function": {"name": "zoom", "arguments": '{"box":[0,0,4,4]}'}}]}
    string_render = canonical.apply_chat_template(
        as_string, tools=_AGENTIC_TOOLS, tokenize=False, add_generation_prompt=True
    )
    assert '"arguments": {"box": [0, 0, 4, 4]}' in served_prefix and '"arguments": {"box":[0,0,4,4]}' in string_render
    served_ids = tokenizer(served_prefix, add_special_tokens=False)["input_ids"]
    string_ids = tokenizer(string_render, add_special_tokens=False)["input_ids"]
    first = next(i for i, (a, b) in enumerate(zip(served_ids, string_ids)) if a != b)
    assert tokenizer.decode(served_ids[:first]).endswith('"arguments": {"box":'), (
        "the first difference is the first argument"
    )
    parsed_again = [dict(message) for message in replayed]
    parsed_again[1] = {
        **replayed[1],
        "tool_calls": [{"function": {"name": "zoom", "arguments": {"box": [0, 0, 4, 4]}}}],
    }
    assert (
        tokenizer(
            canonical.apply_chat_template(
                parsed_again, tools=_AGENTIC_TOOLS, tokenize=False, add_generation_prompt=True
            ),
            add_special_tokens=False,
        )["input_ids"]
        == served_ids
    ), "the parsed-object row matches the served prefix id for id"
    # (b) "trimmed" means str.strip() (Jinja's trim), not one newline removed at each end: a trailing space survives the latter
    spaced = [dict(message) for message in replayed]
    spaced[1] = {**replayed[1], "reasoning_content": "\nLet me read TASK.md. \n"}
    assert (
        canonical.apply_chat_template(spaced, tools=_AGENTIC_TOOLS, tokenize=False, add_generation_prompt=True)
        == stripped
    )
    one_newline_removed = [dict(message) for message in replayed]
    one_newline_removed[1] = {**replayed[1], "reasoning_content": "Let me read TASK.md. "}
    through_base = verbatim.apply_chat_template(
        one_newline_removed, tools=_AGENTIC_TOOLS, tokenize=False, add_generation_prompt=True
    )
    assert through_base != stripped and "TASK.md. \n</think>" in through_base
    fully_stripped = [dict(message) for message in replayed]
    fully_stripped[1] = {**replayed[1], "reasoning_content": "\nLet me read TASK.md. \n".strip()}
    assert (
        verbatim.apply_chat_template(fully_stripped, tools=_AGENTIC_TOOLS, tokenize=False, add_generation_prompt=True)
        == stripped
    )
    # review R149B-01 (fix round T6e): the string form of condition (a). A string equal to json.dumps(obj, ensure_ascii=False)
    # (HF's tojson form) renders byte-equal to the parsed object; json.dumps(obj) with the default ensure_ascii=True escapes
    # non-ASCII and differs, as does the model's compact string.
    accented = {"box": [0, 0, 4, 4], "note": "café"}
    forms = {}
    for label, arguments in (
        ("object", accented),
        ("tojson_string", json.dumps(accented, ensure_ascii=False)),
        ("ascii_string", json.dumps(accented)),
        ("compact_string", json.dumps(accented, ensure_ascii=False, separators=(",", ":"))),
    ):
        row = [dict(message) for message in replayed]
        row[1] = {**replayed[1], "tool_calls": [{"function": {"name": "zoom", "arguments": arguments}}]}
        forms[label] = canonical.apply_chat_template(
            row, tools=_AGENTIC_TOOLS, tokenize=False, add_generation_prompt=True
        )
    assert forms["tojson_string"] == forms["object"] and '"note": "café"' in forms["object"]
    assert forms["ascii_string"] != forms["object"] and '"note": "caf\\u00e9"' in forms["ascii_string"]
    assert forms["compact_string"] != forms["object"] and '"box":[0,0,4,4]' in forms["compact_string"]


# --- Row-level opener: ``SFTTarget.opener`` (fix round T6c, row rule R4) --------------------------------------------

_REASONING_THREE_TURN = [
    {"role": "user", "content": "Use a tool."},
    {
        "role": "assistant",
        "content": None,
        "reasoning_content": "Look.",
        "tool_calls": [{"function": {"name": "lookup", "arguments": {"query": "value"}}}],
    },
    {"role": "tool", "content": "found"},
    {"role": "assistant", "content": "done", "reasoning_content": "Sure."},
]


def test_opener_prefilled_three_turn_row_exact_loss_counts_under_the_four_combinations(
    text_packer: QwenSFTDataPacker,
) -> None:
    """Fixture for ``SFTTarget.opener`` (fix round T6c, row rule R4): a three-turn row through the toy tokenizer (one
    token per ASCII byte, ``<think>`` / ``</think>`` one token each, the toy template's ``<think>\n`` + reasoning +
    ``\n</think>\n\n`` before the body) under supervise x opener. Under ``all_assistant`` the row carries reasoning in both
    assistant turns, so every supervised block has an opener: turn 1 = 1 + 1 + 5 (``Look.``) + 1 + 1 + 2 + 90 (the tool
    call) + 1 (``<|im_end|>``) = 102, turn 2 = 1 + 1 + 5 (``Sure.``) + 1 + 1 + 2 + 4 (``done``) + 1 = 16: ``self`` 118 =
    [102, 16], ``prefilled`` 116 = [101, 15]. Under ``last_assistant`` the same conversation with the history turn
    reasoning-free, as review R151-04 requires (the full row is refused there, asserted): ``self`` 16, ``prefilled`` 15.
    The prefilled counts are the self counts minus one token per supervised block, the difference exactly the ``<think>``
    opener of each supervised block with the target beginning at the newline after it; within each supervision value the
    ids are identical; the contract strings gain ``_opener_prefilled_v1``. On the parent commit the field does not exist."""
    from cosmos_framework.posttrain.data.packer.sft import masking_contract

    think = text_packer.tokenizer.convert_tokens_to_ids("<think>")
    history_free = [
        _REASONING_THREE_TURN[0],
        {key: value for key, value in _REASONING_THREE_TURN[1].items() if key != "reasoning_content"},
        *_REASONING_THREE_TURN[2:],
    ]
    rows = {"all_assistant": _REASONING_THREE_TURN, "last_assistant": history_free}
    with pytest.raises(ValueError, match="history assistant turn 1 carries reasoning"):
        text_packer.sft_process_sample(
            SFTTarget(_REASONING_THREE_TURN, tools=_THREE_TURN_TOOLS, supervise="last_assistant")
        )
    samples = {
        (supervise, opener): text_packer.sft_process_sample(
            SFTTarget(rows[supervise], tools=_THREE_TURN_TOOLS, supervise=supervise, opener=opener)
        )
        for supervise in rows
        for opener in ("self", "prefilled")
    }
    for supervise in rows:
        assert samples[(supervise, "self")].input_ids == samples[(supervise, "prefilled")].input_ids, "labels only"
    counts = {key: _loss_tokens_per_block(text_packer, sample.labels) for key, sample in samples.items()}
    assert counts[("all_assistant", "self")] == [102, 16] and _loss_count(samples[("all_assistant", "self")]) == 118
    assert counts[("all_assistant", "prefilled")] == [101, 15]
    assert _loss_count(samples[("all_assistant", "prefilled")]) == 116
    assert counts[("last_assistant", "self")] == [16] and counts[("last_assistant", "prefilled")] == [15]
    for supervise, expected in (("all_assistant", [102, 16]), ("last_assistant", [91, 16])):
        rendered = text_packer._render(
            text_packer._conversation(SFTTarget(rows[supervise], tools=_THREE_TURN_TOOLS)), _THREE_TURN_TOOLS
        )
        assert _independent_block_counts(text_packer, rendered) == expected, "the self counts are the blocks' counts"
    for supervise, expected_openers in (("all_assistant", 2), ("last_assistant", 1)):
        self_labels = samples[(supervise, "self")].labels
        prefilled_labels = samples[(supervise, "prefilled")].labels
        ids = samples[(supervise, "self")].input_ids
        changed = [i for i, (a, b) in enumerate(zip(self_labels, prefilled_labels, strict=True)) if a != b]
        openers = [i for i, token in enumerate(ids) if token == think and self_labels[i] == think]
        assert changed == openers and len(changed) == expected_openers, "the opener of each supervised block only"
        assert all(prefilled_labels[i] == -100 for i in changed)
        assert all(prefilled_labels[i + 1] != -100 for i in changed), "the target begins with the newline after it"
    assert samples[("all_assistant", "self")].masking == "qwen_chatml_all_assistant_bodies_and_end_v2"
    assert samples[("last_assistant", "self")].masking == "qwen_chatml_last_assistant_body_and_end_v1"
    assert samples[("all_assistant", "prefilled")].masking == (
        "qwen_chatml_all_assistant_bodies_and_end_v2_opener_prefilled_v1"
    )
    assert samples[("last_assistant", "prefilled")].masking == (
        "qwen_chatml_last_assistant_body_and_end_v1_opener_prefilled_v1"
    )
    assert masking_contract("all_assistant", "prefilled") == samples[("all_assistant", "prefilled")].masking
    assert _supervised(text_packer, samples[("last_assistant", "prefilled")]) == "\nSure.\n</think>\n\ndone<|im_end|>"
    assert _supervised(text_packer, samples[("last_assistant", "self")]) == "<think>\nSure.\n</think>\n\ndone<|im_end|>"


def test_opener_prefilled_and_the_global_mask_are_one_rule(text_packer: QwenSFTDataPacker) -> None:
    """A row with ``opener: prefilled`` under a run with ``mask_think_opener=True`` is masked ONCE: its labels equal the
    row-only and the knob-only packings, the count is unchanged (116 on the reasoning three-turn row); the row's contract
    string carries the row's rule, the packer identity the run's knob. Fix round T6c."""
    knob = QwenSFTDataPacker(text_packer.tokenizer, max_length=4096, mask_think_opener=True)
    prefilled_row = SFTTarget(_REASONING_THREE_TURN, tools=_THREE_TURN_TOOLS, opener="prefilled")
    self_row = SFTTarget(_REASONING_THREE_TURN, tools=_THREE_TURN_TOOLS)
    row_only = text_packer.sft_process_sample(prefilled_row)
    knob_only = knob.sft_process_sample(self_row)
    both = knob.sft_process_sample(prefilled_row)
    assert row_only.input_ids == knob_only.input_ids == both.input_ids
    assert row_only.labels == knob_only.labels == both.labels
    assert (
        _loss_count(both) == _loss_count(row_only) == 116 == _loss_count(text_packer.sft_process_sample(self_row)) - 2
    )
    assert row_only.masking == both.masking == "qwen_chatml_all_assistant_bodies_and_end_v2_opener_prefilled_v1"
    assert knob_only.masking == "qwen_chatml_all_assistant_bodies_and_end_v2"
    assert knob.validation_semantics_identity()["mask_think_opener"] is True
    assert "mask_think_opener" not in text_packer.validation_semantics_identity()


def test_opener_refuses_unknown_values_and_a_tokenizer_without_think(text_packer: QwenSFTDataPacker) -> None:
    """Fail closed: the schema refuses any value outside the two at construction, the packer refuses a target mutated after
    construction and a ``prefilled`` row on a tokenizer whose vocabulary has no ``<think>``; ``masking_contract`` refuses
    an unknown opener. Fix round T6c."""
    from cosmos_framework.posttrain.data.packer.sft import masking_contract

    with pytest.raises(ValueError, match="opener must be one of"):
        SFTTarget(_REASONING_THREE_TURN, opener="prefill")
    with pytest.raises(ValueError, match="opener must be one of"):
        SFTTarget(_REASONING_THREE_TURN, opener=None)  # type: ignore[arg-type]
    mutated = SFTTarget(_REASONING_THREE_TURN, tools=_THREE_TURN_TOOLS)
    mutated.opener = "none"
    with pytest.raises(ValueError, match="opener must be one of"):
        text_packer.sft_process_sample(mutated)
    no_think = QwenSFTDataPacker(local_tokenizer_without_think(), max_length=4096)
    with pytest.raises(ValueError, match="opener=prefilled requires a tokenizer whose vocabulary has <think>"):
        no_think.sft_process_sample(SFTTarget(_REASONING_THREE_TURN, tools=_THREE_TURN_TOOLS, opener="prefilled"))
    assert masking_contract("all_assistant", "self") == "qwen_chatml_all_assistant_bodies_and_end_v2"
    with pytest.raises(ValueError, match="opener must be one of"):
        masking_contract("all_assistant", "none")
    assert SFTTarget(_REASONING_THREE_TURN).opener == "self" and SFTPolicyInput([1], [1]).masking.endswith("_end_v2")


def test_mask_think_openers_reads_the_boundary_from_the_original_labels(text_packer: QwenSFTDataPacker) -> None:
    """``_mask_think_openers`` read the supervision boundary from the list it was mutating, so a body that begins with
    two consecutive openers masked both (the first, once ignored, became the ignored neighbour of the second). The
    boundary is read from the ORIGINAL labels: ids ``[5, T, T, 7]`` with labels ``[-100, T, T, 7]`` give
    ``[-100, -100, T, 7]`` (on c0d4306271 and 0153cf2f7727: ``[-100, -100, -100, 7]``). Controls that must not change:
    an inline ``<think>`` after text stays supervised; two blocks with one opener each mask one per block; an opener at
    position 0 is masked."""
    packer = QwenSFTDataPacker(text_packer.tokenizer, max_length=4096, mask_think_opener=True)
    think = packer.tokenizer.convert_tokens_to_ids("<think>")
    a, b, c = 1001, 1002, 1003  # filler ids: the mask compares ids with the opener id only (``<think>`` is 7 here)
    assert think not in (a, b, c)
    assert packer._mask_think_openers([a, think, think, b], [-100, think, think, b]) == [-100, -100, think, b]
    assert packer._mask_think_openers([a, think, c, think, b], [-100, think, c, think, b]) == [-100, -100, c, think, b]
    assert packer._mask_think_openers([think, c, a, think, c], [think, c, -100, think, c]) == [-100, c, -100, -100, c]
    assert packer._mask_think_openers([a, c, think], [-100, c, think]) == [-100, c, think], "inline opener stays"


def test_consecutive_think_openers_mask_exactly_one_token_per_supervised_block(text_packer: QwenSFTDataPacker) -> None:
    """The original-labels boundary of ``_mask_think_openers``, the row through the packer: an assistant body that
    starts with ``<think><think>`` under the run-level ``mask_think_opener=True`` changes exactly ONE label (the first
    opener); the second ``<think>`` stays supervised. The same row as an ``opener=prefilled`` row packs with one removal
    in its block (on c0d4306271 the knob dropped both labels silently and the prefilled row was refused for two
    removals)."""
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "<think><think>\nReason.\n</think>\n\nAnswer."},
    ]
    think = text_packer.tokenizer.convert_tokens_to_ids("<think>")
    default = text_packer.sft_process_sample(SFTTarget(messages))
    knob = QwenSFTDataPacker(text_packer.tokenizer, max_length=4096, mask_think_opener=True)
    masked = knob.sft_process_sample(SFTTarget(messages))
    assert masked.input_ids == default.input_ids
    openers = [i for i, t in enumerate(masked.input_ids) if t == think]
    assert len(openers) == 2 and openers[1] == openers[0] + 1, "two consecutive opener tokens in one body"
    assert default.labels[openers[0]] == think and default.labels[openers[1]] == think
    changed = [i for i, (a, b) in enumerate(zip(default.labels, masked.labels, strict=True)) if a != b]
    assert changed == [openers[0]], "the first opener and only the first opener"
    assert masked.labels[openers[1]] == think, "the second <think> stays supervised"
    assert _loss_count(masked) == _loss_count(default) - 1
    prefilled = text_packer.sft_process_sample(SFTTarget(messages, opener="prefilled"))
    assert prefilled.labels == masked.labels and prefilled.masking.endswith("_opener_prefilled_v1")


def test_opener_prefilled_refuses_a_supervised_block_without_an_opener(text_packer: QwenSFTDataPacker) -> None:
    """Review R149-01 (fix round T6d): a ``prefilled`` row whose supervised assistant block carries no ``<think>`` opener
    is REFUSED, naming the bare block(s) and the per-block mask counts, instead of packing as a self row under a
    ``_opener_prefilled_v1`` contract string. B1: ``all_assistant`` with no reasoning turn at all (both blocks bare);
    B2: ``last_assistant`` with a bare last turn; B3: mixed, turn 1 with reasoning and turn 2 bare. The same rows pack as
    self rows, and the run-level ``mask_think_opener`` on the mixed row masks the one opener it finds and records the run's
    variant in the packer identity, not a prefilled row contract. On cd2f6fddac67 all three rows pack (DID NOT RAISE)."""
    bare_last = [
        _REASONING_THREE_TURN[0],
        {key: value for key, value in _REASONING_THREE_TURN[1].items() if key != "reasoning_content"},
        _REASONING_THREE_TURN[2],
        {"role": "assistant", "content": "done"},
    ]
    mixed = [*_REASONING_THREE_TURN[:3], {"role": "assistant", "content": "done"}]
    with pytest.raises(
        ValueError, match=r"opener=prefilled: supervised assistant block\(s\) \[0, 1\] .*exactly one <think> opener"
    ):
        text_packer.sft_process_sample(
            SFTTarget(_THREE_TURN_MESSAGES, tools=_THREE_TURN_TOOLS, opener="prefilled")
        )  # B1
    with pytest.raises(ValueError, match=r"block\(s\) \[1\] .*masked per supervised block \[1\]: \[0\]"):
        text_packer.sft_process_sample(
            SFTTarget(bare_last, tools=_THREE_TURN_TOOLS, supervise="last_assistant", opener="prefilled")
        )  # B2
    with pytest.raises(ValueError, match=r"block\(s\) \[1\] .*masked per supervised block \[0, 1\]: \[1, 0\]"):
        text_packer.sft_process_sample(SFTTarget(mixed, tools=_THREE_TURN_TOOLS, opener="prefilled"))  # B3
    for messages, supervise in (
        (_THREE_TURN_MESSAGES, "all_assistant"),
        (bare_last, "last_assistant"),
        (mixed, "all_assistant"),
    ):
        sample = text_packer.sft_process_sample(SFTTarget(messages, tools=_THREE_TURN_TOOLS, supervise=supervise))
        assert not sample.masking.endswith("_opener_prefilled_v1"), "a self row never claims the prefilled contract"
    knob = QwenSFTDataPacker(text_packer.tokenizer, max_length=4096, mask_think_opener=True)
    knob_sample = knob.sft_process_sample(SFTTarget(mixed, tools=_THREE_TURN_TOOLS))
    self_sample = text_packer.sft_process_sample(SFTTarget(mixed, tools=_THREE_TURN_TOOLS))
    assert _loss_count(knob_sample) == _loss_count(self_sample) - 1, "the knob masks the one opener the row has"
    assert knob_sample.masking == "qwen_chatml_all_assistant_bodies_and_end_v2"
    assert knob.validation_semantics_identity()["mask_think_opener"] is True
    packed = text_packer.sft_process_sample(
        SFTTarget(_REASONING_THREE_TURN, tools=_THREE_TURN_TOOLS, opener="prefilled")
    )
    assert packed.masking.endswith("_opener_prefilled_v1") and _loss_count(packed) == 116, (
        "reasoning in every block: packs"
    )


def test_unified_export_tokenizer_loads_the_canonical_text_and_a_divergent_jinja_sidecar_is_refused(
    tmp_path: Path,
) -> None:
    """A tokenizer loaded from a unified export carries the canonical text, and the reproduction -- a divergent
    ``chat_template.jinja`` beside an accepted base -- is what the tokenizer loader prefers over both canonical JSON
    files, so the materializer refuses that base (``unified_template_conflicts`` names the file) and the rule refuses
    such an export; a Jinja equal to the canonical text is allowed and effective. Real loader, toy tokenizer
    (``local_tokenizer().save_pretrained``): the base ships its templates in the two JSON files, as the base of record
    does. On 7e5523cabe5 ``unified_template_conflicts`` does not exist."""
    import shutil

    from transformers import AutoTokenizer

    from cosmos_framework.posttrain.checkpoint.conversion.hf import (
        ConversionError,
        canonicalize_processor_template,
        unified_template_conflicts,
        verify_sidecars,
        write_unified_chat_template,
    )

    verbatim_text = official_qwen3_vl_chat_template()
    canonical_text = canonicalize_processor_template(verbatim_text)
    base, export = tmp_path / "base", tmp_path / "export"
    export.mkdir()
    local_tokenizer().save_pretrained(base)
    # a base that ships its templates in the two JSON files: the toy template into tokenizer_config.json, no Jinja sidecar
    if (base / "chat_template.jinja").exists():
        (base / "chat_template.jinja").unlink()
    config = json.loads((base / "tokenizer_config.json").read_text())
    config["chat_template"] = "TOKENIZER-TEMPLATE"
    (base / "tokenizer_config.json").write_text(json.dumps(config, indent=2))
    (base / "chat_template.json").write_text(json.dumps({"chat_template": verbatim_text}))
    assert unified_template_conflicts(str(base)) == ([], [])
    for name in os.listdir(base):
        if (base / name).is_file():
            shutil.copy2(base / name, export / name)  # the sidecars as sync_sidecars copies them
    record = write_unified_chat_template(str(export), str(base))
    assert record["files"] == ["tokenizer_config.json", "chat_template.json"]
    assert AutoTokenizer.from_pretrained(str(export)).chat_template == canonical_text, (
        "the loaded tokenizer is canonical"
    )
    verify_sidecars(str(export), str(base), unified_chat_template=True)
    # the reproduction: the base's verbatim processor template as a Jinja sidecar
    (base / "chat_template.jinja").write_text(verbatim_text)
    conflicts, allowed = unified_template_conflicts(str(base))
    assert [c["file"] for c in conflicts] == ["chat_template.jinja"] and allowed == []
    assert conflicts[0]["equals_canonical"] is False and "tokenizer loader" in conflicts[0]["preferred_by"]
    shutil.copy2(base / "chat_template.jinja", export / "chat_template.jinja")  # what sync_sidecars would do
    loaded = AutoTokenizer.from_pretrained(str(export)).chat_template
    assert loaded == verbatim_text and loaded != canonical_text, "the Jinja shadows both canonical JSON files"
    with pytest.raises(ConversionError, match="chat_template.jinja whose text is not the canonical processor template"):
        verify_sidecars(str(export), str(base), unified_chat_template=True)
    # equal to the canonical text: allowed, recorded, effective
    (base / "chat_template.jinja").write_text(canonical_text)
    shutil.copy2(base / "chat_template.jinja", export / "chat_template.jinja")
    conflicts, allowed = unified_template_conflicts(str(base))
    assert (
        conflicts == [] and [a["file"] for a in allowed] == ["chat_template.jinja"] and allowed[0]["equals_canonical"]
    )
    assert AutoTokenizer.from_pretrained(str(export)).chat_template == canonical_text
    verify_sidecars(str(export), str(base), unified_chat_template=True)
