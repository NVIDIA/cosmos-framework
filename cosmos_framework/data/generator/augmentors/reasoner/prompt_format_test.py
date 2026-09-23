# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

from cosmos_framework.data.generator.augmentors.reasoner.prompt_format import PromptFormat

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_strip_thinking_removes_reasoning_and_inline_trace() -> None:
    formatter = PromptFormat(strip_thinking_prob=1.0)

    result = formatter(
        {
            "texts": [
                {"role": "user", "content": "Question"},
                {
                    "role": "assistant",
                    "reasoning_content": "Hidden reasoning",
                    "content": "<think>Inline reasoning</think>\nFinal answer",
                },
            ]
        }
    )

    assert result is not None
    assert result["is_thinking_stripped"] is True
    assert result["conversation"][0]["content"] == [{"type": "text", "text": "Question"}]
    assert result["conversation"][1]["content"] == [{"type": "text", "text": "Final answer"}]
    assert "reasoning_content" not in result["conversation"][1]


def test_zero_probability_preserves_thinking_without_prompt_injection() -> None:
    formatter = PromptFormat(strip_thinking_prob=0.0)

    result = formatter(
        {
            "texts": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "reasoning_content": "Reasoning", "content": "Final answer"},
            ]
        }
    )

    assert result is not None
    assert result["is_thinking_stripped"] is False
    assert result["conversation"][0]["content"] == [{"type": "text", "text": "Question"}]
    assert result["conversation"][1]["content"] == [
        {"type": "text", "text": "<think>\n"},
        {"type": "text", "text": "Reasoning"},
        {"type": "text", "text": "\n</think>\n\n"},
        {"type": "text", "text": "Final answer"},
    ]


def test_strip_thinking_drops_sample_without_assistant_supervision() -> None:
    formatter = PromptFormat(strip_thinking_prob=1.0)

    result = formatter(
        {
            "texts": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "reasoning_content": "Only reasoning", "content": ""},
            ]
        }
    )

    assert result is None


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_strip_thinking_rejects_invalid_probability(probability: float) -> None:
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        PromptFormat(strip_thinking_prob=probability)
