# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Restore stored agentic fields and preserve per-turn supervision for Qwen templates."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

REASONING_PREFILL_KEY = "_reasoning_prefix_in_prompt"


def restore_agentic_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Extract the tool menu and restore observation roles without changing storage."""
    restored = copy.deepcopy(messages)
    tools = restored[0].pop("tools", None) if restored else None
    for message in restored:
        if message.get("role") == "user" and message.get("tool_call_id"):
            message["role"] = "tool"
    return restored, tools


def assistant_body_spans(tokens: list[int] | np.ndarray, tokenizer: Any) -> list[tuple[int, int]]:
    """Find assistant bodies, retaining a BPE token shared with source whitespace."""
    values = np.asarray(tokens)
    start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    role = tokenizer.encode("assistant", add_special_tokens=False)
    header = tokenizer.encode("assistant\n", add_special_tokens=False)
    starts = np.flatnonzero(values == start_id)
    ends = np.flatnonzero(values == end_id)
    if len(starts) != len(ends):
        raise ValueError("Unbalanced chat boundaries while applying source supervision")
    spans = []
    previous_end = -1
    for start, end in zip(starts, ends):
        if not previous_end < start < end:
            raise ValueError("Nested or reordered chat boundaries")
        previous_end = end
        body = int(start + 1 + len(header))
        if values[start + 1 : body].tolist() != header:
            if values[start + 1 : start + 1 + len(role)].tolist() != role:
                continue
            candidate = values[start + 1 : min(body + 4, end)].tolist()
            decoded = tokenizer.decode(candidate, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if not decoded.startswith("assistant\n"):
                raise ValueError("Unexpected assistant role header")
            whole = 0
            for size in range(1, len(candidate) + 1):
                partial = tokenizer.decode(
                    candidate[:size], skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                if not "assistant\n".startswith(partial):
                    break
                whole = size
            if whole == 0:
                raise ValueError("No complete assistant role token")
            body = int(start + 1 + whole)
        spans.append((body, int(end + 1)))
    return spans


def restrict_assistant_loss(
    tokens: list[int] | np.ndarray,
    mask: list[bool] | np.ndarray,
    messages: list[dict[str, Any]],
    tokenizer: Any,
) -> np.ndarray:
    """Only remove loss: context assistant turns and inserted reasoning prefills.

    Fail on mismatched boundaries instead of allowing a context turn into loss.
    This runs after modality token expansion, using the final token sequence.
    """
    values = np.asarray(tokens)
    result = np.asarray(mask, dtype=bool).copy()
    if values.ndim != 1 or result.shape != values.shape:
        raise ValueError("Expected matching one-dimensional token and loss arrays")
    assistants = [message for message in messages if message.get("role") == "assistant"]
    if not any(message.get("loss") is False or message.get(REASONING_PREFILL_KEY) for message in assistants):
        return result
    spans = assistant_body_spans(values, tokenizer)
    if len(spans) != len(assistants):
        raise ValueError("Assistant message count differs from rendered assistant frames")
    for (start, end), message in zip(spans, assistants):
        if message.get("loss") is False:
            result[start:end] = False
        elif message.get(REASONING_PREFILL_KEY):
            prefix_text = "<think>\n"
            prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
            if values[start : start + len(prefix)].tolist() == prefix:
                result[start : start + len(prefix)] = False
                continue
            # BPE can merge the inserted newline with source-leading whitespace.
            # Mask only tokens wholly inside the inserted prefix. Keep a shared
            # boundary token supervised so no source reasoning token is lost.
            candidate = values[start : min(start + len(prefix) + 4, end)].tolist()
            decoded = tokenizer.decode(candidate, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if not decoded.startswith(prefix_text):
                raise ValueError("Structured reasoning prefill differs from tokenizer output")
            prefix_tokens = 0
            for size in range(1, len(candidate) + 1):
                partial = tokenizer.decode(
                    candidate[:size], skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                if not prefix_text.startswith(partial):
                    break
                prefix_tokens = size
            if prefix_tokens == 0:
                raise ValueError("No complete token inside structured reasoning prefill")
            result[start : start + prefix_tokens] = False
    return result
