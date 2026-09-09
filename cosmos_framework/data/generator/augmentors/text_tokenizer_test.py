# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from typing import Any

import pytest
import torch

from cosmos_framework.data.generator.augmentors import text_tokenizer

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class _FakeProcessor:
    calls: list[tuple[str, dict[str, Any]]]

    def __init__(self) -> None:
        self.calls = []

    def tokenize_text(self, caption: str, **kwargs: Any) -> list[int]:
        self.calls.append((caption, kwargs))
        return list(range(len(caption.split()) + 1))


def _make_transfer_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    processor: _FakeProcessor,
    *,
    cfg_dropout_rate: float = 0.0,
    tokenize_separately: bool = True,
    task: str | None = None,
) -> text_tokenizer.TextTokenizerTransformForTransfer:
    monkeypatch.setattr(text_tokenizer, "lazy_instantiate", lambda _config: processor)
    args: dict[str, object] = {
        "tokenizer_config": object(),
        "cfg_dropout_rate": cfg_dropout_rate,
        "tokenize_separately": tokenize_separately,
    }
    if task is not None:
        args["task"] = task
    return text_tokenizer.TextTokenizerTransformForTransfer(
        input_keys=["ai_caption"],
        output_keys=["text_token_ids", "text_token_lengths"] if tokenize_separately else ["text_token_ids"],
        args=args,
    )


def test_transfer_tokenizer_tokenizes_each_view_independently(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = _FakeProcessor()
    tokenizer = _make_transfer_tokenizer(monkeypatch, processor)
    second_caption = {"caption": "a longer side view"}

    result = tokenizer({"ai_caption": ["front view", second_caption], "sample_n_views": 2})

    assert result is not None
    assert result["ai_caption"] == ["front view", json.dumps(second_caption)]
    assert len(result["text_token_ids"]) == 2
    assert all(isinstance(tokens, torch.Tensor) for tokens in result["text_token_ids"])
    assert result["text_token_lengths"] == [3, 6]
    assert [caption for caption, _kwargs in processor.calls] == result["ai_caption"]
    assert all(
        call_kwargs["system_prompt"] == text_tokenizer._SYSTEM_PROMPT_TRANSFER for _, call_kwargs in processor.calls
    )


def test_transfer_tokenizer_accepts_av_multiview_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = _FakeProcessor()
    tokenizer = _make_transfer_tokenizer(monkeypatch, processor, task="av_multiview_transfer")

    result = tokenizer({"ai_caption": ["front view", "rear view"], "sample_n_views": 2})

    assert result is not None
    assert all(
        call_kwargs["system_prompt"] == text_tokenizer._SYSTEM_PROMPT_AV_MULTIVIEW_TRANSFER
        for _, call_kwargs in processor.calls
    )


def test_separate_transfer_tokenizer_applies_cfg_dropout_once_per_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = _FakeProcessor()
    tokenizer = _make_transfer_tokenizer(monkeypatch, processor, cfg_dropout_rate=0.5)
    random_calls = 0

    def sample_dropout() -> float:
        nonlocal random_calls
        random_calls += 1
        return 0.0

    monkeypatch.setattr(text_tokenizer.random, "random", sample_dropout)

    result = tokenizer({"ai_caption": ["front view", "rear view"], "sample_n_views": 2})

    assert result is not None
    assert random_calls == 1
    assert result["ai_caption"] == ["", ""]
    assert [caption for caption, _kwargs in processor.calls] == ["", ""]


def test_transfer_tokenizer_keeps_legacy_scalar_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = _FakeProcessor()
    tokenizer = _make_transfer_tokenizer(monkeypatch, processor, tokenize_separately=False)
    caption = {"caption": "front view"}

    result = tokenizer({"ai_caption": caption})

    assert result is not None
    assert result["ai_caption"] == json.dumps(caption)
    assert isinstance(result["text_token_ids"], torch.Tensor)
    assert "text_token_lengths" not in result


def test_single_view_separate_tokenization_matches_legacy_token_values(monkeypatch: pytest.MonkeyPatch) -> None:
    caption = {"caption": "front view"}
    legacy_processor = _FakeProcessor()
    legacy_tokenizer = _make_transfer_tokenizer(monkeypatch, legacy_processor, tokenize_separately=False)
    legacy_result = legacy_tokenizer({"ai_caption": caption})

    separate_processor = _FakeProcessor()
    separate_tokenizer = _make_transfer_tokenizer(monkeypatch, separate_processor)
    separate_result = separate_tokenizer({"ai_caption": [caption], "sample_n_views": 1})

    assert legacy_result is not None
    assert separate_result is not None
    assert torch.equal(separate_result["text_token_ids"][0], legacy_result["text_token_ids"])
    assert separate_result["text_token_lengths"] == [int(legacy_result["text_token_ids"].shape[0])]
    assert legacy_processor.calls == separate_processor.calls


def test_separate_transfer_tokenizer_requires_view_count_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _make_transfer_tokenizer(monkeypatch, _FakeProcessor())

    with pytest.raises(ValueError, match="requires sample_n_views metadata"):
        tokenizer({"ai_caption": ["front view"]})


def test_separate_transfer_tokenizer_requires_one_caption_per_view(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _make_transfer_tokenizer(monkeypatch, _FakeProcessor())

    with pytest.raises(ValueError, match="captions=1, sample_n_views=2"):
        tokenizer({"ai_caption": ["front view"], "sample_n_views": 2})
