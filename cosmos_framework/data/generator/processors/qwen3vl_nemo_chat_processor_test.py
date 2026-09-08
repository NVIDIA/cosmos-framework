# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

import cosmos_framework.data.generator.processors as processors
from cosmos_framework.data.generator.processors.qwen3vl_nemo_chat_processor import Qwen3VLNemoChatProcessor
from cosmos_framework.data.generator.processors.qwen3vl_processor import Qwen3VLProcessor

_BOS_ID: int = 10
_EOS_ID: int = 11
_ASSISTANT_ID: int = 12
_THINK_START_ID: int = 20
_THINK_END_ID: int = 21
_NEWLINE_ID: int = 22
_HEADER_SEPARATOR_ID: int = 30
_ANSWER_ID: int = 40
_REASONING_ID: int = 41
_USER_ID: int = 50
_QUESTION_ID: int = 51
_NEMO_CHAT_V3_MODEL: str = "Qwen/Qwen3-VL-8B-Instruct-Nemo-Chat-v3"


def _make_loss_mask_processor_stub(
    assistant_role_ids: list[int] | None = None,
) -> Qwen3VLNemoChatProcessor:
    processor = Qwen3VLNemoChatProcessor.__new__(Qwen3VLNemoChatProcessor)
    processor.processor = Mock()
    tokenizer = Mock()
    role_ids = assistant_role_ids or [_ASSISTANT_ID]
    token_ids = {
        "<|im_start|>": _BOS_ID,
        "<|im_end|>": _EOS_ID,
        "assistant": role_ids[0] if len(role_ids) == 1 else -1,
        "<think>": _THINK_START_ID,
    }
    tokenizer.convert_tokens_to_ids.side_effect = token_ids.__getitem__

    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return {"assistant": role_ids, "\n": [_NEWLINE_ID]}[text]

    tokenizer.encode.side_effect = encode
    processor.processor.tokenizer = tokenizer
    processor.think_end_id = _THINK_END_ID
    return processor


@pytest.mark.L0
def test_nemo_chat_processor_records_think_end_id(monkeypatch: pytest.MonkeyPatch) -> None:
    hf_processor = Mock()
    hf_processor.tokenizer.convert_tokens_to_ids.return_value = _THINK_END_ID

    def initialize_base_processor(
        processor: Qwen3VLProcessor,
        name: str,
        credentials: str,
        bucket: str,
        cache_dir: str | None,
    ) -> None:
        del name, credentials, bucket, cache_dir
        processor.processor = hf_processor

    monkeypatch.setattr(Qwen3VLProcessor, "__init__", initialize_base_processor)

    processor = Qwen3VLNemoChatProcessor()

    assert processor.think_end_id == _THINK_END_ID
    hf_processor.tokenizer.convert_tokens_to_ids.assert_called_once_with("</think>")


@pytest.mark.L0
def test_build_processor_uses_nemo_chat_wrapper_for_v3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other_model = "Qwen/Qwen3-VL-8B-Instruct"
    nemo_processor = object()
    other_processor = object()
    nemo_processor_class = Mock(return_value=nemo_processor)
    qwen_processor_class = Mock(return_value=other_processor)
    monkeypatch.setattr(processors, "Qwen3VLNemoChatProcessor", nemo_processor_class)
    monkeypatch.setattr(processors, "Qwen3VLProcessor", qwen_processor_class)

    assert processors.build_processor(_NEMO_CHAT_V3_MODEL, credentials="credentials", bucket="bucket") is nemo_processor
    assert processors.build_processor(other_model, credentials="credentials", bucket="bucket") is other_processor
    nemo_processor_class.assert_called_once_with(
        _NEMO_CHAT_V3_MODEL,
        credentials="credentials",
        bucket="bucket",
        cache_dir=None,
    )
    qwen_processor_class.assert_called_once_with(
        other_model,
        credentials="credentials",
        bucket="bucket",
        cache_dir=None,
    )


@pytest.mark.L0
def test_build_processor_selects_nemo_chat_wrapper_for_local_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / _NEMO_CHAT_V3_MODEL
    model_path.mkdir(parents=True)
    expected_processor = object()
    processor_class = Mock(return_value=expected_processor)
    monkeypatch.setattr(processors, "Qwen3VLNemoChatProcessor", processor_class)

    processor = processors.build_processor(str(model_path), cache_dir="cache")

    assert processor is expected_processor
    processor_class.assert_called_once_with(str(model_path), cache_dir="cache")


@pytest.mark.L0
@pytest.mark.parametrize(
    ("assistant_content", "expected_content_mask"),
    [
        ([_THINK_START_ID, _THINK_END_ID, _ANSWER_ID], [False, False, True]),
        (
            [_THINK_START_ID, _NEWLINE_ID, _THINK_END_ID, _ANSWER_ID],
            [False, False, False, True],
        ),
        ([_THINK_START_ID, _REASONING_ID, _THINK_END_ID, _ANSWER_ID], [False, True, True, True]),
        (
            [_THINK_START_ID, _NEWLINE_ID, _REASONING_ID, _THINK_END_ID, _ANSWER_ID],
            [False, True, True, True, True],
        ),
        ([_ANSWER_ID], [True]),
    ],
)
def test_nemo_chat_loss_mask_depends_on_leading_think_tags(
    assistant_content: list[int],
    expected_content_mask: list[bool],
) -> None:
    processor = _make_loss_mask_processor_stub()
    tokens = [_BOS_ID, _ASSISTANT_ID, _HEADER_SEPARATOR_ID, *assistant_content, _EOS_ID]

    mask = processor.add_assistant_tokens_mask(tokens)

    assert mask == [False, False, False, *expected_content_mask, True]


@pytest.mark.L0
def test_nemo_chat_loss_mask_supports_batched_multi_turn_tokens() -> None:
    processor = _make_loss_mask_processor_stub()
    tokens = torch.tensor(
        [
            [
                _BOS_ID,
                _USER_ID,
                _HEADER_SEPARATOR_ID,
                _QUESTION_ID,
                _EOS_ID,
                _BOS_ID,
                _ASSISTANT_ID,
                _HEADER_SEPARATOR_ID,
                _THINK_START_ID,
                _NEWLINE_ID,
                _THINK_END_ID,
                _ANSWER_ID,
                _EOS_ID,
            ],
            [
                _BOS_ID,
                _USER_ID,
                _HEADER_SEPARATOR_ID,
                _QUESTION_ID,
                _EOS_ID,
                _BOS_ID,
                _ASSISTANT_ID,
                _HEADER_SEPARATOR_ID,
                _THINK_START_ID,
                _NEWLINE_ID,
                _REASONING_ID,
                _THINK_END_ID,
                _EOS_ID,
            ],
        ]
    )  # [B,N_token]
    expected = torch.tensor(
        [
            [False, False, False, False, False, False, False, False, False, False, False, True, True],
            [False, False, False, False, False, False, False, False, False, True, True, True, True],
        ]
    )  # [B,N_token]

    mask = processor.add_assistant_tokens_mask(tokens)  # [B,N_token]

    assert isinstance(mask, torch.Tensor)
    assert torch.equal(mask, expected)


@pytest.mark.L0
def test_nemo_chat_loss_mask_supports_multi_token_assistant_role() -> None:
    assistant_role_ids = [_ASSISTANT_ID, _ASSISTANT_ID + 1]
    processor = _make_loss_mask_processor_stub(assistant_role_ids)
    tokens = [
        _BOS_ID,
        *assistant_role_ids,
        _HEADER_SEPARATOR_ID,
        _THINK_START_ID,
        _THINK_END_ID,
        _ANSWER_ID,
        _EOS_ID,
    ]

    mask = processor.add_assistant_tokens_mask(tokens)

    assert mask == [False, False, False, False, False, False, True, True]
