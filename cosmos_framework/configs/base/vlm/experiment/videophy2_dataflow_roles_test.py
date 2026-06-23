# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import torch

from cosmos_framework.configs.base.vlm.experiment.videophy2_dataflow_roles import (
    VideoPhy2Processor,
)
from cosmos_framework.utils.vlm.constant import IGNORE_INDEX


class _FakeTok:
    pad_token_id = 7


class _FakeProcessor:
    tokenizer = _FakeTok()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return {
            "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "attention_mask": torch.ones(4, dtype=torch.bool),
        }

    def add_assistant_tokens_mask(self, input_ids):
        return torch.tensor([False, False, True, True])


def _item():
    return {
        "texts": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ],
        "media": {},
    }


def test_videophy2_processor_emits_per_sample_constants():
    proc = VideoPhy2Processor(processor=_FakeProcessor())
    out = proc.process(_item())

    assert out["pad_token_id"] == 7
    assert out["ignore_index"] == IGNORE_INDEX
    assert "token_mask" in out
    assert out["token_mask"].dtype == torch.bool
    assert out["labels"].tolist() == [IGNORE_INDEX, IGNORE_INDEX, 3, 4]
