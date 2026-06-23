# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import torch

from cosmos_framework.configs.base.vlm.experiment.dataflow_roles import (
    VLMProcessor,
    VLMCollator,
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
        # first two tokens are prompt (masked), last two are assistant (kept)
        return torch.tensor([False, False, True, True])


def _item():
    return {
        "conversations": [
            {"from": "human", "value": "hello"},
            {"from": "gpt", "value": "world"},
        ],
        "image": None,
    }


def test_processor_emits_per_sample_constants():
    proc = VLMProcessor(processor=_FakeProcessor())
    out = proc.process(_item())

    assert out["pad_token_id"] == 7
    assert out["ignore_index"] == IGNORE_INDEX
    assert "token_mask" in out
    assert out["token_mask"].dtype == torch.bool
    # labels are -100 where token_mask is False
    assert out["labels"].tolist() == [IGNORE_INDEX, IGNORE_INDEX, 3, 4]


def _sample(n: int, pad_id: int = 0, vision: bool = False, img_tokens: int = 4):
    s = {
        "input_ids": torch.arange(1, n + 1, dtype=torch.long),
        "labels": torch.arange(1, n + 1, dtype=torch.long),
        "attention_mask": torch.ones(n, dtype=torch.bool),
        "token_mask": torch.ones(n, dtype=torch.bool),
        "pad_token_id": pad_id,
        "ignore_index": IGNORE_INDEX,
    }
    if vision:
        s["pixel_values"] = torch.randn(img_tokens, 8)
        s["image_grid_thw"] = torch.tensor([[1, 2, 2]])
    return s


def test_collate_bs1_padded_to_multiple_of_16():
    out = VLMCollator().collate([_sample(5, pad_id=9)])
    assert out["input_ids"].shape == (1, 16)        # 5 -> 16
    assert out["labels"].shape == (1, 16)
    # real tokens preserved, then pad_token_id
    assert out["input_ids"][0, :5].tolist() == [1, 2, 3, 4, 5]
    assert out["input_ids"][0, 5:].eq(9).all()
    # labels padded with ignore_index
    assert out["labels"][0, 5:].eq(IGNORE_INDEX).all()
    assert out["sample_worker_id"].shape == (1,)
    assert out["collated"] is True


def test_collate_bs4_shapes_and_right_pad_fill():
    samples = [_sample(3, pad_id=9), _sample(20, pad_id=9),
               _sample(7, pad_id=9), _sample(12, pad_id=9)]
    out = VLMCollator().collate(samples)
    assert out["input_ids"].shape == (4, 32)        # max 20 -> 32
    # attention_mask / token_mask padded with False past real length
    assert out["attention_mask"][0, :3].all()
    assert (~out["attention_mask"][0, 3:]).all()
    # row 1 (len 20) real tokens intact
    assert out["input_ids"][1, :20].tolist() == list(range(1, 21))
    assert out["input_ids"][1, 20:].eq(9).all()
    # length-B resume meta
    assert out["sample_epoch"].tolist() == [0, 0, 0, 0]
    assert out["sample_index"].shape == (4,)


def test_collate_vision_flat_concat():
    samples = [_sample(5, vision=True, img_tokens=4),
               _sample(6, vision=True, img_tokens=9)]
    out = VLMCollator().collate(samples)
    # pixel_values concatenated on dim 0 (4 + 9), NOT stacked on a batch axis
    assert out["pixel_values"].shape[0] == 13
    # image_grid_thw concatenated: 2 rows of [1,2,2]
    assert out["image_grid_thw"].shape == (2, 3)


def test_collate_uses_per_sample_pad_token_id():
    # Distinct pad ids per sample: each row's pad region must use its OWN pad id,
    # not a single global value taken from samples[0].
    samples = [_sample(3, pad_id=11), _sample(8, pad_id=22)]
    out = VLMCollator().collate(samples)
    assert out["input_ids"].shape == (2, 16)        # max 8 -> 16
    # row 0 (len 3) real tokens, then its own pad id 11
    assert out["input_ids"][0, :3].tolist() == [1, 2, 3]
    assert out["input_ids"][0, 3:].eq(11).all()
    # row 1 (len 8) real tokens, then its own pad id 22
    assert out["input_ids"][1, :8].tolist() == list(range(1, 9))
    assert out["input_ids"][1, 8:].eq(22).all()
