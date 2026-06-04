# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for VLMProcessor / VLMCollator extracted from VLMDataPacker."""

from __future__ import annotations

import torch

from cosmos_framework.configs.base.vlm.experiment.dataflow_roles import VLMCollator, VLMProcessor


class _FakeProcessor:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return {
            "input_ids": torch.arange(6),
            "pixel_values": torch.zeros(4, 8),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }

    def add_assistant_tokens_mask(self, input_ids):
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        mask[3:] = True
        return mask


def _item():
    return {
        "image": None,
        "conversations": [
            {"from": "human", "value": "hi <image>"},
            {"from": "gpt", "value": "hello"},
        ],
    }


def test_vlmprocessor_builds_input_ids_and_masked_labels():
    p = VLMProcessor(processor=_FakeProcessor(), ignore_index=-100)
    s = p.process(_item())
    assert s["input_ids"].tolist() == [0, 1, 2, 3, 4, 5]
    assert s["labels"].tolist() == [-100, -100, -100, 3, 4, 5]
    assert "pixel_values" in s and "image_grid_thw" in s


def test_vlmcollator_adds_batch_dim_and_resume_meta():
    p = VLMProcessor(processor=_FakeProcessor(), ignore_index=-100)
    s = p.process(_item())
    batch = VLMCollator().collate([s])
    assert batch["input_ids"].shape == (1, 6)
    assert batch["labels"].shape == (1, 6)
    assert batch["pixel_values"].shape == (4, 8)
    assert batch["image_grid_thw"].shape == (1, 3)
    assert batch["sample_epoch"].tolist() == [0]
    assert batch["sample_index"].tolist() == [0]
