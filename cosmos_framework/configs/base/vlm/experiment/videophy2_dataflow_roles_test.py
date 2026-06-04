# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for VideoPhy2Processor (extracted from VideoPhy2DataPacker)."""

from __future__ import annotations

import pytest
import torch

from cosmos_framework.configs.base.vlm.experiment.videophy2_dataflow_roles import VideoPhy2Processor


class _FakeProcessor:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return {"input_ids": torch.arange(6), "pixel_values_videos": torch.zeros(3, 8)}

    def add_assistant_tokens_mask(self, input_ids):
        m = torch.zeros_like(input_ids, dtype=torch.bool)
        m[3:] = True
        return m


def test_videophy2_processor_builds_masked_labels():
    p = VideoPhy2Processor(processor=_FakeProcessor(), ignore_index=-100)
    s = p.process({
        "texts": [
            {"role": "user", "content": "describe this"},
            {"role": "assistant", "content": "a ball falls"},
        ],
        "media": {},
    })
    assert s["input_ids"].tolist() == [0, 1, 2, 3, 4, 5]
    assert s["labels"].tolist() == [-100, -100, -100, 3, 4, 5]
    assert "pixel_values_videos" in s


def test_videophy2_processor_rejects_non_list_texts():
    p = VideoPhy2Processor(processor=_FakeProcessor())
    with pytest.raises(TypeError):
        p.process({"texts": "not-a-list", "media": {}})
