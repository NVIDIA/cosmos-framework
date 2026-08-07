# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import torch

from cosmos_framework.utils.generator.lora import (
    LoraInjectedLinear,
    apply_lora_trainable_scope,
    inject_lora_pre_fsdp,
    merge_lora_adapters_,
)


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 3, bias=True)
        self.lm_head = torch.nn.Linear(3, 2, bias=False)


def test_vlm_lora_preserves_base_keys_and_reports_trainable_scope() -> None:
    model = _Tiny()
    base_keys = set(model.state_dict())
    inject_lora_pre_fsdp(
        model,
        lora_rank=2,
        lora_alpha=4,
        lora_target_modules="q_proj",
        lora_dropout=0.0,
        lora_bias="lora_only",
        lora_modules_to_save="lm_head",
    )

    assert isinstance(model.q_proj, LoraInjectedLinear)
    assert base_keys <= set(model.state_dict())
    assert model.q_proj.weight.requires_grad is False
    assert model.q_proj.bias.requires_grad is True
    assert model.lm_head.weight.requires_grad is True
    assert model.q_proj.lora_A.weight.requires_grad is True
    assert model._tao_peft_parameter_summary == {
        "training_mode": "peft",
        "trainable_parameters": 23,
        "total_parameters": 35,
        "frozen_parameters": 12,
        "trainable_parameter_tensors": 4,
        "adapter_module_count": 1,
        "adapter_modules": ["q_proj"],
    }

    model.requires_grad_(True)
    summary = apply_lora_trainable_scope(
        model,
        lora_target_modules="q_proj",
        lora_bias="lora_only",
        lora_modules_to_save="lm_head",
    )
    assert summary == model._tao_peft_parameter_summary
    assert model.q_proj.weight.requires_grad is False
    assert model.q_proj.bias.requires_grad is True
    assert model.lm_head.weight.requires_grad is True
    assert model.q_proj.lora_A.weight.requires_grad is True


def test_rslora_merge_matches_unmerged_forward() -> None:
    model = _Tiny()
    inject_lora_pre_fsdp(
        model,
        lora_rank=2,
        lora_alpha=4,
        lora_target_modules="q_proj",
        lora_use_rslora=True,
    )
    torch.nn.init.normal_(model.q_proj.lora_A.weight)
    torch.nn.init.normal_(model.q_proj.lora_B.weight)
    values = torch.randn(2, 4)
    expected = model.q_proj(values)
    assert merge_lora_adapters_(model) == 1
    with torch.no_grad():
        model.q_proj.lora_A.weight.zero_()
        model.q_proj.lora_B.weight.zero_()
    torch.testing.assert_close(model.q_proj(values), expected)
