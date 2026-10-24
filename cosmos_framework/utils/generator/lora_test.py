# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import pytest
import torch

from cosmos_framework.utils.generator.lora import (
    LoraInjectedLinear,
    apply_lora_trainable_scope,
    inject_lora_pre_fsdp,
    merge_and_strip_lora_adapters_,
    merge_lora_adapters_,
)

pytestmark = pytest.mark.L0


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
    assert model._cosmos_peft_parameter_summary == {
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
    assert summary == model._cosmos_peft_parameter_summary
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


@pytest.mark.parametrize("use_rslora", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_export_merge_preserves_precision_and_training_weights(dtype, use_rslora):
    torch.manual_seed(13)
    layer = LoraInjectedLinear(
        torch.nn.Linear(4, 3, dtype=dtype), rank=2, alpha=3, use_rslora=use_rslora, adapter_dtype=torch.float32
    )
    base = layer.weight.detach().clone()
    adapter_a = layer.lora_A.weight.detach().clone()
    adapter_b = layer.lora_B.weight.detach().clone()
    scale = 3 / (2**0.5 if use_rslora else 2)
    accumulation_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    expected = base.to(accumulation_dtype) + scale * (
        adapter_b.to(accumulation_dtype) @ adapter_a.to(accumulation_dtype)
    )
    # A higher-precision export must not first round the merged weight to BF16.
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        exported = layer.merged_weight(base, adapter_a, adapter_b, dtype=accumulation_dtype)
    torch.testing.assert_close(exported, expected, rtol=0, atol=0)
    for actual, original in ((layer.weight, base), (layer.lora_A.weight, adapter_a), (layer.lora_B.weight, adapter_b)):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


@pytest.mark.parametrize("merge_api", ["layer", "network", "offline"])
@pytest.mark.parametrize(
    "dtype,autocast,expected_inplace,expected_export",
    [
        (torch.bfloat16, False, 1.0, 1.0078125),
        (torch.bfloat16, True, 1.0, 1.0078125),
        (torch.float32, False, 1.00392, 1.00392),
        (torch.float32, True, 1.00390625, 1.00392),
    ],
)
@torch.no_grad()
def test_existing_merge_apis_retain_rounding_and_autocast(
    merge_api, dtype, autocast, expected_inplace, expected_export
):
    layer = LoraInjectedLinear(
        torch.nn.Linear(1, 1, bias=False, dtype=dtype), rank=1, alpha=1, adapter_dtype=torch.float32
    )
    layer.weight.fill_(1.0)
    layer.lora_A.weight.fill_(1.0)
    layer.lora_B.weight.fill_(0.00392)
    weight = layer.weight
    network = torch.nn.ModuleDict({"layer": layer})
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=autocast):
        exported = layer.merged_weight(layer.weight, layer.lora_A.weight, layer.lora_B.weight)
        if merge_api == "layer":
            layer.merge_adapter_()
        elif merge_api == "network":
            assert merge_lora_adapters_(network) == 1
        else:
            assert merge_and_strip_lora_adapters_(network) == 1
            assert type(network["layer"]) is torch.nn.Linear
    # Deliberate boundary: legacy APIs round the delta before adding it, and
    # honor ambient autocast. Automatic export rounds only the completed sum.
    assert network["layer"].weight is weight
    torch.testing.assert_close(weight, torch.full_like(weight, expected_inplace), rtol=0, atol=0)
    torch.testing.assert_close(exported, torch.full_like(weight, expected_export), rtol=0, atol=0)
