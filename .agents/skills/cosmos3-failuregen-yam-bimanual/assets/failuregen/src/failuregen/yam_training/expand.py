# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Append new domain rows after loading a checkpoint at its original shape."""

import math

import torch
from torch import nn


def expand_action_domains(net: nn.Module, new_count: int, seed: int = 42) -> dict:
    from cosmos_framework.model.generator.mot.domain_aware_linear import DomainAwareLinear

    projectors = [getattr(net, name, None) for name in ("action2llm", "llm2action")]
    if not all(isinstance(layer, DomainAwareLinear) for layer in projectors):
        raise ValueError("Expected domain-aware action input/output projections; inspect checkpoint architecture")
    old_count = projectors[0].num_domains
    if any(layer.num_domains != old_count for layer in projectors) or new_count <= old_count:
        raise ValueError(f"Expansion must append rows after all {old_count} existing domains")
    if any(layer.fc.weight.device.type != "cpu" for layer in projectors):
        raise ValueError("Checkpoint domain expansion must run on CPU")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for layer in projectors:
            old_weight, old_bias = layer.fc.weight.detach(), layer.bias.weight.detach()
            weight = old_weight.new_empty((new_count, old_weight.shape[1]))
            bias = old_bias.new_zeros((new_count, old_bias.shape[1]))
            std = 1.0 / math.sqrt(layer.input_size)
            nn.init.trunc_normal_(weight[old_count:], std=std, a=-3 * std, b=3 * std)
            weight[:old_count].copy_(old_weight)
            bias[:old_count].copy_(old_bias)
            if not torch.equal(weight[:old_count], old_weight) or not torch.equal(bias[:old_count], old_bias):
                raise AssertionError("Existing action domains changed during expansion")
            layer.fc = nn.Embedding.from_pretrained(weight, freeze=False)
            layer.bias = nn.Embedding.from_pretrained(bias, freeze=False)
            layer.num_domains = new_count
    net.num_embodiment_domains = new_count
    return {"old_domain_count": old_count, "new_domain_count": new_count, "seed": seed, "existing_rows_preserved": True}
