# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L0 tests for `OptimizerConfig.decay_groups`: `""` keeps the single-group construction, `"hf"` reproduces
`transformers.Trainer.get_decay_parameter_names` (checked against HF's own method) and its group order."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cosmos_framework.posttrain.config.build import build_optimizer
from cosmos_framework.posttrain.config.rl import OptimizerConfig


class _AdaNormLike(nn.Module):
    """A Linear under an attribute named `norm1`, as diffusers' AdaLayerNorm has: `norm1.linear.weight`
    does NOT match HF's `(?:^|\\.)norm(?:$|\\.)` pattern, so it decays."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = _AdaNormLike()
        self.ln = nn.LayerNorm(4)  # excluded by TYPE, not by name
        self.embed = nn.Embedding(8, 4)
        self.proj = nn.Linear(4, 4)
        self.b = nn.Parameter(torch.zeros(4))  # a bare `b`, as CategorySpecificLinear has: no pattern matches it
        self.frozen = nn.Linear(4, 4)
        self.frozen.requires_grad_(False)


def _hf_decay_names(module: nn.Module) -> set[str]:
    from transformers import Trainer

    # `get_decay_parameter_names` reads only its `model` argument, never `self`.
    return set(Trainer.get_decay_parameter_names(None, module))


def _names_by_id(net: nn.Module) -> dict[int, str]:
    return {id(p): n for n, p in net.named_parameters()}


@pytest.mark.L0
@pytest.mark.CPU
def test_unknown_decay_groups_mode_is_rejected_at_config_time() -> None:
    with pytest.raises(ValueError, match="decay_groups"):
        OptimizerConfig(decay_groups="huggingface")


@pytest.mark.L0
@pytest.mark.CPU
def test_empty_mode_builds_one_group_carrying_the_configured_decay() -> None:
    net = _Toy()
    opt = build_optimizer(net, OptimizerConfig(weight_decay=0.123))
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.123)
    assert len(opt.param_groups[0]["params"]) == len([p for p in net.parameters() if p.requires_grad])


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_mode_reproduces_the_transformers_partition_and_group_order() -> None:
    net = _Toy()
    opt = build_optimizer(net, OptimizerConfig(decay_groups="hf", weight_decay=0.5))

    # HF emits [decay, no-decay]; checkpointed optimizer state is indexed by group position.
    assert [g["weight_decay"] for g in opt.param_groups] == [pytest.approx(0.5), 0.0]
    decay_names, by_id = _hf_decay_names(net), _names_by_id(net)
    trainable = {n for n, p in net.named_parameters() if p.requires_grad}
    assert {by_id[id(p)] for p in opt.param_groups[0]["params"]} == trainable & decay_names
    assert {by_id[id(p)] for p in opt.param_groups[1]["params"]} == trainable - decay_names
    # The unobvious placements, per HF: `norm1.linear.weight` and the bare `b` decay; a real bias does not.
    assert {"norm1.linear.weight", "b"} <= decay_names and "norm1.linear.bias" not in decay_names


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_mode_never_groups_a_parameter_the_freeze_excluded() -> None:
    """`train_keys` runs first; grouping partitions the survivors, not a set of its own."""
    net = _Toy()
    opt = build_optimizer(net, OptimizerConfig(decay_groups="hf", train_keys=["norm1", "proj"]))
    by_id = _names_by_id(net)
    grouped = {by_id[id(p)] for group in opt.param_groups for p in group["params"]}
    assert grouped == {"norm1.linear.weight", "norm1.linear.bias", "proj.weight", "proj.bias"}
    assert not net.embed.weight.requires_grad and not net.b.requires_grad
