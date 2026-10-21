# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L0 tests for the GR00T-N1.7 training adapter on a toy stand-in: the `action_head.*` wire scope, the mode
propagation of `flow_matching_loss`, and an Isaac-free module import. The numbers are the parity gate's job."""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch
import torch.nn as nn

from cosmos_framework.posttrain.model.gr00t.policy import (
    BF16_PRECISION,
    OFFICIAL_PRECISION,
    Gr00tHeads,
    Gr00tN1d7TrainPolicy,
)
from cosmos_framework.posttrain.model.precision import compute_context
from cosmos_framework.posttrain.model.protocol import SupervisedFlowMatchingPolicy


class _ToyGr00t(nn.Module):
    """A `Gr00tN1d7` stand-in: frozen backbone, trainable action head, dict-returning forward."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4).requires_grad_(False)
        self.action_head = nn.Linear(4, 4)
        self.forward_modes: list[bool] = []
        self.autocast_seen: list[bool] = []
        self.input_dtypes: list[torch.dtype] = []

    def forward(self, inputs: dict) -> dict:
        self.forward_modes.append(self.training)
        self.autocast_seen.append(torch.is_autocast_enabled("cpu"))
        self.input_dtypes.append(inputs["x"].dtype)
        loss = self.action_head(inputs["x"]).sum()
        return {"loss": loss, "action_loss": loss.detach(), "action_mask": torch.ones_like(inputs["x"])}


def _policy() -> Gr00tN1d7TrainPolicy:
    """The adapter around a toy model, bypassing the Isaac factory in `__init__`."""
    policy = object.__new__(Gr00tN1d7TrainPolicy)
    policy._ppo_enabled = False
    policy.gr00t = _ToyGr00t()
    policy.backbone = policy.gr00t.backbone
    policy.net = Gr00tHeads(policy.gr00t.action_head)
    policy.precision = OFFICIAL_PRECISION
    policy.compute_ctx = compute_context(OFFICIAL_PRECISION.mode, getattr(torch, OFFICIAL_PRECISION.param), "cpu")
    return policy


@pytest.mark.L0
@pytest.mark.CPU
def test_heads_expose_exactly_the_action_head_under_the_wire_prefix() -> None:
    from cosmos_framework.posttrain.model.gr00t.serving.body import ACTION_HEAD_PREFIX

    model = _ToyGr00t()
    net = Gr00tHeads(model.action_head)
    assert net.action_head is model.action_head
    keys = list(net.state_dict())
    assert keys and all(k.startswith(ACTION_HEAD_PREFIX) for k in keys), keys


@pytest.mark.L0
@pytest.mark.CPU
def test_flow_matching_loss_propagates_the_mode_and_only_the_head_gets_gradients() -> None:
    policy = _policy()
    assert isinstance(policy, SupervisedFlowMatchingPolicy)
    inputs = {"x": torch.ones(2, 4)}

    policy.net.eval()
    policy.flow_matching_loss(inputs)
    policy.net.train()
    out = policy.flow_matching_loss(inputs)
    assert policy.gr00t.forward_modes == [False, True]

    out.loss.backward()
    assert all(p.grad is not None for p in policy.net.parameters())
    assert all(p.grad is None for p in policy.backbone.parameters())


@pytest.mark.L0
@pytest.mark.CPU
def test_module_import_does_not_pull_in_isaac_gr00t_or_vllm() -> None:
    code = (
        "import sys;"
        "import cosmos_framework.posttrain.model.gr00t;"
        "leaked = sorted(k for k in sys.modules if k.split('.')[0] in ('gr00t', 'vllm', 'vllm_omni'));"
        "assert not leaked, leaked"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.L0
@pytest.mark.CPU
def test_shard_plan_delivers_no_compute_dtype_under_autocast() -> None:
    """Under the official (autocast) preset the FSDP wrap must be dtype-transparent, so the plan carries
    `param_dtype=None`; under the all-bf16 `fsdp` preset it carries the compute dtype. The block type is Isaac's, so
    this runs where Isaac-GR00T is installed (the gr00t venv) and is not in the CI manifest."""
    pytest.importorskip("gr00t.model.modules.dit", reason="Isaac-GR00T is only installed in the gr00t venv")
    from cosmos_framework.posttrain.model.gr00t.policy import gr00t_shard_plan

    plan = gr00t_shard_plan(OFFICIAL_PRECISION)
    assert plan.param_dtype is None and plan.reduce_dtype is torch.float32, plan

    fsdp = BF16_PRECISION
    plan = gr00t_shard_plan(fsdp)
    assert plan.param_dtype is torch.bfloat16 and plan.reduce_dtype is torch.bfloat16, plan


@pytest.mark.L0
@pytest.mark.CPU
def test_precision_mode_selects_the_forward_context() -> None:
    """The official recipe (autocast) runs an fp32 model under autocast; mode "fsdp" runs a bf16 model with no
    autocast and casts the floating inputs to the parameter dtype itself (what FSDP2's `cast_forward_inputs`
    would do for a wrapped head). Observable on CPU because the device type follows the model."""
    policy = _policy()
    inputs = {"x": torch.ones(2, 4)}
    out = policy.flow_matching_loss(inputs)
    assert policy.gr00t.autocast_seen == [True] and policy.gr00t.input_dtypes == [torch.float32]
    assert out.loss.dtype is torch.bfloat16

    fsdp = BF16_PRECISION
    policy.precision = fsdp
    policy.compute_ctx = compute_context(fsdp.mode, getattr(torch, fsdp.param), "cpu")
    policy.gr00t.to(torch.bfloat16)  # what `build_official_gr00t_model` does for the bf16 preset
    out = policy.flow_matching_loss(inputs)
    assert policy.gr00t.autocast_seen == [True, False] and policy.gr00t.input_dtypes[-1] is torch.bfloat16
    assert out.loss.dtype is torch.bfloat16


@pytest.mark.L0
@pytest.mark.CPU
def test_bf16_preset_is_the_all_bf16_fsdp_recipe_with_no_master_to_wire_gap() -> None:
    """One owner for the all-bf16 preset: the parity CLIs and the weight-sync driver import it from here."""
    from cosmos_framework.posttrain.config.build import Precision

    assert BF16_PRECISION == Precision(
        master="bfloat16", param="bfloat16", reduce="bfloat16", transfer="bfloat16", mode="fsdp"
    )
    assert BF16_PRECISION.transfer == BF16_PRECISION.master  # identity cast on the wire: zero gap by construction
    assert OFFICIAL_PRECISION.transfer == "bfloat16" != OFFICIAL_PRECISION.master  # the official recipe rounds
