# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real NF4 task-gradient and auxiliary-artifact checks, not family quality evidence."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.model.quantization.family import FamilyBackboneConfig, install_family_backbone

pytestmark = [pytest.mark.GPU, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA kernels")]


class TinyActionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(128, 128, bias=False, dtype=torch.bfloat16)
        self.head = nn.Linear(128, 7, dtype=torch.float32)

    def forward(self, observation):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.head(self.backbone(observation).tanh())


@pytest.fixture
def owned_device(tmp_path):
    from cosmos_framework.posttrain.utils.determinism import enable_determinism

    enable_determinism(strict=True)
    created = not torch.distributed.is_initialized()
    if created:
        torch.distributed.init_process_group(
            "nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1
        )
    try:
        yield torch.device("cuda", torch.cuda.current_device())
    finally:
        if created:
            torch.distributed.destroy_process_group()


@pytest.mark.parametrize("mode", ["nf4_lora", "bf16_lora", "nf4_frozen", "bf16_frozen"])
def test_real_task_gradient_auxiliary_update_and_frozen_reconstruction(mode, owned_device):
    with torch.random.fork_rng():
        torch.manual_seed(19)
        net = TinyActionPolicy()
    source = copy.deepcopy(net)
    args = {
        "config": FamilyBackboneConfig(mode=mode, rank=8, alpha=16, seed=17),
        "targets": {"backbone": (128, 128)},
        "auxiliary_parameters": ("head.weight", "head.bias"),
        "source_identity": {"family": "kernel-test-only", "seed": 19},
        "task_identity": {"loss": "test-action-mse", "dimensions": 7},
        "device": owned_device,
    }
    state = install_family_backbone(net, **args)
    optimizer = torch.optim.AdamW(state.codec.parameters.values(), lr=1e-3, weight_decay=0, fused=True)
    state.assert_optimizer(optimizer)
    before = {name: parameter.detach().clone() for name, parameter in state.codec.parameters.items()}
    generator = torch.Generator(device=owned_device).manual_seed(42)
    observation = torch.randn(2, 5, 128, device=owned_device, dtype=torch.bfloat16, generator=generator)
    observation.requires_grad_(True)
    target = torch.randn(2, 5, 7, device=owned_device, dtype=torch.bfloat16, generator=generator)
    assert net.backbone.weight.dtype == (torch.uint8 if mode.startswith("nf4") else torch.bfloat16)
    assert bool(any("lora_" in name for name in state.codec.parameters)) == mode.endswith("lora")
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = (net(observation).float() - target.float()).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert observation.grad is not None and torch.isfinite(observation.grad).all()
        assert torch.count_nonzero(observation.grad)
        for name, parameter in state.codec.parameters.items():
            assert parameter.dtype == torch.float32
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            if step > 0 or ".lora_A." not in name:
                assert torch.count_nonzero(parameter.grad), name
        optimizer.step()
        state.validate_frozen_base()
    state.assert_optimizer(optimizer)
    assert set(optimizer.state) == set(state.codec.parameters.values())
    for name, parameter in state.codec.parameters.items():
        assert not torch.equal(parameter, before[name]), name
        assert optimizer.state[parameter]["exp_avg"].dtype == torch.float32
        assert optimizer.state[parameter]["exp_avg_sq"].dtype == torch.float32
    rebuilt = install_family_backbone(source, **args)
    rebuilt.codec.validate_metadata(state.codec.metadata())
    rebuilt.codec.view.load_state_dict(state.codec.view.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(source(observation), net(observation), rtol=0, atol=0)
    rebuilt.validate_frozen_base()
    net.to(device=owned_device)
    with pytest.raises(ValueError, match="dtype|conversion"):
        net.bfloat16()
