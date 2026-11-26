# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Mixed BF16/NF4 matrix smoke test on a real CUDA device."""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from cosmos_framework.posttrain.tests.qlora.helpers import make_lifecycle_fixture

pytestmark = [
    pytest.mark.L0,
    pytest.mark.GPU,
    pytest.mark.gpus(1),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA kernels"),
]


def test_actual_mixed_nf4_dense_coverage_forward_optimizer_and_rebuild(tmp_path, monkeypatch):
    from cosmos_framework.posttrain.model.quantization.loader import quantized_state_record
    from cosmos_framework.posttrain.utils.determinism import enable_determinism

    enabled, warn_only = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    old_deterministic, old_benchmark = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    for name in ("FLASH_ATTENTION_DETERMINISTIC", "CUBLAS_WORKSPACE_CONFIG"):
        previous = os.environ.get(name)
        monkeypatch.setenv(name, previous if previous is not None else "")
        if previous is None:
            del os.environ[name]
    try:
        enable_determinism(strict=True)
        net, life, source, _ = make_lifecycle_fixture(tmp_path, monkeypatch, (6, 15), real_cuda=True)
        assert len(life.quantized_targets) == 238 and len(net._cosmos_adapter_parameters) == 504
        dense_name = "model.language_model.layers.6.self_attn.q_proj"
        packed_name = "model.language_model.layers.0.self_attn.q_proj"
        dense, packed = life.targets[dense_name], life.targets[packed_name]
        assert dense.weight.dtype == torch.bfloat16 and packed.weight.dtype == torch.uint8
        assert torch.equal(dense.weight.cpu(), source[dense_name + ".weight"])
        fingerprint = quantized_state_record(packed)
        generator = torch.Generator().manual_seed(93)
        x = torch.randn(2, 4, 128, generator=generator, dtype=torch.bfloat16).to(life.device).requires_grad_(True)
        assert torch.equal(dense(x), F.linear(x, dense.weight))
        exercised = {
            f"{kind}.{name}": parameter
            for kind, module in (("dense", dense), ("packed", packed))
            for name, parameter in module.adapter_parameters().items()
        }
        initial = {name: parameter.detach().clone() for name, parameter in exercised.items()}
        # Disable weight decay so an A change proves a gradient-driven update,
        # while allowing the legitimate zero A gradient on the first zero-B step.
        optimizer = torch.optim.AdamW(net._cosmos_adapter_parameters.values(), lr=1e-3, weight_decay=0)
        optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        adapter_ids = {id(parameter) for parameter in net._cosmos_adapter_parameters.values()}
        assert len(optimizer_parameters) == len(adapter_ids) == 504
        assert {id(parameter) for parameter in optimizer_parameters} == adapter_ids
        assert all(id(module.weight) not in adapter_ids for module in life.targets.values())
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = dense(packed(x)).float().square().mean()
            loss.backward()
            for module in (dense, packed):
                for parameter in module.adapter_parameters().values():
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                assert module.weight.grad is None and not module.weight.requires_grad
            assert x.grad is not None and torch.isfinite(x.grad).all()
            optimizer.step()
        assert set(optimizer.state) == set(exercised.values())
        for name, parameter in exercised.items():
            assert not torch.equal(parameter, initial[name]), name
            state = optimizer.state[parameter]
            assert state["step"].item() == 3
            for key in ("exp_avg", "exp_avg_sq"):
                assert state[key].dtype == torch.float32
                assert torch.isfinite(state[key]).all() and torch.count_nonzero(state[key]), (name, key)
        net._cosmos_adapter_checkpoint_codec.assert_live(optimizer=optimizer)
        net._cosmos_validate_frozen_base()
        assert quantized_state_record(packed) == fingerprint
        assert torch.equal(dense.weight.cpu(), source[dense_name + ".weight"])
        restored, rebuilt, _, _ = make_lifecycle_fixture(tmp_path, monkeypatch, (6, 15), real_cuda=True)
        restored._cosmos_adapter_checkpoint_codec.validate_metadata(net._cosmos_adapter_checkpoint_codec.metadata())
        restored._cosmos_adapter_checkpoint_codec.view.load_state_dict(
            net._cosmos_adapter_checkpoint_codec.view.state_dict()
        )
        with torch.no_grad():
            expected = dense(packed(x))
            actual = rebuilt.targets[dense_name](rebuilt.targets[packed_name](x))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        restored._cosmos_validate_frozen_base()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.deterministic = old_deterministic
        torch.backends.cudnn.benchmark = old_benchmark
