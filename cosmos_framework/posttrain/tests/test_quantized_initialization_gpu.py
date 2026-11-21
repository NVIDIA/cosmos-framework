# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Separate actual-NF4 residual canary; no task-quality or model parity claim."""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from cosmos_framework.posttrain.config.model import NativeLoRAConfig, QuantizedBaseTrainingConfig

pytestmark = [pytest.mark.GPU, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA kernels")]


def test_actual_nf4_residual_scaling_forward_and_trainability(record_property, monkeypatch):
    from cosmos_framework.posttrain.model.quantization.bnb import NativeLoraLinear4bit
    from cosmos_framework.posttrain.model.quantization.initialization import residual_factors
    from cosmos_framework.posttrain.model.quantization.loader import quantized_state_record
    from cosmos_framework.posttrain.utils.determinism import enable_determinism

    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    for name in ("FLASH_ATTENTION_DETERMINISTIC", "CUBLAS_WORKSPACE_CONFIG"):
        previous = os.environ.get(name)
        monkeypatch.setenv(name, previous if previous is not None else "")
        if previous is None:
            del os.environ[name]
    try:
        enable_determinism(strict=True)
        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator().manual_seed(734)
        source = torch.randn(128, 128, generator=generator, dtype=torch.bfloat16) * 0.02
        adapter = NativeLoRAConfig(rank=8, alpha=24)
        module = NativeLoraLinear4bit(
            torch.nn.Linear(128, 128, bias=False, device="meta"), adapter, QuantizedBaseTrainingConfig()
        )
        module.load_dense_weight_once(source, device)
        packed = quantized_state_record(module)
        effective = module.effective_weight_for_initialization()
        cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
        a, b, report = residual_factors(source, effective, rank=8, alpha=24, seed=42, target="canary")
        a_repeat, b_repeat, repeated = residual_factors(source, effective, rank=8, alpha=24, seed=42, target="canary")
        assert torch.equal(a, a_repeat) and torch.equal(b, b_repeat) and report == repeated
        assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
        assert report["after_squared_frobenius"] < report["before_squared_frobenius"]
        residual = source.to(device).float() - effective.float()
        torch.testing.assert_close(
            (residual - (adapter.alpha / adapter.rank) * (b @ a)).square().sum(),
            torch.tensor(report["after_squared_frobenius"], device=device),
            rtol=2e-6,
            atol=1e-8,
        )
        module.lora_A.weight = torch.nn.Parameter(a)
        module.lora_B.weight = torch.nn.Parameter(b)
        x = torch.randn(2, 7, 128, generator=generator, dtype=torch.bfloat16).to(device).requires_grad_(True)
        # Match the real native arithmetic: BF16 backend base result, FP32
        # adapter GEMMs, cast the delta to BF16, then scale and add. This is
        # intentionally not a dense-model or zero-LoRA parity assertion.
        base = super(NativeLoraLinear4bit, module).forward(x)
        explicit = base + (adapter.alpha / adapter.rank) * F.linear(F.linear(x.float(), a), b).to(base.dtype)
        actual = module(x)
        torch.testing.assert_close(actual, explicit, rtol=0, atol=0)
        assert not torch.equal(actual, base)
        actual.float().square().mean().backward()
        for gradient in (module.lora_A.weight.grad, module.lora_B.weight.grad, x.grad):
            assert gradient is not None and torch.isfinite(gradient).all() and torch.count_nonzero(gradient)
        assert module.weight.grad is None and not module.weight.requires_grad
        optimizer = torch.optim.AdamW(module.adapter_parameters().values(), lr=1e-3)
        optimizer.step()
        assert quantized_state_record(module) == packed
        assert not torch.equal(module.lora_A.weight, a_repeat) and not torch.equal(module.lora_B.weight, b_repeat)
        import json

        record_property("nf4_residual_reconstruction", json.dumps(report, sort_keys=True))
        record_property("scope", "actual NF4 fixed-base matrix and native delta canary; no model/task-quality parity")
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
