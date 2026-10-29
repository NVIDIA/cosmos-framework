# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real CUDA NF4 + SDPA gates for opt-in AC/compile; native checkpoint qualification is separate."""

from __future__ import annotations

import os

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from cosmos_framework.model.generator.parallelize_vlm import _apply_ac_to_transformer_block
from cosmos_framework.posttrain.config.model import (
    NATIVE_ADAPTER_SAC_SAVE_OPS,
    NativeLoRAConfig,
    QuantizedBaseTrainingConfig,
)
from cosmos_framework.posttrain.utils.native_compile import guard_native_compiled_call

pytestmark = [pytest.mark.GPU, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA kernels")]


@pytest.fixture(autouse=True)
def strict_determinism(monkeypatch):
    from cosmos_framework.posttrain.utils.determinism import enable_determinism

    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    for name in ("FLASH_ATTENTION_DETERMINISTIC", "CUBLAS_WORKSPACE_CONFIG"):
        original = os.environ.get(name)
        monkeypatch.setenv(name, original or "")
        if original is None:
            del os.environ[name]
    try:
        enable_determinism(strict=True)
        with torch._dynamo.config.patch(suppress_errors=False):
            yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark


@pytest.mark.parametrize("mode", ["none", "full", "selective"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "inductor"])
def test_nf4_bf16_adapter_ac_compile_preserves_frozen_state_and_repeated_updates(mode, compiled):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from cosmos_framework.posttrain.model.quantization.bnb import NativeLoraLinear4bit
    from cosmos_framework.posttrain.model.quantization.loader import quantized_state_record

    device = torch.device("cuda", torch.cuda.current_device())
    source_generator = torch.Generator(device="cpu").manual_seed(913)
    weight = torch.randn(128, 128, generator=source_generator, dtype=torch.bfloat16) * 0.02
    inputs = torch.randn(2, 7, 128, generator=source_generator, dtype=torch.bfloat16).to(device)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = NativeLoraLinear4bit(
                nn.Linear(128, 128, bias=False, device="meta"),
                NativeLoRAConfig(),
                QuantizedBaseTrainingConfig(),
                adapter_compute_dtype=torch.bfloat16,
            )
            self.projection.lora_A.weight = nn.Parameter(torch.empty(16, 128, device=device, dtype=torch.float32))
            self.projection.lora_B.weight = nn.Parameter(torch.empty(128, 16, device=device, dtype=torch.float32))
            self.projection.load_dense_weight_once(weight, device)
            self.projection.reset_adapter_parameters(generator=torch.Generator(device=device).manual_seed(17))

        def forward(self, x):
            hidden = self.projection(x)
            heads = hidden.reshape(x.shape[0], x.shape[1], 4, 32).transpose(1, 2)
            attended = F.scaled_dot_product_attention(heads, heads, heads, is_causal=True)
            return attended.transpose(1, 2).reshape_as(hidden)

    def train_once():
        block = Block()
        parameters = [block.projection.lora_A.weight, block.projection.lora_B.weight]
        initial = [p.detach().clone() for p in parameters]
        frozen = quantized_state_record(block.projection)
        forward = block
        if mode != "none":
            forward = _apply_ac_to_transformer_block(
                forward,
                ActivationCheckpointingConfig(mode=mode, save_ops_regex=list(NATIVE_ADAPTER_SAC_SAVE_OPS)),
            )
        if compiled:
            forward = guard_native_compiled_call(torch.compile(forward, dynamic=True, fullgraph=False))
        assert {id(p) for p in forward.parameters() if p.requires_grad} == {id(p) for p in parameters}
        optimizer = torch.optim.AdamW(parameters, lr=0.01, fused=True)
        losses = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                loss = forward(inputs).float().square().mean()
                assert torch.isfinite(loss)
                loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
            optimizer.step()
            losses.append(loss.detach().clone())
        assert inputs.requires_grad is False
        assert all(p.dtype == torch.float32 for p in parameters)
        assert all(not torch.equal(old, p) for old, p in zip(initial, parameters, strict=True))
        assert quantized_state_record(block.projection) == frozen
        assert not block.projection.weight.requires_grad and block.projection.weight not in optimizer.state
        for state in optimizer.state.values():
            assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
        return torch.stack(losses), [p.detach().clone() for p in parameters]

    with torch.random.fork_rng(devices=[device.index]):
        first_loss, first_parameters = train_once()
        second_loss, second_parameters = train_once()
    torch.testing.assert_close(first_loss, second_loss, rtol=0, atol=0)
    for first, second in zip(first_parameters, second_parameters, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
