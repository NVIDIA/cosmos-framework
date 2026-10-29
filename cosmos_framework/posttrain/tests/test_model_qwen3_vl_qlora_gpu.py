# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real accelerator gates; full native WTS training remains a separate qualification."""

from __future__ import annotations

import os

import pytest
import torch

from cosmos_framework.posttrain.config.model import NativeLoRAConfig, QuantizedBaseTrainingConfig

pytestmark = [pytest.mark.GPU, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA kernels")]


@pytest.fixture(autouse=True)
def restore_determinism_state(monkeypatch):
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    for name in ("FLASH_ATTENTION_DETERMINISTIC", "CUBLAS_WORKSPACE_CONFIG"):
        previous = os.environ.get(name)
        # Register the original environment with monkeypatch while preserving
        # absence for enable_determinism's existing late-CUDA-init guard.
        monkeypatch.setenv(name, previous if previous is not None else "")
        if previous is None:
            del os.environ[name]
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark


@pytest.mark.parametrize("adapter_compute_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compile_model", [False, True], ids=["eager", "compiled"])
def test_nf4_kernel_training_canary_preserves_rng_and_global_flags(adapter_compute_dtype, compile_model):
    # Import failure/version mismatch must fail a CUDA qualification, never skip
    # to a different backend or a mocked/dense calculation.
    from cosmos_framework.posttrain.utils.determinism import enable_determinism

    enable_determinism(strict=True)
    from cosmos_framework.posttrain.model.quantization.bnb import kernel_preflight

    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.init()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    precision = torch.get_float32_matmul_precision()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    assert deterministic and not warn_only
    result = kernel_preflight(
        device,
        NativeLoRAConfig(),
        QuantizedBaseTrainingConfig(),
        adapter_compute_dtype=adapter_compute_dtype,
        compile_model=compile_model,
    )
    assert result["backend"] == "bitsandbytes" and result["version"] == "0.49.2"
    assert result["steps"] == 3
    assert result["compiled"] is compile_model
    assert result["adapter_compute_dtype"] == str(adapter_compute_dtype).removeprefix("torch.")
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    assert precision == torch.get_float32_matmul_precision()
    assert deterministic == torch.are_deterministic_algorithms_enabled()
    assert warn_only == torch.is_deterministic_algorithms_warn_only_enabled()


@pytest.mark.parametrize(
    ("heads", "length", "head_dim", "causal"),
    [(32, 349, 128, True), (16, 240, 72, False)],
    ids=["language", "vision_chunk"],
)
def test_strict_flash_sdpa_repeats_bf16_forward_and_backward_exactly(heads, length, head_dim, causal):
    """Exercise Flash itself: warn-only mode can retain its nondeterministic dQ accumulation."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from cosmos_framework.posttrain.utils.determinism import enable_determinism

    enable_determinism(strict=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cuda.is_flash_attention_available()
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(2026)
    shape = (1, heads, length, head_dim)
    source = [torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator) for _ in range(3)]
    upstream = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    reference = None
    # Pin only this kernel test. The native trainer retains its ordinary SDPA
    # dispatch, whose full-model save/resume comparison is qualified separately.
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for _ in range(3):
            query, key, value = [tensor.detach().clone().requires_grad_(True) for tensor in source]
            output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=causal
            )
            gradients = torch.autograd.grad(output, (query, key, value), upstream)
            observed = (output.detach(), *gradients)
            assert all(torch.isfinite(tensor).all() for tensor in observed)
            assert all(torch.count_nonzero(tensor) > 0 for tensor in gradients)
            if reference is None:
                reference = tuple(tensor.clone() for tensor in observed)
            else:
                for actual, expected in zip(observed, reference, strict=True):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
