# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real accelerator gates; full native WTS training remains a separate qualification."""

from __future__ import annotations

import pytest
import torch

from cosmos_framework.posttrain.config.model import NativeLoRAConfig, QuantizedBaseTrainingConfig

pytestmark = [pytest.mark.GPU, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA kernels")]


def test_nf4_kernel_training_canary_preserves_rng_and_global_flags():
    # Import failure/version mismatch must fail a CUDA qualification, never skip
    # to a different backend or a mocked/dense calculation.
    from cosmos_framework.posttrain.utils.determinism import enable_determinism

    enable_determinism()
    from cosmos_framework.posttrain.model.quantization.bnb import kernel_preflight

    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.init()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    precision = torch.get_float32_matmul_precision()
    deterministic = torch.are_deterministic_algorithms_enabled()
    result = kernel_preflight(device, NativeLoRAConfig(), QuantizedBaseTrainingConfig())
    assert result["backend"] == "bitsandbytes" and result["version"] == "0.49.2"
    assert result["steps"] == 3
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    assert precision == torch.get_float32_matmul_precision()
    assert deterministic == torch.are_deterministic_algorithms_enabled()
