# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Actual NF4 replacement/restoration canary, required before full sensitivity scan."""

from __future__ import annotations

import json

import pytest
import torch

pytestmark = [pytest.mark.GPU, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA/NF4")]


def test_actual_nf4_dense_restoration_forward_and_exception_safety(record_property):
    from cosmos_framework.posttrain.config.model import NativeLoRAConfig, QuantizedBaseTrainingConfig
    from cosmos_framework.posttrain.model.checkpoint import AdapterCheckpointCodec
    from cosmos_framework.posttrain.model.quantization.bnb import NativeLoraLinear4bit
    from cosmos_framework.posttrain.model.quantization.loader import quantized_state_record
    from tools.native_qlora_qualification.layer_sensitivity import restore_block

    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator().manual_seed(91)
    source = torch.randn(128, 128, dtype=torch.bfloat16, generator=generator) / 20
    original = NativeLoraLinear4bit(
        torch.nn.Linear(128, 128, bias=False, device="meta"),
        NativeLoRAConfig(rank=8, alpha=24),
        QuantizedBaseTrainingConfig(),
    )
    original.load_dense_weight_once(source, device)
    original.lora_A.weight = torch.nn.Parameter(torch.randn(8, 128, generator=generator).to(device) / 20)
    original.lora_B.weight = torch.nn.Parameter(torch.randn(128, 8, generator=generator).to(device) / 20)
    net = torch.nn.Sequential(original).eval()
    net._cosmos_adapter_parameters = {f"0.{k}": v for k, v in original.adapter_parameters().items()}
    codec = AdapterCheckpointCodec(net, identity={"test": "actual small NF4 canary"})
    packed = quantized_state_record(original)
    x = torch.randn(2, 7, 128, dtype=torch.bfloat16, generator=generator).to(device)
    with torch.no_grad():
        baseline = net(x)
        expected = torch.nn.functional.linear(x, source.to(device)) + original.adapter_delta(
            x, output_dtype=torch.bfloat16
        )
        for fail in (False, True):
            try:
                with restore_block(net, {"canary": (net, "0", original)}, lambda _: source):
                    assert net[0].lora_A is original.lora_A and net[0].lora_B is original.lora_B
                    torch.testing.assert_close(net(x), expected, rtol=0, atol=0)
                    with pytest.raises(RuntimeError, match="model parameters changed"):
                        codec.metadata()
                    if fail:
                        raise OSError("canary interruption")
            except OSError:
                assert fail
            assert net[0] is original
            codec.assert_live()
            torch.testing.assert_close(net(x), baseline, rtol=0, atol=0)
            assert quantized_state_record(original) == packed
    record_property(
        "layer_sensitivity_cuda_canary",
        json.dumps(
            {
                "real_nf4": True,
                "nonzero_adapter": True,
                "exact_native_dense_forward": True,
                "restored_original_forward": True,
                "exception_restored": True,
                "checkpoint_refused_while_swapped": True,
                "source_shape": [128, 128],
            },
            sort_keys=True,
        ),
    )
