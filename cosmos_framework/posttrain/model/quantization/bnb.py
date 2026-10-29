# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""The explicitly qualified bitsandbytes representation, without PEFT preparation."""

from __future__ import annotations

from importlib.metadata import version
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos_framework.utils.generator.lora import NativeLoRAMixin

if version("bitsandbytes") != "0.49.2":
    raise RuntimeError("native NF4 training requires bitsandbytes==0.49.2; no backend fallback is permitted")

import bitsandbytes as bnb


class NativeLoraLinear4bit(NativeLoRAMixin, bnb.nn.Linear4bit):
    """Meta-only replacement preserving the native base and adapter key layout.

    The dense superclass constructor must never run on an accelerator. A source
    tensor is quantized once, and its parameter and complete QuantState are
    attached together. No model-wide conversion is involved.
    """

    def __init__(self, base: nn.Linear, adapter: Any, quantization: Any) -> None:
        if not base.weight.is_meta or base.bias is not None:
            raise ValueError("native NF4 replacement requires an unmaterialized, bias-free linear")
        super().__init__(
            base.in_features,
            base.out_features,
            bias=False,
            device="meta",
            compute_dtype=torch.bfloat16,
            compress_statistics=True,
            quant_type="nf4",
            quant_storage=torch.uint8,
        )
        self.weight = bnb.nn.Params4bit(
            base.weight.detach(),
            requires_grad=False,
            blocksize=quantization.block_size,
            compress_statistics=True,
            quant_type="nf4",
            quant_storage=torch.uint8,
            module=self,
        )
        self.lora_A = nn.Linear(base.in_features, adapter.rank, bias=False, device="meta", dtype=torch.float32)
        self.lora_B = nn.Linear(adapter.rank, base.out_features, bias=False, device="meta", dtype=torch.float32)
        self.lora_dropout = nn.Dropout(adapter.dropout) if adapter.dropout else nn.Identity()
        self._lora_rank = adapter.rank
        self._lora_alpha = adapter.alpha
        self._lora_use_rslora = False
        self.logical_shape = (base.out_features, base.in_features)

    @torch.no_grad()
    def load_dense_weight_once(self, source: torch.Tensor, device: torch.device) -> None:
        if not self.weight.is_meta or self.weight.bnb_quantized or self.quant_state is not None:
            raise RuntimeError("attempted to load/quantize an already materialized NF4 weight")
        if source.device.type != "cpu" or source.dtype != torch.bfloat16 or tuple(source.shape) != self.logical_shape:
            raise ValueError("NF4 source must be the exact logical BF16 CPU weight")
        if device.type != "cuda":
            raise ValueError("native NF4 training only supports the owned CUDA device")
        self.weight = bnb.nn.Params4bit(
            source,
            requires_grad=False,
            blocksize=64,
            compress_statistics=True,
            quant_type="nf4",
            quant_storage=torch.uint8,
            module=self,
        ).to(device=device)
        self.validate_quantization_state()

    def validate_quantization_state(self) -> None:
        weight, state = self.weight, self.weight.quant_state
        if (
            not isinstance(weight, bnb.nn.Params4bit)
            or not weight.bnb_quantized
            or weight.requires_grad
            or weight.dtype != torch.uint8
            or weight.device.type != "cuda"
            or state is None
            or self.quant_state is not state
            or state.quant_type != "nf4"
            or state.blocksize != 64
            or state.dtype != torch.bfloat16
            or tuple(state.shape) != self.logical_shape
            or not state.nested
            or state.state2 is None
            or state.offset is None
            or self.compute_dtype != torch.bfloat16
        ):
            raise RuntimeError("incomplete or incompatible native NF4 weight/QuantState")
        for value in (state.absmax, state.code, state.offset, state.state2.absmax, state.state2.code):
            if not isinstance(value, torch.Tensor) or value.is_meta or value.device != weight.device:
                raise RuntimeError("all NF4 and nested scale metadata must reside on the owned device")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16 or x.device != self.weight.device:
            raise ValueError("native NF4 expects BF16 activations on its owned CUDA device")
        base = super().forward(x)
        return base + self.adapter_delta(x, output_dtype=base.dtype)


def kernel_preflight(device: torch.device, adapter: Any, quantization: Any) -> dict[str, Any]:
    """Exercise real NF4 forward/input backward and several FP32 adapter updates.

    Fork RNG and leave backend/determinism settings unchanged. This is a kernel
    canary, not full-model numerical, task-quality, or memory qualification.
    """
    from cosmos_framework.posttrain.model.quantization.loader import quantized_state_record, tensor_digest

    with torch.random.fork_rng(devices=[device.index]):
        generator = torch.Generator(device="cpu").manual_seed(9817)
        weight = torch.randn(128, 128, generator=generator, dtype=torch.bfloat16) * 0.02
        module = NativeLoraLinear4bit(nn.Linear(128, 128, bias=False, device="meta"), adapter, quantization)
        module.lora_A.weight = nn.Parameter(torch.empty(adapter.rank, 128, dtype=torch.float32, device=device))
        module.lora_B.weight = nn.Parameter(torch.empty(128, adapter.rank, dtype=torch.float32, device=device))
        module.load_dense_weight_once(weight, device)
        module.reset_adapter_parameters(generator=torch.Generator(device=device).manual_seed(9817))
        before = quantized_state_record(module)
        x = torch.randn(2, 7, 128, generator=generator, dtype=torch.bfloat16).to(device).requires_grad_(True)
        control_x = x.detach().clone().requires_grad_(True)
        effective = bnb.functional.dequantize_4bit(module.weight.data, module.weight.quant_state).detach()
        actual, expected = module(x), F.linear(control_x, effective)
        if not torch.allclose(actual, expected, rtol=0.02, atol=0.02):
            raise RuntimeError("NF4 canary forward differs from its dequantized effective base")
        actual_grad = torch.autograd.grad(actual.float().square().mean(), x)[0]
        expected_grad = torch.autograd.grad(expected.float().square().mean(), control_x)[0]
        if not torch.allclose(actual_grad, expected_grad, rtol=0.03, atol=0.0002):
            raise RuntimeError("NF4 canary input backward differs from its dequantized effective base")
        initial_a = tensor_digest(module.lora_A.weight)
        initial_b = tensor_digest(module.lora_B.weight)
        optimizer = torch.optim.AdamW(module.adapter_parameters().values(), lr=1e-3, fused=True)
        saw_a_gradient = False
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = module(x.detach()).float().square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite NF4 canary loss")
            loss.backward()
            if any(
                parameter.grad is None or not torch.isfinite(parameter.grad).all()
                for parameter in module.adapter_parameters().values()
            ):
                raise RuntimeError("NF4 canary did not propagate finite adapter gradients")
            saw_a_gradient |= bool(torch.count_nonzero(module.lora_A.weight.grad).item())
            optimizer.step()
        if (
            not saw_a_gradient
            or initial_a == tensor_digest(module.lora_A.weight)
            or initial_b == tensor_digest(module.lora_B.weight)
        ):
            raise RuntimeError("NF4 canary did not update both adapter factors over multiple steps")
        if before != quantized_state_record(module) or module.weight.grad is not None:
            raise RuntimeError("NF4 canary changed the frozen base or its quantization metadata")
        # Params4bit.to must retain packed storage rather than interpreting it as dense input.
        moved = module.weight.to(device=device)
        if (
            moved.data_ptr() != module.weight.data_ptr()
            or not moved.bnb_quantized
            or moved.quant_state is not module.weight.quant_state
        ):
            raise RuntimeError("same-device Params4bit placement changed quantized storage/state")
        torch.cuda.synchronize(device)
        return {
            "backend": "bitsandbytes",
            "version": bnb.__version__,
            "steps": 3,
            "forward_max_abs": float((actual - expected).abs().max().item()),
            "input_gradient_max_abs": float((actual_grad - expected_grad).abs().max().item()),
        }
