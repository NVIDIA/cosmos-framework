# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""The explicitly qualified bitsandbytes representation, without PEFT preparation."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from importlib.metadata import version
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos_framework.utils.generator.lora import NativeLoRAMixin

if version("bitsandbytes") != "0.49.2":
    raise RuntimeError("native NF4 training requires bitsandbytes==0.49.2; no backend fallback is permitted")

import bitsandbytes as bnb

_CHECKPOINTED_DEQUANT_OWNERS: ContextVar[frozenset[int]] = ContextVar(
    "native_checkpointed_dequant_owners", default=frozenset()
)


@contextmanager
def _checkpointed_dequant_scope(owners: frozenset[int]) -> Iterator[None]:
    if _CHECKPOINTED_DEQUANT_OWNERS.get():
        raise RuntimeError("nested checkpointed dequantization block scopes are unsupported")
    token = _CHECKPOINTED_DEQUANT_OWNERS.set(owners)
    try:
        yield
    finally:
        _CHECKPOINTED_DEQUANT_OWNERS.reset(token)


def checkpointed_dequant_contexts(modules: tuple[nn.Module, ...]):
    """Contexts for one verified full nonreentrant language-block checkpoint.

    The wrapper must supply this as its ``context_fn``, so both original forward
    and recomputation enter the same narrowly owned scope. No dense weight is
    cached on a module. A backward exception remains fatal to the training
    process; this interface does not implement an in-process retry protocol.
    """
    if not modules or any(
        not isinstance(module, NativeLoraLinear4bit) or module.quantized_linear_execution != "checkpointed_dequant"
        for module in modules
    ):
        raise ValueError("checkpointed dequantization scope requires its exact candidate linears")
    owners = frozenset(id(module) for module in modules)
    if len(owners) != len(modules):
        raise ValueError("checkpointed dequantization scope contains duplicate linears")
    return _checkpointed_dequant_scope(owners), _checkpointed_dequant_scope(owners)


class NativeLoraLinear4bit(NativeLoRAMixin, bnb.nn.Linear4bit):
    """Meta-only replacement preserving the native base and adapter key layout.

    The dense superclass constructor must never run on an accelerator. A source
    tensor is quantized once, and its parameter and complete QuantState are
    attached together. No model-wide conversion is involved.
    """

    def __init__(
        self,
        base: nn.Linear,
        adapter: Any,
        quantization: Any,
        *,
        adapter_compute_dtype: torch.dtype = torch.float32,
        quantized_linear_execution: str = "bnb",
    ) -> None:
        if type(quantized_linear_execution) is not str or quantized_linear_execution not in {
            "bnb",
            "checkpointed_dequant",
        }:
            raise ValueError("unsupported native quantized linear execution")
        if quantized_linear_execution == "checkpointed_dequant" and adapter_compute_dtype != torch.float32:
            raise ValueError("checkpointed dequantization requires FP32 adapters")
        if adapter_compute_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("native adapter compute dtype must be float32 or bfloat16")
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
        self._adapter_compute_dtype = adapter_compute_dtype
        self.quantized_linear_execution = quantized_linear_execution
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

    @torch.no_grad()
    def effective_weight_for_initialization(self) -> torch.Tensor:
        """Dequantize only this actual attached matrix for explicit preparation.

        The returned BF16 tensor is temporary. Neither packed storage nor nested
        QuantState is replaced, requantized, or inferred from another backend.
        Model construction and checkpoint restore never call this method.
        """
        self.validate_quantization_state()
        effective = bnb.functional.dequantize_4bit(self.weight.data, self.weight.quant_state)
        if (
            effective.dtype != torch.bfloat16
            or effective.device != self.weight.device
            or tuple(effective.shape) != self.logical_shape
        ):
            raise RuntimeError("actual NF4 dequantization differs from its logical BF16 matrix")
        return effective

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16 or x.device != self.weight.device:
            raise ValueError("native NF4 expects BF16 activations on its owned CUDA device")
        if self.quantized_linear_execution == "checkpointed_dequant" and self.training and torch.is_grad_enabled():
            if id(self) not in _CHECKPOINTED_DEQUANT_OWNERS.get():
                raise RuntimeError("checkpointed dequantization requires its full nonreentrant block scope")
            # Detach only dequantization. F.linear participates in autograd so
            # checkpoint saved-tensor hooks discard/recompute its actual BF16
            # weight. Retaining it on this module would defeat block AC.
            with torch.no_grad():
                weight = bnb.functional.dequantize_4bit(self.weight.data, self.weight.quant_state)
            if (
                weight.dtype != torch.bfloat16
                or weight.device != x.device
                or tuple(weight.shape) != self.logical_shape
                or weight.requires_grad
            ):
                raise RuntimeError("checkpointed dequantization changed the logical frozen BF16 weight")
            base = F.linear(x, weight)
        else:
            # Keep upstream no-grad and single-vector inference behavior.
            base = super().forward(x)
        return base + self.adapter_delta(x, output_dtype=base.dtype)


class NativeFrozenLinear4bit(bnb.nn.Linear4bit):
    """The same frozen NF4 representation without unused adapter parameters.

    Used by the explicit feature-extractor control in native action policies.
    Input backward remains enabled so an earlier trainable operation is never
    implicitly detached merely because this particular matrix is frozen.
    """

    load_dense_weight_once = NativeLoraLinear4bit.load_dense_weight_once
    validate_quantization_state = NativeLoraLinear4bit.validate_quantization_state
    effective_weight_for_initialization = NativeLoraLinear4bit.effective_weight_for_initialization

    def __init__(self, base: nn.Linear) -> None:
        if not base.weight.is_meta or base.bias is not None:
            raise ValueError("native frozen NF4 requires an unmaterialized, bias-free linear")
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
        self.logical_shape = (base.out_features, base.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16 or x.device != self.weight.device:
            raise ValueError("native frozen NF4 expects BF16 activations on its owned CUDA device")
        return super().forward(x)


def kernel_preflight(
    device: torch.device,
    adapter: Any,
    quantization: Any,
    *,
    adapter_compute_dtype: torch.dtype = torch.float32,
    compile_model: bool = False,
    quantized_linear_execution: str = "bnb",
) -> dict[str, Any]:
    """Exercise real NF4 forward/input backward and several FP32 adapter updates.

    Fork RNG and leave backend/determinism settings unchanged. This is a kernel
    canary, not full-model numerical, task-quality, or memory qualification.
    """
    from cosmos_framework.posttrain.model.quantization.loader import quantized_state_record, tensor_digest

    if quantized_linear_execution == "checkpointed_dequant" and compile_model:
        raise ValueError("checkpointed dequantization requires eager full checkpointing")
    if compile_model and torch._dynamo.config.suppress_errors:
        raise ValueError("native NF4 compile qualification forbids suppress_errors and silent eager fallback")
    with torch.random.fork_rng(devices=[device.index]):
        generator = torch.Generator(device="cpu").manual_seed(9817)
        weight = torch.randn(128, 128, generator=generator, dtype=torch.bfloat16) * 0.02
        module = NativeLoraLinear4bit(
            nn.Linear(128, 128, bias=False, device="meta"),
            adapter,
            quantization,
            adapter_compute_dtype=adapter_compute_dtype,
            quantized_linear_execution=quantized_linear_execution,
        )
        module.lora_A.weight = nn.Parameter(torch.empty(adapter.rank, 128, dtype=torch.float32, device=device))
        module.lora_B.weight = nn.Parameter(torch.empty(128, adapter.rank, dtype=torch.float32, device=device))
        module.load_dense_weight_once(weight, device)
        module.reset_adapter_parameters(generator=torch.Generator(device=device).manual_seed(9817))
        forward = torch.compile(module, dynamic=True, fullgraph=False) if compile_model else module
        if quantized_linear_execution == "checkpointed_dequant":
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

            forward = checkpoint_wrapper(
                module, preserve_rng_state=True, context_fn=partial(checkpointed_dequant_contexts, (module,))
            )
        if compile_model:
            from cosmos_framework.posttrain.utils.native_compile import guard_native_compiled_call

            guard_native_compiled_call(forward)
        before = quantized_state_record(module)
        x = torch.randn(2, 7, 128, generator=generator, dtype=torch.bfloat16).to(device).requires_grad_(True)
        control_x = x.detach().clone().requires_grad_(True)
        effective = bnb.functional.dequantize_4bit(module.weight.data, module.weight.quant_state).detach()
        actual, expected = forward(x), F.linear(control_x, effective)
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
            loss = forward(x.detach()).float().square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite NF4 canary loss")
            loss.backward()
            if any(
                parameter.dtype != torch.float32
                or parameter.grad is None
                or parameter.grad.dtype != torch.float32
                or not torch.isfinite(parameter.grad).all()
                for parameter in module.adapter_parameters().values()
            ):
                raise RuntimeError("NF4 canary did not propagate finite adapter gradients")
            saw_a_gradient |= bool(torch.count_nonzero(module.lora_A.weight.grad).item())
            optimizer.step()
            if any(
                state[name].dtype != torch.float32
                for state in optimizer.state.values()
                for name in ("exp_avg", "exp_avg_sq")
            ):
                raise RuntimeError("NF4 canary optimizer moments must remain FP32")
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
        result = {
            "backend": "bitsandbytes",
            "version": bnb.__version__,
            "steps": 3,
            "adapter_compute_dtype": str(adapter_compute_dtype).removeprefix("torch."),
            "compiled": compile_model,
            "forward_max_abs": float((actual - expected).abs().max().item()),
            "input_gradient_max_abs": float((actual_grad - expected_grad).abs().max().item()),
        }
        if quantized_linear_execution != "bnb":
            result["quantized_linear_execution"] = quantized_linear_execution
        return result
