# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""ModelOpt FP8 wrapper for the packed grouped-mm MoE experts.

ModelOpt's generic ``_QuantFusedExperts`` claims any module with 3-D ``gate_up_proj`` /
``down_proj`` parameters but only intercepts ``F.linear``, so on the packed grouped-mm experts
(which run ``torch._grouped_mm``) it would quantize nothing. :class:`QuantPackedExperts` keeps
the experts packed and quantizes them in place: one per-expert weight quantizer per packed
parameter and one shared input quantizer per GEMM, matching what vLLM's FP8 MoE kernels consume.

The quantizers act on tensors inside the expert computation, so its ``forward`` routes the tokens
and runs both grouped GEMMs itself.
"""

from __future__ import annotations

import torch
from modelopt.torch.quantization.nn import QuantModule, QuantModuleRegistry, TensorQuantizer
from torch import nn

from cosmos_framework.model.generator.moe.interface import MoEExperts
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.moe import Qwen3VLMoeTextExpertsGroupedMm

_PACKED_EXPERT_TYPES = (Qwen3VLMoeTextExpertsGroupedMm, MoEExperts)


class QuantPackedExperts(QuantModule):
    """Quantized packed grouped-mm experts; the packed parameters stay untouched."""

    def _setup(self):
        self.gate_up_proj_input_quantizer = TensorQuantizer()
        self.gate_up_proj_weight_quantizer = TensorQuantizer()
        self.down_proj_input_quantizer = TensorQuantizer()
        self.down_proj_weight_quantizer = TensorQuantizer()

    def forward(
        self,
        hidden_states: torch.Tensor,  # [num_tokens,hidden_size]
        topk_scores: torch.Tensor,  # [num_tokens,top_k]
        expert_indices: torch.Tensor,  # [num_tokens,top_k]
        num_tokens_per_expert: torch.Tensor,  # [num_experts]
        token_mask: torch.Tensor | None = None,  # [num_tokens]
    ) -> torch.Tensor:  # [num_tokens,hidden_size]
        # Group the real (token, slot) pairs by expert; ``num_tokens_per_expert`` counts exactly these.
        top_k = expert_indices.shape[1]
        slots = torch.arange(expert_indices.numel(), device=expert_indices.device)  # [num_tokens*top_k]
        if token_mask is not None:
            slots = slots[token_mask.repeat_interleave(top_k)]  # [num_routed]
        if slots.numel() == 0:
            return torch.zeros_like(hidden_states)  # [num_tokens,hidden_size]
        slots = slots[torch.argsort(expert_indices.reshape(-1)[slots], stable=True)]  # [num_routed]
        tokens = slots // top_k  # [num_routed]
        scores = topk_scores.reshape(-1)[slots].to(hidden_states.dtype)  # [num_routed]
        x = hidden_states[tokens]  # [num_routed,hidden_size]
        routed = self._quantized_experts(x, num_tokens_per_expert, scores)  # [num_routed,hidden_size]
        return torch.zeros_like(hidden_states).index_add_(0, tokens, routed)  # [num_tokens,hidden_size]

    def _quantized_experts(
        self,
        x: torch.Tensor,  # [num_routed,hidden_size]  (grouped by expert)
        num_tokens_per_expert: torch.Tensor,  # [num_experts]
        scores: torch.Tensor,  # [num_routed]
    ) -> torch.Tensor:  # [num_routed,hidden_size]
        # a2 observes ``act(gate) * up`` before the routing scores, because vLLM applies the scores
        # after the down GEMM.
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)  # [num_experts]
        x = self.gate_up_proj_input_quantizer(x)
        h = torch._grouped_mm(x, self.gate_up_proj_weight_quantizer(self.gate_up_proj), offs=offsets)
        g, u = torch.chunk(h, chunks=2, dim=-1)  # 2x [num_routed,moe_intermediate_size]
        a = self.down_proj_input_quantizer(self.act_fn(g) * u)  # [num_routed,moe_intermediate_size]
        return torch._grouped_mm(
            a * scores.unsqueeze(-1), self.down_proj_weight_quantizer(self.down_proj), offs=offsets
        )  # [num_routed,hidden_size]


def register_packed_experts() -> None:
    """Register :class:`QuantPackedExperts` before ``mtq.quantize`` so ``_QuantFusedExperts`` skips it."""
    pending: dict[type[nn.Module], str] = {}
    for expert_cls in _PACKED_EXPERT_TYPES:
        registered = QuantModuleRegistry.get(expert_cls)
        if registered is not None and issubclass(registered, QuantPackedExperts):
            continue
        if registered is not None:
            # An earlier ``mtq.quantize`` that saw this type unregistered has globally registered
            # ModelOpt's generic ``_QuantFusedExperts`` for it, which would quantize nothing here.
            QuantModuleRegistry.unregister(expert_cls)
        pending[expert_cls] = expert_cls.__name__
    if pending:
        QuantModuleRegistry.register(pending)(QuantPackedExperts)


def has_packed_experts(model: nn.Module) -> bool:
    return any(isinstance(module, _PACKED_EXPERT_TYPES) for module in model.modules())


def assert_quantizers_calibrated(model: nn.Module) -> None:
    """Fail export instead of writing a missing or zero scale for an enabled quantizer."""
    problems = []
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and not module.is_enabled:
            continue
        if isinstance(module, TensorQuantizer):
            amax = getattr(module, "amax", None)
            if amax is None:
                problems.append(f"{name}: no amax")
            elif not torch.isfinite(amax).all():
                problems.append(f"{name}: non-finite amax")
            elif name.endswith("input_quantizer") and not (amax > 0).all():
                problems.append(f"{name}: non-positive amax")
        elif (
            isinstance(getattr(module, "gate_up_proj", None), nn.Parameter)
            and module.gate_up_proj.ndim == 3
            and not isinstance(module, QuantPackedExperts)
        ):
            problems.append(f"{name}: packed experts {type(module).__name__} are not quantized")
    if problems:
        shown = "; ".join(problems[:10])
        raise ValueError(f"{len(problems)} quantizer problem(s), first {min(len(problems), 10)}: {shown}")


_FP8_MAX = 448.0  # Mirrors ``export.FP8_MAX``; export imports this module, so it is not imported back.
_PACKED_EXPERT_QUANTIZERS = (
    "gate_up_proj_input_quantizer",
    "gate_up_proj_weight_quantizer",
    "down_proj_input_quantizer",
    "down_proj_weight_quantizer",
)
_PACKED_EXPERT_KEY_SUFFIXES = tuple(
    f"experts.{name}{suffix}"
    for name in ("gate_up_proj", "down_proj")
    for suffix in ("", ".weight_scale", ".input_scale")
)


def is_packed_expert_key(key: str) -> bool:
    """True for a packed expert parameter or its scale, e.g. ``...experts.gate_up_proj.weight_scale``."""
    return key.endswith(_PACKED_EXPERT_KEY_SUFFIXES)


def _check_per_expert_fp8(name: str, module: QuantPackedExperts) -> None:
    """Refuse any layout but the one written below: FP8 E4M3, per-expert weight amax, scalar input amax."""
    for quantizer_name in _PACKED_EXPERT_QUANTIZERS:
        quantizer = getattr(module, quantizer_name)
        amax_shape = tuple(quantizer.amax.shape)
        if quantizer.num_bits != (4, 3) or quantizer.block_sizes:
            found = f"num_bits={quantizer.num_bits}, block_sizes={quantizer.block_sizes}"
        elif quantizer_name.endswith("weight_quantizer"):
            num_experts = getattr(module, quantizer_name.removesuffix("_weight_quantizer")).shape[0]
            if amax_shape == (num_experts, 1, 1):
                continue
            found = f"amax shape {amax_shape}, expected ({num_experts}, 1, 1)"
        elif quantizer.amax.numel() == 1:
            continue
        else:
            found = f"amax shape {amax_shape}, expected a single element"
        raise ValueError(f"{name}.{quantizer_name}: {found}; packed-expert export supports per-expert FP8 only")


def export_packed_experts(net: nn.Module) -> dict[str, torch.Tensor]:
    """FP8 packed experts in the source layout, with per-expert weight scales and scalar input scales."""
    if not has_packed_experts(net):
        return {}  # Dense models export exactly as before, without the calibration check.
    assert_quantizers_calibrated(net)
    exported = {}
    for name, module in net.named_modules():
        if not isinstance(module, QuantPackedExperts):
            continue
        enabled = [getattr(module, quantizer).is_enabled for quantizer in _PACKED_EXPERT_QUANTIZERS]
        if not any(enabled):
            continue  # Skipped by the coverage filter; the experts stay BF16 from the source.
        if not all(enabled):
            raise ValueError(f"{name or 'experts'}: packed-expert quantizers are only partly enabled: {enabled}")
        _check_per_expert_fp8(name or "experts", module)
        for param_name in ("gate_up_proj", "down_proj"):
            # BF16 models keep BF16 amax; the scales are FP32 on disk.
            weight_amax = getattr(module, f"{param_name}_weight_quantizer").amax.detach().float()
            weight_scale = torch.where(weight_amax == 0, 1.0, weight_amax / _FP8_MAX)  # [num_experts,1,1]
            weight = getattr(module, param_name).detach().float()
            fp8 = (weight / weight_scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
            input_amax = getattr(module, f"{param_name}_input_quantizer").amax.detach().float()
            key = f"{name}.{param_name}" if name else param_name
            exported[key] = fp8.contiguous().cpu()
            exported[f"{key}.weight_scale"] = weight_scale.contiguous().cpu()
            exported[f"{key}.input_scale"] = (input_amax / _FP8_MAX).reshape(()).contiguous().cpu()
    return exported
