# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Real-INT8 GEMM linear used as a *speed proxy* for the INT8 plan.

``Int8SpeedLinear`` stores the weight as INT8 with one scale per output channel and, on
every call, quantizes the input per token (dynamic absmax), runs ``torch._int_mm``
(cuBLASLt INT8 x INT8 -> INT32) and rescales the INT32 result to the compute dtype.
It is a plain ``nn.Module`` -- no tensor subclass -- so torch.compile captures it whole
and CUDA graphs can replay it.

Scale granularity is per token / per channel because cuBLASLt exposes no block-scaled
INT8 MMA; the accuracy plan's group-64 scales would need a custom kernel. Use this
module to measure GEMM-side speed, not to judge accuracy.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

_QMAX = 127.0


class Int8SpeedLinear(nn.Module):
    """W8A8 linear on ``torch._int_mm`` with per-channel weight and per-token activation scales."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.compute_dtype = linear.weight.dtype
        with torch.no_grad():
            w32 = linear.weight.detach().float()  # [N,K]
            w_scale = (w32.abs().amax(dim=1) / _QMAX).clamp_(min=torch.finfo(torch.float32).tiny)  # [N]
            w_q = torch.round(w32 / w_scale[:, None]).clamp_(-_QMAX, _QMAX).to(torch.int8)
        # _int_mm wants b as [K,N]; keep a contiguous transposed copy
        self.register_buffer("weight_int8_t", w_q.t().contiguous(), persistent=False)  # [K,N] int8
        self.register_buffer("weight_scale", w_scale, persistent=False)  # [N] fp32
        self.bias = linear.bias
        del linear.weight  # free the bf16 weight

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.reshape(-1, self.in_features)
        m = x.shape[0]
        if m <= 16:  # cuBLASLt INT8 GEMM requires M > 16; tiny batches fall back to bf16 math
            w = (self.weight_int8_t.t().float() * self.weight_scale[:, None]).to(self.compute_dtype)
            out = F.linear(x, w, self.bias)
            return out.reshape(*inputs.shape[:-1], self.out_features)
        x32 = x.float()
        x_scale = (x32.abs().amax(dim=1, keepdim=True) / _QMAX).clamp_(min=torch.finfo(torch.float32).tiny)  # [M,1]
        x_q = torch.round(x32 / x_scale).clamp_(-_QMAX, _QMAX).to(torch.int8)
        acc = torch._int_mm(x_q, self.weight_int8_t)  # [M,N] int32
        out = (acc.float() * x_scale * self.weight_scale[None, :]).to(self.compute_dtype)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*inputs.shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, w8a8 int_mm (per-channel W, per-token A)"


def swap_int8_speed_linears(model: nn.Module, target_fqns: list[str]) -> list[str]:
    """Replace the given loaded ``nn.Linear`` modules with :class:`Int8SpeedLinear` (in place)."""
    swapped: list[str] = []
    for fqn in sorted(target_fqns):
        module = model.get_submodule(fqn)
        if not isinstance(module, nn.Linear):
            raise KeyError(f"int8_speed target {fqn!r} is not an nn.Linear module")
        if module.in_features % 8 or module.out_features % 8:
            raise ValueError(f"int8_speed target {fqn!r}: in/out features must be multiples of 8 for _int_mm")
        replacement = Int8SpeedLinear(module).to(
            module.weight_int8_t.device if hasattr(module, "weight_int8_t") else "cuda"
        )
        parent_fqn, _, child = fqn.rpartition(".")
        setattr(model.get_submodule(parent_fqn) if parent_fqn else model, child, replacement)
        swapped.append(fqn)
    return swapped


__all__ = ["Int8SpeedLinear", "swap_int8_speed_linears"]
