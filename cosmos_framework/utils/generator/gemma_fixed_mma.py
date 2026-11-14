# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Experimental Gemma projection/norm arithmetic; attention remains FP32.

Reuse the fixed A100 MMA kernel and separate adapter algebra already used by
SmolLM. Gemma's RMSNorm has a zero-centered weight and an FP32 ``1 + weight``
boundary; preserve it instead of substituting Llama's direct weight scale.
No owned parameter, buffer, low-rank branch, or persistent cache is replaced.
"""

import torch


def _project(inputs, weight, bias=None):
    from cosmos_framework.utils.generator.smollm_fixed_mma import linear

    return linear(inputs, weight, bias)


def _check(module, inputs):
    from cosmos_framework.utils.generator.gemma_sdpa_layout import _guard

    _guard(module)
    if inputs.dtype != torch.float32:
        raise ValueError("gemma_fp32_fixed_mma_v1 requires FP32 operator inputs")


def linear(module, inputs):
    _check(module, inputs)
    return _project(inputs, module.weight, module.bias).to(inputs.dtype)


def adapter(module, inputs):
    from cosmos_framework.utils.generator.smollm_consistent import _adapter_result

    _check(module, inputs)
    # The common algebra retains base, A/B, PiSSA initial branches and DoRA
    # magnitude. The DoRA direction is only a norm reference, never a dense
    # effective-weight forward. Reuse its explicit FP64 transient boundaries.
    return _adapter_result(module, inputs, _project)


def norm(module, inputs):
    _check(module, inputs)
    x = inputs.double()
    normalized = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + module.eps)).to(inputs.dtype)
    # Match Gemma's original FP32 offset-weight addition and final multiply.
    # Moving the +1 into FP64 would independently change the represented scale.
    return normalized * (1.0 + module.weight.float())
