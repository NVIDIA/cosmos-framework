# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Fixed FP64 MMA linear candidate; every row executes identical K tiles."""

import triton.experimental.gluon.language as tl
from triton.experimental import gluon
from triton.experimental.gluon.language import BlockedLayout, DotOperandLayout, NVMMADistributedLayout, SliceLayout
from triton.experimental.gluon.language.nvidia.ampere import mma_v2


@gluon.jit
def fixed_linear(
    X,
    W,
    B,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    L: tl.constexpr = BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    MMA: tl.constexpr = NVMMADistributedLayout(version=[2, 0], warps_per_cta=[2, 2], instr_shape=[16, 8])
    rows = tl.program_id(0) * BM + tl.arange(0, BM, layout=SliceLayout(1, L))
    cols = tl.program_id(1) * BN + tl.arange(0, BN, layout=SliceLayout(0, L))
    inner_x = tl.arange(0, BK, layout=SliceLayout(0, L))
    inner_w = tl.arange(0, BK, layout=SliceLayout(1, L))
    acc = tl.full((BM, BN), 0.0, tl.float64, layout=MMA)
    for start in range(tl.cdiv(K, BK)):
        kx = start * BK + inner_x
        kw = start * BK + inner_w
        x = tl.load(X + rows[:, None] * K + kx[None, :], mask=(rows[:, None] < M) & (kx[None, :] < K), other=0).to(
            tl.float64
        )
        w = tl.load(W + cols[None, :] * K + kw[:, None], mask=(cols[None, :] < N) & (kw[:, None] < K), other=0).to(
            tl.float64
        )
        acc = mma_v2(
            tl.convert_layout(x, DotOperandLayout(0, MMA, 1)), tl.convert_layout(w, DotOperandLayout(1, MMA, 1)), acc
        )
    result = tl.convert_layout(acc, L)
    if HAS_BIAS:
        result = result + tl.load(B + cols, mask=cols < N, other=0).to(tl.float64)[None, :]
    tl.store(Y + rows[:, None] * N + cols[None, :], result, mask=(rows[:, None] < M) & (cols[None, :] < N))


def linear(inputs, weight, bias=None):
    """Fixed A100 FP64 MMA order, independent of row count and tail padding.

    Parameters remain FP32; conversion occurs in each tile. Return FP64 so the
    shared factorized implementation rounds only at its original FP32 boundary.
    No derived weight/cache or parameter replacement is created.
    """
    import torch
    import triton

    if torch.is_grad_enabled() or torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"):
        raise RuntimeError("fixed MMA inference requires no_grad and no autocast")
    if (
        inputs.device != weight.device
        or inputs.device.type != "cuda"
        or torch.cuda.get_device_capability(inputs.device) != (8, 0)
    ):
        raise ValueError("fixed MMA inference requires one A100 device")
    allowed = (torch.float32, torch.float64)
    if (
        inputs.dtype not in allowed
        or weight.dtype not in allowed
        or weight.ndim != 2
        or inputs.ndim < 2
        or inputs.shape[-1] != weight.shape[1]
        or not weight.is_contiguous()
    ):
        raise ValueError("fixed MMA requires contiguous FP32/FP64 weights and compatible input dimensions")
    if bias is not None and (
        bias.shape != (weight.shape[0],)
        or bias.device != weight.device
        or bias.dtype not in allowed
        or not bias.is_contiguous()
    ):
        raise ValueError("fixed MMA requires a compatible contiguous bias")
    x = inputs.reshape(-1, inputs.shape[-1]).contiguous()
    m, k = x.shape
    n = weight.shape[0]
    output = torch.empty((m, n), dtype=torch.float64, device=inputs.device)
    if m:
        with torch.cuda.device(inputs.device):
            fixed_linear[(triton.cdiv(m, 16), triton.cdiv(n, 32))](
                x,
                weight,
                bias if bias is not None else x,
                output,
                m,
                n,
                k,
                bias is not None,
                16,
                32,
                32,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
            )
    return output.reshape(*inputs.shape[:-1], n)
