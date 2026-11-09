# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Versioned long-prefill FP64 MMA attention; fixed 16-query/64-key tiles.

Used only by qwen3_vl_bf16_tensorcore_prefill_v1 on SM80. Decode and short
inputs retain the reference kernel. This is the preserved candidate2 arithmetic.
"""

import triton.experimental.gluon.language as tl
from triton.experimental import gluon
from triton.experimental.gluon.language import BlockedLayout as Blocked
from triton.experimental.gluon.language import DotOperandLayout, NVMMADistributedLayout, SliceLayout
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
from triton.language.extra.cuda import libdevice


@gluon.jit
def _attention_tensorcore_kernel(
    Q,
    K,
    V,
    O,
    Mask,
    q0,
    q1,
    q2,
    q3,
    k0,
    k1,
    k2,
    k3,
    v0,
    v1,
    v2,
    v3,
    m0,
    m2,
    m3,
    M,
    N,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BOOL_MASK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    L: tl.constexpr = Blocked([1, 4], [4, 8], [4, 1], [1, 0])
    MMA: tl.constexpr = NVMMADistributedLayout(version=[2, 0], warps_per_cta=[2, 2], instr_shape=[16, 8])
    rows = tl.program_id(0) * BM + tl.arange(0, BM, layout=SliceLayout(1, L))
    h = tl.program_id(1)
    b = tl.program_id(2)
    dims = tl.arange(0, BD, layout=SliceLayout(0, L))
    cols = tl.arange(0, BN, layout=SliceLayout(0, L))
    q = tl.load(
        Q + b * q0 + h * q1 + rows[:, None] * q2 + dims[None, :] * q3,
        mask=(rows[:, None] < M) & (dims[None, :] < D),
        other=0,
    ).to(tl.float64)
    q = tl.convert_layout(q, DotOperandLayout(0, MMA, 1))
    running_max = tl.full((BM,), float("-inf"), tl.float64, layout=SliceLayout(1, L))
    total = tl.full((BM,), 0.0, tl.float64, layout=SliceLayout(1, L))
    acc = tl.full((BM, BD), 0.0, tl.float64, layout=L)
    end = N if HAS_MASK else tl.minimum(N, N - M + (tl.program_id(0) + 1) * BM)
    for start in range(0, tl.cdiv(end, BN)):
        keys = start * BN + cols
        k = tl.load(
            K
            + b * k0
            + (h // (HQ // HK)) * k1
            + keys[None, :] * k2
            + tl.convert_layout(dims, SliceLayout(1, L))[:, None] * k3,
            mask=(keys[None, :] < N) & (tl.convert_layout(dims, SliceLayout(1, L))[:, None] < D),
            other=0,
        ).to(tl.float64)
        score = (
            tl.convert_layout(
                mma_v2(
                    q, tl.convert_layout(k, DotOperandLayout(1, MMA, 1)), tl.full((BM, BN), 0.0, tl.float64, layout=MMA)
                ),
                L,
            )
            * SCALE
        )
        valid = (rows[:, None] < M) & (keys[None, :] < N)
        if HAS_MASK:
            mask = tl.load(
                Mask + b * m0 + rows[:, None] * m2 + keys[None, :] * m3,
                mask=(rows[:, None] < M) & (keys[None, :] < N),
                other=0,
            )
            if BOOL_MASK:
                valid = valid & mask
            else:
                score = score + mask.to(tl.float64)
        else:
            valid = valid & (keys[None, :] <= N - M + rows[:, None])
        score = tl.where(valid, score, float("-inf"))
        new_max = tl.maximum(running_max, tl.max(score, 1))
        safe_max = tl.where(new_max == float("-inf"), 0.0, new_max)
        alpha = tl.where(total > 0, libdevice.exp(running_max - safe_max), 0.0)
        prob = libdevice.exp(score - safe_max[:, None])
        total = total * alpha + tl.sum(prob, 1)
        v = tl.load(
            V
            + b * v0
            + (h // (HQ // HK)) * v1
            + tl.convert_layout(keys, SliceLayout(1, L))[:, None] * v2
            + dims[None, :] * v3,
            mask=(tl.convert_layout(keys, SliceLayout(1, L))[:, None] < N) & (dims[None, :] < D),
            other=0,
        ).to(tl.float64)
        acc = acc * alpha[:, None] + tl.convert_layout(
            mma_v2(
                tl.convert_layout(prob, DotOperandLayout(0, MMA, 1)),
                tl.convert_layout(v, DotOperandLayout(1, MMA, 1)),
                tl.full((BM, BD), 0.0, tl.float64, layout=MMA),
            ),
            L,
        )
        running_max = new_max
    tl.store(
        O + ((b * HQ + h) * M + rows[:, None]) * D + dims[None, :],
        tl.where(total[:, None] > 0, acc / total[:, None], 0.0),
        mask=(rows[:, None] < M) & (dims[None, :] < D),
    )
