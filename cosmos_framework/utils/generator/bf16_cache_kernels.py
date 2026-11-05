# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Versioned fixed-tile inference kernels; imported only by the opt-in CUDA profile.

Changing reduction order or tile constants requires a new execution version.
BF16 linear inputs/weights accumulate in FP32; causal attention reduces fixed
key tiles in FP64 and returns BF16, without constructing a dense score matrix.
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _attention_kernel(
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
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    dims = tl.arange(0, BD)
    cols = tl.arange(0, BN)
    q = tl.load(Q + b * q0 + h * q1 + row * q2 + dims * q3, mask=dims < D, other=0).to(tl.float64)
    running_max = tl.full((), float("-inf"), tl.float64)
    total = tl.full((), 0.0, tl.float64)
    acc = tl.full((BD,), 0.0, tl.float64)
    # Explicit masks are authoritative, as in stock SDPA. In particular a
    # finite additive bias is not a boolean exclusion; even fully biased rows
    # must retain their reference softmax semantics.
    end = N if HAS_MASK else N - M + row + 1
    for start in range(0, tl.cdiv(end, BN)):
        keys = start * BN + cols
        k = tl.load(
            K + b * k0 + (h // (HQ // HK)) * k1 + keys[:, None] * k2 + dims[None, :] * k3,
            mask=(keys[:, None] < N) & (dims[None, :] < D),
            other=0,
        ).to(tl.float64)
        score = tl.sum(q[None, :] * k, 1) * SCALE
        valid = (keys < N) & (keys < end)
        if HAS_MASK:
            mask = tl.load(Mask + b * m0 + row * m2 + keys * m3, mask=keys < N, other=0)
            if BOOL_MASK:
                valid = valid & mask
            else:
                score = score + mask.to(tl.float64)
        score = tl.where(valid, score, float("-inf"))
        new_max = tl.maximum(running_max, tl.max(score, 0))
        safe_max = tl.where(new_max == float("-inf"), 0.0, new_max)
        alpha = tl.where(total > 0, libdevice.exp(running_max - safe_max), 0.0)
        prob = libdevice.exp(score - safe_max)
        total = total * alpha + tl.sum(prob, 0)
        v = tl.load(
            V + b * v0 + (h // (HQ // HK)) * v1 + keys[:, None] * v2 + dims[None, :] * v3,
            mask=(keys[:, None] < N) & (dims[None, :] < D),
            other=0,
        ).to(tl.float64)
        acc = acc * alpha + tl.sum(prob[:, None] * v, 0)
        running_max = new_max
    tl.store(O + ((b * HQ + h) * M + row) * D + dims, tl.where(total > 0, acc / total, 0.0), mask=dims < D)


@triton.jit
def _linear_kernel(
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
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0.0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + inner
        x = tl.load(X + rows[:, None] * K + k[None, :], mask=(rows[:, None] < M) & (k[None, :] < K), other=0)
        w = tl.load(W + cols[None, :] * K + k[:, None], mask=(cols[None, :] < N) & (k[:, None] < K), other=0)
        acc = tl.dot(x, w, acc)
    if HAS_BIAS:
        acc += tl.load(B + cols, mask=cols < N, other=0)[None, :].to(tl.float32)
    tl.store(Y + rows[:, None] * N + cols[None, :], acc, mask=(rows[:, None] < M) & (cols[None, :] < N))
