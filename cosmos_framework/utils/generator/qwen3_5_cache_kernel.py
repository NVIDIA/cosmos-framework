# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Versioned gated-delta recurrence: FP64 reduction, FP32 state after every token."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _recurrence_kernel(
    Q,
    K,
    V,
    G,
    Beta,
    Initial,
    Output,
    Final,
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
    g0,
    g1,
    g2,
    b0,
    b1,
    b2,
    T,
    H: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    SAVE_FINAL: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    keys = tl.arange(0, BK)
    values = block * BV + tl.arange(0, BV)
    pos = ((batch * H + head) * DK + keys[:, None]) * DV + values[None, :]
    valid = (keys[:, None] < DK) & (values[None, :] < DV)
    if HAS_INITIAL:
        state = tl.load(Initial + pos, mask=valid, other=0).to(tl.float64)
    else:
        state = tl.full((BK, BV), 0.0, tl.float64)
    for t in range(T):
        q = tl.load(Q + batch * q0 + t * q1 + head * q2 + keys * q3, mask=keys < DK, other=0).to(tl.float64) * (
            DK**-0.5
        )
        k = tl.load(K + batch * k0 + t * k1 + head * k2 + keys * k3, mask=keys < DK, other=0).to(tl.float64)
        v = tl.load(V + batch * v0 + t * v1 + head * v2 + values * v3, mask=values < DV, other=0).to(tl.float64)
        g = tl.load(G + batch * g0 + t * g1 + head * g2).to(tl.float64)
        beta = tl.load(Beta + batch * b0 + t * b1 + head * b2).to(tl.float64)
        state = state * libdevice.exp(g)
        memory = tl.sum(state * k[:, None], axis=0)
        delta = (v - memory) * beta
        state = (state + k[:, None] * delta[None, :]).to(tl.float32).to(tl.float64)
        value = tl.sum(state * q[:, None], axis=0)
        tl.store(Output + ((batch * T + t) * H + head) * DV + values, value, mask=values < DV)
    if SAVE_FINAL:
        tl.store(Final + pos, state, mask=valid)
