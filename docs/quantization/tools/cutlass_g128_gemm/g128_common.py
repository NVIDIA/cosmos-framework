# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared definitions for the INT8 g128 GEMM work: the project's exact quantization math, an fp64 reference,
the SGLang Triton block-INT8 kernel (per-token 1x128 activation scale, per-column 1x128 weight scale), and the
flush-L2 / median timing helper used by all benches in docs/quantization/tools/.

g128 (per-col) as defined in int8_handoff.md 2.1 / 3.6:
  A[M,K] int8, s_a[M, K/128] fp32   (dynamic, per token, per 128 consecutive K; absmax/127, rint, clamp +-127)
  W[N,K] int8, s_w[N, K/128] fp32   (static, per output channel, per 128 K)
  Y[m,n] = sum_g s_a[m,g] * s_w[n,g] * sum_{k in g} qa[m,k]*qw[n,k]   (int32 inside a group, fp32 across groups, bf16 out)
"""
from __future__ import annotations

import statistics

import torch

G = 128
QMAX = 127.0
FLUSH_BYTES = 512 << 20


def quant_int8_groups(x: torch.Tensor, group: int = G):
    """Symmetric INT8 with one fp32 scale per `group` consecutive elements of the last dim.
    Matches cosmos_framework.utils.generator.quantization.fake_quant_int8(per_row=True, group_size=group):
    scale = absmax/127 (fp32, floored at fp32 tiny), q = round_half_even(x/scale) clamped to [-127, 127]."""
    rows, k = x.shape
    assert k % group == 0
    x32 = x.float().reshape(rows, k // group, group)
    scale = (x32.abs().amax(dim=-1) / QMAX).clamp_(min=torch.finfo(torch.float32).tiny)  # [rows, k/group]
    q = torch.round(x32 / scale[..., None]).clamp_(-QMAX, QMAX).to(torch.int8).reshape(rows, k)
    return q, scale.contiguous()


def reference_fp64(qa: torch.Tensor, sa: torch.Tensor, qw: torch.Tensor, sw: torch.Tensor, rows=None, group: int = G):
    """Exact reference Y = sum_g sa[m,g] sw[n,g] <qa[m,g,:], qw[n,g,:]> in fp64 for the given row subset."""
    if rows is not None:
        qa, sa = qa[rows], sa[rows]
    m, k = qa.shape
    n = qw.shape[0]
    a = qa.double().reshape(m, k // group, group)
    w = qw.double().reshape(n, k // group, group)
    # per-group int dot products (exact in fp64: |sum| <= 128*127*127 < 2^53)
    dots = torch.einsum("mgk,ngk->mng", a, w)  # [m, n, k/group]
    return torch.einsum("mng,mg,ng->mn", dots, sa.double(), sw.double())


def check_bf16(y: torch.Tensor, ref: torch.Tensor, tol: float = 1.5e-2):
    rel = (y.double() - ref).abs() / ref.abs().clamp(min=1e-3)
    return rel.max().item(), bool((rel <= tol).all())


def time_fn(fn, iters: int = 50, warmup: int = 10, flush: torch.Tensor | None = None):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for i in range(iters):
        if flush is not None:
            flush.fill_(i & 0xFF)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(ts), statistics.fmean(ts)


def sglang_block_int8(qa, sa, qw, sw, out_dtype=torch.bfloat16, cfg=None):
    """SGLang _w8a8_block_int8_matmul with block_size=[1, 128] (per-column weight scale), explicit tile config.
    W is passed as the [N,K] row-major tensor and indexed with transposed strides (K-major B), as in the H100 bench."""
    import triton
    from ref.sglang_int8_kernel import _w8a8_block_int8_matmul  # noqa: WPS433

    cfg = cfg or dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=8, num_warps=4, num_stages=4)
    M, K = qa.shape
    N = qw.shape[0]
    C = torch.empty(M, N, device=qa.device, dtype=out_dtype)
    grid = (triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),)
    _w8a8_block_int8_matmul[grid](
        qa, qw, C, sa, sw, M, N, K, 1, G,
        qa.stride(0), qa.stride(1), qw.stride(1), qw.stride(0), C.stride(0), C.stride(1),
        sa.stride(0), sa.stride(1), sw.stride(1), sw.stride(0),
        BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"], BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"], BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"], num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    return C
