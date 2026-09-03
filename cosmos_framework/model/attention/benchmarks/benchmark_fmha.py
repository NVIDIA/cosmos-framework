# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Benchmark the unified ``cosmos_framework.model.attention.attention`` frontend across a
user-specified list of backends, for a single sequence-length (and optional
context-parallel shard shape).

Dense / batched attention ONLY: this benchmark deliberately has no varlen
(sequence-packed) support. Inputs are ``[B, S, H, D]``.

This single script combines two previously-separate benchmarks:

1. Backend sweep (the former ``benchmark_fmha``): compare backends at a single query
   length (``--seqlen``), with optional cross-attention (``--kv-seqlen``), causal
   masking, backward-pass timing, MLA-style value head dim (``--head-dim-v``), and
   per-backend rejection diagnostics (``--debug``).

2. Context-parallel per-rank shape study :
   isolate the per-rank attention kernel shape a CP strategy would produce, via
   ``--shard-mode`` + ``--cp-size``. ``head`` divides Q/KV heads by ``cp_size`` and
   keeps the full KV length; ``sequence`` keeps the heads and splits the KV length
   across ``cp_size`` shards (Q is intentionally NOT divided — each rank computes a
   partial result over a local KV shard). There is intentionally no all-to-all,
   all-gather, or all-reduce in the timed region: the intent is to study local kernel
   scalability. The shard transform is applied to the ``(q_len, kv_len)`` shape. With
   ``--shard-mode none`` (default) no sharding is applied.

The script runs single-process by default, but also runs under ``torchrun`` so a real
multi-rank CP shape can be measured (``sequence`` mode is rank-dependent). Metrics
reported per row: average latency, achieved TFLOP/s (on the LOCAL shape), tokens/s,
and peak allocated memory. Only rank 0 prints.

FlashInfer options are also available when Cosmos3's optional
``flashinfer`` dependency group is installed on GB200/SM100:
* ``flashinfer-trtllm-gen``: dense BF16 FlashInfer TRTLLM-GEN attention.
* ``flashinfer-trtllm-gen-sage``: BSHD-to-THD layout, SAGE quantization/scales,
  ragged metadata, and fused TRTLLM-GEN SAGE attention.
They call FlashInfer directly rather than registering a production
``cosmos_framework.model.attention`` backend. They are forward-only, batch-one, BF16,
non-causal benchmark paths. The SAGE row reports ``-`` for TFLOP/s because its
timed scope includes quantization and metadata work, not only attention FLOPs.
Implementation references for the exploratory FlashInfer measurements:
* FlashInfer TRTLLM-GEN attention API:
  https://docs.flashinfer.ai/api/attention.html

By default this is self-attention (``q_len == kv_len``). Cross-attention with a
different KV length can be benchmarked via ``--kv-seqlen``, but only when ``--causal``
is NOT set: causal masking here uses ``CausalType.DontCare``, which requires
``q_len == kv_len``. When ``--causal`` is set, ``kv_len`` always equals ``q_len``.

Examples:

    # Backend sweep, self-attention, causal, forward-only:
    python -m cosmos_framework.model.attention.benchmarks.benchmark_fmha \
        --backends natten cudnn \
        --seqlen 3093 \
        --batch 1 --num-q-heads 16 --num-kv-heads 8 --head-dim 128 \
        --dtype bf16 --causal

    # Cross-attention (non-causal), distinct q_len and kv_len:
    python -m cosmos_framework.model.attention.benchmarks.benchmark_fmha \
        --backends natten cudnn \
        --seqlen 4096 --kv-seqlen 1024 \
        --num-q-heads 16 --num-kv-heads 16 --head-dim 128 --dtype bf16

    # One-GPU mock of a CP=4 per-rank head-sharded shape:
    python -m cosmos_framework.model.attention.benchmarks.benchmark_fmha \
        --backends natten cudnn \
        --seqlen 396 --kv-seqlen 177771 \
        --num-q-heads 32 --num-kv-heads 8 --head-dim 128 \
        --shard-mode head --cp-size 4 --compile

    # True four-GPU CP head-sharded run:
    torchrun --standalone --nproc_per_node=4 \
        -m cosmos_framework.model.attention.benchmarks.benchmark_fmha \
        --seqlen 396 --kv-seqlen 177771 \
        --num-q-heads 32 --num-kv-heads 8 --head-dim 128 \
        --shard-mode head --cp-size 4

    # One-GPU mock of a CP=4 per-rank sequence-sharded (split-KV) shape.
    # KV length is divided by cp_size while heads are kept; --causal is not allowed:
    python -m cosmos_framework.model.attention.benchmarks.benchmark_fmha \
        --backends natten cudnn \
        --seqlen 396 --kv-seqlen 177771 \
        --num-q-heads 32 --num-kv-heads 8 --head-dim 128 \
        --shard-mode sequence --cp-size 4

    # Compare the current frontend with FlashInfer's dense and SAGE paths.
    python -m cosmos_framework.model.attention.benchmarks.benchmark_fmha \\
        --backends auto flashinfer-trtllm-gen flashinfer-trtllm-gen-sage \\
        --seqlen 1576 --kv-seqlen 26549 \\
        --batch 1 --num-q-heads 32 --num-kv-heads 8 --head-dim 128 \\
        --dtype bf16 --q-block-size 1 --k-block-size 16

    # True four-GPU CP sequence-sharded run (each rank holds a different KV shard):
    torchrun --standalone --nproc_per_node=4 \
        -m cosmos_framework.model.attention.benchmarks.benchmark_fmha \
        --seqlen 396 --kv-seqlen 177771 \
        --num-q-heads 32 --num-kv-heads 8 --head-dim 128 \
        --shard-mode sequence --cp-size 4

Pass ``--debug`` to report the concrete reason an incompatible backend was rejected
(rather than the generic "incompatible" message) and to print DEBUG-level
backend-selection logs. Pass ``--json`` to emit one JSON object per result row instead
of the formatted table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor

from cosmos_framework.model.attention import attention
from cosmos_framework.model.attention.backends import choose_backend
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.attention.utils import get_arch_tag

DTYPE_MAP: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
FLASHINFER_BACKENDS = frozenset({"flashinfer-trtllm-gen", "flashinfer-trtllm-gen-sage"})
FLASHINFER_SAGE_BACKEND = "flashinfer-trtllm-gen-sage"
FLASHINFER_WORKSPACE_BYTES = 128 * 1024**2
SM100_COMPUTE_CAPABILITY = (10, 0)


@dataclass(frozen=True)
class LocalAttentionShape:
    """Per-rank attention shape after applying the CP shard transform to a global pair."""

    q_len: int
    kv_len: int
    num_q_heads: int
    num_kv_heads: int
    sequence_shard_index: int
    sequence_shard_start: int


@dataclass(frozen=True)
class BenchmarkRun:
    """Average per-forward timing result for one frontend or direct FlashInfer backend."""

    average_latency_ms: float
    peak_bytes: int
    error: str | None
    scope: str


@dataclass(frozen=True)
class FlashInferApis:
    """Lazily imported FlashInfer entrypoints used by the optional benchmark paths."""

    ragged_attention: Callable[..., Tensor]
    trtllm_gen_module: object
    get_device_sm_count: Callable[[torch.device], int]


@dataclass
class FlashInferSageRunner:
    """Reusable SAGE buffers that model the steady-state Cosmos adapter scope."""

    ragged_attention: Callable[..., Tensor]
    sage_quantize: Callable[..., object]
    sm_count: int
    workspace: Tensor  # [workspace_bytes]
    seq_lens: Tensor  # [B]
    cumulative_query: Tensor  # [B+1]
    cumulative_key_value: Tensor  # [B+1]
    q_quant: Tensor  # [S_q,H_q,D]
    k_quant: Tensor  # [S_kv,H_kv,D]
    v_quant: Tensor  # [S_kv,H_kv,D]
    q_scale: Tensor  # [H_q,S_q]
    k_scale: Tensor  # [H_kv,ceil(S_kv/K_block)]
    v_scale: Tensor  # [H_kv,D]
    output: Tensor  # [S_q,H_q,D]

    def run(
        self,
        query: Tensor,  # [1,S_q,H_q,D]
        key: Tensor,  # [1,S_kv,H_kv,D]
        value: Tensor,  # [1,S_kv,H_kv,D]
        q_block_size: int,
        k_block_size: int,
        query_tokens: int,
        key_value_tokens: int,
        head_dim: int,
    ) -> Tensor:  # [S_q,H_q,D]
        """Run the SAGE adapter's full steady-state per-call work."""
        # FlashInfer consumes THD. These conversions intentionally remain in
        # the timed SAGE path because Cosmos AR attention provides BSHD Q/K/V.
        query_thd = query.squeeze(0).contiguous()  # [S_q,H_q,D]
        key_thd = key.squeeze(0).contiguous()  # [S_kv,H_kv,D]
        value_thd = value.squeeze(0).contiguous()  # [S_kv,H_kv,D]
        self.sage_quantize(
            self.q_quant,
            self.k_quant,
            self.v_quant,
            self.q_scale,
            self.k_scale,
            self.v_scale,
            query_thd,
            key_thd,
            value_thd,
            q_block_size,
            k_block_size,
            self.sm_count,
        )
        return run_flashinfer_attention(
            ragged_attention=self.ragged_attention,
            query=self.q_quant,
            key=self.k_quant,
            value=self.v_quant,
            workspace=self.workspace,
            seq_lens=self.seq_lens,
            cumulative_query=self.cumulative_query,
            cumulative_key_value=self.cumulative_key_value,
            query_tokens=query_tokens,
            key_value_tokens=key_value_tokens,
            head_dim=head_dim,
            sage_scales=(self.q_scale, self.k_scale, None, self.v_scale),
            sage_block_elements=(q_block_size, k_block_size, 0, 1),
            out=self.output,
        )


def local_shape_for_pair(
    q_len: int,
    kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    shard_mode: str,
    cp_size: int,
    rank: int,
) -> LocalAttentionShape:
    """Transform a global ``(q_len, kv_len, heads)`` pair into the LOCAL per-rank shape.

    - ``none``: identity (cp_size must be 1).
    - ``head``: divide Q and KV heads by ``cp_size``, keep the full KV length. All ranks
      see the same local shape (only head counts shrink).
    - ``sequence``: keep the heads, split the KV length across ``cp_size`` shards; the
      local KV length depends on ``rank % cp_size`` (remainder distributed to the low
      shards). Q is intentionally not divided.
    """
    if shard_mode == "none":
        if cp_size != 1:
            raise ValueError(f"--shard-mode none requires --cp-size 1, got {cp_size}")
        return LocalAttentionShape(q_len, kv_len, num_q_heads, num_kv_heads, 0, 0)

    if shard_mode == "head":
        if num_q_heads % cp_size != 0:
            raise ValueError(f"num_q_heads={num_q_heads} must be divisible by cp_size={cp_size}")
        if num_kv_heads % cp_size != 0:
            raise ValueError(f"num_kv_heads={num_kv_heads} must be divisible by cp_size={cp_size}")
        local_q_heads = num_q_heads // cp_size
        local_kv_heads = num_kv_heads // cp_size
        if local_q_heads % local_kv_heads != 0:
            raise ValueError(f"local_q_heads={local_q_heads} must be divisible by local_kv_heads={local_kv_heads}")
        return LocalAttentionShape(q_len, kv_len, local_q_heads, local_kv_heads, 0, 0)

    if shard_mode == "sequence":
        shard_index = rank % cp_size
        base_kv_len = kv_len // cp_size
        remainder = kv_len % cp_size
        local_kv_len = base_kv_len + int(shard_index < remainder)
        if local_kv_len <= 0:
            raise ValueError(
                f"sequence-sharded local_kv_len must be positive, got {local_kv_len}; "
                f"kv_len={kv_len}, cp_size={cp_size}, shard_index={shard_index}"
            )
        sequence_shard_start = shard_index * base_kv_len + min(shard_index, remainder)
        return LocalAttentionShape(q_len, local_kv_len, num_q_heads, num_kv_heads, shard_index, sequence_shard_start)

    raise ValueError(f"Unsupported shard_mode={shard_mode!r}")


def attention_flops(
    batch: int,
    q_len: int,
    kv_len: int,
    num_q_heads: int,
    head_dim: int,
    head_dim_v: int,
    is_causal: bool,
    include_backward: bool,
) -> float:
    """
    Approximate attention FLOPs (supports q_len != kv_len for cross-attention).

    Forward is 2 GEMMs (QK^T and P@V); each multiply-add counts as 2 FLOPs.
    Backward is ~2x the forward cost. Causal masking roughly halves the work
    (only applicable when q_len == kv_len).
    """
    qk = 2.0 * batch * num_q_heads * q_len * kv_len * head_dim  # [B,Hq,Sq,Skv] scores
    pv = 2.0 * batch * num_q_heads * q_len * kv_len * head_dim_v  # [B,Hq,Sq,Dv] output
    flops = qk + pv
    if is_causal:
        flops *= 0.5
    if include_backward:
        flops *= 3.0  # fwd (1x) + bwd (~2x)
    return flops


def make_inputs(
    batch: int,
    q_len: int,
    kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    head_dim_v: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create heads-last dense QKV tensors (q_len may differ from kv_len)."""
    q = torch.randn(
        batch, q_len, num_q_heads, head_dim, dtype=dtype, device=device, generator=generator
    ).requires_grad_(requires_grad)  # [B,Sq,Hq,D]
    k = torch.randn(
        batch, kv_len, num_kv_heads, head_dim, dtype=dtype, device=device, generator=generator
    ).requires_grad_(requires_grad)  # [B,Skv,Hkv,D]
    v = torch.randn(
        batch, kv_len, num_kv_heads, head_dim_v, dtype=dtype, device=device, generator=generator
    ).requires_grad_(requires_grad)  # [B,Skv,Hkv,Dv]
    return q, k, v


def load_flashinfer_apis() -> FlashInferApis:
    """Load the optional FlashInfer APIs only when one of its benchmark paths is selected."""
    try:
        from flashinfer.prefill import get_trtllm_gen_fmha_module, trtllm_ragged_attention_deepseek
        from flashinfer.utils import get_device_sm_count
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "FlashInfer is unavailable. Create the Cosmos3 environment with "
            "--group=flashinfer before selecting a flashinfer-trtllm-gen backend."
        ) from error
    return FlashInferApis(
        ragged_attention=trtllm_ragged_attention_deepseek,
        trtllm_gen_module=get_trtllm_gen_fmha_module(),
        get_device_sm_count=get_device_sm_count,
    )


def flashinfer_compatibility_reason(
    q: Tensor,  # [B,S_q,H_q,D]
    k: Tensor,  # [B,S_kv,H_kv,D]
    v: Tensor,  # [B,S_kv,H_kv,D]
    *,
    is_causal: bool,
    include_backward: bool,
    compile_fn: bool,
) -> str | None:
    """Return an unmet direct-FlashInfer benchmark constraint, if any.

    Supported inputs are GB200/SM100, BF16, batch-one, non-causal, forward-only
    attention with matching Q/K/V head dimensions and grouped-query-compatible
    head counts. Direct FlashInfer calls are not benchmarked with ``--compile``.
    """
    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) != SM100_COMPUTE_CAPABILITY:
        return "requires a GB200/SM100 CUDA device"
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        return "requires --dtype bf16"
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        return "requires --batch 1"
    if v.shape[-1] != q.shape[-1]:
        return "this FlashInfer benchmark path requires --head-dim-v to equal --head-dim"
    if q.shape[1] <= 0 or k.shape[1] <= 0:
        return "requires non-empty Q and KV sequences"
    if q.shape[2] % k.shape[2] != 0:
        return "requires Q heads divisible by KV heads"
    if is_causal:
        return "requires non-causal attention"
    if include_backward:
        return "requires forward-only attention"
    if compile_fn:
        return (
            "does not support --compile: FlashInfer's public TRTLLM-GEN wrapper "
            "lazily loads a TVM-FFI artifact that Dynamo cannot trace; run this "
            "benchmark eagerly"
        )
    return None


def run_flashinfer_attention(
    ragged_attention: Callable[..., Tensor],
    query: Tensor,  # [S_q,H_q,D]
    key: Tensor,  # [S_kv,H_kv,D]
    value: Tensor,  # [S_kv,H_kv,D]
    workspace: Tensor,  # [workspace_bytes]
    seq_lens: Tensor,  # [B]
    cumulative_query: Tensor,  # [B+1]
    cumulative_key_value: Tensor,  # [B+1]
    query_tokens: int,
    key_value_tokens: int,
    head_dim: int,
    sage_scales: tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None],
    sage_block_elements: tuple[int, int, int, int],
    *,
    out: Tensor | None = None,  # [S_q,H_q,D] or None
) -> Tensor:  # [S_q,H_q,D]
    """Dispatch one dense or already-quantized SAGE FlashInfer TRTLLM call."""
    return ragged_attention(
        query,
        key,
        value,
        workspace,
        seq_lens,
        query_tokens,
        key_value_tokens,
        1.0 / math.sqrt(head_dim),
        1.0,
        1.0,
        1,
        -1,
        cumulative_query,
        cumulative_key_value,
        enable_pdl=None,
        is_causal=False,
        return_lse=False,
        sage_attn_sfs=sage_scales,
        num_elts_per_sage_attn_blk=sage_block_elements,
        out=out,
        backend="trtllm-gen",
    )


def compatibility_reason(
    q: Tensor,  # [B,S,Hq,D]
    k: Tensor,  # [B,S,Hkv,D]
    v: Tensor,  # [B,S,Hkv,Dv]
    backend: str | None,
    is_causal: bool,
) -> str | None:
    """
    Surface the concrete reason a backend was rejected for this use case / device.

    The frontend runs backend selection with ``raise_error=False`` and only reports a
    generic "incompatible" message, swallowing the underlying reason. Re-run
    ``choose_backend`` with ``raise_error=True`` so named backends retain their
    concrete validator error and ``backend=None`` preserves frontend ``auto``
    selection.

    Returns the reason string, or None if the backend is compatible.
    """
    causal_type = CausalType.DontCare if is_causal else None
    try:
        choose_backend(
            query_shape=q.shape,
            key_shape=k.shape,
            value_shape=v.shape,
            dtype=q.dtype,
            device=q.device,
            requires_grad=q.requires_grad or k.requires_grad or v.requires_grad,
            is_causal=is_causal,
            causal_type=causal_type,
            is_varlen=False,
            deterministic=False,
            backend=backend,
            raise_error=True,
        )
        return None
    except Exception as e:  # noqa: BLE001 - surface the concrete rejection reason
        return f"{type(e).__name__}: {e}"


def run_once(
    q: Tensor,  # [B,S,Hq,D]
    k: Tensor,  # [B,S,Hkv,D]
    v: Tensor,  # [B,S,Hkv,Dv]
    backend: str | None,
    is_causal: bool,
    include_backward: bool,
) -> Tensor:
    """Run a single forward (and optional backward) attention call.

    Returns the attention output. The caller MUST keep/consume the returned tensor:
    under ``torch.compile`` an unused output lets Inductor dead-code-eliminate the whole
    attention (yielding absurd, ~PFLOP/s "timings"), so the output is returned as a graph
    output to keep the kernel live.
    """
    causal_type = CausalType.DontCare if is_causal else None
    out = attention(
        query=q,
        key=k,
        value=v,
        is_causal=is_causal,
        causal_type=causal_type,
        backend=backend,
        return_lse=False,
    )  # [B,S,Hq,Dv]
    if include_backward:
        grad = torch.ones_like(out)  # [B,S,Hq,Dv]
        out.backward(grad)
    return out


def benchmark_backend(
    q: Tensor,  # [B,S,Hq,D]
    k: Tensor,  # [B,S,Hkv,D]
    v: Tensor,  # [B,S,Hkv,Dv]
    backend: str | None,
    is_causal: bool,
    include_backward: bool,
    warmup: int,
    iters: int,
    compile_fn: bool,
    shard_mode: str,
    cp_size: int,
    debug: bool,
) -> BenchmarkRun:
    """
    Time a single backend on a single (already-sharded) input shape.

    If the backend is incompatible or errors out, ``error`` describes why. When
    ``debug`` is set, an incompatible backend's concrete rejection reason is
    reported instead of the generic frontend message.

    The forward (+ optional backward) is wrapped in an NVTX range so the timed region is
    easy to find in Nsight Systems. Backward requires grad, so ``inference_mode`` is only
    used for the forward-only path. ``compile_fn`` compiles the call before timing.

    Returns:
        ``BenchmarkRun`` containing per-iteration latency and peak allocation on
        success, or an error string when the selected backend is incompatible or
        raises during setup or execution.
    """
    if debug:
        reason = compatibility_reason(q, k, v, backend, is_causal)
        if reason is not None:
            return BenchmarkRun(0.0, 0, reason, "attention only")

    def _call() -> Tensor:
        return run_once(q, k, v, backend, is_causal, include_backward)

    call = torch.compile(_call, fullgraph=not include_backward) if compile_fn else _call

    # inference_mode is incompatible with autograd; only use it for forward-only timing.
    ctx = torch.inference_mode() if not include_backward else torch.enable_grad()

    if dist.is_initialized():
        dist.barrier()

    # Sink holding the latest output so Inductor cannot dead-code-eliminate the attention
    # (an unused compiled output would be optimized away, giving impossible TFLOP/s).
    sink: Tensor | None = None
    try:
        with ctx:
            for _ in range(warmup):
                sink = call()
            torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats(q.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.nvtx.range_push(f"fmha.{shard_mode}.cp{cp_size}.{backend or 'auto'}")
            # Enqueue all measured forwards before synchronizing. This avoids CPU/GPU
            # lockstep between individual launches; event time divided by ``iters`` is
            # the steady-state average latency per forward.
            start.record()
            for _ in range(iters):
                sink = call()
            end.record()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            average_latency_ms = start.elapsed_time(end) / iters
            # Force a host-side read of the result so the compiled output is genuinely
            # materialized (belt-and-suspenders against dead-code elimination).
            if sink is not None:
                _ = sink.sum().item()
    except Exception as e:  # noqa: BLE001 - report and skip incompatible backends
        return BenchmarkRun(0.0, 0, f"{type(e).__name__}: {e}", "attention only")

    peak_bytes = torch.cuda.max_memory_allocated(q.device)
    return BenchmarkRun(average_latency_ms, peak_bytes, None, "attention only")


def benchmark_flashinfer_backend(
    q: Tensor,  # [B,S_q,H_q,D]
    k: Tensor,  # [B,S_kv,H_kv,D]
    v: Tensor,  # [B,S_kv,H_kv,D]
    backend: str,
    *,
    is_causal: bool,
    include_backward: bool,
    compile_fn: bool,
    warmup: int,
    iters: int,
    q_block_size: int,
    k_block_size: int,
    shard_mode: str,
    cp_size: int,
) -> BenchmarkRun:
    """Time a direct FlashInfer dense or full SAGE-adapter backend.

    Workspace, ragged metadata, and SAGE reusable buffers are allocated before
    warmup. Dense timing is the TRTLLM kernel call only; SAGE deliberately
    includes its BSHD-to-THD conversion and Q/K/V quantization every iteration.
    """
    scope = "attention only" if backend != FLASHINFER_SAGE_BACKEND else "layout+quant+metadata+attention"
    reason = flashinfer_compatibility_reason(
        q,
        k,
        v,
        is_causal=is_causal,
        include_backward=include_backward,
        compile_fn=compile_fn,
    )
    if reason is not None:
        return BenchmarkRun(0.0, 0, reason, scope)

    try:
        flashinfer_apis = load_flashinfer_apis()
        # Dense FlashInfer accepts THD. squeeze is a zero-copy view of BSHD
        # batch-one inputs, so it is intentionally outside the dense timing.
        query_thd = q.squeeze(0)  # [S_q,H_q,D]
        key_thd = k.squeeze(0)  # [S_kv,H_kv,D]
        value_thd = v.squeeze(0)  # [S_kv,H_kv,D]
        # TRTLLM-GEN uses caller-owned GPU scratch space for internal softmax statistics
        # and temporary kernel state. Allocate it once and reuse it across warmup and
        # timed calls: this is neither Cosmos KV-cache storage nor per-call allocation.
        workspace = torch.empty(FLASHINFER_WORKSPACE_BYTES, device=q.device, dtype=torch.uint8)  # [workspace_bytes]
        # FlashInfer's ragged API consumes device-resident request metadata. ``seq_lens``
        # holds each request's KV length; the two cumulative arrays are flattened-token
        # boundaries. With this benchmark's B=1 input, they are [KV_len], [0,Q_len],
        # and [0,KV_len], respectively.
        seq_lens = torch.tensor([k.shape[1]], device=q.device, dtype=torch.int32)  # [B]
        cumulative_query = torch.tensor([0, q.shape[1]], device=q.device, dtype=torch.int32)  # [B+1]
        cumulative_key_value = torch.tensor([0, k.shape[1]], device=q.device, dtype=torch.int32)  # [B+1]

        if backend == FLASHINFER_SAGE_BACKEND:
            sage_quantize = getattr(flashinfer_apis.trtllm_gen_module, "trtllm_sage_attention_quantize", None)
            if not callable(sage_quantize):
                return BenchmarkRun(0.0, 0, "FlashInfer TRTLLM-Gen module does not expose SAGE quantization", scope)
            sage_runner = FlashInferSageRunner(
                ragged_attention=flashinfer_apis.ragged_attention,
                sage_quantize=sage_quantize,
                sm_count=flashinfer_apis.get_device_sm_count(q.device),
                workspace=workspace,
                seq_lens=seq_lens,
                cumulative_query=cumulative_query,
                cumulative_key_value=cumulative_key_value,
                q_quant=torch.empty(
                    (q.shape[1], q.shape[2], q.shape[3]), device=q.device, dtype=torch.int8
                ),  # [S_q,H_q,D]
                k_quant=torch.empty(
                    (k.shape[1], k.shape[2], k.shape[3]), device=q.device, dtype=torch.int8
                ),  # [S_kv,H_kv,D]
                v_quant=torch.empty(
                    (v.shape[1], v.shape[2], v.shape[3]), device=q.device, dtype=torch.float8_e4m3fn
                ),  # [S_kv,H_kv,D]
                q_scale=torch.empty((q.shape[2], q.shape[1]), device=q.device, dtype=torch.float32),  # [H_q,S_q]
                k_scale=torch.empty(
                    (k.shape[2], math.ceil(k.shape[1] / k_block_size)), device=q.device, dtype=torch.float32
                ),  # [H_kv,ceil(S_kv/K_block)]
                v_scale=torch.empty((v.shape[2], v.shape[3]), device=q.device, dtype=torch.float32),  # [H_kv,D]
                output=torch.empty((q.shape[1], q.shape[2], q.shape[3]), device=q.device, dtype=q.dtype),  # [S_q,H_q,D]
            )

            def call() -> Tensor:  # [S_q,H_q,D]
                return sage_runner.run(q, k, v, q_block_size, k_block_size, q.shape[1], k.shape[1], q.shape[3])

        else:

            def call() -> Tensor:  # [S_q,H_q,D]
                return run_flashinfer_attention(
                    ragged_attention=flashinfer_apis.ragged_attention,
                    query=query_thd,
                    key=key_thd,
                    value=value_thd,
                    workspace=workspace,
                    seq_lens=seq_lens,
                    cumulative_query=cumulative_query,
                    cumulative_key_value=cumulative_key_value,
                    query_tokens=q.shape[1],
                    key_value_tokens=k.shape[1],
                    head_dim=q.shape[3],
                    sage_scales=(None, None, None, None),
                    sage_block_elements=(0, 0, 0, 0),
                )

        if dist.is_initialized():
            dist.barrier()
        sink: Tensor | None = None
        with torch.inference_mode():
            for _ in range(warmup):
                sink = call()
            torch.cuda.synchronize(q.device)
            torch.cuda.reset_peak_memory_stats(q.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.nvtx.range_push(f"fmha.{shard_mode}.cp{cp_size}.{backend}")
            # Enqueue all measured forwards before synchronizing. This avoids CPU/GPU
            # lockstep between individual launches; event time divided by ``iters`` is
            # the steady-state average latency per forward.
            start.record()
            for _ in range(iters):
                sink = call()
            end.record()
            torch.cuda.synchronize(q.device)
            torch.cuda.nvtx.range_pop()
            average_latency_ms = start.elapsed_time(end) / iters
            if sink is not None:
                _ = sink.sum().item()
    except Exception as error:  # noqa: BLE001 - report optional-backend setup/runtime failures in the table
        return BenchmarkRun(0.0, 0, f"{type(error).__name__}: {error}", scope)

    return BenchmarkRun(average_latency_ms, torch.cuda.max_memory_allocated(q.device), None, scope)


def format_row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(cell.ljust(w) for cell, w in zip(cells, widths))


def init_distributed() -> tuple[int, int, int]:
    """Read torchrun env (defaults to single process) and init NCCL when world_size > 1."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    return rank, local_rank, world_size


def resolve_seqlen_pair(seqlen: int, kv_seqlen: int | None, causal: bool) -> tuple[int, int]:
    """Resolve the single (q_len, kv_len) shape. Causal masking here uses
    CausalType.DontCare, which requires q_len == kv_len, so a distinct KV length is only
    allowed when not causal."""
    if kv_seqlen is None:
        return (seqlen, seqlen)
    if causal:
        raise SystemExit(
            "--kv-seqlen is only valid without --causal: causal masking here uses CausalType.DontCare, "
            "which requires q_len == kv_len. Drop --kv-seqlen, or drop --causal."
        )
    return (seqlen, kv_seqlen)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark cosmos_framework.model.attention (dense / non-varlen only) across backends, "
        "sequence lengths, and optional context-parallel shard shapes."
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["auto"],
        help="Backends to benchmark. Use 'auto' to let the frontend choose. "
        "Frontend choices include flash2, flash3, natten, cudnn. Benchmark-only "
        "FlashInfer choices: flashinfer-trtllm-gen, flashinfer-trtllm-gen-sage.",
    )
    parser.add_argument("--seqlen", type=int, default=4096, help="Query sequence length.")
    parser.add_argument(
        "--kv-seqlen",
        type=int,
        default=None,
        help="KV sequence length for cross-attention (q_len != kv_len). Only allowed when "
        "--causal is NOT set. If omitted, kv_len == q_len (self-attention).",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num-q-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--head-dim-v", type=int, default=None, help="Value head dim (defaults to --head-dim; set differently for MLA)."
    )
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="bf16")
    parser.add_argument(
        "--q-block-size",
        choices=(1, 4, 16),
        type=int,
        default=1,
        help="FlashInfer SAGE Q quantization block size; ignored by other backends.",
    )
    parser.add_argument(
        "--k-block-size",
        choices=(1, 4, 16),
        type=int,
        default=16,
        help="FlashInfer SAGE K quantization block size; ignored by other backends.",
    )
    parser.add_argument("--causal", action="store_true", help="Enable causal masking (CausalType.DontCare).")
    parser.add_argument("--backward", action="store_true", help="Include a backward pass (training) in timing.")
    parser.add_argument(
        "--shard-mode",
        choices=("none", "head", "sequence"),
        default="none",
        help="Context-parallel per-rank shape to simulate. none: no sharding; "
        "head: divide Q/KV heads by --cp-size (full KV length); "
        "sequence: keep heads and split KV length across --cp-size shards.",
    )
    parser.add_argument(
        "--cp-size", type=int, default=1, help="Context-parallel sharding factor (>1 needs --shard-mode)."
    )
    parser.add_argument("--compile", action="store_true", help="torch.compile the attention call before benchmarking.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Number of forwards timed as one asynchronous batch; reports their average latency.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for input generation.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", action="store_true", help="Emit one JSON object per result row instead of a table.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Report the concrete reason a backend is rejected (instead of the generic "
        "'incompatible' message) and enable DEBUG-level backend-selection logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.debug:
        # Raise the logger to DEBUG so the per-backend selection diagnostics (emitted by the
        # attention checks via log.debug) are printed, then re-initialize the stdout sink.
        from cosmos_framework.utils import log

        log.LEVEL = "DEBUG"
        log.init_loguru_stdout()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this benchmark requires a GPU.")

    if args.cp_size < 1:
        raise SystemExit(f"--cp-size must be >= 1, got {args.cp_size}")
    if args.iters < 1:
        raise SystemExit(f"--iters must be >= 1, got {args.iters}")
    if args.shard_mode == "sequence" and args.causal:
        raise SystemExit(
            "--shard-mode sequence is incompatible with --causal: a KV-split shard holds a "
            "partial key range, so a causal mask over the local shard is not well-defined here."
        )

    rank, local_rank, world_size = init_distributed()
    is_main = rank == 0
    device = torch.device("cuda", local_rank) if args.device == "cuda" else torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]
    head_dim_v = args.head_dim_v if args.head_dim_v is not None else args.head_dim
    backends: list[str | None] = [None if b == "auto" else b for b in args.backends]
    generator = torch.Generator(device=device).manual_seed(args.seed)

    q_len, kv_len = resolve_seqlen_pair(args.seqlen, args.kv_seqlen, args.causal)
    local = local_shape_for_pair(
        q_len=q_len,
        kv_len=kv_len,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        shard_mode=args.shard_mode,
        cp_size=args.cp_size,
        rank=rank,
    )
    seq_label = str(local.q_len) if local.q_len == local.kv_len else f"{local.q_len}/{local.kv_len}"
    heads_label = f"{local.num_q_heads}/{local.num_kv_heads}"
    shard_label = "-" if args.shard_mode == "none" else f"{args.shard_mode}/{args.cp_size}"

    if is_main and not args.json:
        arch_tag = get_arch_tag(device)
        print(f"Device: {torch.cuda.get_device_name(device)} (arch_tag={arch_tag}, sm_{arch_tag})")
        print(
            f"Config: batch={args.batch} Hq={args.num_q_heads} Hkv={args.num_kv_heads} "
            f"D={args.head_dim} Dv={head_dim_v} dtype={args.dtype} causal={args.causal} "
            f"backward={args.backward} compile={args.compile}"
        )
        print(
            f"Shard: mode={args.shard_mode} cp_size={args.cp_size} world_size={world_size} | "
            f"Timing: warmup={args.warmup} iters={args.iters} (average per forward)\n"
        )
        headers = [
            "backend",
            "seq(q/kv)",
            "heads(q/kv)",
            "shard",
            "scope",
            "avg(ms)",
            "TFLOP/s",
            "tok/s",
            "peakMB",
            "status",
        ]
        widths = [25, 15, 12, 10, 33, 11, 9, 12, 9, 40]
        print(format_row(headers, widths))
        print(format_row(["-" * w for w in widths], widths))
    else:
        widths = [25, 15, 12, 10, 33, 11, 9, 12, 9, 40]

    for backend in backends:
        backend_label = backend if backend is not None else "auto"

        q, k, v = make_inputs(
            batch=args.batch,
            q_len=local.q_len,
            kv_len=local.kv_len,
            num_q_heads=local.num_q_heads,
            num_kv_heads=local.num_kv_heads,
            head_dim=args.head_dim,
            head_dim_v=head_dim_v,
            dtype=dtype,
            device=device,
            requires_grad=args.backward,
            generator=generator,
        )

        if backend in FLASHINFER_BACKENDS:
            run = benchmark_flashinfer_backend(
                q,
                k,
                v,
                backend,
                is_causal=args.causal,
                include_backward=args.backward,
                compile_fn=args.compile,
                warmup=args.warmup,
                iters=args.iters,
                q_block_size=args.q_block_size,
                k_block_size=args.k_block_size,
                shard_mode=args.shard_mode,
                cp_size=args.cp_size,
            )
        else:
            run = benchmark_backend(
                q,
                k,
                v,
                backend=backend,
                is_causal=args.causal,
                include_backward=args.backward,
                warmup=args.warmup,
                iters=args.iters,
                compile_fn=args.compile,
                shard_mode=args.shard_mode,
                cp_size=args.cp_size,
                debug=args.debug,
            )

        if not is_main:
            continue

        if run.error is not None:
            if args.json:
                print(json.dumps({"backend": backend_label, "seq": seq_label, "scope": run.scope, "status": run.error}))
            else:
                print(
                    format_row(
                        [
                            backend_label,
                            seq_label,
                            heads_label,
                            shard_label,
                            run.scope,
                            "-",
                            "-",
                            "-",
                            "-",
                            run.error,
                        ],
                        widths,
                    )
                )
            continue

        average_latency_ms = run.average_latency_ms
        flops = attention_flops(
            batch=args.batch,
            q_len=local.q_len,
            kv_len=local.kv_len,
            num_q_heads=local.num_q_heads,
            head_dim=args.head_dim,
            head_dim_v=head_dim_v,
            is_causal=args.causal,
            include_backward=args.backward,
        )
        tflops = None if backend == FLASHINFER_SAGE_BACKEND else flops / (average_latency_ms * 1e-3) / 1e12
        tokens_per_s = (args.batch * local.q_len) / (average_latency_ms * 1e-3)
        peak_mb = run.peak_bytes / (1024**2)

        if args.json:
            print(
                json.dumps(
                    {
                        "backend": backend_label,
                        "q_len": local.q_len,
                        "kv_len": local.kv_len,
                        "num_q_heads": local.num_q_heads,
                        "num_kv_heads": local.num_kv_heads,
                        "shard_mode": args.shard_mode,
                        "cp_size": args.cp_size,
                        "average_latency_ms": average_latency_ms,
                        "tflops": tflops,
                        "tokens_per_s": tokens_per_s,
                        "peak_mb": peak_mb,
                        "scope": run.scope,
                        "status": "ok",
                    }
                )
            )
        else:
            print(
                format_row(
                    [
                        backend_label,
                        seq_label,
                        heads_label,
                        shard_label,
                        run.scope,
                        f"{average_latency_ms:.4f}",
                        "-" if tflops is None else f"{tflops:.1f}",
                        f"{tokens_per_s:.0f}",
                        f"{peak_mb:.1f}",
                        "ok",
                    ],
                    widths,
                )
            )

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
