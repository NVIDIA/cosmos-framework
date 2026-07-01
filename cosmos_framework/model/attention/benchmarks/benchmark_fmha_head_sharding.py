# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Standalone FMHA microbenchmark for CP attention sharding characteristics.

This script isolates per-rank attention kernel shapes used by possible context
parallelism strategies. ``--shard-mode head`` matches the current
CP shape where ``--cp-size`` only changes the local head counts:

    CP=1: q [1,S_q,H,D],       k/v [1,S_kv,H_kv,D]
    CP=4: q [1,S_q,H/4,D],     k/v [1,S_kv,H_kv/4,D]

``--shard-mode sequence`` models the local split-KV kernel shape:

    CP=1: q [1,S_q,H,D],       k/v [1,S_kv,H_kv,D]
    CP=4: q [1,S_q,H,D],       k/v [1,S_kv/4,H_kv,D]

In sequence mode, Q is intentionally not divided. Each rank would compute a
partial result for the same query tokens over a local KV shard, and a complete
distributed implementation would merge partial outputs using LSEs. This
benchmark times only the local attention kernel.

There is intentionally no all-to-all, all-gather, or all-reduce in the timed
region since the intent is to study kernel scalability.

Example production-like one-GPU run:

    torchrun --standalone --nproc_per_node=1 \
        -m cosmos_framework.model.attention.benchmarks.benchmark_fmha_head_sharding \
        --cp-size 1 \
        --q-len 396 \
        --kv-len 177771 \
        --num-q-heads 32 \
        --num-kv-heads 8 \
        --head-dim 128 \
        --warmup-iters 20 \
        --iters 100

Example true four-GPU CP run:

    torchrun --standalone --nproc_per_node=4 \
        -m cosmos_framework.model.attention.benchmarks.benchmark_fmha_head_sharding \
        --shard-mode head \
        --cp-size 4 \
        --q-len 396 \
        --kv-len 177771 \
        --num-q-heads 32 \
        --num-kv-heads 8 \
        --head-dim 128 \
        --warmup-iters 20 \
        --iters 100

Example one-GPU local sequence-shard mock of a CP=4 split-KV attention shape:

    torchrun --standalone --nproc_per_node=1 \
        -m cosmos_framework.model.attention.benchmarks.benchmark_fmha_head_sharding \
        --shard-mode sequence \
        --cp-size 4 \
        --q-len 396 \
        --kv-len 177771 \
        --num-q-heads 32 \
        --num-kv-heads 8 \
        --head-dim 128 \
        --warmup-iters 20 \
        --iters 100

Example one-GPU local-head mock of the CP=4 per-rank attention shape:

    torchrun --standalone --nproc_per_node=1 \
        -m cosmos_framework.model.attention.benchmarks.benchmark_fmha_head_sharding \
        --shard-mode head \
        --cp-size 1 \
        --q-len 396 \
        --kv-len 177771 \
        --num-q-heads 8 \
        --num-kv-heads 2 \
        --head-dim 128 \
        --warmup-iters 20 \
        --iters 100

The mock uses one process and sets the global head counts to the local head
counts that a CP=4 rank would see. It creates q [1,396,8,128] and k/v
[1,177771,2,128] without launching four ranks. This isolates whether the
attention kernel scales with the smaller local-head shape, independent of
distributed launch overhead, communication, or multi-GPU contention.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.distributed as dist

from cosmos_framework.model.attention import attention


@dataclass(frozen=True)
class BenchmarkConfig:
    q_len: int
    kv_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    cp_size: int
    shard_mode: str
    batch_size: int
    warmup_iters: int
    iters: int
    dtype: str
    backend: str | None
    compile: bool
    seed: int


@dataclass(frozen=True)
class LocalAttentionShape:
    q_len: int
    kv_len: int
    num_q_heads: int
    num_kv_heads: int
    sequence_shard_index: int
    sequence_shard_start: int


def _parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-len", type=int, default=512, help="Current-frame query token count.")
    parser.add_argument("--kv-len", type=int, default=262144, help="Total key/value token count including history.")
    parser.add_argument("--num-q-heads", type=int, default=32, help="Global query head count before head sharding.")
    parser.add_argument("--num-kv-heads", type=int, default=8, help="Global KV head count before head sharding.")
    parser.add_argument("--head-dim", type=int, default=128, help="Per-head dimension.")
    parser.add_argument("--cp-size", type=int, default=1, help="Context-parallel sharding factor to simulate.")
    parser.add_argument(
        "--shard-mode",
        choices=("head", "sequence"),
        default="head",
        help=(
            "head: divide Q/KV heads by cp_size and keep full KV length; "
            "sequence: keep heads and split KV length across cp_size shards."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for the dense attention call.")
    parser.add_argument("--warmup-iters", type=int, default=20, help="Untimed warmup iterations.")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations.")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Q/K/V dtype.",
    )
    parser.add_argument("--backend", type=str, default=None, help="Optional cosmos_framework attention backend override.")
    parser.add_argument("--compile", action="store_true", help="Compile the attention call before benchmarking.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    args = parser.parse_args()
    return BenchmarkConfig(
        q_len=args.q_len,
        kv_len=args.kv_len,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        cp_size=args.cp_size,
        shard_mode=args.shard_mode,
        batch_size=args.batch_size,
        warmup_iters=args.warmup_iters,
        iters=args.iters,
        dtype=args.dtype,
        backend=args.backend,
        compile=args.compile,
        seed=args.seed,
    )


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype={name!r}")


def _init_rank() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    return rank, local_rank, world_size


def _validate_config(config: BenchmarkConfig, rank: int) -> LocalAttentionShape:
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.batch_size}")
    if config.q_len <= 0:
        raise ValueError(f"q_len must be positive, got {config.q_len}")
    if config.kv_len <= 0:
        raise ValueError(f"kv_len must be positive, got {config.kv_len}")
    if config.head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {config.head_dim}")
    if config.warmup_iters < 0:
        raise ValueError(f"warmup_iters must be non-negative, got {config.warmup_iters}")
    if config.iters <= 0:
        raise ValueError(f"iters must be positive, got {config.iters}")
    if config.cp_size <= 0:
        raise ValueError(f"cp_size must be positive, got {config.cp_size}")
    if config.num_q_heads % config.num_kv_heads != 0:
        raise ValueError(f"num_q_heads={config.num_q_heads} must be divisible by num_kv_heads={config.num_kv_heads}")
    if config.shard_mode == "head":
        if config.num_q_heads % config.cp_size != 0:
            raise ValueError(f"num_q_heads={config.num_q_heads} must be divisible by cp_size={config.cp_size}")
        if config.num_kv_heads % config.cp_size != 0:
            raise ValueError(f"num_kv_heads={config.num_kv_heads} must be divisible by cp_size={config.cp_size}")
        local_q_heads = config.num_q_heads // config.cp_size
        local_kv_heads = config.num_kv_heads // config.cp_size
        if local_q_heads % local_kv_heads != 0:
            raise ValueError(f"local_q_heads={local_q_heads} must be divisible by local_kv_heads={local_kv_heads}")
        return LocalAttentionShape(
            q_len=config.q_len,
            kv_len=config.kv_len,
            num_q_heads=local_q_heads,
            num_kv_heads=local_kv_heads,
            sequence_shard_index=0,
            sequence_shard_start=0,
        )
    if config.shard_mode == "sequence":
        shard_index = rank % config.cp_size
        base_kv_len = config.kv_len // config.cp_size
        remainder = config.kv_len % config.cp_size
        local_kv_len = base_kv_len + int(shard_index < remainder)
        if local_kv_len <= 0:
            raise ValueError(
                f"sequence-sharded local_kv_len must be positive, got {local_kv_len}; "
                f"kv_len={config.kv_len}, cp_size={config.cp_size}, shard_index={shard_index}"
            )
        sequence_shard_start = shard_index * base_kv_len + min(shard_index, remainder)
        return LocalAttentionShape(
            q_len=config.q_len,
            kv_len=local_kv_len,
            num_q_heads=config.num_q_heads,
            num_kv_heads=config.num_kv_heads,
            sequence_shard_index=shard_index,
            sequence_shard_start=sequence_shard_start,
        )
    raise ValueError(f"Unsupported shard_mode={config.shard_mode!r}")


def _make_inputs(
    config: BenchmarkConfig,
    local_shape: LocalAttentionShape,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    query = torch.randn(
        config.batch_size,
        local_shape.q_len,
        local_shape.num_q_heads,
        config.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )  # [B,S_q,H_local,D]
    key = torch.randn(
        config.batch_size,
        local_shape.kv_len,
        local_shape.num_kv_heads,
        config.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )  # [B,S_kv,H_kv_local,D]
    value = torch.randn(
        config.batch_size,
        local_shape.kv_len,
        local_shape.num_kv_heads,
        config.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )  # [B,S_kv,H_kv_local,D]
    return query, key, value


def _run_attention(
    query: torch.Tensor,  # [B,S_q,H_local,D]
    key: torch.Tensor,  # [B,S_kv,H_kv_local,D]
    value: torch.Tensor,  # [B,S_kv,H_kv_local,D]
    config: BenchmarkConfig,
) -> torch.Tensor:  # [B,S_q,H_local,D]
    out = attention(
        query=query,
        key=key,
        value=value,
        is_causal=False,
        backend=config.backend,
        return_lse=False,
    )  # [B,S_q,H_local,D]
    assert isinstance(out, torch.Tensor)
    return out


def _benchmark(
    query: torch.Tensor,  # [B,S_q,H_local,D]
    key: torch.Tensor,  # [B,S_kv,H_kv_local,D]
    value: torch.Tensor,  # [B,S_kv,H_kv_local,D]
    config: BenchmarkConfig,
) -> dict[str, Any]:
    def run_once(
        local_query: torch.Tensor,  # [B,S_q,H_local,D]
        local_key: torch.Tensor,  # [B,S_kv,H_kv_local,D]
        local_value: torch.Tensor,  # [B,S_kv,H_kv_local,D]
    ) -> torch.Tensor:  # [B,S_q,H_local,D]
        return _run_attention(local_query, local_key, local_value, config)  # [B,S_q,H_local,D]

    if config.compile:
        run_once = torch.compile(run_once, fullgraph=True)

    with torch.inference_mode():
        for _ in range(config.warmup_iters):
            _warmup_out = run_once(query, key, value)  # [B,S_q,H_local,D]
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats(query.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.nvtx.range_push(f"fmha_{config.shard_mode}_sharding.cp{config.cp_size}")
        start.record()
        for _ in range(config.iters):
            out = run_once(query, key, value)  # [B,S_q,H_local,D]
        end.record()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    tokens = config.batch_size * config.q_len
    checksum = float(out.float().mean().item())  # []
    return {
        "elapsed_ms": elapsed_ms,
        "iters": config.iters,
        "avg_ms": elapsed_ms / config.iters,
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(query.device),
        "tokens_per_second": (tokens * config.iters) / (elapsed_ms / 1000.0),
        "checksum": checksum,
    }


def main() -> None:
    config = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this FMHA benchmark")
    rank, local_rank, world_size = _init_rank()
    try:
        local_shape = _validate_config(config, rank)
        device = torch.device("cuda", local_rank)
        dtype = _dtype_from_name(config.dtype)
        query, key, value = _make_inputs(config, local_shape, device, dtype)
        # query: [B,S_q,H_local,D], key/value: [B,S_kv,H_kv_local,D]

        if dist.is_initialized():
            dist.barrier()
        result = _benchmark(query, key, value, config)
        if dist.is_initialized():
            dist.barrier()

        payload = {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "device": torch.cuda.get_device_name(device),
            "config": asdict(config),
            "local_shape": asdict(local_shape),
            "local_q_heads": local_shape.num_q_heads,
            "local_kv_heads": local_shape.num_kv_heads,
            "local_q_len": local_shape.q_len,
            "local_kv_len": local_shape.kv_len,
            "query_shape": list(query.shape),
            "key_shape": list(key.shape),
            "value_shape": list(value.shape),
            "result": result,
        }
        if rank == 0:
            print(json.dumps(payload, sort_keys=True))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
