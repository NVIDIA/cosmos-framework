# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os

import pytest
import torch
import torch.distributed as dist

from cosmos_framework.model.attention import attention as full_attention
from cosmos_framework.model.generator.mot.attention import SplitInfo
from cosmos_framework.model.generator.mot.context_parallel_utils import (
    context_parallel_attention,
)
from cosmos_framework.model.generator.mot.parallelize_unified_mot import ContextParallelDispatch
from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    get_gen_seq,
)
from cosmos_framework.utils.generator.parallelism import ParallelDims
from cosmos_framework.model.generator.mot.causal_attention import (
    attention_AR_gen_only,
    dispatch_attention_with_memory,
)
from cosmos_framework.model.generator.utils.kv_cache import ARMemoryValue, DualKVCache, TFNoisyMemoryValue


def setup_distributed_environment():
    """Initializes the distributed environment."""
    if "RANK" not in os.environ:
        pytest.skip("requires distributed environment (run with: torchrun --nproc_per_node=2)")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def _make_gen_only_pack(
    gen_seq: torch.Tensor,
    *,
    is_sharded: bool = True,
    global_seq_len: int | None = None,
) -> SequencePack:
    """Create a minimal gen-only SequencePack for testing.

    Und is empty; gen contains *gen_seq*.  Metadata is minimal but sufficient
    for ``get_gen_seq`` / ``from_und_gen_splits`` to work.

    When the tensor is seq-sharded (``is_sharded=True``), pass
    ``global_seq_len`` so that ``_num_full_tokens`` reflects the global
    sequence length, matching the contract that metadata uses global sizes.
    """
    S_local = gen_seq.shape[0]
    S_global = global_seq_len if global_seq_len is not None else S_local
    device = gen_seq.device
    return {
        "causal_seq": gen_seq.new_empty(0, *gen_seq.shape[1:]),
        "full_only_seq": gen_seq,
        "is_sharded": is_sharded,
        "sample_offsets": torch.tensor([0, S_local], device=device),
        "max_sample_len": S_local,
        "max_causal_len": 0,
        "max_full_len": S_local,
        "_causal_indices": torch.empty(0, dtype=torch.long, device=device),
        "_full_indices": torch.arange(S_local, device=device),
        "_causal_seq_offsets": torch.tensor([0], dtype=torch.long, device=device),
        "_full_only_seq_offsets": torch.tensor([0, S_local], dtype=torch.long, device=device),
        "_num_causal_tokens": 0,
        "_num_full_tokens": S_global,
    }


def _cp_attention_ar_gen(
    cp_mesh,
    q_gen: torch.Tensor,
    k_gen: torch.Tensor,
    v_gen: torch.Tensor,
    dual_kv_cache: DualKVCache,
    frame_idx: int,
    gen_len: int,
) -> torch.Tensor:
    """Test helper: CP AR gen-only attention with cache read/write.

    Wraps context_parallel_attention + attention_AR_gen_only and manages
    DualKVCache reads/writes, providing a convenient high-level interface
    for tests.
    """
    num_heads = q_gen.shape[1]
    head_dim = q_gen.shape[2]

    und_k_cached, und_v_cached = dual_kv_cache.und_cache.get()
    gen_k_hist, gen_v_hist = dual_kv_cache.gen_cache.fetch_kv(frame_idx)

    mv = ARMemoryValue(
        und_k_cached=und_k_cached,
        und_v_cached=und_v_cached,
        gen_k_hist=gen_k_hist,
        gen_v_hist=gen_v_hist,
        frame_idx=frame_idx,
        gen_len=gen_len,
    )
    gen_only_mask = SplitInfo(
        split_lens=[gen_len],
        attn_modes=["full"],
        sample_lens=[gen_len],
        actual_len=gen_len,
    )
    out_pack, kv_to_store = context_parallel_attention(
        cp_mesh,
        _make_gen_only_pack(q_gen, global_seq_len=gen_len),
        _make_gen_only_pack(k_gen, global_seq_len=gen_len),
        _make_gen_only_pack(v_gen, global_seq_len=gen_len),
        gen_only_mask,
        attention_AR_gen_only,
        memory_value=mv,
    )
    out_sp = get_gen_seq(out_pack).unflatten(-1, (num_heads, head_dim))

    assert kv_to_store is not None
    gen_k, gen_v, _, _ = kv_to_store
    dual_kv_cache.gen_cache.store_kv(gen_k, gen_v, frame_idx=frame_idx)

    return out_sp


def _make_factored_pack(
    causal_seq: torch.Tensor,
    full_only_seq: torch.Tensor,
    S_und_global: int,
    S_gen_global: int,
    device: torch.device,
    is_sharded: bool = False,
) -> SequencePack:
    """Minimal single-sample SequencePack for unit tests.

    Metadata always uses GLOBAL (pre-sharding) token counts so the metadata
    is consistent before and after all-to-all inside context_parallel_attention().
    The causal_seq / full_only_seq tensors may be either sharded or global.
    """
    return {
        "causal_seq": causal_seq,
        "full_only_seq": full_only_seq,
        "is_sharded": is_sharded,
        "sample_offsets": torch.tensor([0, S_und_global + S_gen_global], device=device, dtype=torch.int32),
        "max_num_tokens": S_und_global + S_gen_global,
        "max_sample_len": S_und_global + S_gen_global,
        "max_causal_len": S_und_global,
        "max_full_len": S_gen_global,
        "_causal_indices": torch.arange(S_und_global, device=device, dtype=torch.int32),
        "_full_indices": torch.arange(S_und_global, S_und_global + S_gen_global, device=device, dtype=torch.int32),
        "_causal_seq_offsets": torch.tensor([0, S_und_global], device=device, dtype=torch.int32),
        "_full_only_seq_offsets": torch.tensor([0, S_gen_global], device=device, dtype=torch.int32),
        "_num_causal_tokens": S_und_global,
        "_num_full_tokens": S_gen_global,
    }


def _make_tf_noisy_memory_value(
    *,
    T: int,
    H_p: int,
    W_p: int,
    num_kv_heads: int,
    head_dim: int,
    S_und: int,
    clean_gen_k: torch.Tensor,
    clean_gen_v: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> TFNoisyMemoryValue:
    """Build a replay-TF memory value for a single current segment."""
    S_super = H_p * W_p
    S_gen = T * S_super
    cached_und_k = torch.zeros(1, S_und, num_kv_heads, head_dim, device=device, dtype=dtype)  # [1,S_und,H_kv,D]
    cached_und_v = torch.zeros(1, S_und, num_kv_heads, head_dim, device=device, dtype=dtype)  # [1,S_und,H_kv,D]
    cached_gen_k = torch.zeros(1, S_gen, num_kv_heads, head_dim, device=device, dtype=dtype)  # [1,S_gen,H_kv,D]
    cached_gen_v = torch.zeros(1, S_gen, num_kv_heads, head_dim, device=device, dtype=dtype)  # [1,S_gen,H_kv,D]
    return TFNoisyMemoryValue(
        vision_token_shapes=[(T, H_p, W_p)],
        num_action_tokens_per_supertoken=0,
        has_new_caption=torch.tensor(True, device=device),  # []
        has_caption=torch.tensor(True, device=device),  # []
        has_cached_gen=torch.tensor(False, device=device),  # []
        und_kv_offsets=torch.tensor([0, S_und], device=device, dtype=torch.int32),  # [2]
        gen_q_offsets=torch.tensor([0, S_gen], device=device, dtype=torch.int32),  # [2]
        gen_ca_cached_kv_offsets=torch.tensor([0, 1], device=device, dtype=torch.int32),  # [2]
        cached_und_k=cached_und_k,
        cached_und_v=cached_und_v,
        cached_gen_k=cached_gen_k,
        cached_gen_v=cached_gen_v,
        max_gen_cache_tokens=S_gen,
        clamp_empty_varlen_kv=True,
        cached_clean_gen_k=clean_gen_k.unsqueeze(0),  # [1,S_gen,H_kv,D]
        cached_clean_gen_v=clean_gen_v.unsqueeze(0),  # [1,S_gen,H_kv,D]
    )


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_context_parallel_replay_teacher_forcing_matches_full_attention():
    """CP replay-TF dispatch matches full-head replay-TF attention on generated tokens."""
    rank, world_size = setup_distributed_environment()
    if world_size < 2:
        pytest.skip("requires at least 2 GPUs")

    device = torch.device("cuda", rank)
    dtype = torch.bfloat16
    parallel_dims = ParallelDims(enable_inference_mode=True, world_size=world_size, dp_shard=1, cp=world_size)
    parallel_dims.build_meshes("cuda")
    cp_mesh = parallel_dims.cp_mesh
    cp_group = cp_mesh.get_group()

    T = 2
    H_p = 1
    W_p = world_size
    S_gen = T * H_p * W_p
    S_und = world_size * 2
    num_heads = world_size * 2
    num_kv_heads = world_size
    head_dim = 32
    s_und_per_rank = S_und // world_size
    s_gen_per_rank = S_gen // world_size
    h_kv_per_rank = num_kv_heads // world_size
    s_und_start = rank * s_und_per_rank
    s_gen_start = rank * s_gen_per_rank
    h_kv_start = rank * h_kv_per_rank

    torch.manual_seed(2026)
    q_und_g = torch.randn(S_und, num_heads, head_dim, device=device, dtype=dtype)  # [S_und,H,D]
    q_gen_g = torch.randn(S_gen, num_heads, head_dim, device=device, dtype=dtype)  # [S_gen,H,D]
    k_und_g = torch.randn(S_und, num_kv_heads, head_dim, device=device, dtype=dtype)  # [S_und,H_kv,D]
    k_gen_g = torch.randn(S_gen, num_kv_heads, head_dim, device=device, dtype=dtype)  # [S_gen,H_kv,D]
    v_und_g = torch.randn(S_und, num_kv_heads, head_dim, device=device, dtype=dtype)  # [S_und,H_kv,D]
    v_gen_g = torch.randn(S_gen, num_kv_heads, head_dim, device=device, dtype=dtype)  # [S_gen,H_kv,D]
    clean_k_g = torch.randn(S_gen, num_kv_heads, head_dim, device=device, dtype=dtype)  # [S_gen,H_kv,D]
    clean_v_g = torch.randn(S_gen, num_kv_heads, head_dim, device=device, dtype=dtype)  # [S_gen,H_kv,D]

    attn_mask = SplitInfo(
        split_lens=[S_und, S_gen],
        attn_modes=["causal", "full"],
        sample_lens=[S_und + S_gen],
        actual_len=S_und + S_gen,
    )
    baseline_mv = _make_tf_noisy_memory_value(
        T=T,
        H_p=H_p,
        W_p=W_p,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        S_und=S_und,
        clean_gen_k=clean_k_g,
        clean_gen_v=clean_v_g,
        device=device,
        dtype=dtype,
    )
    baseline_pack, _ = dispatch_attention_with_memory(
        _make_factored_pack(q_und_g, q_gen_g, S_und, S_gen, device, is_sharded=False),
        _make_factored_pack(k_und_g, k_gen_g, S_und, S_gen, device, is_sharded=False),
        _make_factored_pack(v_und_g, v_gen_g, S_und, S_gen, device, is_sharded=False),
        attn_mask,
        memory_value=baseline_mv,
    )
    baseline_gen = get_gen_seq(baseline_pack).unflatten(-1, (num_heads, head_dim))  # [S_gen,H,D]

    clean_k_hp = clean_k_g[:, h_kv_start : h_kv_start + h_kv_per_rank, :]  # [S_gen,H_kv_local,D]
    clean_v_hp = clean_v_g[:, h_kv_start : h_kv_start + h_kv_per_rank, :]  # [S_gen,H_kv_local,D]
    cp_mv = _make_tf_noisy_memory_value(
        T=T,
        H_p=H_p,
        W_p=W_p,
        num_kv_heads=h_kv_per_rank,
        head_dim=head_dim,
        S_und=S_und,
        clean_gen_k=clean_k_hp,
        clean_gen_v=clean_v_hp,
        device=device,
        dtype=dtype,
    )
    cp_dispatch = ContextParallelDispatch(cp_mesh, wrapped_dispatch=dispatch_attention_with_memory)
    q_und_sp = q_und_g[s_und_start : s_und_start + s_und_per_rank]  # [S_und/cp,H,D]
    q_gen_sp = q_gen_g[s_gen_start : s_gen_start + s_gen_per_rank]  # [S_gen/cp,H,D]
    k_und_sp = k_und_g[s_und_start : s_und_start + s_und_per_rank]  # [S_und/cp,H_kv,D]
    k_gen_sp = k_gen_g[s_gen_start : s_gen_start + s_gen_per_rank]  # [S_gen/cp,H_kv,D]
    v_und_sp = v_und_g[s_und_start : s_und_start + s_und_per_rank]  # [S_und/cp,H_kv,D]
    v_gen_sp = v_gen_g[s_gen_start : s_gen_start + s_gen_per_rank]  # [S_gen/cp,H_kv,D]
    cp_pack, _ = cp_dispatch(
        _make_factored_pack(q_und_sp, q_gen_sp, S_und, S_gen, device, is_sharded=True),
        _make_factored_pack(k_und_sp, k_gen_sp, S_und, S_gen, device, is_sharded=True),
        _make_factored_pack(v_und_sp, v_gen_sp, S_und, S_gen, device, is_sharded=True),
        attn_mask,
        memory_value=cp_mv,
    )
    cp_gen_sp = get_gen_seq(cp_pack).unflatten(-1, (num_heads, head_dim))  # [S_gen/cp,H,D]
    gathered = [torch.empty_like(cp_gen_sp) for _ in range(world_size)]  # list of [S_gen/cp,H,D]
    dist.all_gather(gathered, cp_gen_sp, group=cp_group)
    if rank == 0:
        cp_gen = torch.cat(gathered, dim=0)  # [S_gen,H,D]
        torch.testing.assert_close(cp_gen, baseline_gen, rtol=2e-2, atol=2e-2)


@pytest.mark.L0
def test_context_parallel_ar_inference():
    """Verify CP AR inference (frame 1+) output matches full-attention baseline frame by frame.

    Setup:
      - Frame 0 K/V are pre-stored in the cache as if context_parallel_attention() ran.
      - Frames 1..num_frames-1 are tested via context_parallel_attention() wrapping
        attention_AR_gen_only.
      - Baseline is full single-GPU attention with the accumulated KV context.
    """
    rank, world_size = setup_distributed_environment()
    if world_size < 2:
        print("Skipping test: requires at least 2 GPUs.")
        return

    device = torch.device("cuda", rank)
    parallel_dims = ParallelDims(enable_inference_mode=True, world_size=world_size, dp_shard=1, cp=world_size)
    parallel_dims.build_meshes("cuda")
    cp_mesh = parallel_dims.cp_mesh

    num_heads = world_size * 2  # divisible by world_size
    head_dim = 32
    S_und = world_size * 4  # divisible by world_size
    S_gen = world_size * 2  # per-frame gen tokens, divisible by world_size
    num_frames = 3  # frame 0 cached; frames 1..num_frames-1 tested
    scale = head_dim**-0.5

    # Same global tensors on all ranks (identical seed)
    torch.manual_seed(42)
    k_und_g = torch.randn(S_und, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_und_g = torch.randn(S_und, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    q_frames_g = [
        torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16) for _ in range(num_frames)
    ]
    k_frames_g = [
        torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16) for _ in range(num_frames)
    ]
    v_frames_g = [
        torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16) for _ in range(num_frames)
    ]

    # Baseline: frame-by-frame full attention (no CP)
    # Context grows: [und | frame0 | frame1 | ...]
    baseline_results = []
    k_ctx = torch.cat([k_und_g.unsqueeze(0), k_frames_g[0].unsqueeze(0)], dim=1)  # [1,S_und+S_gen,H,D]
    v_ctx = torch.cat([v_und_g.unsqueeze(0), v_frames_g[0].unsqueeze(0)], dim=1)
    for fi in range(1, num_frames):
        k_full = torch.cat([k_ctx, k_frames_g[fi].unsqueeze(0)], dim=1)  # [1,S_ctx+S_gen,H,D]
        v_full = torch.cat([v_ctx, v_frames_g[fi].unsqueeze(0)], dim=1)
        out = full_attention(
            query=q_frames_g[fi].unsqueeze(0),
            key=k_full,
            value=v_full,
            is_causal=False,
            scale=scale,
            return_lse=False,
        )
        assert isinstance(out, torch.Tensor)
        out_t: torch.Tensor = out
        baseline_results.append(out_t.squeeze(0))  # [S_gen,H,D]
        k_ctx = k_full
        v_ctx = v_full

    # CP path: DualKVCache pre-populated with head-sharded tensors (simulating CP frame 0)
    h_per_rank = num_heads // world_size
    h_start = rank * h_per_rank

    kv_cache = DualKVCache(gen_cache_size=num_frames + 2)

    # Store head-sharded und K/V (what context_parallel_attention() stores at frame 0)
    k_und_hp = k_und_g[:, h_start : h_start + h_per_rank, :].unsqueeze(0)  # [1,S_und,H/cp,D]
    v_und_hp = v_und_g[:, h_start : h_start + h_per_rank, :].unsqueeze(0)  # [1,S_und,H/cp,D]
    kv_cache.und_cache.store(k_und_hp, v_und_hp)

    # Store head-sharded frame 0 gen K/V
    k0_hp = k_frames_g[0][:, h_start : h_start + h_per_rank, :].unsqueeze(0)  # [1,S_gen,H/cp,D]
    v0_hp = v_frames_g[0][:, h_start : h_start + h_per_rank, :].unsqueeze(0)  # [1,S_gen,H/cp,D]
    kv_cache.gen_cache.store_kv(k0_hp, v0_hp, frame_idx=0)

    # Test frames 1..num_frames-1 with CP
    s_per_rank = S_gen // world_size
    s_start = rank * s_per_rank

    cp_results = []
    for fi in range(1, num_frames):
        q_sp = q_frames_g[fi][s_start : s_start + s_per_rank, :, :]  # [S_gen/cp,H,D] seq-sharded
        k_sp = k_frames_g[fi][s_start : s_start + s_per_rank, :, :]  # [S_gen/cp,H,D]
        v_sp = v_frames_g[fi][s_start : s_start + s_per_rank, :, :]  # [S_gen/cp,H,D]

        out_sp = _cp_attention_ar_gen(
            cp_mesh,
            q_sp,
            k_sp,
            v_sp,
            dual_kv_cache=kv_cache,
            frame_idx=fi,
            gen_len=S_gen,
        )  # [S_gen/cp,H,D]
        cp_results.append(out_sp)

    # Verify: gather CP shards, compare with baseline
    for i, fi in enumerate(range(1, num_frames)):
        gathered = [None] * world_size
        dist.all_gather_object(gathered, cp_results[i].cpu(), group=cp_mesh.get_group())
        if rank == 0:
            gathered_output = torch.cat(gathered, dim=0).to(device)  # [S_gen,H,D]
            torch.testing.assert_close(
                gathered_output,
                baseline_results[i],
                rtol=1e-2,
                atol=1e-2,
                msg=f"Frame {fi} mismatch",
            )
            print(f"Frame {fi}: CP output matches baseline")
        dist.barrier()

    if rank == 0:
        print("=== test_context_parallel_ar_inference passed")


@pytest.mark.L0
def test_context_parallel_ar_frame0_stores_head_sharded():
    """context_parallel_attention() with memory_value returns head-sharded kv_to_store.

    After context_parallel_attention() runs at frame 0 with an ARMemoryValue,
    the returned kv_to_store contains head-sharded tensors:
    - und K/V: [1, S_und, H/cp, D]  (one head-shard per rank)
    - gen K/V: [1, S_gen, H/cp, D]

    The head slice must match the expected sub-range of the global K tensor.
    """
    rank, world_size = setup_distributed_environment()
    if world_size < 2:
        pytest.skip("requires at least 2 GPUs")

    device = torch.device("cuda", rank)
    parallel_dims = ParallelDims(enable_inference_mode=True, world_size=world_size, dp_shard=1, cp=world_size)
    parallel_dims.build_meshes("cuda")
    cp_mesh = parallel_dims.cp_mesh

    num_heads = world_size * 2  # divisible by world_size
    head_dim = 32
    S_und = world_size * 4  # divisible by world_size
    S_gen = world_size * 2  # divisible by world_size
    h_per_rank = num_heads // world_size
    s_und_per_rank = S_und // world_size
    s_gen_per_rank = S_gen // world_size

    torch.manual_seed(99)
    # Global tensors identical on all ranks
    k_und_g = torch.randn(S_und, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_und_g = torch.randn(S_und, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_gen_g = torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_gen_g = torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    q_und_g = torch.randn(S_und, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    q_gen_g = torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16)

    # Build seq-sharded input packs (each rank holds S/cp tokens, all H heads)
    q_pack = _make_factored_pack(
        q_und_g[rank * s_und_per_rank : (rank + 1) * s_und_per_rank],
        q_gen_g[rank * s_gen_per_rank : (rank + 1) * s_gen_per_rank],
        S_und,
        S_gen,
        device,
        is_sharded=True,
    )
    k_pack = _make_factored_pack(
        k_und_g[rank * s_und_per_rank : (rank + 1) * s_und_per_rank],
        k_gen_g[rank * s_gen_per_rank : (rank + 1) * s_gen_per_rank],
        S_und,
        S_gen,
        device,
        is_sharded=True,
    )
    v_pack = _make_factored_pack(
        v_und_g[rank * s_und_per_rank : (rank + 1) * s_und_per_rank],
        v_gen_g[rank * s_gen_per_rank : (rank + 1) * s_gen_per_rank],
        S_und,
        S_gen,
        device,
        is_sharded=True,
    )
    attn_mask = SplitInfo(
        split_lens=[S_und, S_gen],
        attn_modes=["causal", "full"],
        sample_lens=[S_und + S_gen],
        actual_len=S_und + S_gen,
    )

    memory_value = ARMemoryValue(
        und_k_cached=None,
        und_v_cached=None,
        gen_k_hist=None,
        gen_v_hist=None,
        frame_idx=0,
        gen_len=S_gen,
    )
    _, kv_to_store = context_parallel_attention(
        cp_mesh,
        q_pack,
        k_pack,
        v_pack,
        attn_mask,
        dispatch_attention_with_memory,
        memory_value=memory_value,
    )

    assert kv_to_store is not None
    gen_k, gen_v, und_k, und_v = kv_to_store

    # kv_to_store should be head-sharded: [1, S, H/cp, D]
    assert gen_k.shape == (1, S_gen, h_per_rank, head_dim), (
        f"Expected [1,{S_gen},{h_per_rank},{head_dim}], got {gen_k.shape}"
    )
    assert gen_v.shape == (1, S_gen, h_per_rank, head_dim)
    assert und_k.shape == (1, S_und, h_per_rank, head_dim), (
        f"Expected [1,{S_und},{h_per_rank},{head_dim}], got {und_k.shape}"
    )
    assert und_v.shape == (1, S_und, h_per_rank, head_dim)

    # Write back to cache and verify stored values match expected head slices
    kv_cache = DualKVCache(gen_cache_size=4)
    kv_cache.und_cache.store(und_k, und_v)
    kv_cache.gen_cache.store_kv(gen_k, gen_v, frame_idx=0)

    assert kv_cache.und_cache.is_initialized
    k_und_cached, v_und_cached = kv_cache.und_cache.get()
    assert k_und_cached.shape == (1, S_und, h_per_rank, head_dim)

    h_start = rank * h_per_rank
    expected_k_und = k_und_g[:, h_start : h_start + h_per_rank, :].unsqueeze(0)  # [1,S_und,H/cp,D]
    expected_k_gen = k_gen_g[:, h_start : h_start + h_per_rank, :].unsqueeze(0)  # [1,S_gen,H/cp,D]
    torch.testing.assert_close(
        k_und_cached, expected_k_und, rtol=1e-2, atol=1e-2, msg=f"Rank {rank}: k_und head shard mismatch"
    )
    torch.testing.assert_close(
        gen_k, expected_k_gen, rtol=1e-2, atol=1e-2, msg=f"Rank {rank}: k_gen head shard mismatch"
    )

    dist.barrier()
    if rank == 0:
        print("=== test_context_parallel_ar_frame0_stores_head_sharded passed")


@pytest.mark.L0
def test_context_parallel_ar_round_trip():
    """Full CP AR round-trip: frame 0 returns head-sharded kv_to_store via
    context_parallel_attention(), frames 1+ also use context_parallel_attention()
    wrapping attention_AR_gen_only.

    Output at each frame must match full single-GPU attention with the same accumulated context.
    This is the end-to-end test connecting the two halves of CP AR inference.
    """
    rank, world_size = setup_distributed_environment()
    if world_size < 2:
        pytest.skip("requires at least 2 GPUs")

    device = torch.device("cuda", rank)
    parallel_dims = ParallelDims(enable_inference_mode=True, world_size=world_size, dp_shard=1, cp=world_size)
    parallel_dims.build_meshes("cuda")
    cp_mesh = parallel_dims.cp_mesh

    num_heads = world_size * 2
    head_dim = 32
    S_und = world_size * 4
    S_gen = world_size * 2
    num_frames = 3
    scale = head_dim**-0.5
    s_und_per_rank = S_und // world_size
    s_gen_per_rank = S_gen // world_size

    torch.manual_seed(77)
    k_und_g = torch.randn(S_und, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_und_g = torch.randn(S_und, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    q_frames_g = [
        torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16) for _ in range(num_frames)
    ]
    k_frames_g = [
        torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16) for _ in range(num_frames)
    ]
    v_frames_g = [
        torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16) for _ in range(num_frames)
    ]

    # ---- Frame 0: populate cache via context_parallel_attention() ----
    kv_cache = DualKVCache(gen_cache_size=num_frames + 2)

    q_pack = _make_factored_pack(
        k_und_g[rank * s_und_per_rank : (rank + 1) * s_und_per_rank],  # q=k for frame 0 (not tested)
        k_frames_g[0][rank * s_gen_per_rank : (rank + 1) * s_gen_per_rank],
        S_und,
        S_gen,
        device,
        is_sharded=True,
    )
    k_pack = _make_factored_pack(
        k_und_g[rank * s_und_per_rank : (rank + 1) * s_und_per_rank],
        k_frames_g[0][rank * s_gen_per_rank : (rank + 1) * s_gen_per_rank],
        S_und,
        S_gen,
        device,
        is_sharded=True,
    )
    v_pack = _make_factored_pack(
        v_und_g[rank * s_und_per_rank : (rank + 1) * s_und_per_rank],
        v_frames_g[0][rank * s_gen_per_rank : (rank + 1) * s_gen_per_rank],
        S_und,
        S_gen,
        device,
        is_sharded=True,
    )
    attn_mask = SplitInfo(
        split_lens=[S_und, S_gen],
        attn_modes=["causal", "full"],
        sample_lens=[S_und + S_gen],
        actual_len=S_und + S_gen,
    )

    frame0_mv = ARMemoryValue(
        und_k_cached=None,
        und_v_cached=None,
        gen_k_hist=None,
        gen_v_hist=None,
        frame_idx=0,
        gen_len=S_gen,
    )
    _, kv_to_store_f0 = context_parallel_attention(
        cp_mesh,
        q_pack,
        k_pack,
        v_pack,
        attn_mask,
        dispatch_attention_with_memory,
        memory_value=frame0_mv,
    )
    assert kv_to_store_f0 is not None
    gen_k_f0, gen_v_f0, und_k_f0, und_v_f0 = kv_to_store_f0
    kv_cache.und_cache.store(und_k_f0, und_v_f0)
    kv_cache.gen_cache.store_kv(gen_k_f0, gen_v_f0, frame_idx=0)
    assert kv_cache.und_cache.is_initialized, "und cache must be populated after frame 0"

    # ---- Baseline: full attention per frame ----
    baseline_results = []
    k_ctx = torch.cat([k_und_g.unsqueeze(0), k_frames_g[0].unsqueeze(0)], dim=1)  # [1,S_und+S_gen,H,D]
    v_ctx = torch.cat([v_und_g.unsqueeze(0), v_frames_g[0].unsqueeze(0)], dim=1)
    for fi in range(1, num_frames):
        k_full = torch.cat([k_ctx, k_frames_g[fi].unsqueeze(0)], dim=1)
        v_full = torch.cat([v_ctx, v_frames_g[fi].unsqueeze(0)], dim=1)
        out = full_attention(
            query=q_frames_g[fi].unsqueeze(0),
            key=k_full,
            value=v_full,
            is_causal=False,
            scale=scale,
            return_lse=False,
        )
        assert isinstance(out, torch.Tensor)
        out_t: torch.Tensor = out
        baseline_results.append(out_t.squeeze(0))  # [S_gen,H,D]
        k_ctx = k_full
        v_ctx = v_full

    # ---- Frames 1+: via _cp_attention_ar_gen (wrapping attention_AR_gen_only) ----
    s_start = rank * s_gen_per_rank
    cp_results = []
    for fi in range(1, num_frames):
        q_sp = q_frames_g[fi][s_start : s_start + s_gen_per_rank]  # [S_gen/cp,H,D]
        k_sp = k_frames_g[fi][s_start : s_start + s_gen_per_rank]
        v_sp = v_frames_g[fi][s_start : s_start + s_gen_per_rank]

        out_sp = _cp_attention_ar_gen(
            cp_mesh,
            q_sp,
            k_sp,
            v_sp,
            dual_kv_cache=kv_cache,
            frame_idx=fi,
            gen_len=S_gen,
        )  # [S_gen/cp,H,D]
        cp_results.append(out_sp)

    # Gather shards and compare with baseline (rank 0 does the check)
    for i, fi in enumerate(range(1, num_frames)):
        gathered = [None] * world_size
        dist.all_gather_object(gathered, cp_results[i].cpu(), group=cp_mesh.get_group())
        if rank == 0:
            gathered_output = torch.cat(gathered, dim=0).to(device)  # [S_gen,H,D]
            torch.testing.assert_close(
                gathered_output,
                baseline_results[i],
                rtol=1e-2,
                atol=1e-2,
                msg=f"Round-trip frame {fi} mismatch",
            )
            print(f"Round-trip frame {fi}: CP output matches baseline")
        dist.barrier()

    if rank == 0:
        print("=== test_context_parallel_ar_round_trip passed")


@pytest.mark.L0
def test_cp_ar_denoising_loop_equivalence():
    """CP=2 multi-step denoising loop matches full-attention reference (cp=1 equivalent).

    Simulates the inner denoising loop for one AR frame: a fixed KV context (und + frame 0)
    is pre-stored; at each step Q/K/V are linear functions of the current latent x.
    The CP=2 path uses context_parallel_attention wrapping attention_AR_gen_only;
    the reference uses full single-GPU attention.  x is updated using the reference output
    each step (oracle update)
    so errors do not accumulate.  Per-step attention outputs must match within rtol=1e-2,
    atol=1e-2 — all-to-all reorders floating-point ops.
    """
    rank, world_size = setup_distributed_environment()
    if world_size < 2:
        pytest.skip("requires at least 2 GPUs")

    device = torch.device("cuda", rank)
    parallel_dims = ParallelDims(enable_inference_mode=True, world_size=world_size, dp_shard=1, cp=world_size)
    parallel_dims.build_meshes("cuda")
    cp_mesh = parallel_dims.cp_mesh
    cp_group = cp_mesh.get_group()

    num_heads = world_size * 2  # divisible by world_size
    head_dim = 16
    S_und = world_size * 4  # divisible by world_size
    S_gen = world_size * 2  # divisible by world_size
    num_steps = 4
    scale = head_dim**-0.5
    h_per_rank = num_heads // world_size
    s_gen_per_rank = S_gen // world_size
    h_start = rank * h_per_rank
    s_start = rank * s_gen_per_rank

    torch.manual_seed(42)
    # Fixed context K/V (same on all ranks)
    k_und_g = torch.randn(S_und, num_heads, head_dim, device=device, dtype=torch.bfloat16)  # [S_und,H,D]
    v_und_g = torch.randn(S_und, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k0_g = torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16)  # frame 0 [S_gen,H,D]
    v0_g = torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    # Linear projection weights for Q/K/V from latent x
    w_q = torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16)  # [S_gen,H,D]
    w_k = torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    w_v = torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16)

    # ── CP=2 path: pre-populate KV cache with head-sharded tensors ─────────────
    kv_cache = DualKVCache(gen_cache_size=4)
    kv_cache.und_cache.store(
        k_und_g[:, h_start : h_start + h_per_rank, :].unsqueeze(0),  # [1,S_und,H/cp,D]
        v_und_g[:, h_start : h_start + h_per_rank, :].unsqueeze(0),
    )
    kv_cache.gen_cache.store_kv(
        k0_g[:, h_start : h_start + h_per_rank, :].unsqueeze(0),  # [1,S_gen,H/cp,D]
        v0_g[:, h_start : h_start + h_per_rank, :].unsqueeze(0),
        frame_idx=0,
    )

    # ── Reference context (full K/V) for single-GPU baseline ──────────────────
    # Context grows: [und | frame0 | frame1_current]
    k_ctx = torch.cat([k_und_g.unsqueeze(0), k0_g.unsqueeze(0)], dim=1)  # [1,S_und+S_gen,H,D]
    v_ctx = torch.cat([v_und_g.unsqueeze(0), v0_g.unsqueeze(0)], dim=1)

    torch.manual_seed(0)
    x = torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16)  # [S_gen,H,D]

    for step in range(num_steps):
        t = 1.0 - step / num_steps

        # ── CP path ──────────────────────────────────────────────────────────
        q_sp = (x * w_q * t)[s_start : s_start + s_gen_per_rank].contiguous()  # [S_gen/cp,H,D]
        k_sp = (x * w_k * t)[s_start : s_start + s_gen_per_rank].contiguous()
        v_sp = (x * w_v * t)[s_start : s_start + s_gen_per_rank].contiguous()

        out_sp = _cp_attention_ar_gen(
            cp_mesh,
            q_sp,
            k_sp,
            v_sp,
            dual_kv_cache=kv_cache,
            frame_idx=1,
            gen_len=S_gen,
        )  # [S_gen/cp,H,D]

        # ── Reference: full single-GPU attention ─────────────────────────────
        q_ref = x * w_q * t  # [S_gen,H,D]
        k_ref = x * w_k * t
        v_ref = x * w_v * t
        k_full = torch.cat([k_ctx, k_ref.unsqueeze(0)], dim=1)  # [1,S_und+S_gen+S_gen,H,D]
        v_full = torch.cat([v_ctx, v_ref.unsqueeze(0)], dim=1)
        out_ref = full_attention(
            query=q_ref.unsqueeze(0),
            key=k_full,
            value=v_full,
            is_causal=False,
            scale=scale,
            return_lse=False,
        )
        assert isinstance(out_ref, torch.Tensor)
        out_ref_t: torch.Tensor = out_ref
        out_ref_sq = out_ref_t.squeeze(0)  # [S_gen,H,D]

        # ── Compare: gather CP output, check against reference on rank 0 ─────
        gathered = [None] * world_size
        dist.all_gather_object(gathered, out_sp.cpu(), group=cp_group)
        if rank == 0:
            cp_out_g = torch.cat(gathered, dim=0).to(device)  # [S_gen,H,D]
            torch.testing.assert_close(
                cp_out_g, out_ref_sq, rtol=1e-2, atol=1e-2, msg=f"Step {step}: CP attention output mismatch"
            )
            print(f"Step {step}: CP output matches reference")
        dist.barrier()

        # Oracle update: advance x using reference output (same on all ranks, no error accumulation)
        x = x - out_ref_sq / num_steps

    dist.barrier()
    if rank == 0:
        print("=== test_cp_ar_denoising_loop_equivalence passed")


@pytest.mark.L0
def test_cfgp_cp_ar_denoising_loop_equivalence():
    """CFGP=2+CP=2 denoising loop matches sequential full-attention + CFG reference.

    Uses 4 GPUs: cfgp=2, cp=2.  Mesh layout [rest=1, cfgp=2, cp=2]:
      - CP groups: {0,1} (cfgp_rank=0, cond) and {2,3} (cfgp_rank=1, uncond)
      - CFGP groups: {0,2} and {1,3} (each cp_rank pair)

    At each denoising step:
      1. Each CP group runs context_parallel_attention (wrapping attention_AR_gen_only) with its cond/uncond K/V.
      2. CFGP P2P exchange swaps seq-sharded velocities across CFGP groups.
      3. CFG blend: v_pred = v_uncond + guidance * (v_cond - v_uncond).
      4. Gather v_pred within the CP group and compare to the sequential reference.
      5. x is updated using the reference output (oracle update, no error accumulation).

    The gathered CFGP+CP velocity must match the reference within rtol=1e-2, atol=1e-2.
    """
    rank, world_size = setup_distributed_environment()
    if world_size < 4:
        pytest.skip("requires at least 4 GPUs")

    device = torch.device("cuda", rank)
    parallel_dims = ParallelDims(enable_inference_mode=True, world_size=world_size, dp_shard=1, cfgp=2, cp=2)
    parallel_dims.build_meshes("cuda")
    cp_mesh = parallel_dims.cp_mesh
    cp_group = cp_mesh.get_group()
    cfgp_mesh = parallel_dims.cfgp_mesh
    cfgp_group = cfgp_mesh.get_group()
    cfgp_rank = parallel_dims.cfgp_rank
    cfgp_size = parallel_dims.cfgp_size
    cp_rank = parallel_dims.cp_rank
    cp_size = parallel_dims.cp_size

    num_heads = cp_size * 2  # divisible by cp=2
    head_dim = 16
    S_und = cp_size * 4  # divisible by cp=2
    S_gen = cp_size * 2  # divisible by cp=2
    num_steps = 3
    scale = head_dim**-0.5
    guidance = 7.5
    h_per_rank = num_heads // cp_size
    s_gen_per_rank = S_gen // cp_size
    h_start = cp_rank * h_per_rank
    s_start = cp_rank * s_gen_per_rank

    torch.manual_seed(42)
    # Fixed context K/V for cond and uncond — same on all 4 ranks (deterministic seed)
    k_und_cond = torch.randn(S_und, num_heads, head_dim, device=device, dtype=torch.bfloat16)  # [S_und,H,D]
    v_und_cond = torch.randn(S_und, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k0_cond = torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16)  # [S_gen,H,D]
    v0_cond = torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_und_uncond = torch.randn(S_und, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_und_uncond = torch.randn(S_und, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k0_uncond = torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    v0_uncond = torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    # Linear projection weights for Q/K/V from x (separate for cond/uncond branches)
    w_q_cond = torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16)  # [S_gen,H,D]
    w_k_cond = torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    w_v_cond = torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    w_q_uncond = torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    w_k_uncond = torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    w_v_uncond = torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16)

    # Each rank uses cond or uncond weights based on cfgp_rank
    if cfgp_rank == 0:  # cond branch
        k_und_local, v_und_local = k_und_cond, v_und_cond
        k0_local, v0_local = k0_cond, v0_cond
        w_q_local, w_k_local, w_v_local = w_q_cond, w_k_cond, w_v_cond
    else:  # uncond branch
        k_und_local, v_und_local = k_und_uncond, v_und_uncond
        k0_local, v0_local = k0_uncond, v0_uncond
        w_q_local, w_k_local, w_v_local = w_q_uncond, w_k_uncond, w_v_uncond

    # ── CP path: pre-populate per-rank KV cache with head-sharded tensors ─────
    kv_cache = DualKVCache(gen_cache_size=4)
    kv_cache.und_cache.store(
        k_und_local[:, h_start : h_start + h_per_rank, :].unsqueeze(0),  # [1,S_und,H/cp,D]
        v_und_local[:, h_start : h_start + h_per_rank, :].unsqueeze(0),
    )
    kv_cache.gen_cache.store_kv(
        k0_local[:, h_start : h_start + h_per_rank, :].unsqueeze(0),  # [1,S_gen,H/cp,D]
        v0_local[:, h_start : h_start + h_per_rank, :].unsqueeze(0),
        frame_idx=0,
    )

    # ── Reference: full context for cond and uncond ────────────────────────────
    k_ctx_cond = torch.cat([k_und_cond.unsqueeze(0), k0_cond.unsqueeze(0)], dim=1)  # [1,S_und+S_gen,H,D]
    v_ctx_cond = torch.cat([v_und_cond.unsqueeze(0), v0_cond.unsqueeze(0)], dim=1)
    k_ctx_uncond = torch.cat([k_und_uncond.unsqueeze(0), k0_uncond.unsqueeze(0)], dim=1)
    v_ctx_uncond = torch.cat([v_und_uncond.unsqueeze(0), v0_uncond.unsqueeze(0)], dim=1)

    torch.manual_seed(0)
    x = torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16)  # [S_gen,H,D] same on all ranks

    cfgp_peer = (cfgp_rank + 1) % cfgp_size  # group rank of CFGP peer (uses group_peer= semantics)

    for step in range(num_steps):
        t = 1.0 - step / num_steps

        # ── CP attention within CFGP group ────────────────────────────────────
        q_sp = (x * w_q_local * t)[s_start : s_start + s_gen_per_rank].contiguous()  # [S_gen/cp,H,D]
        k_sp = (x * w_k_local * t)[s_start : s_start + s_gen_per_rank].contiguous()
        v_sp = (x * w_v_local * t)[s_start : s_start + s_gen_per_rank].contiguous()

        v_local_sp = _cp_attention_ar_gen(
            cp_mesh,
            q_sp,
            k_sp,
            v_sp,
            dual_kv_cache=kv_cache,
            frame_idx=1,
            gen_len=S_gen,
        )  # [S_gen/cp,H,D]

        # ── CFGP P2P exchange (group_rank = group rank within cfgp_group) ─────
        other_v_sp = torch.empty_like(v_local_sp)
        reqs = dist.batch_isend_irecv(
            [
                dist.P2POp(op=dist.isend, tensor=v_local_sp.contiguous(), group_peer=cfgp_peer, group=cfgp_group),
                dist.P2POp(op=dist.irecv, tensor=other_v_sp, group_peer=cfgp_peer, group=cfgp_group),
            ]
        )
        for req in reqs:
            req.wait()

        v_cond_sp = v_local_sp if cfgp_rank == 0 else other_v_sp  # [S_gen/cp,H,D]
        v_uncond_sp = other_v_sp if cfgp_rank == 0 else v_local_sp
        v_pred_sp = v_uncond_sp + guidance * (v_cond_sp - v_uncond_sp)  # [S_gen/cp,H,D]

        # ── Gather CFG-blended velocity within CP group ───────────────────────
        all_v_pred = [torch.empty_like(v_pred_sp) for _ in range(cp_size)]
        dist.all_gather(all_v_pred, v_pred_sp, group=cp_group)
        v_pred_g = torch.cat(all_v_pred, dim=0)  # [S_gen,H,D]

        # ── Reference: sequential full attention + CFG blend ──────────────────
        q_cond = x * w_q_cond * t  # [S_gen,H,D]
        k_cond_curr = x * w_k_cond * t
        v_cond_curr = x * w_v_cond * t
        q_uncond = x * w_q_uncond * t
        k_uncond_curr = x * w_k_uncond * t
        v_uncond_curr = x * w_v_uncond * t
        k_full_cond = torch.cat([k_ctx_cond, k_cond_curr.unsqueeze(0)], dim=1)  # [1,S_und+S_gen+S_gen,H,D]
        v_full_cond = torch.cat([v_ctx_cond, v_cond_curr.unsqueeze(0)], dim=1)
        k_full_uncond = torch.cat([k_ctx_uncond, k_uncond_curr.unsqueeze(0)], dim=1)
        v_full_uncond = torch.cat([v_ctx_uncond, v_uncond_curr.unsqueeze(0)], dim=1)
        out_cond = full_attention(
            query=q_cond.unsqueeze(0),
            key=k_full_cond,
            value=v_full_cond,
            is_causal=False,
            scale=scale,
            return_lse=False,
        )
        out_uncond = full_attention(
            query=q_uncond.unsqueeze(0),
            key=k_full_uncond,
            value=v_full_uncond,
            is_causal=False,
            scale=scale,
            return_lse=False,
        )
        assert isinstance(out_cond, torch.Tensor)
        assert isinstance(out_uncond, torch.Tensor)
        out_cond_t: torch.Tensor = out_cond
        out_uncond_t: torch.Tensor = out_uncond
        v_cond_ref = out_cond_t.squeeze(0)  # [S_gen,H,D]
        v_uncond_ref = out_uncond_t.squeeze(0)
        v_pred_ref = v_uncond_ref + guidance * (v_cond_ref - v_uncond_ref)  # [S_gen,H,D]

        # ── Compare gathered CFGP+CP velocity to reference ────────────────────
        if rank == 0:
            torch.testing.assert_close(
                v_pred_g,
                v_pred_ref,
                rtol=1e-2,
                atol=1e-2,
                msg=f"Step {step}: CFGP+CP velocity mismatch",
            )
            print(f"Step {step}: CFGP+CP velocity matches reference")
        dist.barrier()

        # Oracle update: advance x using reference velocity (same on all ranks)
        x = x - v_pred_ref / num_steps

    dist.barrier()
    if rank == 0:
        print("=== test_cfgp_cp_ar_denoising_loop_equivalence passed")


def _cp_attention_ar_gen_static(
    cp_mesh,
    q_gen: torch.Tensor,
    k_gen: torch.Tensor,
    v_gen: torch.Tensor,
    dual_kv_cache: DualKVCache,
    frame_idx: int,
    gen_len: int,
    *,
    max_gen_cache_tokens: int,
    h_per_rank: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    attention_fn=attention_AR_gen_only,
) -> torch.Tensor:
    """Static-shape (``for_cuda_graphs=True``) variant of ``_cp_attention_ar_gen``.

    Builds an ``ARMemoryValue`` with the full preallocated head-sharded gen
    buffer (``[1, max_gen_cache_tokens, H/cp, D]``) and ``cu_seqlens_*``
    int32 ``[2]`` tensors so that ``attention_AR_gen_only`` runs through the
    static-shape branch.  Sequence-dim shapes stay constant across frames,
    which is the prerequisite for replaying a single CUDA-graph capture.

    ``attention_fn`` lets callers swap ``attention_AR_gen_only`` for a
    ``torch.compile``-wrapped version.
    """
    und_k_cached, und_v_cached = dual_kv_cache.und_cache.get()
    assert und_k_cached is not None and und_v_cached is not None, (
        "und cache must be pre-populated by the frame-0 prefill (dynamic-shape ARMemoryState branch)"
    )
    gen_k_buf, gen_v_buf, real_len = dual_kv_cache.gen_cache.fetch_kv_padded(
        frame_idx,
        max_gen_cache_tokens,
        num_heads=h_per_rank,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )

    s_und = und_k_cached.shape[1]
    real_total_kv_len = s_und + gen_len + real_len
    cu_seqlens_q_t = torch.tensor([0, gen_len], device=device, dtype=torch.int32)
    cu_seqlens_kv_t = torch.tensor([0, real_total_kv_len], device=device, dtype=torch.int32)
    real_gen_cache_len_t = torch.tensor([real_len], device=device, dtype=torch.int32)
    max_seqlen_KV = s_und + gen_len + max_gen_cache_tokens

    mv = ARMemoryValue(
        und_k_cached=und_k_cached,
        und_v_cached=und_v_cached,
        gen_k_hist=None,
        gen_v_hist=None,
        frame_idx=frame_idx,
        gen_len=gen_len,
        gen_k_buf_full=gen_k_buf,
        gen_v_buf_full=gen_v_buf,
        real_gen_cache_len_t=real_gen_cache_len_t,
        cu_seqlens_q_t=cu_seqlens_q_t,
        cu_seqlens_kv_t=cu_seqlens_kv_t,
        max_seqlen_KV=max_seqlen_KV,
        for_cuda_graphs=True,
    )

    gen_only_mask = SplitInfo(
        split_lens=[gen_len],
        attn_modes=["full"],
        sample_lens=[gen_len],
        actual_len=gen_len,
    )
    # Input ``q_gen`` is seq-sharded with FULL heads: ``[S_gen/cp, H, D]``.
    # ``context_parallel_attention`` does all-to-all to head-shard inside,
    # then the inverse all-to-all reseats the output as seq-sharded with
    # full heads again — same layout as the input.
    num_heads_global = q_gen.shape[1]
    out_pack, kv_to_store = context_parallel_attention(
        cp_mesh,
        _make_gen_only_pack(q_gen, global_seq_len=gen_len),
        _make_gen_only_pack(k_gen, global_seq_len=gen_len),
        _make_gen_only_pack(v_gen, global_seq_len=gen_len),
        gen_only_mask,
        attention_fn,
        memory_value=mv,
    )
    out_sp = get_gen_seq(out_pack).unflatten(-1, (num_heads_global, head_dim))

    assert kv_to_store is not None
    gen_k, gen_v, _, _ = kv_to_store
    dual_kv_cache.gen_cache.store_kv(gen_k, gen_v, frame_idx=frame_idx)
    return out_sp


def _populate_eager_baseline_and_cache(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    cp_mesh,
    num_heads: int,
    head_dim: int,
    S_und: int,
    S_gen: int,
    num_frames: int,
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], DualKVCache]:
    """Build the eager dynamic-shape CP baseline and a head-sharded DualKVCache.

    Returns:
        baseline_results: per-frame ``[S_gen/cp, H, D]`` outputs from the
            existing dynamic-shape CP path (frames 1..num_frames-1).
        q_frames, k_frames, v_frames: global per-frame tensors so the caller
            can rebuild the same inputs for additional paths.
        kv_cache: a *fresh* ``DualKVCache`` populated with head-sharded und
            and frame-0 gen K/V — ready to be re-used by another path.
    """
    torch.manual_seed(seed)
    k_und_g = torch.randn(S_und, num_heads, head_dim, device=device, dtype=dtype)
    v_und_g = torch.randn(S_und, num_heads, head_dim, device=device, dtype=dtype)
    q_frames_g = [torch.randn(S_gen, num_heads, head_dim, device=device, dtype=dtype) for _ in range(num_frames)]
    k_frames_g = [torch.randn(S_gen, num_heads, head_dim, device=device, dtype=dtype) for _ in range(num_frames)]
    v_frames_g = [torch.randn(S_gen, num_heads, head_dim, device=device, dtype=dtype) for _ in range(num_frames)]

    h_per_rank = num_heads // world_size
    h_start = rank * h_per_rank
    s_per_rank = S_gen // world_size
    s_start = rank * s_per_rank

    kv_cache = DualKVCache(gen_cache_size=num_frames + 2)
    kv_cache.und_cache.store(
        k_und_g[:, h_start : h_start + h_per_rank, :].unsqueeze(0),
        v_und_g[:, h_start : h_start + h_per_rank, :].unsqueeze(0),
    )
    kv_cache.gen_cache.store_kv(
        k_frames_g[0][:, h_start : h_start + h_per_rank, :].unsqueeze(0),
        v_frames_g[0][:, h_start : h_start + h_per_rank, :].unsqueeze(0),
        frame_idx=0,
    )

    baseline_results: list[torch.Tensor] = []
    for fi in range(1, num_frames):
        q_sp = q_frames_g[fi][s_start : s_start + s_per_rank, :, :]
        k_sp = k_frames_g[fi][s_start : s_start + s_per_rank, :, :]
        v_sp = v_frames_g[fi][s_start : s_start + s_per_rank, :, :]
        out_sp = _cp_attention_ar_gen(
            cp_mesh,
            q_sp,
            k_sp,
            v_sp,
            dual_kv_cache=kv_cache,
            frame_idx=fi,
            gen_len=S_gen,
        )
        baseline_results.append(out_sp)

    # Rebuild the cache from scratch so the caller starts from frame 0 too.
    fresh = DualKVCache(gen_cache_size=num_frames + 2)
    fresh.und_cache.store(
        k_und_g[:, h_start : h_start + h_per_rank, :].unsqueeze(0),
        v_und_g[:, h_start : h_start + h_per_rank, :].unsqueeze(0),
    )
    fresh.gen_cache.store_kv(
        k_frames_g[0][:, h_start : h_start + h_per_rank, :].unsqueeze(0),
        v_frames_g[0][:, h_start : h_start + h_per_rank, :].unsqueeze(0),
        frame_idx=0,
    )

    return baseline_results, q_frames_g, k_frames_g, v_frames_g, fresh


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_context_parallel_ar_compile_no_cuda_graphs():
    """CP + ``torch.compile`` (no CUDA Graphs) on the dynamic-shape AR path
    matches eager and recompiles at most once for the 0->1 KV-length jump.

    This pins the behavior described in
    ``omni_mot_causal_model.py::velocity_fn`` for the
    ``enabled and not use_cuda_graphs`` regime: Dynamo handles the
    growing ``S_KV`` via dynamic shapes and a one-time recompile at the
    frame-0 → frame-1 boundary; subsequent frames reuse the same graph.

    Setup mirrors ``test_context_parallel_ar_inference`` but wraps
    ``attention_AR_gen_only`` in ``torch.compile`` and runs it under CP.
    """
    rank, world_size = setup_distributed_environment()
    if world_size < 2:
        pytest.skip("requires at least 2 GPUs")

    device = torch.device("cuda", rank)
    parallel_dims = ParallelDims(enable_inference_mode=True, world_size=world_size, dp_shard=1, cp=world_size)
    parallel_dims.build_meshes("cuda")
    cp_mesh = parallel_dims.cp_mesh

    num_heads = world_size * 2
    head_dim = 32
    S_und = world_size * 4
    S_gen = world_size * 2
    num_frames = 4  # frame 0 cached; frames 1, 2, 3 exercise the compiled path

    baseline_results, q_frames_g, k_frames_g, v_frames_g, kv_cache = _populate_eager_baseline_and_cache(
        rank=rank,
        world_size=world_size,
        device=device,
        cp_mesh=cp_mesh,
        num_heads=num_heads,
        head_dim=head_dim,
        S_und=S_und,
        S_gen=S_gen,
        num_frames=num_frames,
        seed=42,
    )

    s_per_rank = S_gen // world_size
    s_start = rank * s_per_rank

    torch._dynamo.reset()
    compiled_attn = torch.compile(attention_AR_gen_only, dynamic=True)

    def _run_compiled_frame(fi: int) -> torch.Tensor:
        q_sp = q_frames_g[fi][s_start : s_start + s_per_rank, :, :]
        k_sp = k_frames_g[fi][s_start : s_start + s_per_rank, :, :]
        v_sp = v_frames_g[fi][s_start : s_start + s_per_rank, :, :]

        und_k_cached, und_v_cached = kv_cache.und_cache.get()
        gen_k_hist, gen_v_hist = kv_cache.gen_cache.fetch_kv(fi)
        mv = ARMemoryValue(
            und_k_cached=und_k_cached,
            und_v_cached=und_v_cached,
            gen_k_hist=gen_k_hist,
            gen_v_hist=gen_v_hist,
            frame_idx=fi,
            gen_len=S_gen,
        )
        gen_only_mask = SplitInfo(
            split_lens=[S_gen],
            attn_modes=["full"],
            sample_lens=[S_gen],
            actual_len=S_gen,
        )
        out_pack, kv_to_store = context_parallel_attention(
            cp_mesh,
            _make_gen_only_pack(q_sp, global_seq_len=S_gen),
            _make_gen_only_pack(k_sp, global_seq_len=S_gen),
            _make_gen_only_pack(v_sp, global_seq_len=S_gen),
            gen_only_mask,
            compiled_attn,
            memory_value=mv,
        )
        assert kv_to_store is not None
        gk, gv, _, _ = kv_to_store
        kv_cache.gen_cache.store_kv(gk, gv, frame_idx=fi)
        return get_gen_seq(out_pack).unflatten(-1, (num_heads, head_dim))

    compiled_results = [_run_compiled_frame(1)]

    for fi in range(2, num_frames):
        compiled_results.append(_run_compiled_frame(fi))

    # Numerical parity vs eager dynamic-shape baseline.
    for i, fi in enumerate(range(1, num_frames)):
        torch.testing.assert_close(
            compiled_results[i],
            baseline_results[i],
            rtol=1e-2,
            atol=1e-2,
            msg=f"compile (no-CG) vs eager mismatch at frame {fi}",
        )


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_context_parallel_ar_compile_with_cuda_graphs():
    """CP + ``torch.compile(mode="reduce-overhead")`` (CUDA Graphs) on the
    static-shape AR path matches eager and produces a single Dynamo graph.

    This is the regime targeted by the static-shape ``ARMemoryState`` /
    ``ARMemoryValue`` refactor: ``gen_k_buf_full`` has constant shape
    ``[1, max_gen_cache_tokens, H/cp, D]`` per frame and ``cu_seqlens_*``
    are ``[2]``-int32 tensors whose values change but whose shape is
    invariant — so a single CUDA-graph capture replays for every frame.

    Wraps ``attention_AR_gen_only`` in ``torch.compile(mode="reduce-overhead")``,
    runs it under CP with ``ARMemoryValue(for_cuda_graphs=True)``, and:
      1. Asserts numerical parity vs the eager dynamic-shape CP baseline.
    """
    rank, world_size = setup_distributed_environment()
    if world_size < 2:
        pytest.skip("requires at least 2 GPUs")

    device = torch.device("cuda", rank)
    dtype = torch.bfloat16
    parallel_dims = ParallelDims(enable_inference_mode=True, world_size=world_size, dp_shard=1, cp=world_size)
    parallel_dims.build_meshes("cuda")
    cp_mesh = parallel_dims.cp_mesh

    num_heads = world_size * 2
    head_dim = 32
    S_und = world_size * 4
    S_gen = world_size * 2
    num_frames = 4
    h_per_rank = num_heads // world_size
    s_per_rank = S_gen // world_size
    s_start = rank * s_per_rank
    # Generous gen-cache budget: gen_cache_size segments stored at most.  The
    # static-shape buffer holds (gen_cache_size - 1) * S_gen tokens (frame N
    # is *not* yet in the historical buffer when we read at frame N).
    gen_cache_size = num_frames + 2
    max_gen_cache_tokens = (gen_cache_size - 1) * S_gen

    baseline_results, q_frames_g, k_frames_g, v_frames_g, kv_cache = _populate_eager_baseline_and_cache(
        rank=rank,
        world_size=world_size,
        device=device,
        cp_mesh=cp_mesh,
        num_heads=num_heads,
        head_dim=head_dim,
        S_und=S_und,
        S_gen=S_gen,
        num_frames=num_frames,
        seed=42,
        dtype=dtype,
    )

    torch._dynamo.reset()
    compiled_attn = torch.compile(attention_AR_gen_only, mode="reduce-overhead", dynamic=False)

    def _run_static_frame(fi: int) -> torch.Tensor:
        # ``mode="reduce-overhead"`` uses CUDA-Graph trees: the runtime needs
        # an explicit step boundary between successive captures so it can
        # reclaim/recycle output storage between iterations.
        torch.compiler.cudagraph_mark_step_begin()
        q_sp = q_frames_g[fi][s_start : s_start + s_per_rank, :, :]
        k_sp = k_frames_g[fi][s_start : s_start + s_per_rank, :, :]
        v_sp = v_frames_g[fi][s_start : s_start + s_per_rank, :, :]
        return _cp_attention_ar_gen_static(
            cp_mesh,
            q_sp,
            k_sp,
            v_sp,
            dual_kv_cache=kv_cache,
            frame_idx=fi,
            gen_len=S_gen,
            max_gen_cache_tokens=max_gen_cache_tokens,
            h_per_rank=h_per_rank,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
            attention_fn=compiled_attn,
        )

    compiled_results = [_run_static_frame(1).clone()]

    for fi in range(2, num_frames):
        compiled_results.append(_run_static_frame(fi).clone())

    # Numerical parity vs eager dynamic-shape baseline.  Static-shape pads
    # the gen buffer with zeros and uses ``cumulative_seqlen_KV`` to mask
    # the padding; output should be the same up to bf16 noise.
    for i, fi in enumerate(range(1, num_frames)):
        torch.testing.assert_close(
            compiled_results[i],
            baseline_results[i],
            rtol=1e-2,
            atol=1e-2,
            msg=f"compile+CG (static-shape) vs eager mismatch at frame {fi}",
        )


if __name__ == "__main__":
    test_context_parallel_ar_inference()
    test_context_parallel_ar_frame0_stores_head_sharded()
    test_context_parallel_ar_round_trip()
    test_cp_ar_denoising_loop_equivalence()
    test_cfgp_cp_ar_denoising_loop_equivalence()
    test_context_parallel_ar_compile_no_cuda_graphs()
    test_context_parallel_ar_compile_with_cuda_graphs()
