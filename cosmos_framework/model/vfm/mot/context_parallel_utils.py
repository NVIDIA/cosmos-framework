# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Context Parallelism Utilities.

Integration Guide:
------------------
1. Shard the Input Sequence:
   Call `get_context_parallel_sharded_sequence` at the start of the forward pass to split
   the global input pack into local shards.

   ```python
   input_pack, position_ids = get_context_parallel_sharded_sequence(
       attn_implementation, input_pack, position_ids, parallel_dims
   )
   ```

2. Apply Context Parallel Attention:
   Use `context_parallel_attention` inside your attention block. It handles All-to-All
   communication (gather seq, scatter heads -> attn -> gather heads, scatter seq).

   ```python
   output, kv_to_store = context_parallel_attention(
       cp_mesh, query_pack, key_pack, value_pack, mask, local_attn_func
   )
   ```

3. Gather Final Hidden States (Optional):
   Use `get_context_parallel_last_hidden_state` if the full global sequence is needed for
   loss or post-processing.
"""

from typing import Callable

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.nn.attention.flex_attention import BlockMask

from cosmos_framework.data.vfm.sequence_packing import (
    FactoredSequencePack,
    JointSequencePack,
    from_mode_splits,
    get_all_seq,
    get_causal_seq,
    get_full_only_seq,
    get_gen_position_ids,
    get_gen_seq,
    get_und_position_ids,
    get_und_seq,
)
from cosmos_framework.model.vfm.mot.attention import SplitInfo
from cosmos_framework.model.vfm.utils.memory import KVToStore, MemoryValue
from cosmos_framework.utils.vfm.parallelism import ParallelDims


def _pad_to_N(N, x: torch.Tensor) -> torch.Tensor:
    assert x.shape[0] <= N
    padded = x.new_zeros((N, *x.shape[1:]))
    padded[: x.shape[0]] = x
    return padded


def _repeat_kv_heads_for_cp(x: torch.Tensor, repeats: int) -> torch.Tensor:
    """Repeat KV heads before CP all-to-all so GQA remains valid after head sharding.

    Args:
        x: KV tensor shaped ``[seq_len,num_kv_heads,head_dim]``.
        repeats: Number of query-head groups served by each KV head.

    Returns:
        KV tensor shaped ``[seq_len,num_kv_heads*repeats,head_dim]``.
    """
    if repeats == 1:
        return x

    seq_len, num_kv_heads, head_dim = x.shape
    x = x[:, :, None, :].expand(seq_len, num_kv_heads, repeats, head_dim)  # [seq_len,num_kv_heads,repeats,head_dim]
    return x.reshape(seq_len, num_kv_heads * repeats, head_dim)  # [seq_len,num_kv_heads*repeats,head_dim]


def context_parallel_broadcast_tensor_list(
    tensors: list[torch.Tensor] | None, parallel_dims: ParallelDims | None
) -> None:
    """Broadcast an in-place tensor list from CP rank 0 to all CP ranks."""
    if tensors is None or parallel_dims is None or not parallel_dims.cp_enabled:
        return

    cp_group = parallel_dims.cp_mesh.get_group()
    global_src_rank = dist.get_global_rank(cp_group, 0)
    for tensor in tensors:
        dist.broadcast(tensor, src=global_src_rank, group=cp_group)


def get_context_parallel_sharded_sequence(
    attn_implementation: str,
    input_pack: FactoredSequencePack,
    position_ids: torch.Tensor,
    parallel_dims: ParallelDims | None,
) -> tuple[FactoredSequencePack, torch.Tensor]:
    """
    Splits the full input_pack into a local shard for Context Parallelism.
    """
    if parallel_dims is None or not parallel_dims.cp_enabled:
        return input_pack, position_ids

    assert attn_implementation in ("two_way", "three_way"), (
        f"Context parallel is only supported for two_way and three_way joint attention modes, "
        f"got {attn_implementation!r}"
    )
    cp_mesh = parallel_dims.cp_mesh
    cp_group = cp_mesh.get_group()
    rank = dist.get_rank(cp_group)
    world_size = dist.get_world_size(cp_group)

    text_seq = get_und_seq(input_pack)
    gen_seq = get_gen_seq(input_pack)
    assert text_seq.shape[0] % world_size == 0, "text_seq.shape[0] must be divisible by world_size"
    assert gen_seq.shape[0] % world_size == 0, "gen_seq.shape[0] must be divisible by world_size"

    text_len = text_seq.shape[0]
    text_shard_len = text_len // world_size
    text_shard = text_seq.narrow(0, rank * text_shard_len, text_shard_len)

    gen_len = gen_seq.shape[0]
    gen_shard_len = gen_len // world_size
    gen_shard = gen_seq.narrow(0, rank * gen_shard_len, gen_shard_len)

    text_position_ids = get_und_position_ids(position_ids, input_pack)
    gen_position_ids = get_gen_position_ids(position_ids, input_pack)

    # Handle 3D mRoPE position IDs: shape (3, L)
    is_mrope = position_ids.dim() == 2 and position_ids.shape[0] == 3
    if is_mrope:
        text_position_ids = text_position_ids.transpose(0, 1)  # [text_len,3]
        gen_position_ids = gen_position_ids.transpose(0, 1)  # [gen_len,3]

    # pad to N
    text_position_ids = _pad_to_N(text_seq.shape[0], text_position_ids)
    gen_position_ids = _pad_to_N(gen_seq.shape[0], gen_position_ids)

    text_position_ids_shard = text_position_ids.narrow(0, rank * text_shard_len, text_shard_len)
    gen_position_ids_shard = gen_position_ids.narrow(0, rank * gen_shard_len, gen_shard_len)

    # create local pack
    local_pack = from_mode_splits(text_shard, gen_shard, input_pack, is_sharded=True)
    local_position_ids = torch.cat(
        [text_position_ids_shard, gen_position_ids_shard], dim=0
    )  # [text_shard_len+gen_shard_len] or [text_shard_len+gen_shard_len,3]

    if is_mrope:
        local_position_ids = local_position_ids.transpose(0, 1)  # [3,text_shard_len+gen_shard_len]

    return local_pack, local_position_ids


def get_context_parallel_last_hidden_state(
    packed_outputs: FactoredSequencePack,
    parallel_dims: ParallelDims | None,
) -> torch.Tensor:
    if parallel_dims is None or not parallel_dims.cp_enabled:
        return get_all_seq(packed_outputs)

    # since unpatchify assumes full images, for now using all_gather to gather the predictions from all context parallel ranks
    # This step can be removed once we make unpatchify work with context parallel local sequences
    und_hidden_seq = get_und_seq(packed_outputs)  # [text_shard_len,hidden_size]
    gen_hidden_seq = get_gen_seq(packed_outputs)  # [gen_shard_len,hidden_size]

    gathered_und_seq = all_gather_tensor(
        und_hidden_seq, gather_dim=0, cp_mesh=parallel_dims.cp_mesh
    )  # [text_len,hidden_size]
    gathered_gen_seq = all_gather_tensor(
        gen_hidden_seq, gather_dim=0, cp_mesh=parallel_dims.cp_mesh
    )  # [gen_len,hidden_size]

    gathered_hidden_pack = from_mode_splits(gathered_und_seq, gathered_gen_seq, packed_outputs, is_sharded=False)
    last_hidden_state = get_all_seq(gathered_hidden_pack)
    return last_hidden_state


def all_to_all_tensor(
    local_input: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    cp_mesh: "DeviceMesh",
) -> torch.Tensor:
    """
    All-to-all via DTensor redistribute.
    Input placement: Shard(gather_dim) -> The dimension we are about to gather was split.
    Output placement: Shard(scatter_dim) -> The dimension we are about to scatter will be split.
    """
    # Wrap local tensor as DTensor with current placement
    # gather_dim is the dimension that is currently sharded locally (so we can gather it to full)
    global_dt = DTensor.from_local(local_input, cp_mesh, [Shard(gather_dim)], run_check=False)

    # Redistribute to new placement (shard scatter_dim)
    new_dt = global_dt.redistribute(cp_mesh, [Shard(scatter_dim)])

    # Convert back to local
    return new_dt.to_local()


def all_gather_tensor(
    local_input: torch.Tensor,
    gather_dim: int,
    cp_mesh: "DeviceMesh",
) -> torch.Tensor:
    """
    All-gather via DTensor redistribute.
    Input placement: Shard(gather_dim) -> The dimension we are about to gather was split.
    Output placement: Replicate() -> Full copy on each rank.
    """
    # Wrap local tensor as DTensor with current placement
    global_dt = DTensor.from_local(local_input, cp_mesh, [Shard(gather_dim)], run_check=False)

    # Redistribute to new placement (Replicate)
    new_dt = global_dt.redistribute(cp_mesh, [Replicate()])

    # Convert back to local
    return new_dt.to_local()


def gather_seq_scatter_heads(
    x: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_mesh: DeviceMesh,
) -> torch.Tensor:
    """
    A func to sync embedding input with alltoall in sequence parallel.
    gather sequence dimension and scatter head dim:
    For example, when seq_dim is 0, head_dim is 1, the transformation is:
    [z, seq/n, h, ...] -> [z, seq, h/n, ...]
    Args:
        x: shape of [z, seq, h, ...]
        seq_dim: the dimension to gather
        head_dim: the dimension to scatter
        cp_mesh: ulysses sequence parallelism size
    Returns:
        torch.Tensor: shape of gathered and scattered tensor
    """
    return all_to_all_tensor(x, head_dim, seq_dim, cp_mesh)


def gather_heads_scatter_seq(
    x: torch.Tensor,
    head_dim: int,
    seq_dim: int,
    cp_mesh: DeviceMesh,
) -> torch.Tensor:
    """
    A func to sync attention result with alltoall in sequence parallel.
    gather head dimension and scatter seq dim:
    For example, when seq_dim is 0, head_dim is 1, the transformation is:
    [seq, h/n, ...] -> [seq/n, h, ...]

    Args:
        x (torch.Tensor): shape of [bsz, seq, h/n, ...]
        head_dim (int): the dimension to gather
        seq_dim (int): the dimension to scatter
        cp_mesh (DeviceMesh): ulysses sequence parallelism size
        splits (List[torch.Tensor], optional): Manual splits for variable length scattering

    Returns:
        torch.Tensor: shape of [bsz, seq/n, h, ...]
    """
    return all_to_all_tensor(x, seq_dim, head_dim, cp_mesh)


def context_parallel_attention(
    cp_mesh: DeviceMesh,
    packed_query_states: FactoredSequencePack,
    packed_key_states: FactoredSequencePack,
    packed_value_states: FactoredSequencePack,
    attention_mask: BlockMask | SplitInfo,
    attention_function: Callable,
    natten_metadata: dict | None = None,
    memory_value: MemoryValue | None = None,
) -> tuple[FactoredSequencePack | JointSequencePack, KVToStore | None]:
    """Ulysses-style context parallel attention for packed und+gen sequences.

    Each rank holds a sequence shard [S/cp, H, D] for Q and [S/cp, H_kv, D]
    for K/V. Two all-to-all calls convert between seq-sharded and head-sharded
    representations:
      1. gather seq, scatter heads → [S, H/cp, D] for Q and [S, H_kv_local, D] for K/V
      2. run attention on full sequence with reduced heads
      3. gather heads, scatter seq → [S/cp, H, D] (seq-sharded)

    When ``memory_value`` is present, produces head-sharded ``kv_to_store``
    from the post-all-to-all K/V tensors for the caller to write back via
    ``MemoryState.write_for_layer()``.  Does **not** write to any cache
    directly.

    Args:
        cp_mesh: Device mesh for context parallelism.
        packed_query_states: Packed Q for both und and gen tokens, seq-sharded [S/cp, H, D].
        packed_key_states: Packed K for both und and gen tokens, seq-sharded [S/cp, H_kv, D].
        packed_value_states: Packed V for both und and gen tokens, seq-sharded [S/cp, H_kv, D].
        attention_mask: Block mask or split info describing causal/full attention pattern.
        attention_function: Callable implementing the actual attention kernel.
        natten_metadata: Optional neighborhood attention metadata.
        memory_value: Optional memory value for KV-cache training / AR inference.

    Returns:
        (output_pack, kv_to_store):
            output_pack: Packed attention output, seq-sharded [S/cp, H, D].
            kv_to_store: Head-sharded ``(gen_k, gen_v, und_k, und_v)`` when
                ``memory_value`` is present, ``None`` otherwise.
    """
    cp_group = cp_mesh.get_group()
    cp_world_size = torch.distributed.get_world_size(cp_group)
    assert cp_world_size > 1, "Context parallel world size must be greater than 1"
    q_und_seq, _ = get_causal_seq(packed_query_states)  # [text_shard_len,H,head_dim]
    q_gen_seq, _ = get_full_only_seq(packed_query_states)  # [gen_shard_len,H,head_dim]
    k_und_seq, _ = get_causal_seq(packed_key_states)  # [text_shard_len,H,head_dim]
    k_gen_seq, _ = get_full_only_seq(packed_key_states)  # [gen_shard_len,H,head_dim]
    v_und_seq, _ = get_causal_seq(packed_value_states)  # [text_shard_len,H,head_dim]
    v_gen_seq, _ = get_full_only_seq(packed_value_states)  # [gen_shard_len,H,head_dim]

    # Check that number of Q heads is divisible by CP world size. K/V heads may be repeated below for GQA.
    q_heads = q_und_seq.shape[1]
    kv_heads = k_und_seq.shape[1]
    assert q_und_seq.shape[1] % cp_world_size == 0, (
        f"Query heads ({q_und_seq.shape[1]}) must be divisible by context parallel world size ({cp_world_size})"
    )
    assert q_gen_seq.shape[1] == q_heads, (
        f"Understanding query heads ({q_heads}) and generation query heads ({q_gen_seq.shape[1]}) must match"
    )
    assert kv_heads == k_gen_seq.shape[1] == v_und_seq.shape[1] == v_gen_seq.shape[1], (
        f"Key/value heads must match across und/gen K/V tensors, got "
        f"k_und={kv_heads}, k_gen={k_gen_seq.shape[1]}, v_und={v_und_seq.shape[1]}, v_gen={v_gen_seq.shape[1]}"
    )
    assert q_heads % kv_heads == 0, f"Query heads ({q_heads}) must be divisible by KV heads ({kv_heads})"

    kv_head_repeats = max(cp_world_size // kv_heads, 1)
    repeated_kv_heads = kv_heads * kv_head_repeats
    assert repeated_kv_heads % cp_world_size == 0, (
        f"Repeated KV heads ({repeated_kv_heads}) must be divisible by context parallel world size "
        f"({cp_world_size}); got KV heads={kv_heads}, repeats={kv_head_repeats}"
    )
    q_heads_per_rank = q_heads // cp_world_size
    kv_heads_per_rank = repeated_kv_heads // cp_world_size
    assert q_heads_per_rank % kv_heads_per_rank == 0, (
        f"Local query heads ({q_heads_per_rank}) must be divisible by local KV heads ({kv_heads_per_rank})"
    )


    # when doing AR-inference with a KV-cache.

    if kv_head_repeats > 1:
        k_und_seq = _repeat_kv_heads_for_cp(k_und_seq, kv_head_repeats)  # [text_shard_len,repeated_kv_heads,head_dim]
        k_gen_seq = _repeat_kv_heads_for_cp(k_gen_seq, kv_head_repeats)  # [gen_shard_len,repeated_kv_heads,head_dim]
        v_und_seq = _repeat_kv_heads_for_cp(v_und_seq, kv_head_repeats)  # [text_shard_len,repeated_kv_heads,head_dim]
        v_gen_seq = _repeat_kv_heads_for_cp(v_gen_seq, kv_head_repeats)  # [gen_shard_len,repeated_kv_heads,head_dim]

    # all2all: gather sequence, scatter heads → head-sharded
    q_und_seq = gather_seq_scatter_heads(
        q_und_seq, seq_dim=0, head_dim=1, cp_mesh=cp_mesh
    )  # [text_len,H_local,head_dim]
    q_gen_seq = gather_seq_scatter_heads(
        q_gen_seq, seq_dim=0, head_dim=1, cp_mesh=cp_mesh
    )  # [gen_len,H_local,head_dim]
    k_und_seq = gather_seq_scatter_heads(
        k_und_seq, seq_dim=0, head_dim=1, cp_mesh=cp_mesh
    )  # [text_len,H_kv_local,head_dim]
    k_gen_seq = gather_seq_scatter_heads(
        k_gen_seq, seq_dim=0, head_dim=1, cp_mesh=cp_mesh
    )  # [gen_len,H_kv_local,head_dim]
    v_und_seq = gather_seq_scatter_heads(
        v_und_seq, seq_dim=0, head_dim=1, cp_mesh=cp_mesh
    )  # [text_len,H_kv_local,head_dim]
    v_gen_seq = gather_seq_scatter_heads(
        v_gen_seq, seq_dim=0, head_dim=1, cp_mesh=cp_mesh
    )  # [gen_len,H_kv_local,head_dim]

    # Build head-sharded kv_to_store when memory is active.
    kv_to_store: KVToStore | None = None
    if memory_value is not None:
        und_len = packed_key_states["_num_causal_tokens"]
        gen_len = packed_key_states["_num_full_tokens"]
        kv_to_store = (
            k_gen_seq[:gen_len].unsqueeze(0),
            v_gen_seq[:gen_len].unsqueeze(0),
            k_und_seq[:und_len].unsqueeze(0),
            v_und_seq[:und_len].unsqueeze(0),
        )

    q_und_seq_len = q_und_seq.shape[0]
    q_gen_seq_len = q_gen_seq.shape[0]
    meta = dict(packed_query_states)
    packed_query_states_ = from_mode_splits(q_und_seq, q_gen_seq, meta, is_sharded=False)
    packed_key_states_ = from_mode_splits(k_und_seq, k_gen_seq, meta, is_sharded=False)
    packed_value_states_ = from_mode_splits(v_und_seq, v_gen_seq, meta, is_sharded=False)

    # dispatch_attention returns (output, kv_to_store | None)
    attn_output_pack_hp, _inner_kv_to_store = attention_function(
        packed_query_states_,
        packed_key_states_,
        packed_value_states_,
        attention_mask,
        natten_metadata=natten_metadata,
        memory_value=memory_value,
    )

    attn_output_und_hp = get_und_seq(attn_output_pack_hp)  # [text_len,H_local,head_dim]
    attn_output_gen_hp = get_gen_seq(attn_output_pack_hp)  # [gen_len,H_local,head_dim]

    attn_output_und_hp = attn_output_und_hp[:q_und_seq_len].contiguous()  # [text_len,H_local,head_dim]
    attn_output_gen_hp = attn_output_gen_hp[:q_gen_seq_len].contiguous()  # [gen_len,H_local,head_dim]

    # all2all: gather heads, scatter seq → seq-sharded
    attn_output_und_sp = gather_heads_scatter_seq(
        attn_output_und_hp,
        seq_dim=0,
        head_dim=1,
        cp_mesh=cp_mesh,
    )  # [text_shard_len,H,head_dim]
    attn_output_gen_sp = gather_heads_scatter_seq(
        attn_output_gen_hp,
        seq_dim=0,
        head_dim=1,
        cp_mesh=cp_mesh,
    )  # [gen_shard_len,H,head_dim]

    final_output_pack_sp = from_mode_splits(
        attn_output_und_sp, attn_output_gen_sp, packed_query_states, is_sharded=True
    )

    return final_output_pack_sp, kv_to_store
