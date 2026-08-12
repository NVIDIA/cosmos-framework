# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Replicated attention-I/O context parallelism for interactive AR models."""

from collections.abc import Callable

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from cosmos_framework.utils import log
from cosmos_framework.model.generator.mot.attention import SplitInfo, dispatch_attention
from cosmos_framework.model.generator.utils.memory import KVToStore, MemoryValue
from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    from_und_gen_splits,
    get_gen_seq,
    get_und_seq,
)
from cosmos_framework.utils.generator.parallelism import ParallelDims


class ARReplicatedIODispatch(nn.Module):
    """AR CP dispatch for replicated attention I/O with local-head attention.

    ``Replicated I/O`` means the caller-side tensors at the attention boundary
    are replicated across CP ranks. It does not mean attention compute is
    replicated. For AR frame 1+, this wrapper slices the replicated current
    Q/K/V to this rank's local Q/KV heads and runs attention against the local
    KV-head cache.

    Shape flow for AR frame 1+:
        before slicing:
            q: [S,H,D], k/v: [S,H_kv,D], cached k/v: [B,S_hist,H_kv/CP,D]
        after local head slicing:
            q: [S,H/CP,D], k/v: [S,H_kv/CP,D], cached k/v: [B,S_hist,H_kv/CP,D]
        after local attention:
            out_local: [S,H/CP*D]
        after sharded o_proj in PackedAttentionMoT:
            out: [S,hidden_size]

    Current-frame hidden states stay replicated. For AR frame 1+, this wrapper
    delegates to the existing memory-aware AR attention for local heads, then
    returns the local current-frame attention output so ``PackedAttentionMoT``
    can apply the corresponding ``o_proj`` column slice. Frame 0 and non-AR
    paths delegate unchanged; frame 0 seeds the local KV-head cache through
    ``ARMemoryState.write_for_layer``.
    """

    cp_mesh: DeviceMesh
    wrapped_dispatch: Callable

    def __init__(
        self,
        cp_mesh: DeviceMesh,
        wrapped_dispatch: Callable = dispatch_attention,
    ) -> None:
        super().__init__()
        self.cp_mesh = cp_mesh
        self.wrapped_dispatch = wrapped_dispatch

    def _head_slices(self, q_heads: int, kv_heads: int) -> tuple[slice, slice]:
        cp_group = self.cp_mesh.get_group()
        cp_rank = torch.distributed.get_rank(cp_group)
        cp_size = torch.distributed.get_world_size(cp_group)
        assert kv_heads % cp_size == 0, (
            f"replicated attention_io_layout requires num_kv_heads({kv_heads}) % cp_size({cp_size}) == 0. "
            f"num_kv_heads={kv_heads} is the upper bound for useful local-head attention CP."
        )
        assert q_heads % kv_heads == 0, f"Query heads ({q_heads}) must be divisible by KV heads ({kv_heads})"
        kv_heads_per_rank = kv_heads // cp_size
        q_heads_per_kv_head = q_heads // kv_heads
        q_heads_per_rank = kv_heads_per_rank * q_heads_per_kv_head
        kv_start = cp_rank * kv_heads_per_rank
        kv_end = kv_start + kv_heads_per_rank
        q_start = cp_rank * q_heads_per_rank
        q_end = q_start + q_heads_per_rank
        return slice(q_start, q_end), slice(kv_start, kv_end)

    def _slice_local_heads(
        self,
        packed_query_states: SequencePack,
        packed_key_states: SequencePack,
        packed_value_states: SequencePack,
    ) -> tuple[SequencePack, SequencePack, SequencePack]:
        # Input heads are full and sequence-replicated on every CP rank:
        # q: [S,H,D], k/v: [S,H_kv,D].
        q_und_seq = get_und_seq(packed_query_states)  # [S_und,H,D]
        q_gen_seq = get_gen_seq(packed_query_states)  # [S_curr,H,D]
        k_und_seq = get_und_seq(packed_key_states)  # [S_und,H_kv,D]
        k_gen_seq = get_gen_seq(packed_key_states)  # [S_curr,H_kv,D]
        v_und_seq = get_und_seq(packed_value_states)  # [S_und,H_kv,D]
        v_gen_seq = get_gen_seq(packed_value_states)  # [S_curr,H_kv,D]

        # Slice the contiguous Q-head group that corresponds to this rank's
        # contiguous KV-head group: q -> [S,H/CP,D], k/v -> [S,H_kv/CP,D].
        q_slice, kv_slice = self._head_slices(q_gen_seq.shape[1], k_gen_seq.shape[1])
        q_und_local = q_und_seq[:, q_slice, :].contiguous()  # [S_und,H_local,D]
        q_gen_local = q_gen_seq[:, q_slice, :].contiguous()  # [S_curr,H_local,D]
        k_und_local = k_und_seq[:, kv_slice, :].contiguous()  # [S_und,H_kv_local,D]
        k_gen_local = k_gen_seq[:, kv_slice, :].contiguous()  # [S_curr,H_kv_local,D]
        v_und_local = v_und_seq[:, kv_slice, :].contiguous()  # [S_und,H_kv_local,D]
        v_gen_local = v_gen_seq[:, kv_slice, :].contiguous()  # [S_curr,H_kv_local,D]

        local_query_pack = from_und_gen_splits(  # [S_und+S_curr,H_local,D]
            q_und_local, q_gen_local, packed_query_states
        )
        local_key_pack = from_und_gen_splits(  # [S_und+S_curr,H_kv_local,D]
            k_und_local, k_gen_local, packed_key_states
        )
        local_value_pack = from_und_gen_splits(  # [S_und+S_curr,H_kv_local,D]
            v_und_local, v_gen_local, packed_value_states
        )
        return local_query_pack, local_key_pack, local_value_pack

    def forward(
        self,
        packed_query_states: SequencePack,
        packed_key_states: SequencePack,
        packed_value_states: SequencePack,
        attention_mask: SplitInfo,
        natten_metadata: dict | None = None,
        memory_value: MemoryValue | None = None,
        packed_key_states_normalized: SequencePack | None = None,
    ) -> tuple[SequencePack, KVToStore | None]:
        # Check the stable post-saturation flag first so the statically compiled
        # path does not guard on the changing Python ``frame_idx`` value and
        # recompile once per AR refresh/denoise call.
        is_ar_gen_only = memory_value is not None and (
            getattr(memory_value, "post_saturation_static_compile", False) or getattr(memory_value, "frame_idx", 0) > 0
        )
        if not is_ar_gen_only:
            return self.wrapped_dispatch(
                packed_query_states,
                packed_key_states,
                packed_value_states,
                attention_mask,
                natten_metadata=natten_metadata,
                memory_value=memory_value,
                packed_key_states_normalized=packed_key_states_normalized,
            )
        if getattr(memory_value, "for_cuda_graphs", False):
            raise ValueError("replicated attention_io_layout does not support ARMemoryState(for_cuda_graphs=True)")

        local_query_pack, local_key_pack, local_value_pack = self._slice_local_heads(
            packed_query_states,
            packed_key_states,
            packed_value_states,
        )
        local_key_pack_normalized: SequencePack | None = None
        if packed_key_states_normalized is not None:
            _, local_key_pack_normalized, _ = self._slice_local_heads(
                packed_query_states,
                packed_key_states_normalized,
                packed_value_states,
            )
        local_output_pack, kv_to_store = self.wrapped_dispatch(  # local_output_pack: [S_curr,H_local*D]
            local_query_pack,
            local_key_pack,
            local_value_pack,
            attention_mask,
            natten_metadata=natten_metadata,
            memory_value=memory_value,
            packed_key_states_normalized=local_key_pack_normalized,
        )
        return local_output_pack, kv_to_store


def apply_replicated_attention_io_cp(
    model: nn.Module,
    parallel_dims: ParallelDims,
) -> nn.Module:
    """Install replicated-attention-I/O context parallelism on every attention layer."""
    cp_mesh = parallel_dims.cp_mesh
    assert cp_mesh is not None, "replicated attention I/O requires a CP mesh"
    cp_size = parallel_dims.cp_size
    first_block = next(iter(model.model.layers.children()))
    first_attn = first_block.self_attn
    num_kv_heads = int(first_attn.num_key_value_heads)
    num_attention_heads = int(first_attn.num_attention_heads)
    assert num_kv_heads % cp_size == 0, (
        f"replicated attention_io_layout requires num_kv_heads({num_kv_heads}) % cp_size({cp_size}) == 0. "
        f"num_kv_heads={num_kv_heads} is the upper bound for useful local-head attention CP."
    )
    log.info(
        "[replicated attention I/O CP] enabled "
        f"cp_size={cp_size}, num_kv_heads={num_kv_heads}, num_attention_heads={num_attention_heads}, "
        f"kv_heads_per_rank={num_kv_heads // cp_size}, max_useful_cp_size={num_kv_heads}",
        rank0_only=True,
    )
    for _, block in model.model.layers.named_children():
        attn = block.self_attn
        attn.replicated_attention_io_local_head_o_proj = True
        attn.replicated_attention_io_cp_mesh = cp_mesh
        attn.dispatch_attention_fn = ARReplicatedIODispatch(
            cp_mesh,
            wrapped_dispatch=attn.dispatch_attention_fn,
        )
    return model
