# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""FSDP / activation-checkpointing / torch.compile pass for the unified MoT.

The activation-checkpointing implementation here mirrors the torchtitan SAC
design (``torchtitan/distributed/activation_checkpoint.py``):

  * Per-op selective AC saves a curated set of compute and communication ops
    (SDPA variants, FlexAttention, ``aten.linear``, NCCL collectives,
    DeepEP/HybridEP) and recomputes everything else.
"""

import re
from typing import Callable

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.fsdp import fully_shard, register_fsdp_forward_method
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from cosmos_framework.utils import log
from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.model.generator.mot.attention import SplitInfo, dispatch_attention
from cosmos_framework.model.generator.mot.context_parallel_utils import context_parallel_attention
from cosmos_framework.model.generator.utils.memory import KVToStore, MemoryValue
from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    from_und_gen_splits,
    get_gen_seq,
    get_und_seq,
)
from cosmos_framework.utils.generator.parallelism import ParallelDims


class ContextParallelDispatch(nn.Module):
    """CP-aware wrapper for the installed attention dispatch function.

    Installed on ``PackedAttentionMoT.dispatch_attention_fn`` when context
    parallelism is enabled, replacing whatever dispatch function was there
    previously.  The call signature of :meth:`forward` matches
    ``dispatch_attention`` so the two are interchangeable.

    All paths delegate to :func:`context_parallel_attention`, which wraps
    the inner ``wrapped_dispatch`` with Ulysses-style all-to-all
    communication.  This includes the AR frame 1+ gen-only path — the inner
    dispatch routes to ``attention_AR_gen_only`` which operates on the
    local-head tensors produced by the all-to-all.

    All cache writes flow through the ``MemoryState`` interface; neither this
    class nor the CP attention functions write to the cache directly.
    """

    def __init__(
        self,
        cp_mesh,
        wrapped_dispatch: Callable = dispatch_attention,
    ):
        super().__init__()
        self.cp_mesh = cp_mesh
        self.wrapped_dispatch = wrapped_dispatch

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
        if memory_value is not None and not memory_value.supports_context_parallel_attention:
            raise ValueError("Context-parallel doesn't work when training with a KV-cache.")

        return context_parallel_attention(
            self.cp_mesh,
            packed_query_states,
            packed_key_states,
            packed_value_states,
            attention_mask,
            attention_function=self.wrapped_dispatch,
            natten_metadata=natten_metadata,
            memory_value=memory_value,
            packed_key_states_normalized=packed_key_states_normalized,
        )


class ARReplicatedIODispatch(nn.Module):
    """AR CP dispatch for replicated attention I/O with local-head attention.

    ``Replicated I/O`` means the caller-side tensors at the attention boundary
    are replicated across CP ranks.  It does **not** mean attention compute is
    replicated.  For AR frame 1+, this wrapper slices the replicated current
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

    Current-frame hidden states stay replicated.  For AR frame 1+, this wrapper
    delegates to the existing memory-aware AR attention for local heads, then
    returns the local current-frame attention output so ``PackedAttentionMoT``
    can apply the corresponding ``o_proj`` column slice.  Frame 0 and non-AR
    paths delegate unchanged; frame 0 seeds the local KV-head cache through
    ``ARMemoryState.write_for_layer``.
    """

    def __init__(
        self,
        cp_mesh,
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

        local_query_pack = from_und_gen_splits(q_und_local, q_gen_local, packed_query_states)
        local_key_pack = from_und_gen_splits(k_und_local, k_gen_local, packed_key_states)
        local_value_pack = from_und_gen_splits(v_und_local, v_gen_local, packed_value_states)
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
        if memory_value is None or getattr(memory_value, "frame_idx", 0) <= 0:
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
        local_output_pack, kv_to_store = self.wrapped_dispatch(
            local_query_pack,
            local_key_pack,
            local_value_pack,
            attention_mask,
            natten_metadata=natten_metadata,
            memory_value=memory_value,
            packed_key_states_normalized=local_key_pack_normalized,
        )
        return local_output_pack, kv_to_store


def _apply_selective_ac(
    module: nn.Module,
    ac: ActivationCheckpointingConfig,
) -> nn.Module:
    """Apply per-op selective activation checkpointing to ``module``."""
    save_ops_regex = [re.compile(pattern) for pattern in ac.save_ops_regex]

    def _get_custom_policy():
        def wrapped_policy(ctx, func, *args, **kwargs) -> CheckpointPolicy:
            op_name = getattr(func, "__name__", str(func))
            if any(pattern.search(op_name) for pattern in save_ops_regex):
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.MUST_RECOMPUTE

        return wrapped_policy

    return ptd_checkpoint_wrapper(
        module,
        context_fn=lambda: create_selective_checkpoint_contexts(_get_custom_policy()),
        preserve_rng_state=ac.preserve_rng_state,
        determinism_check=ac.determinism_check,
    )


def _apply_full_ac(
    module: nn.Module,
    config: ActivationCheckpointingConfig,
) -> nn.Module:
    """Apply full activation checkpointing to ``module``."""
    return ptd_checkpoint_wrapper(
        module,
        preserve_rng_state=config.preserve_rng_state,
        determinism_check=config.determinism_check,
    )


def _apply_ac_to_transformer_block(
    module: nn.Module,
    config: ActivationCheckpointingConfig,
) -> nn.Module:
    if config.mode == "full":
        return _apply_full_ac(module, config)
    elif config.mode == "selective":
        return _apply_selective_ac(module, config)
    else:
        raise ValueError(f"Invalid AC mode: {config.mode}.")


def apply_ac(
    model: nn.Module,
    config: ActivationCheckpointingConfig,
) -> None:
    """Apply activation checkpointing to ``model.model.layers``.

    Args:
        model: The unified MoT model whose ``model.layers.*`` blocks will be
            wrapped (or whose compiled region will be tagged with a memory
            budget for the partitioner).
        config: AC policy (``OmniMoTModelConfig.activation_checkpointing``).
    """
    if config.mode == "none":
        return

    layers = model.model.layers
    for layer_id, transformer_block in layers.named_children():
        transformer_block = _apply_ac_to_transformer_block(
            transformer_block,
            config,
        )
        layers.register_module(layer_id, transformer_block)


def apply_compile(model: nn.Module, config: CompileConfig) -> None:
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    compile_options = {}
    if config.max_autotune_pointwise:
        compile_options["max_autotune_pointwise"] = True
    if config.coordinate_descent_tuning:
        compile_options["coordinate_descent_tuning"] = True

    for layer_id, block in model.model.layers.named_children():
        block = torch.compile(
            block,
            fullgraph=True,
            dynamic=config.compile_dynamic,
            mode="reduce-overhead" if config.use_cuda_graphs else None,
            options=compile_options or None,
        )
        model.model.layers.register_module(layer_id, block)


def apply_cp(
    model: nn.Module,
    parallel_dims: ParallelDims,
) -> nn.Module:
    """Install :class:`ContextParallelDispatch` on every attention layer.

    Walks the unified-MoT decoder stack and wraps each
    ``self_attn.dispatch_attention_fn`` with a CP-aware dispatcher that
    pre/post-pends Ulysses-style all-to-all communication around the
    inner attention.  The wrapper carries its own reference to
    ``cp_mesh`` (captured in :meth:`ContextParallelDispatch.__init__`),
    so the CP-aware dispatch path never has to read a mesh attribute
    off the attention module itself.

    Must run BEFORE :func:`apply_ac`, :func:`apply_compile`, and
    :func:`apply_fsdp` so the activation-checkpoint wrapper / compiled
    graph / FSDP unit each see the CP-aware dispatch in place; rewiring
    ``dispatch_attention_fn`` after compile would silently regress to
    the non-CP path inside the traced kernel.

    Args:
        model: The unified-MoT model whose
            ``model.model.layers[*].self_attn`` will be CP-wrapped.
        parallel_dims: Parallelism dims with ``cp_enabled`` already
            checked by the caller; ``cp_mesh`` is guaranteed non-``None``
            here because ``build_meshes`` populates it whenever
            ``cp_enabled``.
    """
    cp_mesh = parallel_dims.cp_mesh
    for _, block in model.model.layers.named_children():
        attn = block.self_attn
        attn.dispatch_attention_fn = ContextParallelDispatch(
            cp_mesh,
            wrapped_dispatch=attn.dispatch_attention_fn,
        )
    return model


def apply_replicated_attention_io_cp(
    model: nn.Module,
    parallel_dims: ParallelDims,
) -> nn.Module:
    """Install replicated-attention-IO context parallelism on every attention layer."""
    cp_mesh = parallel_dims.cp_mesh
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


def apply_fsdp(
    model: nn.Module,
    parallel_dims: ParallelDims,
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Also registers each decoder block's ``reasoner_forward`` (used by the
    AR text-generation loop in ``unified_mot._impl_generate_reasoner_text``)
    as an FSDP2 forward-equivalent so its pre-forward unshard / post-forward
    reshard hooks fire on every call.  Without this registration the AR
    loop touches ``layer.input_layernorm.weight`` et al. while they are
    still ``DTensor`` shards and raises ``RuntimeError: aten.mul.Tensor:
    got mixed torch.Tensor and DTensor`` — the per-block companion to the
    top-level ``register_fsdp_forward_method(model, "generate_reasoner_text")``
    in ``parallelize_vfm_network``.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        parallel_dims (ParallelDims): The device mesh to use for data parallelism and expert parallel.
    """
    for _, block in model.model.layers.named_children():
        fully_shard(block, mesh=parallel_dims.dp_mesh)
        register_fsdp_forward_method(block, "reasoner_forward")


def parallelize_unified_mot(
    model: nn.Module,
    parallel_dims: ParallelDims | None,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    attention_io_layout: str = "sequence_sharded",
) -> nn.Module:
    """Optimize the model using CP, FSDP, activation checkpointing, and torch.compile.

    Context parallelism is installed first (before AC / compile / FSDP)
    so the CP-aware ``dispatch_attention_fn`` is captured by every
    downstream wrapper.  FSDP reduces memory usage by sharding the model
    parameters across multiple GPUs.  Activation checkpointing reduces
    memory usage by selectively checkpointing only the outputs of each
    layer. Torch.compile compiles the model for faster training.

    Args:
        model: The unified MoT (typically ``omni_model.language_model``).
        parallel_dims: Device mesh / parallelism descriptor.
        compile_config: Compile switches (enabled, dynamic, autotune).
        ac_config: Selective activation-checkpointing policy. ``None`` falls
            back to the dataclass defaults (mode="selective", save the
            ``save_ops_regex`` ops, mode="full", save only the outputs of
            each transformer block).
        attention_io_layout: Tensor layout at the attention boundary under CP.

    """
    if parallel_dims is not None and parallel_dims.cp_enabled:
        if attention_io_layout == "replicated":
            apply_replicated_attention_io_cp(model, parallel_dims)
        elif attention_io_layout == "sequence_sharded":
            apply_cp(model, parallel_dims)
        else:
            raise ValueError(f"Unsupported attention_io_layout={attention_io_layout!r}")
    apply_ac(model, ac_config)
    if compile_config.enabled:
        apply_compile(model, compile_config)
    if parallel_dims is not None and parallel_dims.dp_enabled:
        apply_fsdp(model, parallel_dims)
    return model
