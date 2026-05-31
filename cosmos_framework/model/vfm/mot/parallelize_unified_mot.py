# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Callable

"""FSDP / activation-checkpointing / torch.compile pass for the unified MoT.

The activation-checkpointing implementation here mirrors the torchtitan SAC
design (``torchtitan/distributed/activation_checkpoint.py``):

  * Per-op selective AC saves a curated set of compute and communication ops
    (SDPA variants, FlexAttention, ``aten.linear``, NCCL collectives,
    DeepEP/HybridEP) and recomputes everything else.
"""

import re

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.fsdp import fully_shard, register_fsdp_forward_method
from torch.nn.attention.flex_attention import BlockMask
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.data.vfm.sequence_packing import (
    FactoredSequencePack,
    JointSequencePack,
)
from cosmos_framework.model.vfm.mot.attention import SplitInfo, dispatch_attention
from cosmos_framework.model.vfm.mot.context_parallel_utils import context_parallel_attention
from cosmos_framework.model.vfm.utils.memory import KVToStore, MemoryValue
from cosmos_framework.utils.vfm.parallelism import ParallelDims


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
    head-sharded tensors produced by the all-to-all.

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
        packed_query_states: FactoredSequencePack | JointSequencePack,
        packed_key_states: FactoredSequencePack | JointSequencePack,
        packed_value_states: FactoredSequencePack | JointSequencePack,
        attention_mask: BlockMask | SplitInfo,
        natten_metadata: dict | None = None,
        memory_value: MemoryValue | None = None,
    ) -> tuple[FactoredSequencePack | JointSequencePack, KVToStore | None]:
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
        )


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

    """
    if parallel_dims is not None and parallel_dims.cp_enabled:
        apply_cp(model, parallel_dims)
    apply_ac(model, ac_config)
    if compile_config.enabled:
        apply_compile(model, compile_config)
    if parallel_dims is not None and parallel_dims.dp_enabled:
        apply_fsdp(model, parallel_dims)
    return model
