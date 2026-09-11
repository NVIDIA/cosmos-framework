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
from torch._dynamo.decorators import mark_unbacked
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard, register_fsdp_forward_method
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.model.generator.mot.attention import SplitInfo, dispatch_attention
from cosmos_framework.model.generator.mot.context_parallel_utils import context_parallel_attention
from cosmos_framework.model.generator.utils.memory import KVToStore, MemoryValue
from cosmos_framework.data.generator.sequence_packing.runtime import SequencePack
from cosmos_framework.utils.generator.parallelism import ParallelDims, fsdp_mesh
from cosmos_framework.model.generator.mot.replicated_io import apply_replicated_attention_io_cp


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


# Sequence packing hands each transformer block a fresh ``causal_seq`` /
# ``full_only_seq`` token count (and the offset / sample-id tensors that
# describe them) on essentially every step, since the pack layout depends on
# which samples got packed together. Dynamo 0/1-specializes the first shape
# it traces for these and Inductor further specializes codegen (32-bit
# indexing, broadcast layout) on that concrete value, so the next batch with
# a different count fails those guards and forces a recompile. Production
# logs show exactly this: repeated ``Recompiling function forward in
# .../checkpoint_wrapper.py`` with guards like
# ``2 <= causal_seq.storage_offset()``, ``(...) // head_dim != 1``, and
# ``_causal_sample_ids size mismatch (expected 1, actual 45)`` — all
# instances of Dynamo treating a dim it first saw as 0/1 or a fixed value as
# guaranteed to stay that way. These are the dim-0 (token-count) tensors
# that vary per pack; everything else in a ``SequencePack`` (mode lists, the
# ``max_*_len`` bounds, etc.) is a Python scalar or list, which Dynamo
# specializes on regardless of anything done here.
#
# All three offset tensors carry their pad segment in place (see
# ``runtime.has_pad_segment``), so marking them covers the padded path -- the
# common one in production, since alignment and CUDA-graph bucketing pad most
# packs -- without any separate entries.
_PACK_DYNAMIC_LEN_KEYS: tuple[str, ...] = (
    "causal_seq",
    "full_only_seq",
    "_causal_sample_ids",
    "_full_only_sample_ids",
    "_causal_indices",
    "_full_indices",
    "_causal_seq_offsets",
    "_full_only_seq_offsets",
    "sample_offsets",
)


def _mark_pack_unbacked(pack: SequencePack) -> None:
    """Mark ``pack``'s per-step-varying dim-0 tensors as unbacked before a compiled call.

    Must run on every call (not once at compile time): each training step builds a brand new
    ``SequencePack`` with brand new tensor objects, and Dynamo's unbacked marking is stored as
    an attribute on the tensor object itself, not learned from previous calls.

    ``mark_unbacked`` makes the compiler report dim 0's size as "always not equal to zero or
    one" (see ``torch._dynamo.decorators.mark_unbacked``'s docstring) -- a hard-coded assumption,
    not a guess it falls back from. ``sequence_packing/runtime.py`` documents that this dimension
    *can* legitimately be 0 for some real batches (e.g. AR no-text packs carry full splits only,
    so their ``causal_seq`` is empty), so marking a genuinely 0/1-sized tensor here would violate
    that assumption and risk silently wrong compiled output instead of a guard failure. Skip those
    and let Dynamo fall back to its normal backed/specialized handling for that one call --
    correctness over avoiding an occasional recompile.

    Deliberately without ``strict=True``, despite how its docstring reads. ``strict`` does not
    strengthen the marking -- it replaces it. ``mark_unbacked`` returns early under ``strict``,
    recording the index in ``_dynamo_strict_unbacked_indices`` and never in
    ``_dynamo_unbacked_indices``, and only the latter selects ``DimDynamic.UNBACKED`` in
    ``_dynamo/variables/builder.py``. The dim instead falls through to ordinary automatic-dynamic
    and gets a *backed* symbol, which carries a hint -- and a hint is exactly what lets Inductor
    answer size questions in ``broadcast_symbolic_shapes`` and ``scheduler.can_fuse`` and install
    the guards whose failures caused the recompiles this marking exists to remove. What ``strict``
    does supply is a constraint that raises if the dim is *constant-folded*, which is a narrower
    event than being guarded on, so it never fired while the recompiles continued. Measured
    identically on torch 2.9 and 2.13: ``mark_unbacked(x, 0)`` yields ``u0``, ``strict=True``
    yields ``s77``.

    Consumers must therefore be data-dependent-friendly, since an unbacked dim has no value to
    settle a branch or a comparison with. Three kinds of site need care, all of them reached from
    the compiled block: a plain ``if`` on one of these lengths (use ``guard_or_true`` /
    ``guard_or_false``), an invariant the attention stack re-derives and guards on internally
    (state it with ``torch._check``), and an ``int()`` coercion, which demands a concrete value
    outright and has no friendly form; leave the length symbolic instead.

    A consumer that derives a *smaller* count from one of these lengths and uses it as a tensor
    dimension needs the 0/1 exclusion to shift with it, because PyTorch specializes the derived
    dim, not the one marked here: a size-1 derived dim is specialized to the constant 1, which
    pins the marked length to a constant too. ``sample_offsets`` is the one such key -- it carries
    ``N + 1`` offsets for ``N`` samples -- and its consumer,
    ``unified_mot._get_local_sample_ids``, avoids the whole situation by reading the pad-segment
    length instead, so no shifted exclusion is needed here.

    Marks unconditionally: whether a given call should mark at all is
    :func:`_wrap_forward_with_unbacked_pack`'s decision, since both of the conditions that answer it
    are properties of the call rather than of the pack.
    """
    for key in _PACK_DYNAMIC_LEN_KEYS:
        tensor = pack.get(key)
        if not isinstance(tensor, torch.Tensor) or tensor.dim() == 0:
            continue
        if tensor.shape[0] in (0, 1):
            continue
        mark_unbacked(tensor, 0)


def _wrap_forward_with_unbacked_pack(forward: Callable) -> Callable:
    """Wrap a compiled block's ``forward`` to mark its packed-sequence args unbacked first.

    ``input`` is the block's own packed sequence; ``packed_position_embeddings`` is the
    ``(cos, sin)`` pair of packs describing RoPE embeddings over the same layout, so it varies in
    lockstep with ``input`` and needs the same treatment.

    Packs whose ``attention_mask`` carries a FlexAttention block mask are left alone, because that
    path cannot currently be compiled with unbacked lengths. The obstruction is not in this repo:
    the generator's full attention runs over a fused ``[UND | GEN]`` stream, whose outer stride is
    ``heads * head_dim * (u_und + u_gen)``, and Inductor's layout pass sorts strides through
    ``ir.get_fill_order``, which falls back to a plain ``sorted()`` whenever its caller passes no
    ShapeEnv. Comparing that stride against a constant inside ``sorted()`` raises
    ``GuardOnDataDependentSymNode`` from the middle of an Inductor pass. Nothing in this
    repo can annotate a comparison there: ``torch._check`` relates whole lengths, and the ShapeEnv
    does not substitute such a relation back into a stride expression already built from the
    individual terms.

    Marking is also confined to training via ``torch.is_grad_enabled``, on the same reasoning that
    gates ``attention._use_varlen``'s dense shortcut: the recompiles it removes come from batch
    composition varying between training steps, while inference holds its pack layout fixed across
    the denoising loop. Marking there buys nothing and breaks compilation outright, because an
    unbacked length used as a slice offset makes the resulting view's ``storage_offset``
    data-dependent -- ``heads * head_dim * u0`` for the heads-flattened attention output that
    ``o_proj`` consumes.

    Both conditions are tested here, per call, and neither can move to ``apply_compile`` beside
    ``CompileConfig.mark_unbacked``. That flag is read once at setup; these two are properties of
    the individual call. Grad state in particular flips within one process -- the periodic sampling
    callbacks drive these same compiled blocks under ``torch.no_grad`` -- so a setup-time reading
    would say "training" forever and mark straight through the first sampling callback.
    """

    def wrapped(
        input: SequencePack,
        attention_mask,
        packed_position_embeddings: tuple[SequencePack, SequencePack],
        *args,
        **kwargs,
    ):
        if torch.is_grad_enabled() and getattr(attention_mask, "flex_block_mask", None) is None:
            _mark_pack_unbacked(input)
            for pos_emb_pack in packed_position_embeddings:
                _mark_pack_unbacked(pos_emb_pack)
        return forward(input, attention_mask, packed_position_embeddings, *args, **kwargs)

    return wrapped


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
        # Instance-attribute override, not a subclass/wrapper module: OptimizedModule already
        # installs ``forward`` as an instance attribute (see torch._dynamo.eval_frame), so this
        # replaces only the call path and leaves module structure (children, state_dict) untouched.
        #
        # Skipping the wrapper entirely, rather than gating inside it, is what makes
        # ``mark_unbacked=False`` a true restoration of the pre-marking behavior: the block is then
        # the compiled module itself, with no extra frame between it and the caller. This is also
        # the only place the marking is installed, so the flag disables it everywhere.
        if config.mark_unbacked:
            block.forward = _wrap_forward_with_unbacked_pack(block.forward)
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
    mp_policy: MixedPrecisionPolicy | None = None,
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

    The mesh comes from :func:`fsdp_mesh`, not ``parallel_dims.dp_mesh`` directly: the latter
    is 2-D even when ``dp_replicate == 1``, which silently puts a pure-FSDP run on FSDP2's
    HSDP path and pays a one-rank ``all_reduce`` per block per step.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        parallel_dims (ParallelDims): The device mesh to use for data parallelism and expert parallel.
            Called whenever ``dp_enabled`` holds, which for training includes a single-rank
            ``(1, 1)`` mesh: the wrap is what installs ``mp_policy``, so it has to happen
            even where there is no cross-rank sharding to gain.
        mp_policy (MixedPrecisionPolicy | None): Mixed-precision policy for each block's FSDP
            unit. ``None`` means FSDP2's default (no casting): the parameters are used for
            compute and gradient reduction in whatever dtype the module holds them. Pass a
            policy to keep the sharded parameters -- the ones the optimizer steps -- in a
            higher precision than the forward/backward, i.e. ``param_dtype`` is the compute
            (and all-gather) dtype and ``reduce_dtype`` the gradient-reduction dtype, which
            must match the dtype the module's parameters are stored in.
    """
    mesh = fsdp_mesh(parallel_dims)
    for _, block in model.model.layers.named_children():
        fully_shard(
            module=block,
            mesh=mesh,
            mp_policy=mp_policy or MixedPrecisionPolicy(),
        )
        register_fsdp_forward_method(block, "reasoner_forward")


def parallelize_unified_mot(
    model: nn.Module,
    parallel_dims: ParallelDims | None,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    attention_io_layout: str = "sequence_sharded",
    mp_policy: MixedPrecisionPolicy | None = None,
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
        mp_policy: FSDP2 mixed-precision policy, forwarded to :func:`apply_fsdp`.
            ``None`` keeps FSDP2's default of no casting.

    """
    if parallel_dims is not None and parallel_dims.cp_enabled:
        if attention_io_layout == "sequence_sharded":
            apply_cp(model, parallel_dims)
        elif attention_io_layout == "replicated":
            apply_replicated_attention_io_cp(model, parallel_dims)
        else:
            raise ValueError(f"Unsupported attention_io_layout={attention_io_layout!r}")
    apply_ac(model, ac_config)
    if compile_config.enabled:
        apply_compile(model, compile_config)
    if parallel_dims is not None and parallel_dims.dp_enabled:
        apply_fsdp(model, parallel_dims, mp_policy)
    return model
