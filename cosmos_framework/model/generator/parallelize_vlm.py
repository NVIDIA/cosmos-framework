# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Activation checkpointing, ``torch.compile`` and FSDP2 for Cosmos3 VLM ``HFModel``s.

Hosts the single VLM-specific ``parallelize`` entry point used by
``vlm_model.VLMModel._init_vlm``.  Lives under ``cosmos_framework/model/generator/``
so the FSDP wrapping concern sits next to the model class it operates on
(mirroring the layout of ``models/mot/parallelize_unified_mot.py`` for the
MoT path).

Pure parallelism plumbing — :class:`~cosmos_framework.utils.generator.parallelism.ParallelDims`
and its meshes — stays in ``vfm/utils/parallelism.py``.
"""

import re
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from cosmos_framework.utils import log
from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.configs.base.defaults.parallelism import (
    PRECISION_TO_TORCH_DTYPE,
    ParallelismConfig,
)
from cosmos_framework.model.generator.hf_model import HFModel
from cosmos_framework.utils.generator.parallelism import ParallelDims, fsdp_mesh

# (parent, attribute name, module) — the module currently registered at that slot, plus what
# is needed to swap in a replacement. Each pass in `parallelize` replaces the module with a
# wrapper around it (checkpoint wrapper, then OptimizedModule) and hands the updated slots on.
_BlockSlot = tuple[nn.Module, str, nn.Module]


def _collect_repeated_blocks(inner: nn.Module) -> tuple[list[_BlockSlot], set[str]]:
    """Collect the repeated transformer blocks by their ORIGINAL type name.

    Matches ``inner._no_split_modules`` — the decoder layers (+ vision blocks for
    VLMs, e.g. Qwen3-VL ``visual.blocks``). MUST run before any ``fully_shard``
    call: ``fully_shard`` mutates each block's ``__class__`` to a dynamically
    created ``FSDP<OrigName>`` type, after which the typename match finds nothing.
    :func:`apply_ac` replaces each block with a ``CheckpointWrapper``, which also
    defeats the typename match, so this runs exactly ONCE per ``parallelize`` and the
    resulting list is threaded through the later passes.

    ``_no_split_modules`` is unioned over the root AND every submodule, mirroring
    ``PreTrainedModel._get_no_split_modules``. Composite VLMs declare it on the sub-model
    that owns the blocks rather than on the root they are assembled into — Nemotron-3-Dense-VL
    declares it only on its nested ``Siglip2VisionTransformer``, so a root-only lookup would
    find nothing for that family.

    A name match alone is NOT enough to make a module a block. HF's attribute means "never
    split these types across devices under ``device_map``", so a declaration legitimately
    names singletons — embeddings, pooling heads, projectors — beside the repeated layers.
    Treating a singleton as a block is wasted work at best, and ``fully_shard``-ing one breaks
    any parent that reads its parameters outside the child's ``forward``, where FSDP has
    already resharded them. So a match is kept only when it is a child of an ``nn.ModuleList``
    — the shape every supported family uses for its repeated layers, and one that still holds
    for a single-layer config where an "appears more than once" test would collect nothing.

    Returns ``(parent, attr_name, block)`` triples — the parent and attribute name are
    what :func:`apply_ac` needs to swap in the checkpoint wrapper — plus the
    ``_no_split_modules`` names (reported in :func:`apply_ac`'s no-blocks raise so a
    misconfiguration is visible).
    """
    no_split_names: set[str] = set()
    for module in inner.modules():
        no_split_names.update(getattr(module, "_no_split_modules", None) or [])
    collected: list[_BlockSlot] = []
    for name, module in inner.named_modules():
        if type(module).__name__ not in no_split_names:
            continue
        # rpartition splits the dotted path into owner + attribute; get_submodule("") returns
        # inner itself, which the ModuleList check below then rejects — a module registered
        # directly on the root is a singleton, not one of a repeated stack.
        parent_path, _, attr_name = name.rpartition(".")
        parent = inner.get_submodule(parent_path)
        if not isinstance(parent, nn.ModuleList):
            continue
        collected.append((parent, attr_name, module))
    return collected, no_split_names


def _replace_blocks(slots: list[_BlockSlot], transform: Callable[[nn.Module], nn.Module]) -> list[_BlockSlot]:
    """Swap each slot's module for ``transform(module)`` and return the updated slots.

    Shared by :func:`apply_ac` and :func:`apply_compile` so both wrap-and-re-register the
    same way, and so the slot list handed to the next pass always points at the module that
    is actually registered in the tree.
    """
    replaced: list[_BlockSlot] = []
    for parent, attr_name, module in slots:
        replacement = transform(module)
        # register_module, not setattr: nn.ModuleList children are keyed by the stringified
        # index ("0", "1", ...), which is not a settable attribute.
        parent.register_module(attr_name, replacement)
        replaced.append((parent, attr_name, replacement))
    return replaced


def _resolve_reduce_dtype(parallelism_config: ParallelismConfig) -> torch.dtype:
    """Resolve ``MixedPrecisionPolicy.reduce_dtype`` for the gradient reduce-scatter.

    ``fsdp_reduce_dtype`` wins when set; ``None`` falls back to ``fsdp_master_dtype`` so
    the reduction happens in the master-parameter dtype.

    Letting the two differ is safe: FSDP2's ``foreach_reduce`` casts the reduce-scatter
    output back to the sharded parameter's dtype before assigning ``.grad``
    (``_to_dtype_if_needed(reduce_output, orig_dtype)``), so a lower ``reduce_dtype``
    changes only the collective and its staging buffer, not the master copy the optimizer
    steps on.
    """
    return PRECISION_TO_TORCH_DTYPE[parallelism_config.fsdp_reduce_dtype or parallelism_config.fsdp_master_dtype]


def _collect_ignored_audio_params(model: HFModel) -> set[nn.Parameter]:
    """Return immutable audio parameters that must stay outside FSDP.

    The selected audio encoder is always a frozen, artifact-backed feature extractor.
    The projector remains part of the raw HF root (and therefore FSDP-managed)
    even when it is frozen, matching the existing vision merger/projector
    lifecycle and keeping checkpoint tensor layouts uniform.
    """
    if not model.sound_und:
        return set()
    return set(model.model.sound_und_model.encoder.parameters())


def _compile_kwargs(compile_config: CompileConfig) -> dict[str, Any]:
    """Translate a ``CompileConfig`` into ``torch.compile`` keyword arguments.

    ``max_autotune_pointwise`` / ``coordinate_descent_tuning`` map straight to the
    torch.compile ``options=`` dict (same as the MoT apply_compile).

    ``fullgraph=False``, unlike ``parallelize_unified_mot.apply_compile``, because the traced
    unit here is HF ``transformers`` modeling code plus the ``monkey_patch`` overrides rather
    than first-party blocks written to trace cleanly, and it is not established to be
    break-free — so a break must degrade to eager instead of aborting startup. Compile still
    fuses the surrounding pointwise/norm regions (the ~6.8k tiny elementwise kernels) where
    the win is.

    NOT the attention backend, contrary to what this said previously: ``cosmos_framework.model.attention``
    is meta-registered and fullgraph-tested (``cosmos_framework/attention/tests/torch_compile_test.py``,
    ``cosmos_framework/attention/cudnn/functions.py``), and even a genuinely opaque custom op stays in
    the graph as an untraced call when it carries a fake kernel (``mot/dot_product_attention.py``
    registers one for exactly that reason).

    The cost of False is that a newly introduced break is silent, surfacing as lost throughput
    rather than a failure. Enumerate the real breaks with ``TORCH_LOGS=graph_breaks`` before
    flipping this.
    """
    options: dict[str, bool] = {}
    if compile_config.max_autotune_pointwise:
        options["max_autotune_pointwise"] = True
    if compile_config.coordinate_descent_tuning:
        options["coordinate_descent_tuning"] = True
    return {
        "fullgraph": False,
        "dynamic": compile_config.compile_dynamic,
        "options": options or None,
    }


def _apply_selective_ac(module: nn.Module, ac: ActivationCheckpointingConfig) -> nn.Module:
    """Apply per-op selective activation checkpointing to ``module``.

    Mirrors ``parallelize_unified_mot._apply_selective_ac``: ops whose name matches
    ``save_ops_regex`` are saved, everything else is recomputed.
    """
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


def _apply_full_ac(module: nn.Module, ac: ActivationCheckpointingConfig) -> nn.Module:
    """Apply full activation checkpointing to ``module``."""
    return ptd_checkpoint_wrapper(
        module,
        preserve_rng_state=ac.preserve_rng_state,
        determinism_check=ac.determinism_check,
    )


def _apply_ac_to_transformer_block(module: nn.Module, ac: ActivationCheckpointingConfig) -> nn.Module:
    if ac.mode == "full":
        return _apply_full_ac(module, ac)
    elif ac.mode == "selective":
        return _apply_selective_ac(module, ac)
    else:
        raise ValueError(f"Invalid AC mode: {ac.mode}.")


def apply_ac(model: HFModel, ac: ActivationCheckpointingConfig) -> list[_BlockSlot]:
    """Wrap each repeated transformer block in a ``CheckpointWrapper``, if enabled.

    Owns activation checkpointing for the VLM path, replacing HF's
    ``gradient_checkpointing_enable``. Same strategy as the MoT path
    (``parallelize_unified_mot.apply_ac``): the block is swapped for a
    ``ptd_checkpoint_wrapper`` around itself, re-registered under its original attribute
    name so the module tree keeps its shape.

    Why not HF's API. ``GradientCheckpointingLayer.__call__`` invokes the checkpoint from
    INSIDE the block, so the only place left to hang ``torch.compile`` is
    ``_compiled_call_impl``, i.e. inside the checkpoint. That nesting makes the eager
    checkpoint machinery compare its own saved tensors against a separately-traced
    recompute, tripping ``_CheckpointFrame.check_recomputed_tensors_match`` with
    ``CheckpointError: Recomputed values ... have different metadata``. It surfaced in the
    Qwen3-VL vision blocks (``saved dtype: bfloat16`` vs ``recomputed dtype: float32``),
    where ``apply_rotary_pos_emb_vision`` round-trips q/k through float32 and the two
    tracing contexts disagree about which dtype to save; ``Qwen3VLTextRMSNorm`` has the same
    float32 round-trip shape, so restricting compile to one tower would only relocate it.
    A module wrapper inverts the nesting for free: :func:`apply_compile` compiles the
    WRAPPER, so the ``checkpoint`` call is inside the traced region where Dynamo lowers it
    to the activation-checkpoint higher-order op and the compiler owns forward and recompute
    together (Note: [torch.compile and checkpoint] in ``torch/utils/checkpoint.py``).

    It also puts FSDP2's forward hooks OUTSIDE the checkpoint, since :func:`apply_fsdp`
    shards this wrapper (or the compiled wrapper around it, never the block within).
    HF checkpoints ``__call__`` precisely to keep the hooks inside
    (forcing a re-all-gather on recompute); here the backward pre-hook has already
    re-all-gathered by the time the recompute runs, so the collective is not repeated and
    one fewer opaque region sits inside the compiled graph.

    ``CheckpointWrapper`` is transparent to the checkpoint/resume path: its state_dict hooks
    strip ``_checkpoint_wrapped_module.`` on save and re-add it on load, and
    ``safetensors_loader.load_vlm_model`` strips the same prefix itself.

    Two behaviours HF's ``GradientCheckpointingLayer.__call__`` supplied are dropped, both
    harmless here: its ``self.training`` gate (non-reentrant checkpoint is a passthrough
    when grad is disabled, so eval only pays the wrapper call) and its ``use_cache`` /
    ``past_key_values`` sanitization (``HFModel.forward`` already pins
    ``use_cache=False``).

    ``ptd_checkpoint_wrapper`` defaults to ``CheckpointImpl.NO_REENTRANT``, which the MoE
    load-balancing stash requires: it is a side channel out of the block's forward and only
    stays grad-connected under non-reentrant AC (see
    ``VLMModel.compute_load_balancing_loss``).

    Args:
        model: HFModel whose ``model._no_split_modules`` blocks (decoder layers + vision
               blocks for VLMs) are wrapped in place.
        ac:    AC policy (``VLMModelConfig.activation_checkpointing``).

    Returns:
        The block slots the later passes must operate on — holding the checkpoint wrappers
        when AC is enabled, the original blocks when ``mode == "none"``.

    Raises:
        ValueError: No repeated blocks are discoverable. Checked BEFORE the ``mode == "none"``
                    early return, so it gates every run rather than only AC-enabled ones: the
                    declaration this depends on is the same one :func:`apply_compile` and
                    :func:`apply_fsdp` need, so a run that gets past here uncheckpointed would
                    also be uncompiled and sharded as one flat group. Being the first pass in
                    :func:`parallelize` makes this the single startup gate for all three.
    """
    collected, no_split_names = _collect_repeated_blocks(model.model)

    if not collected:
        raise ValueError(
            f"No repeated blocks were found to wrap (_no_split_modules={no_split_names!r} on "
            f"{type(model.model).__name__}); declare _no_split_modules on the model, naming the "
            "repeated layer types. Only nn.ModuleList children qualify, so a declaration that "
            "names solely singletons (embeddings, pooling heads) also lands here. This is "
            "fatal for every activation_checkpointing.mode, 'none' included: the same missing "
            "declaration leaves apply_compile with nothing to compile and apply_fsdp sharding "
            "the root as one flat parameter group for the whole model."
        )

    if ac.mode == "none":
        return collected

    wrapped = _replace_blocks(collected, lambda block: _apply_ac_to_transformer_block(block, ac))
    log.info(f"parallelize: {ac.mode} activation checkpointing applied to {len(wrapped)} block(s)")
    return wrapped


def apply_compile(slots: list[_BlockSlot], compile_config: CompileConfig | None) -> list[_BlockSlot]:
    """``torch.compile`` each repeated transformer block, if enabled.

    No-op when ``compile_config`` is ``None`` or ``compile_config.enabled`` is False.
    ``slots`` is never empty: :func:`apply_ac` raises on an undiscoverable block set for every
    AC mode before this runs, so there is no silently-uncompiled case left to report here.
    Independent of FSDP (peer of :func:`apply_fsdp`), so compile also applies on the
    single-GPU / replicate-only path.

    Wraps with ``torch.compile(module)`` and re-registers the resulting ``OptimizedModule``,
    matching ``parallelize_unified_mot.apply_compile`` — the repeated block compiles once and
    the artifact is reused across all layers.

    Runs AFTER :func:`apply_ac` and on whatever that returned, so under activation
    checkpointing the compiled unit is the ``CheckpointWrapper`` and the ``checkpoint`` call
    lands INSIDE the traced region — the arrangement Dynamo supports, where it becomes the
    activation-checkpoint higher-order op. Compiling the inner block instead would put
    compile inside the checkpoint and reintroduce the recomputed-metadata mismatch; see
    :func:`apply_ac`.

    Wrapping rather than the in-place ``nn.Module.compile`` keeps FSDP genuinely outside the
    compiled region. :func:`apply_fsdp` shards what this returns, so FSDP's forward hooks are
    registered on the ``OptimizedModule`` and run in eager AROUND the traced body. In-place
    ``.compile()`` sets ``_compiled_call_impl``, which wraps ``_call_impl`` — the function
    that RUNS those hooks — putting the collectives inside the traced callable and leaving
    correctness resting on ``torch._dynamo.config.skip_fsdp_hooks``.

    The ``_orig_mod`` rename this introduces (``...layers.N._orig_mod.*``) is handled
    everywhere it is observed, which is why the MoT path can afford the same wrapper:
    DCP's ``get_model_state_dict`` / ``set_model_state_dict`` skip the prefix for both model
    and optimizer state (``_get_fqns(..., skip_compiler_prefix=True)``), so save AND resume
    keys are unchanged by toggling compile; ``safetensors_loader`` and ``hf_export`` strip it
    explicitly; ``vlm_model._canonical_param_name`` strips it before freeze-config regexes
    are matched. ``tie_embeddings`` is unaffected because the tied ``lm_head`` /
    ``embed_tokens`` live at the root, not inside a repeated block.

    Applied BEFORE ``fully_shard`` (called first in :func:`parallelize`): FSDP2
    ``fully_shard`` SWAPS each block's ``__class__`` to a dynamically-created
    ``FSDP<OrigName>`` type, so a typename match against ``_no_split_modules``
    AFTER sharding finds nothing — the blocks must be collected (and compiled)
    while their original types are intact.

    Args:
        slots:          The units to compile, as returned by :func:`apply_ac` — holding the
                        ``CheckpointWrapper``s under AC, the raw ``_no_split_modules`` blocks
                        (decoder layers + vision blocks for VLMs) otherwise.
        compile_config: torch.compile knobs (``compile_dynamic``,
                        ``max_autotune_pointwise``, ``coordinate_descent_tuning``), or
                        ``None`` to skip compilation entirely.

    Returns:
        The block slots :func:`apply_fsdp` must shard — holding the ``OptimizedModule``s when
        compile ran, otherwise ``slots`` unchanged.
    """
    if compile_config is None or not compile_config.enabled:
        return slots

    compile_kwargs = _compile_kwargs(compile_config)
    compiled = _replace_blocks(slots, lambda block: torch.compile(block, **compile_kwargs))
    log.info(
        f"parallelize: torch.compile applied to {len(compiled)} block(s) "
        f"(dynamic={compile_config.compile_dynamic}, options={compile_kwargs['options']})"
    )
    return compiled


def apply_fsdp(
    model: HFModel,
    slots: list[_BlockSlot],
    parallel_dims: ParallelDims,
    parallelism_config: ParallelismConfig,
    precision: str,
) -> None:
    """Apply FSDP2 to an HFModel in-place.

    Uses torch.distributed.fsdp.fully_shard (FSDP2).  Each transformer block is
    sharded individually for fine-grained memory savings; the outer model is then
    wrapped to cover remaining parameters (embeddings, layer norms, lm_head).

    Supported architectures:
    - Language models: ``inner.model.layers`` (standard HF LLM structure)
    - Vision-language models: additionally ``inner.visual.blocks`` (Qwen3-VL)

    No-op only when there is no data-parallel axis at all (single process). Replicate-only
    (``dp_replicate > 1, dp_shard == 1``) is handled here rather than by a separate DDP path:
    :func:`fsdp_mesh` hands over the 2-D mesh, ``fully_shard`` reads a 2-D mesh as HSDP, and a
    shard dim of size 1 makes each rank's "shard" the whole parameter while gradients are
    all-reduced over the replicate dim — DDP semantics through the FSDP2 machinery.

    Preferred over wrapping in DDP because everything downstream keeps working unchanged:
    parameters stay ``DTensor``s (so ``safetensors_loader`` and DCP see one layout in every
    parallelism mode), and the ``MixedPrecisionPolicy`` below still casts parameters to
    ``precision`` for compute and reduces gradients in ``fsdp_reduce_dtype``, which plain DDP
    would not do without extra hooks. The cost is that FSDP2 still runs its all-gather on the
    one-rank shard group, i.e. one copy of each parameter group per step that a DDP wrapper
    would not pay.

    Args:
        model:              HFModel instance (``model`` attribute must be on meta or CPU device).
        slots:              The units to shard individually, as returned by
                            :func:`apply_compile` — outermost first: ``OptimizedModule`` when
                            compiled, else the ``CheckpointWrapper`` under AC, else the raw
                            block. Sharding the OUTERMOST wrapper is what keeps FSDP's
                            forward hooks outside both the compiled region and the
                            checkpoint; see :func:`apply_compile` and :func:`apply_ac`.
        parallel_dims:      ParallelDims with meshes already built via
                            :meth:`ParallelDims.build_meshes`.
        parallelism_config: Source of the FSDP gradient-reduction dtype threaded to
                            ``MixedPrecisionPolicy.reduce_dtype`` — ``fsdp_reduce_dtype``
                            when set, else ``fsdp_master_dtype`` (see
                            :func:`_resolve_reduce_dtype`).
        precision:          FSDP MixedPrecisionPolicy parameter dtype
                            (``"bfloat16"``, ``"float16"``, or ``"float32"``).
    """
    if not parallel_dims.dp_enabled:
        log.info("parallelize: no data-parallel axis (dp_shard == dp_replicate == 1) — skipping FSDP2 wrapping")
        return

    mp_policy = MixedPrecisionPolicy(
        param_dtype=PRECISION_TO_TORCH_DTYPE[precision],
        reduce_dtype=_resolve_reduce_dtype(parallelism_config),
    )

    # 2-D (dp_replicate × dp_shard) mesh for HSDP, or 1-D dp_shard sub-mesh
    # for pure FSDP. In the overlay design cp does NOT fold into the FSDP
    # shard axis; cp/cfgp are handled by separate meshes.
    fsdp_kwargs = {"mesh": fsdp_mesh(parallel_dims), "mp_policy": mp_policy}

    inner = model.model

    # Blocks first, root last (below): fully_shard groups a module's parameters EXCEPT those an
    # earlier call on a submodule already claimed, so sharding the root first would absorb every
    # block into one flat group — one all-gather for the whole model, no layer-by-layer overlap.
    # reversed() covers nesting only, NOT sibling order (FSDP2 prefetches on runtime
    # post-forward order, not wrap order): _collect_repeated_blocks walks named_modules
    # pre-order, so reversing shards a collected block before any collected ANCESTOR of it —
    # which unioning _no_split_modules across submodules makes reachable.
    # Iterating the list threaded down from parallelize (not re-scanning by typename) is
    # required: the AC and compile passes replaced each block with a wrapper, and fully_shard
    # swaps each block's __class__ as it goes — any of those defeats a typename match (see
    # _collect_repeated_blocks).
    for _, _, block in reversed(slots):
        fully_shard(block, **fsdp_kwargs)
    log.info(f"Wrapped {len(slots)} sub-modules.")

    # Wrap the full inner model to cover remaining parameters
    # (embed_tokens, final layer norm, lm_head, visual projector stem, etc.)
    # NOTE: FSDP-2 CPU offload (offload_policy=CPUOffloadPolicy()) was never
    # wired through to any active recipe and the path was untested; see the
    # comment in vlm_model._init_vlm meta-materialize block (search for
    # "FSDP-2 CPU offload") for how to re-enable it.
    ignored_audio_params = _collect_ignored_audio_params(model)
    if ignored_audio_params:
        fully_shard(inner, ignored_params=ignored_audio_params, **fsdp_kwargs)
    else:
        # Preserve the pre-audio call surface exactly when sound_und=False.
        fully_shard(inner, **fsdp_kwargs)
    log.info("parallelize: FSDP2 applied to HFModel.model")


def parallelize(
    model: HFModel,
    parallel_dims: ParallelDims,
    parallelism_config: ParallelismConfig,
    precision: str,
    activation_checkpointing: ActivationCheckpointingConfig,
    compile_config: CompileConfig | None = None,
) -> None:
    """Optimize an HFModel in place: activation checkpointing, ``torch.compile``, FSDP2.

    Mirrors ``parallelize_unified_mot.parallelize``, including its pass ORDER, which is
    load-bearing rather than incidental:

    1. :func:`apply_ac` collects the repeated blocks while their original type names are
       intact and swaps each for a ``CheckpointWrapper``.
    2. :func:`apply_compile` wraps those in ``torch.compile``, so the ``checkpoint`` call
       sits inside the compiled region (the arrangement Dynamo can lower to the AC
       higher-order op) rather than around it.
    3. :func:`apply_fsdp` shards the outermost wrapper, leaving FSDP's collective hooks
       outside both the compiled region and the checkpoint.

    Each pass hands the next the block slots it produced, so every pass wraps the artifact of
    the previous one and FSDP ends up outermost.

    Each pass is a no-op when its feature is disabled (AC when ``mode == "none"``, compile
    when ``compile_config`` is ``None``/disabled, FSDP when dp is off entirely), so this is
    safe to call on the single-GPU path — where it is in fact the only thing that enables
    activation checkpointing. The one unconditional failure is :func:`apply_ac` raising when
    the model exposes no repeated blocks, whatever the AC mode, since all three passes key off
    that same ``_no_split_modules`` declaration.

    Args:
        model:                    HFModel instance (``model`` attribute on meta or CPU
                                  device).
        parallel_dims:            ParallelDims with meshes already built via
                                  :meth:`ParallelDims.build_meshes`.
        parallelism_config:       Source of the FSDP gradient-reduction dtype
                                  (``fsdp_reduce_dtype``, falling back to
                                  ``fsdp_master_dtype``).
        precision:                FSDP MixedPrecisionPolicy parameter dtype.
        activation_checkpointing: AC policy (mode, ``save_ops_regex``,
                                  ``preserve_rng_state``, ``determinism_check``).
        compile_config:           Optional ``CompileConfig``; ``None``/``enabled=False``
                                  skips compile.
    """
    slots = apply_ac(model, activation_checkpointing)
    slots = apply_compile(slots, compile_config)
    apply_fsdp(model, slots, parallel_dims, parallelism_config, precision)
