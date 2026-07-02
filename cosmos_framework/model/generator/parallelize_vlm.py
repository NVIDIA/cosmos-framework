# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""FSDP2 wrapping for Cosmos3 VLM ``HFModel`` instances.

Hosts the single VLM-specific ``parallelize`` entry point used by
``vlm_model.VLMModel._init_vlm``.  Lives under ``cosmos_framework/model/generator/``
so the FSDP wrapping concern sits next to the model class it operates on
(mirroring the layout of ``models/mot/parallelize_unified_mot.py`` for the
MoT path).

Pure parallelism plumbing — :class:`~cosmos_framework.utils.generator.parallelism.ParallelDims`
and its meshes — stays in ``vfm/utils/parallelism.py``.
"""

import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from cosmos_framework.utils import log
from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.configs.base.defaults.parallelism import (
    PRECISION_TO_TORCH_DTYPE,
    ParallelismConfig,
)
from cosmos_framework.model.generator.hf_model import HFModel
from cosmos_framework.utils.generator.parallelism import ParallelDims


def _collect_repeated_blocks(inner: nn.Module) -> tuple[list[nn.Module], set[str]]:
    """Collect the repeated transformer blocks by their ORIGINAL type name.

    Matches ``inner._no_split_modules`` — the decoder layers (+ vision blocks for
    VLMs, e.g. Qwen3-VL ``visual.blocks``). MUST run before any ``fully_shard``
    call: ``fully_shard`` mutates each block's ``__class__`` to a dynamically
    created ``FSDP<OrigName>`` type, after which the typename match finds nothing.
    In-place ``.compile()`` preserves the type, so :func:`apply_fsdp` can collect
    again after :func:`apply_compile` has run.

    Returns the matched blocks and the ``_no_split_modules`` names (the latter is
    reported in the no-blocks warning so a misconfiguration is visible).
    """
    no_split_names = set(getattr(inner, "_no_split_modules", []))
    blocks = [m for m in inner.modules() if type(m).__name__ in no_split_names]
    return blocks, no_split_names


def apply_compile(model: HFModel, compile_config: CompileConfig | None) -> None:
    """In-place ``torch.compile`` each repeated transformer block, if enabled.

    No-op when ``compile_config`` is ``None`` or ``compile_config.enabled`` is
    False. When compile IS enabled but the ``_no_split_modules`` typename scan
    matched nothing, logs a warning and returns rather than silently proceeding
    uncompiled. Independent of FSDP (peer of :func:`apply_fsdp`), so compile also
    applies on the single-GPU / replicate-only path.

    Uses the **in-place** ``nn.Module.compile`` (NOT the ``torch.compile(block)``
    wrapper) on purpose. The wrapper would replace each block with an
    ``OptimizedModule`` whose child is ``_orig_mod``, renaming every parameter to
    ``...layers.N._orig_mod.*``. That rename would break (a) the suffix-matching
    safetensors loader (``HFModel.load_weights``, run AFTER this in
    ``_init_vlm`` step g), (b) ``tie_embeddings``, and (c) DCP checkpoint
    save/resume (key set must match the prod checkpoint). The in-place variant
    compiles ``block._call_impl`` without inserting a wrapper, so ``state_dict``
    keys and module types are unchanged — the typename-based FSDP collection and
    all downstream loaders keep working.

    Applied BEFORE ``fully_shard`` (called first in :func:`parallelize`): FSDP2
    ``fully_shard`` SWAPS each block's ``__class__`` to a dynamically-created
    ``FSDP<OrigName>`` type, so a typename match against ``_no_split_modules``
    AFTER sharding finds nothing — the blocks must be collected (and compiled)
    while their original types are intact. In-place ``.compile()`` sets
    ``_compiled_call_impl`` as an instance attribute, which survives the later
    ``__class__`` swap; ``nn.Module.__call__`` still routes to it. PyTorch dynamo
    skips FSDP2 collective hooks (``torch._dynamo.config.skip_fsdp_hooks`` defaults
    True), so the param all-gather / grad reduce-scatter stay in eager AROUND the
    compiled forward body (FSDP-outside-compile, the torchtitan ordering) without
    the OptimizedModule key-rename.

    Mirrors the per-block strategy of the MoT path
    (``parallelize_unified_mot.apply_compile``): the repeated block compiles once
    and is reused across all layers. ``fullgraph=False`` because the "cosmos"
    attention backend (NATTEN / blackwell-fmha) is an opaque kernel that forces a
    graph break; compile still fuses the surrounding pointwise/norm regions (the
    ~6.8k tiny elementwise kernels) where the win is.

    Args:
        model:          HFModel whose ``model._no_split_modules`` blocks (decoder
                        layers + vision blocks for VLMs) are compiled in place.
        compile_config: torch.compile knobs (``compile_dynamic``,
                        ``max_autotune_pointwise``, ``coordinate_descent_tuning``),
                        or ``None`` to skip compilation entirely.
    """
    if compile_config is None or not compile_config.enabled:
        return

    blocks, no_split_names = _collect_repeated_blocks(model.model)
    if not blocks:
        # Compile was requested but the typename scan matched nothing: either
        # _no_split_modules is empty/absent, or its names don't match any module
        # type. Without this warning the run would silently proceed uncompiled
        # ("0 block(s)" looking like success), so surface the misconfiguration.
        log.warning(
            "parallelize: compile_config.enabled=True but no _no_split_modules "
            f"blocks were found (no_split_names={no_split_names!r}) — torch.compile "
            "is a no-op (the model exposes no compilable repeated blocks)"
        )
        return

    # max_autotune_pointwise / coordinate_descent_tuning map straight to the
    # torch.compile ``options=`` dict (same as the MoT apply_compile).
    options: dict[str, bool] = {}
    if compile_config.max_autotune_pointwise:
        options["max_autotune_pointwise"] = True
    if compile_config.coordinate_descent_tuning:
        options["coordinate_descent_tuning"] = True

    for block in blocks:
        # In-place: sets block._compiled_call_impl; keeps type + state_dict keys.
        block.compile(
            fullgraph=False,
            dynamic=compile_config.compile_dynamic,
            options=options or None,
        )
    log.info(
        f"parallelize: torch.compile applied in-place to {len(blocks)} block(s) "
        f"(dynamic={compile_config.compile_dynamic}, options={options or None})"
    )


def apply_fsdp(
    model: HFModel,
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

    No-op when there is no shard axis (``dp_shard <= 1``): single-GPU, or
    replicate-only (``dp_replicate > 1, dp_shard == 1``) which uses DDP outside
    this function.

    Args:
        model:              HFModel instance (``model`` attribute must be on meta or CPU device).
        parallel_dims:      ParallelDims with meshes already built via
                            :meth:`ParallelDims.build_meshes`.
        parallelism_config: Source of FSDP master dtype (``fsdp_master_dtype``;
                            threaded to ``MixedPrecisionPolicy.reduce_dtype``).
        precision:          FSDP MixedPrecisionPolicy parameter dtype
                            (``"bfloat16"``, ``"float16"``, or ``"float32"``).
    """
    if not parallel_dims.dp_shard_enabled:
        log.info("parallelize: dp_shard <= 1 — skipping FSDP2 wrapping")
        return

    mp_policy = MixedPrecisionPolicy(
        param_dtype=PRECISION_TO_TORCH_DTYPE[precision],
        reduce_dtype=PRECISION_TO_TORCH_DTYPE[parallelism_config.fsdp_master_dtype],
    )

    # 2-D (dp_replicate × dp_shard) mesh for HSDP, or 1-D dp_shard sub-mesh
    # for pure FSDP. In the overlay design cp does NOT fold into the FSDP
    # shard axis; cp/cfgp are handled by separate meshes.
    if parallel_dims.dp_replicate_enabled:
        fsdp_mesh = parallel_dims.dp_mesh
    else:
        fsdp_mesh = parallel_dims.dp_shard_mesh
    fsdp_kwargs = {"mesh": fsdp_mesh, "mp_policy": mp_policy}

    inner = model.model

    # Collect the repeated blocks by their ORIGINAL type name BEFORE any
    # fully_shard call (see _collect_repeated_blocks). apply_compile (if it ran
    # first) used in-place compile, which preserves the type names, so this
    # re-collection still matches.
    blocks, _ = _collect_repeated_blocks(inner)

    # Shard each collected block (reversed = leaf-first), then the root. Iterating
    # the collected list (not re-scanning by typename) is robust to the in-place
    # compile above and to fully_shard's per-block __class__ swap.
    for block in reversed(blocks):
        fully_shard(block, **fsdp_kwargs)
    log.info(f"Wrapped {len(blocks)} sub-modules.")

    # Wrap the full inner model to cover remaining parameters
    # (embed_tokens, final layer norm, lm_head, visual projector stem, etc.)
    # NOTE: FSDP-2 CPU offload (offload_policy=CPUOffloadPolicy()) was never
    # wired through to any active recipe and the path was untested; see the
    # comment in vlm_model._init_vlm meta-materialize block (search for
    # "FSDP-2 CPU offload") for how to re-enable it.
    fully_shard(inner, **fsdp_kwargs)
    log.info("parallelize: FSDP2 applied to HFModel.model")


def parallelize(
    model: HFModel,
    parallel_dims: ParallelDims,
    parallelism_config: ParallelismConfig,
    precision: str,
    compile_config: CompileConfig | None = None,
) -> None:
    """Optimize an HFModel in place: ``torch.compile`` (optional) then FSDP2.

    Mirrors ``parallelize_unified_mot``: :func:`apply_compile` and
    :func:`apply_fsdp` are peer passes, each a no-op when its feature is disabled
    (compile when ``compile_config`` is ``None``/disabled; FSDP when
    ``dp_shard <= 1``). Compile runs FIRST so the in-place block compile is
    collected with the blocks' original type names (before ``fully_shard`` swaps
    them) and dynamo's ``skip_fsdp_hooks`` keeps the FSDP collectives eager around
    the compiled forward body.

    Args:
        model:              HFModel instance (``model`` attribute on meta or CPU device).
        parallel_dims:      ParallelDims with meshes already built via
                            :meth:`ParallelDims.build_meshes`.
        parallelism_config: Source of FSDP master dtype (``fsdp_master_dtype``).
        precision:          FSDP MixedPrecisionPolicy parameter dtype.
        compile_config:     Optional ``CompileConfig``; ``None``/``enabled=False`` skips compile.
    """
    apply_compile(model, compile_config)
    apply_fsdp(model, parallel_dims, parallelism_config, precision)
