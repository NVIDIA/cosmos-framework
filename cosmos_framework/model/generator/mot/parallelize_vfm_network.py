# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard, register_fsdp_forward_method

from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.model.generator.mot.parallelize_unified_mot import (
    apply_ac_to_module,
    parallelize_unified_mot,
)
from cosmos_framework.utils.generator.parallelism import ParallelDims, fsdp_mesh


def apply_compile(model: torch.nn.Module, config: CompileConfig):
    """Apply torch.compile to the VFM encode/decode heads.

    The MoT-side ``compile_dynamic`` knob on ``CompileConfig`` intentionally
    does **not** propagate here.  The VFM encode/decode paths have no graph
    breaks and their input shapes are stable across a prompt, so we always
    trace them as a single dynamic graph (``fullgraph=True, dynamic=True``).
    This keeps AR inference (which sets ``compile_dynamic=False`` on MoT for
    shape-specialized kernels) from accidentally regressing the VFM compile.
    """

    inductor_options = {}
    if config.max_autotune_pointwise:
        inductor_options["max_autotune_pointwise"] = True
    if config.coordinate_descent_tuning:
        inductor_options["coordinate_descent_tuning"] = True

    # Whole-forward CUDA graphs (``cuda_graph_scope="forward"``) capture these heads inside the
    # recorded AR forward: per-region CUDA-graph trees cannot replay inside that capture, and a
    # symbolic-shape wrapper would stage host values through pinned memory that a replay later
    # reads after it is freed.  That scope therefore compiles the heads like the decoder blocks
    # (inductor only, static shapes); every other configuration keeps the dynamic single graph.
    forward_scope_capture = config.use_cuda_graphs and config.cuda_graph_scope == "forward"
    compile_options = {
        "fullgraph": True,
        "dynamic": not forward_scope_capture,
        "mode": "reduce-overhead" if (config.use_cuda_graphs and not forward_scope_capture) else None,
        "options": inductor_options or None,
    }

    model._encode_text = torch.compile(model._encode_text, **compile_options)
    model._encode_vision = torch.compile(model._encode_vision, **compile_options)
    model._encode_action = torch.compile(model._encode_action, **compile_options)
    model._decode_vision = torch.compile(model._decode_vision, **compile_options)
    model._decode_action = torch.compile(model._decode_action, **compile_options)
    return model


# Modules outside ``language_model`` that are worth recomputing rather than
# saving. ``apply_ac`` in ``parallelize_unified_mot`` only reaches the repeated
# decoder blocks, so anything named here would otherwise keep every
# intermediate activation alive from its forward until its backward.
#
# ``time_embedder``: its MLP is two ``[N, hidden]`` matmuls over *one row per
# noised token*, run under a float32 autocast. At the 45k-token-per-rank
# packing budgets used for multiview AV training that is ~0.6M rows, the
# ``nn.Linear`` output and the ``nn.SiLU`` output are ~8.8 GiB *each* in fp32,
# and several packs' worth are live at once across the microbatch. Recomputing
# them costs two small matmuls -- tens of milliseconds against a step measured
# in seconds.
_AC_MODULE_ATTRS = ("time_embedder",)


def apply_ac(model: torch.nn.Module, config: ActivationCheckpointingConfig) -> None:
    """Checkpoint the VFM-level modules listed in ``_AC_MODULE_ATTRS``, in place.

    The ``language_model`` blocks are handled separately by
    ``parallelize_unified_mot.apply_ac``; this covers what that pass cannot see.
    Attributes are optional because the heads they belong to are conditional --
    ``time_embedder`` only exists when ``config.vision_gen`` is set.
    """
    if config.mode == "none":
        return

    for attr in _AC_MODULE_ATTRS:
        module = getattr(model, attr, None)
        if module is None:
            continue
        setattr(model, attr, apply_ac_to_module(module, config))


def parallelize_vfm_network(
    model: torch.nn.Module,
    parallel_dims: ParallelDims | None,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    attention_io_layout: str = "sequence_sharded",
    mp_policy: MixedPrecisionPolicy | None = None,
) -> torch.nn.Module:
    """Optimize the model using FSDP, CP, activation checkpointing, and torch.compile.

    FSDP reduces memory usage by sharding the model parameters across multiple GPUs.
    Activation checkpointing reduces memory usage by selectively checkpointing only
    the outputs of each layer. Torch.compile compiles the model for faster training.

    Args:
        model: The Cosmos3 VFM network.
        parallel_dims: Device mesh / parallelism descriptor. FSDP is applied whenever its
            ``dp_enabled`` holds, which for training includes a single-rank ``(1, 1)`` mesh
            -- at that degree the wrap buys no memory, but it is what installs ``mp_policy``,
            without which the parameters would be computed with in their storage dtype
            rather than cast down.
        compile_config: Compile switches (enabled, compiled_region, etc.).
        ac_config: Activation-checkpointing policy, typically
            ``OmniMoTModelConfig.activation_checkpointing``. Forwarded to
            ``parallelize_unified_mot`` for the decoder blocks, and applied here
            to the VFM-level modules in ``_AC_MODULE_ATTRS``.
        attention_io_layout: Tensor layout at the attention boundary under CP.
        mp_policy: FSDP2 mixed-precision policy applied to every FSDP unit (the
            per-decoder-layer ones inside ``parallelize_unified_mot`` and the root wrap
            here). ``None`` keeps FSDP2's default of no casting, i.e. compute and gradient
            reduction happen in the dtype the parameters are stored in. Pass
            ``MixedPrecisionPolicy(param_dtype=<compute dtype>, reduce_dtype=<param storage
            dtype>)`` to run a low-precision forward/backward over higher-precision
            parameters; the module must already hold its parameters in ``reduce_dtype``
            when this is called, since each FSDP unit records their dtype as it is built.
    """
    model.attention_io_layout = attention_io_layout
    if parallel_dims is not None and parallel_dims.cp_enabled:
        model.parallel_dims = parallel_dims

    model.language_model = parallelize_unified_mot(
        model.language_model,
        parallel_dims=parallel_dims,
        compile_config=compile_config,
        ac_config=ac_config,
        attention_io_layout=attention_io_layout,
        mp_policy=mp_policy,
    )
    apply_ac(model, ac_config)

    if compile_config.enabled and compile_config.compiled_region == "all":
        model = apply_compile(model, compile_config)

    if parallel_dims is not None and parallel_dims.dp_enabled:
        # Same mesh as the per-block wrapping in ``parallelize_unified_mot.apply_fsdp`` (see
        # ``fsdp_mesh``): the root and the blocks must agree, or one model ends up with a
        # mix of HSDP and pure-FSDP units whose gradient reductions differ.
        model = fully_shard(
            module=model,
            mesh=fsdp_mesh(parallel_dims),
            mp_policy=mp_policy or MixedPrecisionPolicy(),
        )

        # Make ``model.generate_reasoner_text(...)`` trigger the same
        # pre-forward unshard / post-forward reshard hooks that
        # ``model.forward(...)`` does.  Without this, the AR-loop
        # reasoner path (``Cosmos3VFMNetwork.generate_reasoner_text``)
        # accesses top-level submodules — ``language_model.model.embed_tokens``,
        # ``language_model.model.norm``, ``language_model.lm_head`` —
        # while their parameters are still ``DTensor`` shards.  Mixing
        # a plain ``input_ids`` tensor with a ``DTensor`` weight raises
        # ``aten.embedding.default: got mixed torch.Tensor and DTensor,
        # need to convert all torch.Tensor to DTensor before calling
        # distributed operators!`` from ``DTensor._op_dispatcher``.
        # ``register_fsdp_forward_method`` is the canonical PyTorch
        # API for opting non-``forward`` entry points (HF ``generate``,
        # custom AR loops, etc.) into FSDP2's unshard/reshard
        # lifecycle, so the AR loop sees fully-materialized weights
        # and standard tensor dispatch on every call.
        #
        # The per-decoder-layer FSDP units (each ``block`` in
        # ``language_model.model.layers``) carry their own params and are
        # handled by the companion
        # ``register_fsdp_forward_method(block, "reasoner_forward")`` in
        # ``parallelize_unified_mot.apply_fsdp``; together the two
        # registrations cover every FSDP-wrapped weight touched on the
        # AR path.
        register_fsdp_forward_method(model, "generate_reasoner_text")

    return model
