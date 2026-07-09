# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch
from torch.distributed.fsdp import fully_shard, register_fsdp_forward_method

from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.model.generator.mot.parallelize_unified_mot import parallelize_unified_mot
from cosmos_framework.utils.generator.parallelism import ParallelDims


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

    compile_options = {
        "fullgraph": True,
        "dynamic": True,
        "mode": "reduce-overhead" if config.use_cuda_graphs else None,
        "options": inductor_options or None,
    }

    model._encode_text = torch.compile(model._encode_text, **compile_options)
    model._encode_vision = torch.compile(model._encode_vision, **compile_options)
    model._encode_action = torch.compile(model._encode_action, **compile_options)
    model._decode_vision = torch.compile(model._decode_vision, **compile_options)
    model._decode_action = torch.compile(model._decode_action, **compile_options)
    return model


def parallelize_vfm_network(
    model: torch.nn.Module,
    parallel_dims: ParallelDims | None,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    attention_io_layout: str = "sequence_sharded",
) -> torch.nn.Module:
    """Optimize the model using FSDP, CP, activation checkpointing, and torch.compile.

    FSDP reduces memory usage by sharding the model parameters across multiple GPUs.
    Activation checkpointing reduces memory usage by selectively checkpointing only
    the outputs of each layer. Torch.compile compiles the model for faster training.

    Args:
        model: The Cosmos3 VFM network.
        parallel_dims: Device mesh / parallelism descriptor.
        compile_config: Compile switches (enabled, compiled_region, etc.).
        ac_config: Selective activation-checkpointing policy, typically
            ``OmniMoTModelConfig.sac``. Forwarded to
            ``parallelize_unified_mot``; ``None`` falls back to the
            ``ActivationCheckpointingConfig`` defaults.
        attention_io_layout: Tensor layout at the attention boundary under CP.
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
    )

    if compile_config.enabled and compile_config.compiled_region == "all":
        model = apply_compile(model, compile_config)

    if parallel_dims is not None and parallel_dims.dp_enabled:
        # Collect parameters to ignore during FSDP wrapping
        ignored_params = set()

        model = fully_shard(
            module=model,
            mesh=parallel_dims.dp_mesh,
            ignored_params=ignored_params,
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
