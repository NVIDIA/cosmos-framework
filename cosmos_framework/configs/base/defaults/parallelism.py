# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""User-facing parallelism degrees shared by VFM and VLM trainers."""

import attrs
import torch

# Canonical mapping from precision string (used in user-facing configs and
# threaded through OmegaConf) to ``torch.dtype``. Consumed by sites that
# need to translate ``precision`` / ``fsdp_master_dtype`` into concrete
# torch dtypes (e.g. ``MixedPrecisionPolicy``, ``HFModel`` meta-init).


PRECISION_TO_TORCH_DTYPE: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@attrs.define(slots=False)
class ParallelismConfig:
    # Number of ranks for sharding the model weights (FSDP). The default -1
    # auto-infers to world_size at runtime via ParallelDims.
    data_parallel_shard_degree: int = -1

    # Number of ranks for replicating the model weights (HSDP outer dim).
    # data_parallel_replicate_degree x data_parallel_shard_degree must divide
    # world_size when both are explicitly set.
    #
    # Pair with data_parallel_shard_degree=1 (and -1 here, which auto-infers to
    # world_size) for a replicate-only run: parameters are not sharded and
    # gradients are all-reduced, i.e. DDP semantics, obtained by handing
    # fully_shard a (dp_replicate, 1) mesh rather than by a separate DDP wrapper.
    # Note that both degrees default such that shard-only is what you get if you
    # set neither.
    data_parallel_replicate_degree: int = 1

    # Number of ranks for context parallelism.
    context_parallel_shard_degree: int = 1

    # Number of ranks for CFG parallelism.
    cfg_parallel_shard_degree: int = 1

    # Size of the group ranks exchange raw video samples within before the VAE encode, to
    # equalize each rank's predicted VAE-encode cost for the step (see
    # models.mot.vae_load_balance and OmniMoTModel._prepare_training_data). 1 (the default)
    # disables load balancing entirely -- same convention as context_parallel_shard_degree
    # and cfg_parallel_shard_degree. Must evenly divide world_size. Unlike cp/cfgp this is
    # NOT an attention-sharding overlay: it only ever runs once per cp-window (on the step
    # that actually calls the VAE encoder), and has no interaction with how the packed
    # sequence is sharded for compute. Size it to stay within one node (NVLink) to keep the
    # raw-pixel exchange cheap -- nothing here enforces that, it is a placement choice.
    vae_load_balance_group_size: int = 1

    # Inference-mode mesh toggle for ParallelDims.
    enable_inference_mode: bool = False

    # Dtype of the FSDP-sharded "master" parameter copy: what nn.Parameter.data
    # holds on each rank, what the optimizer reads/writes against, and what the
    # checkpoint stores. Supplies the sharded-param storage dtype at meta-init
    # (VLM via HFModel, VFM via ``OmniMoTModel.set_up_model``) and, unless
    # ``fsdp_reduce_dtype`` overrides it, MixedPrecisionPolicy.reduce_dtype -- so the
    # reduced gradient lands in the dtype of the shard it writes back into.
    # The forward/backward compute dtype is the separate ``precision`` field on
    # the model config (mapped to MixedPrecisionPolicy.param_dtype).
    # Setting it equal to ``precision`` opts out of mixed precision entirely: no policy is
    # installed and the params are stored, computed with, and reduced in the compute dtype.
    fsdp_master_dtype: str = "float32"

    # Dtype of the gradient reduce-scatter itself (MixedPrecisionPolicy.reduce_dtype),
    # decoupled from the master-parameter dtype above. None (the default) reuses
    # fsdp_master_dtype, i.e. reduce in the master dtype for numerical headroom when
    # summing across a large shard group.
    #
    # Setting "bfloat16" halves the reduce-scatter staging buffer, which FSDP2 sizes at
    # the FULL unsharded gradient of the unit being reduced — 9.3 GiB per decoder layer
    # on Qwen3-VL-235B-A22B. The master copy is unaffected: FSDP2's foreach_reduce casts
    # the reduce output back to the sharded parameter's dtype before assigning .grad, so
    # only the collective and its staging buffer change. The cost is precision, since
    # gradients are then summed across the shard group in bf16.
    fsdp_reduce_dtype: str | None = None
