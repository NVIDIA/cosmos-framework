# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""User-facing parallelism degrees shared by VFM and VLM trainers."""

from typing import Literal

import attrs
import torch

AttentionIOLayout = Literal["sequence_sharded", "replicated"]

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
    data_parallel_replicate_degree: int = 1

    # Number of ranks for context parallelism.
    context_parallel_shard_degree: int = 1

    # Tensor layout at the attention boundary when CP is enabled.  Both
    # layouts may run the attention kernel with head-sharded Q/K/V:
    # ``sequence_sharded`` keeps surrounding projections/MLP sequence-sharded
    # with Ulysses-style all-to-all into/out of attention, while
    # ``replicated`` keeps current-frame hidden states replicated, slices
    # local heads before attention, then reduces/gathers attention output back
    # to replicated hidden states.
    attention_io_layout: AttentionIOLayout = "sequence_sharded"

    # Number of ranks for CFG parallelism.
    cfg_parallel_shard_degree: int = 1

    # Inference-mode mesh toggle for ParallelDims.
    enable_inference_mode: bool = False

    # Dtype of the FSDP-sharded "master" parameter copy: what nn.Parameter.data
    # holds on each rank, what the optimizer reads/writes against, and what the
    # cross-rank gradient reduce-scatter accumulates into. Threaded both to the
    # HFModel meta-init (sharded-param storage dtype) and to
    # MixedPrecisionPolicy.reduce_dtype (gradient comm dtype); these must match
    # because the reduced gradient writes back into the master param's shard.
    # The forward/backward compute dtype is the separate ``precision`` field on
    # the model config (mapped to MixedPrecisionPolicy.param_dtype).
    # NOTE: only used in VLM; VFM has no FSDP master.
    fsdp_master_dtype: str = "float32"
