# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared user-facing parallelism schema for VFM and VLM.

Both project trees (vfm/, vfm/configs/base/vlm/) instantiate the same
ParallelDims runtime at vfm/utils/parallelism.py. They now also share this
single user-facing config schema. Trainer-side translation from the long
descriptive field names here to the short ParallelDims constructor kwargs
happens at the read site (see vfm/models/{omni_mot_model,vlm_model}.py).
"""

import attrs


@attrs.define(slots=False)
class ParallelismConfig:
    # Torch compile is used to compile the model for faster training.
    use_torch_compile: bool = False

    # Whether to use CUDA graphs for faster inference. This option does not work during training.
    use_cuda_graphs: bool = False

    # Whether the entire Cosmos3 VFM network is compiled, or only a specific region is compiled.
    # Use "language" to compile only individual layers in the MOT model.
    # Use "all" to compile the the MOT model, as well as encode/decode functions.
    compiled_region: str = attrs.field(
        default="language",
        validator=attrs.validators.in_({"all", "language"}),
    )

    # Whether torch.compile should generate symbolic-shape (dynamic) kernels
    # (maps to ``torch.compile(dynamic=...)``).  Defaults to True for training,
    # which sees varying shapes across batches (sequence length, CP sharding, ...);
    # specializing would recompile continuously.  See ParallelismOverrides in
    # cosmos_framework/inference/common/args.py for the inference-side rationale
    # (where dynamic=False is preferred for stable AR shapes).
    compile_dynamic: bool = True

    # Enable autotuning for pointwise/reduction Triton kernels (e.g. RMSNorm).
    # Explores 6 candidate configs instead of the default 1, improving kernel performance
    # at the cost of longer first-iteration compilation time.
    max_autotune_pointwise: bool = False

    # Enable coordinate descent tuning after autotuning. Starts from the best autotuned
    # config and explores nearby configs by adjusting one parameter at a time.
    # Requires max_autotune_pointwise=True to have effect on reduction kernels.
    coordinate_descent_tuning: bool = False

    # Whether to enable inference mode.
    enable_inference_mode: bool = False

    # Number of ranks for sharding the model weights (FSDP). The default -1
    # auto-infers to world_size at runtime via ParallelDims.
    data_parallel_shard_degree: int = -1

    # Number of ranks for replicating the model weights (HSDP outer dim).
    # data_parallel_replicate_degree x data_parallel_shard_degree must divide
    # world_size when both are explicitly set.
    data_parallel_replicate_degree: int = 1

    # Number of ranks for context parallelism.
    context_parallel_shard_degree: int = 1

    # Number of ranks for CFG parallelism.
    cfg_parallel_shard_degree: int = 1

    # Precision for the model.
    precision: str = "bfloat16"
