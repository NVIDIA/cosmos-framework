# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""User-facing torch.compile knobs for VFM and VLM training paths."""

from typing import Literal

import attrs

ARPostSaturationMode = Literal[
    "default",
    "static-compile",
    "cuda-graph",
]


@attrs.define(slots=False)
class CompileConfig:
    # Master torch.compile switch. When False, all other CompileConfig fields
    # are inert.
    enabled: bool = False

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
    # packages/cosmos3/cosmos3/common/args.py for the inference-side rationale
    # (where dynamic=False is preferred for stable AR shapes).
    compile_dynamic: bool = True

    # Whether to use CUDA graphs for faster inference. This option does not work during training.
    use_cuda_graphs: bool = False

    # AR-inference-specific behavior once the rolling KV window saturates.
    # "default" uses the global compile settings for the entire generation.
    # "static-compile" keeps the normal pre-saturation path, then uses dedicated
    # static-compiled decoder layers. "cuda-graph" keeps pre-saturation eager,
    # then captures coarse denoise and KV-refresh graphs; it requires both
    # ``enabled`` and ``use_cuda_graphs`` to be False.
    ar_post_saturation_mode: ARPostSaturationMode = attrs.field(
        default="default",
        validator=attrs.validators.in_(
            {
                "default",
                "static-compile",
                "cuda-graph",
            }
        ),
    )

    # Enable autotuning for pointwise/reduction Triton kernels (e.g. RMSNorm).
    # Explores 6 candidate configs instead of the default 1, improving kernel performance
    # at the cost of longer first-iteration compilation time.
    max_autotune_pointwise: bool = False

    # Enable coordinate descent tuning after autotuning. Starts from the best autotuned
    # config and explores nearby configs by adjusting one parameter at a time.
    # Requires max_autotune_pointwise=True to have effect on reduction kernels.
    coordinate_descent_tuning: bool = False
