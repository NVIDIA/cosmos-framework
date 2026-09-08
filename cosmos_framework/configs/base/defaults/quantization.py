# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import attrs


@attrs.define(slots=False)
class QuantizationConfig:
    """Configuration for low-precision quantization of model parameters.

    Controls which quantization method is applied (mxfp8, nvfp4, fp8), and which
    parameters are selected for quantization via include/exclude key filters.
    When ``method`` is None, quantization is disabled and all other fields are
    inert.

    ``mxfp8`` and ``nvfp4`` use block-scaled MX / NVFP4 formats that require
    Blackwell (sm_100) tensor cores. ``fp8`` is plain e4m3 dynamic-activation +
    fp8-weight quantization that also runs on Hopper (sm_90) and Ada (sm_89).
    """

    # Quantization method for the model.
    method: str | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.in_({"mxfp8", "nvfp4", "fp8"})),
    )

    # Scaling granularity for the ``fp8`` method: ``per_row`` (rowwise scales,
    # better accuracy) or ``per_tensor`` (single scale per tensor, slightly
    # faster/simpler). Both are supported on Hopper (sm_90) and Ada (sm_89).
    # Ignored by ``mxfp8`` / ``nvfp4``, which use fixed block-scaled formats.
    fp8_granularity: str = attrs.field(
        default="per_row",
        validator=attrs.validators.in_({"per_row", "per_tensor"}),
    )

    # How to select parameters to select for the quantization. Each key is a
    # regular expression matched against a module's fully-qualified name with
    # `re.search` (a plain substring is still a valid pattern, so substring-style
    # keys keep working, while anchors like `^`/`$`, alternation `a|b`, and
    # character classes are also supported). A module is selected only if its FQN
    # matches at least one pattern in `include_regex` and matches none in
    # `exclude_regex`. If `include_regex` is empty, all parameters are
    # considered as included. If `exclude_regex` is empty, no parameters are
    # considered as excluded.
    include_regex: list[str] = attrs.field(factory=list)
    exclude_regex: list[str] = attrs.field(factory=list)

    # Local root of a ModelOpt static-FP8 diffusers checkpoint. When set, the
    # linears named by ``modelopt_fp8_target_fqns`` are swapped to FP8 modules on
    # the meta device *before* the network is parallelized and materialized, so
    # peak memory follows the FP8 weights rather than their bf16 shapes. This is
    # independent of ``method``, which selects runtime (post-training)
    # quantization; a ModelOpt checkpoint arrives already quantized.
    modelopt_fp8_checkpoint_path: str | None = attrs.field(default=None)

    # Target module FQNs (relative to the VFM network) that the ModelOpt FP8
    # checkpoint carries quantized weights for. Computed from the checkpoint
    # index by the loader, which knows the diffusers key mapping; passed through
    # the config so the meta-device swap in ``build_net`` needs no mapper.
    modelopt_fp8_target_fqns: list[str] = attrs.field(factory=list)

    # Mixed-precision diffusion steps for ModelOpt FP8 checkpoints: the first
    # ``mixed_precision_first_steps`` and last ``mixed_precision_last_steps``
    # denoising steps run generation-path linears with 16-bit activations
    # (W8A16: dequantized FP8 weight + dense GEMM); the middle steps keep the
    # FP8-activation path (W8A8). Both 0 (default) disables the feature.
    mixed_precision_first_steps: int = attrs.field(
        default=0, validator=[attrs.validators.instance_of(int), attrs.validators.ge(0)]
    )
    mixed_precision_last_steps: int = attrs.field(
        default=0, validator=[attrs.validators.instance_of(int), attrs.validators.ge(0)]
    )

    # Reasoner-path (understanding pathway) precision, independent of the step
    # schedule: "high_precision" keeps those linears on W8A16 for every step;
    # "base_precision" keeps them on W8A8.
    mixed_precision_reasoner_policy: str = attrs.field(
        default="high_precision",
        validator=attrs.validators.in_({"high_precision", "base_precision"}),
    )

    # Where W8A16 dense weights come from: "none" dequantizes per call,
    # "generation"/"all" hold resident BF16 caches, "gpu_block"/"cpu_block"
    # stage per-decoder-layer slots through a double buffer. Only "none" is
    # supported when the model is FSDP-sharded. The default deliberately
    # diverges from vllm-omni's "gpu_block": measured wall time of "none" is
    # indistinguishable from the cached modes at first/last-step schedules
    # (t2i, t2v, and FSDP-sharded Super runs), and "none" works everywhere,
    # including multi-GPU FSDP where the cached modes are rejected at load.
    mixed_precision_w8a16_cache: str = attrs.field(
        default="none",
        validator=attrs.validators.in_({"none", "generation", "all", "cpu_block", "gpu_block"}),
    )

    @property
    def mixed_precision_enabled(self) -> bool:
        """Whether mixed-precision diffusion steps are requested."""
        return self.mixed_precision_first_steps + self.mixed_precision_last_steps > 0
