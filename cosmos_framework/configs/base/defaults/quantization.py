# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import attrs


@attrs.define(slots=False)
class QuantizationConfig:
    """Configuration for low-precision quantization of model parameters.

    Controls which quantization method is applied (mxfp8, nvfp4, fp8, int8_sim,
    fp8_sim), and which parameters are selected for quantization via
    include/exclude key filters or an exact ``target_fqns`` list. When ``method``
    is None, quantization is disabled and all other fields are inert.

    ``mxfp8`` and ``nvfp4`` use block-scaled MX / NVFP4 formats that require
    Blackwell (sm_100) tensor cores. ``fp8`` is plain e4m3 dynamic-activation +
    fp8-weight quantization that also runs on Hopper (sm_90) and Ada (sm_89).
    These three run real low-precision GEMMs through torchao.

    ``int8_sim`` and ``fp8_sim`` are quantize-dequantize (Q/DQ) *simulations*:
    the selected linears keep dense GEMMs in the compute dtype (bf16), but their
    weights are fake-quantized once at load time and their input activations
    are fake-quantized dynamically on every call, so the output carries exactly
    the rounding/clipping error of the low-precision format without needing any
    low-precision kernel (or torchao). ``int8_sim`` is symmetric INT8 with
    per-output-channel weight scales and per-token activation scales (the common
    "per-channel weight / per-row activation" recipe). ``fp8_sim`` is E4M3 with
    ``fp8_granularity`` scales (per_row = per-output-channel weight + per-token
    activation; per_tensor = one scale per weight and one per activation
    tensor). All methods share the same module selection, so an ``int8_sim`` and
    an ``fp8``/``fp8_sim`` run with identical selection settings quantize
    exactly the same set of linears.
    """

    # Quantization method for the model.
    method: str | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.in_({"mxfp8", "nvfp4", "fp8", "int8_sim", "fp8_sim"})),
    )

    # Scaling granularity for the ``fp8`` and ``fp8_sim`` methods: ``per_row``
    # (rowwise scales, better accuracy) or ``per_tensor`` (single scale per
    # tensor, slightly faster/simpler). Both are supported on Hopper (sm_90) and
    # Ada (sm_89). Ignored by ``mxfp8`` / ``nvfp4``, which use fixed block-scaled
    # formats, and by ``int8_sim``, which is always per-channel/per-token.
    fp8_granularity: str = attrs.field(
        default="per_row",
        validator=attrs.validators.in_({"per_row", "per_tensor"}),
    )

    # ``int8_sim`` / ``fp8_sim`` only: block size along the input-feature (K)
    # dimension for both operands. 0 (default) keeps one scale per whole row
    # (per output channel for the weight, per token for the activation); ``g``
    # > 0 gives one scale per ``g`` consecutive K elements of every weight row and
    # of every activation token, i.e. the two operands are blocked along the same
    # reduction dimension (like MX formats, but with a free block size). K must
    # be divisible by ``g``. Incompatible with ``fp8_granularity="per_tensor"``.
    qdq_group_size: int = attrs.field(default=0, validator=[attrs.validators.instance_of(int), attrs.validators.ge(0)])

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

    # Exact module selection. When non-empty this replaces the regex filters: a
    # Linear is quantized iff its FQN is listed here, and every listed FQN must
    # resolve to an ``nn.Linear`` in the model (missing or non-Linear entries are
    # an error, so a run can never silently quantize a different set than the
    # one it was handed). FQNs may be given relative to the ``OmniMoTModel``
    # (``net.language_model.model.layers...``) or to the VFM network
    # (``language_model.model.layers...``, the form ModelOpt targets use). Use
    # this to pin two methods (e.g. ``fp8`` and ``int8_sim``) to one
    # identical module set.
    target_fqns: list[str] = attrs.field(factory=list)

    # Optional file the sorted list of actually-quantized module FQNs is written
    # to (one per line) when runtime quantization is applied. Lets two runs be
    # diffed for module-set identity. The inference CLI points this at
    # ``<output_dir>/quantization_matched_fqns.txt``.
    matched_fqns_dump_path: str | None = attrs.field(default=None)

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
