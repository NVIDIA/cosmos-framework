# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Dependency-light model backend schemas shared by posttrain algorithms.

Model selection is a property of the policy, not of an RL objective. Keep the typed load and
parallelism settings here so DPO, GRPO, and OPD roots can share them without putting model-family
details in ``config/rl.py``. Construction remains lazy in each backend's ``model/*.py`` module.

MODULE-LEVEL IMPORTS MUST STAY DEPENDENCY-LIGHT. Roles read a schema from here to decide what to
build, so pulling a model stack in at module scope would tax every process that only wanted the
shape. ``tests/test_config_model_gr00t.py`` asserts this in a subprocess.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from cosmos_framework.posttrain.config.gr00t_bc import Gr00tBackboneAdaptation

# Model backend identifiers used by the top-level selector and its thin dispatch.
I4 = "i4"
HF = "hf"
QWEN3_VL = "qwen3_vl"


@dataclass
class DataPackerConfig:
    """Importable DataPacker factories keyed by model identity.

    ``factory`` constructs the student packer used by rollout, policy, reward filtering, and the
    OPD teacher actor's student-side codec. ``teacher_model_factory`` is reserved for the separate
    teacher model in cross-token multimodal OPD. Empty values retain the built-in model-type
    registry behavior.
    """

    factory: str = ""
    teacher_model_factory: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.factory, str):
            raise TypeError("DataPackerConfig.factory must be a string")
        if not isinstance(self.teacher_model_factory, str):
            raise TypeError("DataPackerConfig.teacher_model_factory must be a string")


# ---------------------------------------------------------------------------
# Native i4 policy backend
# ---------------------------------------------------------------------------


@dataclass
class NativePolicyParallelismConfig:
    """Policy data-parallel mesh for a native i4 policy builder.

    The policy WorkerGroup supplies the world size. ``-1`` asks posttrain to infer the matching
    dimension from that world, using the same convention as :class:`HFParallelismConfig`.
    Model-family restrictions remain the builder's responsibility; for example, the Qwen3-VL
    builder accepts FSDP and HSDP over the policy rank world.
    """

    dp_replicate: int = 1
    dp_shard: int = -1


@dataclass
class NativePolicyBuildSpec:
    """Generic construction seam for an i4-native online policy.

    ``builder`` is an absolute dotted callable with signature
    ``(model_id, model_config, pdims) -> policy``. ``value_builder`` is the matching optional
    callable for algorithms such as PPO that own a learned critic; those algorithms validate that
    it is present before worker allocation. Keeping the callables as strings makes the root
    config dependency-light and defers importing a concrete model family until the policy worker
    starts. ``model_id`` is the single tokenizer / rollout / policy artifact identity used by
    generic online machinery and the builder; ``model_config`` remains opaque to the entrypoint and
    is consumed only by that builder and the precision compatibility check.
    """

    builder: str
    model_id: str
    model_config: Any
    value_builder: str = ""
    parallelism: NativePolicyParallelismConfig = field(default_factory=NativePolicyParallelismConfig)
    requires_collated_policy_forward: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.builder, str) or not self.builder.strip() or "." not in self.builder:
            raise ValueError("NativePolicyBuildSpec.builder must be a non-empty absolute dotted callable")
        if not isinstance(self.value_builder, str) or (self.value_builder and "." not in self.value_builder):
            raise ValueError("NativePolicyBuildSpec.value_builder must be empty or an absolute dotted callable")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("NativePolicyBuildSpec.model_id must be a non-empty model/tokenizer artifact identity")
        if self.model_config is None:
            raise ValueError("NativePolicyBuildSpec.model_config must not be None")
        if not isinstance(self.requires_collated_policy_forward, bool):
            raise TypeError("NativePolicyBuildSpec.requires_collated_policy_forward must be a bool")


# ---------------------------------------------------------------------------
# Hugging Face model backend
# ---------------------------------------------------------------------------


@dataclass
class HFParallelismConfig:
    """Policy data-parallel mesh for the Hugging Face backend.

    These are Torch DeviceMesh dimensions inside the policy execution replica's rank world and are
    unrelated to WorkerGroup execution replicas. Use the mesh names ``dp_replicate`` and
    ``dp_shard`` consistently; ``-1`` asks posttrain to infer that dimension from ``POLICY_RANKS``.
    """

    dp_replicate: int = 1
    dp_shard: int = -1


@dataclass
class HFModelConfig:
    """Hugging Face backend spec (``config.model_hf``), selected via ``model_backend=hf``.

    The loader lives with the backend in ``model/hf.py``. The equivalent i4 parallelism source is
    the inherited LazyConfig tree at ``config.model.config.parallelism``.
    """

    name_or_path: str = ""
    revision: str | None = None
    processor_revision: str | None = None
    # DEPRECATED, superseded by `rl.param_dtype` (compute) + `rl.master_dtype` (storage).
    # This ONE field drove both -- `model/hf.py` used it for the parameter normalization AND for
    # `ShardPlan.param_dtype` -- which makes fp32-storage/bf16-compute inexpressible, and that
    # combination is what the reference GRPO recipe actually runs. Still honoured as the fallback
    # when the `rl.*` fields are unset, so existing recipes and dotlists keep working;
    # `config/build.py::resolve_precision` owns that chain. See `HFRunConfig`.
    dtype: str = "bfloat16"
    attn_impl: str = "flash_attention_2"
    trust_remote_code: bool = False
    auto_class: str = "AutoModelForCausalLM"
    grad_ckpt: bool = False  # drives HFModel.apply_gradient_checkpointing in setup()
    parallelism: HFParallelismConfig = field(default_factory=HFParallelismConfig)
    # Gradient all-reduce dtype for FSDP2 mixed precision (params stay `dtype`); cosmos-rl
    # `fsdp_reduce_dtype`. fp32 reduction over bf16 params is the reference recipe.
    #
    # DEPRECATED, superseded by `rl.reduce_dtype`. It WAS dead -- nothing read it, and `model/hf.py`
    # hardcoded `reduce_dtype=float32`, which was harmless only because the hardcode equalled every
    # recipe's value. `resolve_precision` now honours it as the fallback for `rl.reduce_dtype`, so a
    # config setting it to anything other than "float32" gets what it asked for.
    fsdp_reduce_dtype: str = "float32"


# ---------------------------------------------------------------------------
# Native Qwen3-VL model backend
# ---------------------------------------------------------------------------


@dataclass
class NativeLoRAConfig:
    """Native language-only LoRA; no PEFT preparation or implicit trainable modules.

    The named profile resolves to the seven attention/MLP projections of every dense
    Qwen3-VL-8B language block. The builder verifies that complete set before loading.
    Adapter storage and optimizer moments are FP32, independently of the BF16 base.
    """

    method: str = "lora"
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    parameter_dtype: str = "float32"
    target_profile: str = "qwen3_vl_8b_language_v1"
    bias: str = "none"
    modules_to_save: tuple[str, ...] = ()
    initialization: str = "kaiming_A_zero_B"
    initialization_seed: int = 42
    # The explicit preparation command publishes this immutable factor bundle.
    # Its location is operational; only its content hash belongs in model identity.
    initialization_artifact: str = ""
    initialization_artifact_sha256: str = ""

    def __post_init__(self) -> None:
        required = {
            "method": "lora",
            "parameter_dtype": "float32",
            "target_profile": "qwen3_vl_8b_language_v1",
            "bias": "none",
        }
        for name, expected in required.items():
            if getattr(self, name) != expected:
                raise ValueError(f"NativeLoRAConfig.{name} supports only {expected!r}")
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("NativeLoRAConfig.rank must be a positive integer")
        if type(self.alpha) not in (int, float) or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("NativeLoRAConfig.alpha must be finite and positive")
        if type(self.dropout) not in (int, float) or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("NativeLoRAConfig.dropout must be finite and in [0, 1)")
        if (
            not isinstance(self.modules_to_save, Sequence)
            or isinstance(self.modules_to_save, (str, bytes))
            or self.modules_to_save
        ):
            raise ValueError("NativeLoRAConfig.modules_to_save must be empty; only adapters may be trained")
        self.modules_to_save = ()
        if type(self.initialization_seed) is not int or not 0 <= self.initialization_seed < 2**63:
            raise ValueError("NativeLoRAConfig.initialization_seed must be an integer in [0, 2**63)")
        if self.initialization not in ("kaiming_A_zero_B", "nf4_residual_svd"):
            raise ValueError("NativeLoRAConfig.initialization supports kaiming_A_zero_B|nf4_residual_svd")
        if not isinstance(self.initialization_artifact, str) or not isinstance(
            self.initialization_artifact_sha256, str
        ):
            raise ValueError("NativeLoRAConfig initialization artifact path and SHA256 must be strings")
        if self.initialization == "kaiming_A_zero_B":
            if self.initialization_artifact or self.initialization_artifact_sha256:
                raise ValueError("NativeLoRAConfig zero-B initialization forbids a residual factor artifact")
        elif (
            not os.path.isabs(self.initialization_artifact)
            or re.fullmatch(r"[0-9a-f]{64}", self.initialization_artifact_sha256) is None
        ):
            raise ValueError("NativeLoRAConfig residual initialization requires an absolute artifact path and SHA256")


def native_adapter_config_identity(adapter: NativeLoRAConfig) -> dict[str, Any]:
    """Keep legacy identity unchanged and bind optional initialization by content.

    Moving an identical initializer bundle must not change the numerical model.
    New fields with empty defaults must not invalidate existing zero-B artifacts.
    """
    identity = asdict(adapter)
    identity.pop("initialization_artifact")
    if adapter.initialization == "kaiming_A_zero_B":
        identity.pop("initialization_artifact_sha256")
    return identity


@dataclass
class QuantizedBaseTrainingConfig:
    """One explicitly qualified frozen-base format, separate from adapter configuration.

    uint8 stores packed NF4 values; BF16 describes the source and dequantized compute.
    Metadata and packed weights are immutable during training and never optimizer members.
    """

    backend: str = "bitsandbytes"
    backend_version: str = "0.49.2"
    format: str = "nf4"
    block_size: int = 64
    nested: bool = True
    source_dtype: str = "bfloat16"
    storage_dtype: str = "uint8"
    compute_dtype: str = "bfloat16"
    target_profile: str = "qwen3_vl_8b_language_v1"
    offload: str = "none"
    # Keep at most two complete language blocks dense; their LoRA targets remain.
    excluded_language_layers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        required = {
            "backend": "bitsandbytes",
            "backend_version": "0.49.2",
            "format": "nf4",
            "source_dtype": "bfloat16",
            "storage_dtype": "uint8",
            "compute_dtype": "bfloat16",
            "target_profile": "qwen3_vl_8b_language_v1",
            "offload": "none",
        }
        for name, expected in required.items():
            if getattr(self, name) != expected:
                raise ValueError(f"QuantizedBaseTrainingConfig.{name} supports only {expected!r}")
        if type(self.block_size) is not int or self.block_size != 64:
            raise ValueError("QuantizedBaseTrainingConfig.block_size supports only 64")
        if self.nested is not True:
            raise ValueError("QuantizedBaseTrainingConfig.nested must be True")
        layers = self.excluded_language_layers
        if (
            not isinstance(layers, Sequence)
            or isinstance(layers, (str, bytes))
            or len(layers) > 2
            or any(type(layer) is not int or not 0 <= layer < 36 for layer in layers)
            or len(set(layers)) != len(layers)
        ):
            raise ValueError("excluded_language_layers requires at most two distinct integer block IDs in [0, 35]")
        self.excluded_language_layers = tuple(sorted(layers))


def quantization_config_identity(quantization: QuantizedBaseTrainingConfig | None) -> dict[str, Any] | None:
    """Retain legacy NF4 identities exactly; bind nonempty dense-block choices."""
    if quantization is None:
        return None
    quantization.__post_init__()
    identity = asdict(quantization)
    if not quantization.excluded_language_layers:
        identity.pop("excluded_language_layers")
    return identity


# Native adapter SAC must match the names dispatched by torch's SDPA backend,
# rather than the custom flash-attn names used by other model families. Keep this
# policy dependency-light and versioned in checkpoint execution identities.
NATIVE_ADAPTER_SAC_SAVE_OPS: tuple[str, ...] = (
    r"^_scaled_dot_product_(?:flash_attention|efficient_attention|cudnn_attention|attention_math)(?:\.|$)",
    r"^(?:mm|addmm|bmm)(?:\.|$)",
)


def validate_quantized_linear_execution(cfg: Any) -> None:
    """Reject unqualified dense-weight lifetimes before loading a native base."""
    execution = getattr(cfg, "quantized_linear_execution", "bnb")
    if execution not in ("bnb", "checkpointed_dequant"):
        raise ValueError("model_qwen3_vl.quantized_linear_execution must be bnb|checkpointed_dequant")
    if execution == "bnb":
        return
    adapter = getattr(cfg, "adapter", None)
    quantization = getattr(cfg, "quantized_base", None)
    if adapter is None or quantization is None or getattr(quantization, "format", None) != "nf4":
        raise ValueError("checkpointed_dequant requires an explicit native LoRA adapter and frozen NF4 base")
    if (
        cfg.activation_checkpointing != "full"
        or getattr(cfg, "adapter_compute_dtype", "float32") != "float32"
        or getattr(cfg, "sft_logits_mode", "full") not in ("full", "supervised")
        or getattr(cfg, "compile_language_model", False) is not False
    ):
        raise ValueError(
            "checkpointed_dequant requires full nonreentrant activation checkpointing, "
            "FP32 adapter compute, full or supervised logits and no language compilation"
        )
    if getattr(quantization, "excluded_language_layers", ()):
        raise ValueError("checkpointed_dequant with excluded language layers is not supported")
    if getattr(adapter, "initialization", None) != "kaiming_A_zero_B":
        raise ValueError("checkpointed_dequant requires kaiming_A_zero_B; residual initialization is not supported")


def native_adapter_execution_policy(cfg: Any, *, include_defaults: bool = False) -> dict[str, Any] | None:
    """Describe execution, retaining the absent legacy-default policy for old readers."""
    mode = cfg.activation_checkpointing
    adapter_compute_dtype = getattr(cfg, "adapter_compute_dtype", "float32")
    logits_mode = getattr(cfg, "sft_logits_mode", "full")
    compile_language = getattr(cfg, "compile_language_model", False)
    quantized_execution = getattr(cfg, "quantized_linear_execution", "bnb")
    if quantized_execution not in ("bnb", "checkpointed_dequant"):
        raise ValueError("model_qwen3_vl.quantized_linear_execution must be bnb|checkpointed_dequant")
    if (
        not include_defaults
        and mode == "full"
        and adapter_compute_dtype == "float32"
        and logits_mode == "full"
        and not compile_language
        and quantized_execution == "bnb"
    ):
        return None
    policy = {
        "protocol": "cosmos.native-adapter-execution.v1",
        "activation_checkpointing": {
            "mode": mode,
            "implementation": "none" if mode == "none" else "nonreentrant",
            "preserve_rng_state": True,
            "determinism_check": "default",
            "save_ops_regex": list(NATIVE_ADAPTER_SAC_SAVE_OPS) if mode == "selective" else [],
        },
        "adapter_compute_dtype": adapter_compute_dtype,
        "sft_logits_mode": logits_mode,
        "language_compile": {
            "enabled": compile_language,
            "dynamic": True,
            "fullgraph": False,
            "backend": "inductor",
            "fail_on_recompile_limit_hit": True,
        },
    }
    if quantized_execution == "checkpointed_dequant":
        policy["protocol"] = "cosmos.native-adapter-execution.v2"
        policy["quantized_linear"] = {
            "protocol": "cosmos.checkpointed-dequant.v1",
            "mode": "checkpointed_dequant",
            "scope": "full_nonreentrant_language_block",
            "inference": "bnb",
        }
    return policy


@dataclass
class Qwen3VLModelConfig:
    """Model-family config carried by a Qwen3-VL ``NativePolicyBuildSpec``.

    Dependency-light strings live here; ``model/qwen3_vl.py`` imports the native i4 model,
    activation-checkpointing and FSDP implementations only inside worker setup. The model family is
    resolved from the checkpoint's ``AutoConfig.model_type`` rather than duplicated in this schema.
    ``config.model_qwen3_vl`` retains this type only for the legacy backend alias.
    """

    name_or_path: str = ""
    revision: str | None = None
    trust_remote_code: bool = False
    checkpoint_path: str = ""
    credential_path: str = ""
    dtype: str = "bfloat16"
    attn_impl: str = "eager"
    activation_checkpointing: str = "full"
    # Parameters and optimizer moments stay FP32; only adapter GEMM arithmetic changes.
    # OmegaConf structured configs do not support Literal annotations. Keep the
    # accepted values explicit in __post_init__, as for the other model enums.
    adapter_compute_dtype: str = "float32"
    sft_logits_mode: str = "full"
    compile_language_model: bool = False
    quantized_linear_execution: str = "bnb"
    fsdp_reduce_dtype: str = "float32"
    adapter: NativeLoRAConfig | None = None
    quantized_base: QuantizedBaseTrainingConfig | None = None
    # Declared repository/revision are verified and enriched with file hashes before loading.
    source_identity: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.revision is not None and not isinstance(self.revision, str):
            raise TypeError("model_qwen3_vl.revision must be a string or None")
        if type(self.trust_remote_code) is not bool:
            raise TypeError("model_qwen3_vl.trust_remote_code must be a bool")
        if self.dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"model_qwen3_vl.dtype must be bfloat16|float16|float32, got {self.dtype!r}")
        if self.fsdp_reduce_dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(
                f"model_qwen3_vl.fsdp_reduce_dtype must be bfloat16|float16|float32, got {self.fsdp_reduce_dtype!r}"
            )
        if self.activation_checkpointing not in ("none", "full", "selective"):
            raise ValueError(
                "model_qwen3_vl.activation_checkpointing must be none|full|selective, got "
                f"{self.activation_checkpointing!r}"
            )
        if self.adapter_compute_dtype not in ("float32", "bfloat16"):
            raise ValueError("model_qwen3_vl.adapter_compute_dtype must be float32|bfloat16")
        if self.sft_logits_mode not in ("full", "supervised"):
            raise ValueError("model_qwen3_vl.sft_logits_mode must be full|supervised")
        if type(self.compile_language_model) is not bool:
            raise TypeError("model_qwen3_vl.compile_language_model must be a bool")
        if self.adapter is None and (
            self.adapter_compute_dtype != "float32" or self.sft_logits_mode != "full" or self.compile_language_model
        ):
            raise ValueError("native adapter execution optimizations require an explicit adapter")
        if not isinstance(self.source_identity, Mapping):
            raise TypeError("model_qwen3_vl.source_identity must be a dictionary")
        self.source_identity = dict(self.source_identity)
        if self.adapter is not None:
            if not isinstance(self.adapter, NativeLoRAConfig):
                raise TypeError("model_qwen3_vl.adapter must be NativeLoRAConfig or None")
            self.adapter.__post_init__()
            if self.dtype != "bfloat16" or self.fsdp_reduce_dtype != "float32":
                raise ValueError("native LoRA requires BF16 base dtype and FP32 reduce dtype")
            if self.attn_impl != "sdpa":
                raise ValueError("native LoRA requires attn_impl='sdpa'")
            if self.trust_remote_code:
                raise ValueError("native LoRA does not support trust_remote_code")
        if self.quantized_base is not None:
            if not isinstance(self.quantized_base, QuantizedBaseTrainingConfig):
                raise TypeError("model_qwen3_vl.quantized_base must be QuantizedBaseTrainingConfig or None")
            self.quantized_base.__post_init__()
            if self.adapter is None:
                raise ValueError("quantized_base requires an adapter; frozen-base full fine-tuning is unsupported")
            if self.quantized_base.target_profile != self.adapter.target_profile:
                raise ValueError("quantized_base and adapter target profiles must match")
        if (
            self.adapter is not None
            and self.adapter.initialization == "nf4_residual_svd"
            and self.quantized_base is None
        ):
            raise ValueError("native residual initialization requires its frozen NF4 base")
        if (
            self.adapter is not None
            and self.adapter.initialization == "nf4_residual_svd"
            and self.quantized_base is not None
            and self.quantized_base.excluded_language_layers
        ):
            raise ValueError("residual initialization with excluded language layers is not supported")
        if (
            self.adapter is not None
            and self.adapter.initialization == "nf4_residual_svd"
            and self.adapter_compute_dtype != "float32"
        ):
            raise ValueError("native residual initialization currently requires FP32 adapter computation")
        validate_quantized_linear_execution(self)


# ---------------------------------------------------------------------------
# GR00T-N1.7 VLA backend
# ---------------------------------------------------------------------------

# The base model's DROID pretrain tag. Not "OXE_DROID" -- no such member exists; the real name states
# the action space (relative EEF / relative joint). vLLM-Omni's GR00T pipeline defaults to it too.
OXE_DROID_TAG = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"


@dataclass
class Gr00tN1d7ModelConfig:
    """Everything both the trainer and the rollout must agree on for one GR00T-N1.7 policy.

    Unlike the backends above, this one is not a root-config field: the recipe
    ``configs/gr00t_n1d7_droid.py`` constructs it directly. One recipe is the single source of these
    values; the two roles run in different processes and assert ``identity_hash()`` equality at bind
    time. Fail-fast rather than a log line because every field here is silently corrupting when it
    differs: the loss still descends while the trainer conditions on inputs the rollout never
    produced. Isaac-GR00T (``gr00t.*``) is imported inside methods only; see the module docstring.

    Horizons, and why there are three:
        ``model_action_horizon``  what the checkpoint's action head emits per forward.
        ``training_prefix``       the leading slice the loss is computed over.
        ``execution_horizon``     how many steps the env executes before re-planning.
    They must satisfy ``execution_horizon <= training_prefix <= model_action_horizon``: executing
    past the supervised prefix runs the policy on action steps no loss ever constrained.
    """

    checkpoint_path: str = ""
    # HF revision (commit SHA, tag or branch) of `checkpoint_path`. Empty means "whatever the path
    # holds". The rollout serves `resolved_checkpoint_path()`, never the bare repo id, so a non-empty
    # value pins weights, config.json and processor_config.json to one commit.
    revision: str = ""
    embodiment_tag: str = OXE_DROID_TAG

    model_action_horizon: int = 16
    training_prefix: int = 16
    execution_horizon: int = 8
    action_dim: int = 0

    denoising_steps: int = 4

    # None preserves the loader's buffer precision and historical policy identity.
    backbone_buffer_dtype: str | None = None
    # Opt-in native flow-BC adaptation. None preserves the historical head-only/PPO identity.
    backbone_adaptation: Gr00tBackboneAdaptation | None = None

    # Observation adaptation: GR00T MODALITY KEY -> env observation key. The left side is fixed by the
    # checkpoint's `processor_config.json` and is verified against the loaded checkpoint at startup;
    # the right side is robolab's observation schema, which posttrain cannot import.
    camera_map: dict[str, str] = field(default_factory=dict)
    state_map: dict[str, str] = field(default_factory=dict)
    prompt_key: str = "prompt"
    # Language modality key, also from processor_config.json; verified at startup like the others.
    language_key: str = "annotation.language.language_instruction"

    def __post_init__(self) -> None:
        if self.backbone_adaptation is not None:
            if not isinstance(self.backbone_adaptation, Gr00tBackboneAdaptation):
                raise TypeError("gr00t.backbone_adaptation must be Gr00tBackboneAdaptation or None")
            self.backbone_adaptation.__post_init__()
        if self.backbone_buffer_dtype not in (None, "bfloat16"):
            raise ValueError("gr00t.backbone_buffer_dtype must be None or bfloat16")
        for name in ("model_action_horizon", "training_prefix", "execution_horizon", "denoising_steps"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"gr00t.{name} must be >= 1, got {getattr(self, name)!r}")
        if not (self.execution_horizon <= self.training_prefix <= self.model_action_horizon):
            raise ValueError(
                "gr00t horizons must satisfy execution_horizon <= training_prefix <= "
                f"model_action_horizon; got {self.execution_horizon} <= {self.training_prefix} <= "
                f"{self.model_action_horizon}. Executing past the supervised prefix runs the policy "
                "on action steps the loss never constrained."
            )
        if int(self.action_dim) < 0:
            raise ValueError(f"gr00t.action_dim must be >= 0 (0 = read from checkpoint), got {self.action_dim!r}")
        if not str(self.embodiment_tag).strip():
            raise ValueError("gr00t.embodiment_tag must be a non-empty EmbodimentTag name or value")
        if not self.camera_map:
            raise ValueError("gr00t.camera_map must map at least one GR00T video modality key to an env key")
        if not self.state_map:
            raise ValueError("gr00t.state_map must map at least one GR00T state modality key to an env key")
        if not str(self.language_key).strip():
            raise ValueError("gr00t.language_key must be a non-empty processor language modality key")
        # Normalize so a mapping built from a CLI dotlist and one from a recipe hash identically.
        self.camera_map = {str(k): str(v) for k, v in dict(self.camera_map).items()}
        self.state_map = {str(k): str(v) for k, v in dict(self.state_map).items()}

    def resolve_embodiment_tag(self) -> Any:
        """Return the Isaac-GR00T ``EmbodimentTag``; ``resolve`` accepts the enum name or its value."""
        from gr00t.data.embodiment_tags import EmbodimentTag

        return EmbodimentTag.resolve(self.embodiment_tag)

    def resolved_checkpoint_path(self) -> str:
        """The path the rollout hands its engine: ``checkpoint_path`` pinned to ``revision``.

        Nothing on the serving path takes a revision -- vLLM-Omni's GR00T pipeline loads through its
        ``Gr00tPolicy`` port (a directory, ``from_pretrained``; Isaac-GR00T's original is the same), and
        vLLM-Omni's pre-download ignores ``OmniDiffusionConfig.revision`` -- so pinning happens here. A
        repo id plus a revision resolves to that revision's local HF snapshot, where weights,
        ``config.json`` and ``processor_config.json`` come from one commit, and the engine only ever sees
        a directory it cannot re-resolve. A directory has no revision to pin, so a directory plus a
        revision is refused rather than silently served: ``identity_hash()`` would claim a pin the
        artifact lacks.
        """
        if not self.revision:
            return self.checkpoint_path
        if os.path.isdir(self.checkpoint_path):
            raise ValueError(
                f"gr00t.revision={self.revision!r} cannot pin the local directory {self.checkpoint_path!r}; "
                "point checkpoint_path at the HF repo id, or clear revision"
            )
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=self.checkpoint_path, revision=self.revision)

    def identity(self) -> dict[str, Any]:
        """Effective fields both roles must agree on; an unset buffer override retains legacy hashes."""
        result = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}
        if self.backbone_buffer_dtype is None:
            result.pop("backbone_buffer_dtype")
        if self.backbone_adaptation is None:
            result.pop("backbone_adaptation")
        elif self.backbone_adaptation.action_head_adapter_profile is None:
            result["backbone_adaptation"].pop("action_head_adapter_profile")
        return result

    def identity_hash(self) -> str:
        """Stable hash of ``identity()`` for the trainer/rollout bind-time equality assertion."""
        from cosmos_framework.posttrain.utils.identity import canonical_json_sha256

        return canonical_json_sha256(self.identity(), context="gr00t model config identity")
