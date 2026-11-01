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

import os
from dataclasses import asdict, dataclass, field
from typing import Any

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
# Native policy backend
# ---------------------------------------------------------------------------


@dataclass
class NativePolicyParallelismConfig:
    """Policy data-parallel mesh for a native policy builder.

    The policy WorkerGroup supplies the world size. ``-1`` asks posttrain to infer the matching
    dimension from that world, using the same convention as :class:`HFParallelismConfig`.
    Model-family restrictions remain the builder's responsibility; for example, the Qwen3-VL
    builder accepts FSDP and HSDP over the policy rank world.
    """

    dp_replicate: int = 1
    dp_shard: int = -1


@dataclass
class NativePolicyBuildSpec:
    """Generic construction seam for a native online policy.

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

    The loader lives with the backend in ``model/hf.py``. The equivalent native parallelism source is
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
class Qwen3VLModelConfig:
    """Model-family config carried by a Qwen3-VL ``NativePolicyBuildSpec``.

    Dependency-light strings live here; ``model/qwen3_vl.py`` imports the native model,
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
    fsdp_reduce_dtype: str = "float32"

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

    # Observation adaptation: GR00T MODALITY KEY -> env observation key. The left side is fixed by the
    # checkpoint's `processor_config.json` and is verified against the loaded checkpoint at startup;
    # the right side is robolab's observation schema, which posttrain cannot import.
    camera_map: dict[str, str] = field(default_factory=dict)
    state_map: dict[str, str] = field(default_factory=dict)
    prompt_key: str = "prompt"
    # Language modality key, also from processor_config.json; verified at startup like the others.
    language_key: str = "annotation.language.language_instruction"

    def __post_init__(self) -> None:
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
        return result

    def identity_hash(self) -> str:
        """Stable hash of ``identity()`` for the trainer/rollout bind-time equality assertion."""
        from cosmos_framework.posttrain.utils.identity import canonical_json_sha256

        return canonical_json_sha256(self.identity(), context="gr00t model config identity")
