# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Typed root configuration and cross-role validation for PPO."""

from __future__ import annotations

import math
from collections.abc import Mapping

import attrs

from cosmos_framework.posttrain.config.config import PosttrainConfig, build_root
from cosmos_framework.posttrain.config.model import DataPackerConfig, HFModelConfig
from cosmos_framework.posttrain.config.rl import (
    OPTIMIZER_NAMES,
    PPOConfig,
    PPOGridConfig,
    RolloutConfig,
    normalize_prompt_metadata_columns,
)


@attrs.define(slots=False)
class PosttrainPPOConfig(PosttrainConfig):
    """PPO root schema for policy, rollout, frozen reference, and critic roles."""

    algorithm: str = "ppo"
    model_backend: str = "hf"
    rl: PPOConfig = attrs.field(factory=PPOConfig)
    grid: PPOGridConfig = attrs.field(factory=PPOGridConfig)
    rollout: RolloutConfig = attrs.field(factory=lambda: RolloutConfig(tensor_parallel_size=4))
    data_packer: DataPackerConfig = attrs.field(factory=DataPackerConfig)
    reference_model_hf: HFModelConfig = attrs.field(factory=HFModelConfig)
    critic_model_hf: HFModelConfig = attrs.field(factory=HFModelConfig)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _validate_sampling(rollout: RolloutConfig) -> None:
    temperature = float(rollout.temperature)
    if not math.isfinite(temperature) or temperature < 0.01:
        raise ValueError("rollout.temperature must be finite and >= 0.01")
    if float(rollout.top_p) != 1.0:
        raise ValueError("PPO requires rollout.top_p=1.0")
    if isinstance(rollout.top_k, bool) or not isinstance(rollout.top_k, int) or rollout.top_k > 0:
        raise ValueError("PPO requires rollout.top_k<=0")
    if float(rollout.repetition_penalty) != 1.0:
        raise ValueError("PPO requires rollout.repetition_penalty=1.0")


def _validate_optimizer(optimizer: object, role: str) -> None:
    if getattr(optimizer, "name", None) not in OPTIMIZER_NAMES:
        raise ValueError(f"{role} optimizer must be one of {OPTIMIZER_NAMES}")
    for name in ("lr", "eps"):
        value = getattr(optimizer, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"{role} optimizer {name} must be finite and positive")
    grad_clip = getattr(optimizer, "grad_clip_norm")
    if not math.isfinite(float(grad_clip)) or grad_clip < 0:
        raise ValueError(f"{role} optimizer grad_clip_norm must be finite and non-negative")


def _model_dp_worlds(config: PosttrainPPOConfig) -> tuple[int, int]:
    """Return policy and critic model-DP widths from the selected model backend."""
    from cosmos_framework.posttrain.config.build import (
        native_policy_build_spec,
        normalize_model_backend,
        resolve_hf_model_parallelism,
        resolve_policy_parallelism,
    )

    policy_ranks = int(config.grid.policy_shape()[1])
    critic_ranks = int(config.grid.critic_ranks)
    if normalize_model_backend(config.model_backend) == "hf":
        policy = resolve_hf_model_parallelism(
            config.model_hf,
            policy_ranks,
            field_name="config.model_hf.parallelism",
            ranks_field_name="grid.policy_shape",
        )
        critic = resolve_hf_model_parallelism(
            config.critic_model_hf,
            critic_ranks,
            field_name="config.critic_model_hf.parallelism",
            ranks_field_name="grid.critic_ranks",
        )
        return policy[0] * policy[1], critic[0] * critic[1]

    spec = native_policy_build_spec(config)
    if spec is None or not str(getattr(spec, "value_builder", "")):
        raise ValueError("PPO model_backend='i4' requires a native policy and value builder")
    policy = resolve_policy_parallelism(config, policy_ranks)
    return policy[0] * policy[1], critic_ranks


def validate_ppo_config(config: PosttrainPPOConfig) -> None:
    """Validate only constraints spanning independently owned config sections."""
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.config.build import normalize_model_backend, resolve_precision
    from cosmos_framework.posttrain.config.rl import require_full_emergency_plan
    from cosmos_framework.posttrain.control.placement import (
        resolve_online_weight_placement,
        validate_online_weight_transport,
    )
    from cosmos_framework.posttrain.data.packer import validate_data_packer_factories
    from cosmos_framework.posttrain.roles.env.prompt import validate_dataset_source

    if config.algorithm != "ppo":
        raise ValueError(f"PPO requires algorithm='ppo', got {config.algorithm!r}")
    normalize_model_backend(config.model_backend)
    resolve_precision(config)
    validate_data_packer_factories(instantiate(config.data_packer))
    if config.grid.backend != "ray" or config.grid.grouping != "per_role":
        raise ValueError("PPO requires grid.backend='ray' and grid.grouping='per_role'")
    if config.rl.checkpoint_store.backend != "local":
        raise ValueError("PPO multi-role checkpoints currently require rl.checkpoint_store.backend='local'")
    require_full_emergency_plan(
        config.rl,
        algorithm="PPO",
        reason="its multi-role checkpoint boundary writes every role with the full plan, at the cadence and at a "
        "termination trigger alike (control/checkpointing.py, MultiRoleCheckpointCoordinator._write_boundary)",
    )
    _backend, _gpu_placement, weight_relationship = resolve_online_weight_placement(
        config.grid.backend,
        config.grid.gpu_placement,
    )
    validate_online_weight_transport(weight_relationship, config.rl.weight_sync_transport)

    policy_shape = config.grid.policy_shape()
    rollout_shape = config.grid.rollout_shape()
    if policy_shape[0] != 1:
        raise ValueError("PPO policy must have one execution replica")
    if (
        config.grid.gpu_placement == "colocated"
        and policy_shape[0] * policy_shape[1] != rollout_shape[0] * rollout_shape[1]
    ):
        raise ValueError("colocated PPO requires equal policy and rollout actor counts")
    if _positive_int(config.rollout.tensor_parallel_size, "rollout.tensor_parallel_size") != rollout_shape[1]:
        raise ValueError("rollout.tensor_parallel_size must equal rollout rank width")
    if getattr(config.rollout, "custom_all_reduce", False):
        raise ValueError(
            "PPO does not support rollout.custom_all_reduce: its rollout actors keep one visible GPU, so vLLM "
            "cannot map the tensor-parallel peers' buffers; only GRPO's full-node experience group exposes them"
        )

    prompts = _positive_int(config.rl.prompts_per_iteration, "rl.prompts_per_iteration")
    rollout_replicas = _positive_int(rollout_shape[0], "rollout replicas")
    if prompts % rollout_replicas:
        raise ValueError("rl.prompts_per_iteration must be divisible by rollout replicas")
    minibatches = _positive_int(config.rl.num_minibatches, "rl.num_minibatches")
    microbatch = _positive_int(config.rl.microbatch_size, "rl.microbatch_size")
    policy_dp, critic_dp = _model_dp_worlds(config)
    for role, dp_world in (("policy", policy_dp), ("critic", critic_dp)):
        divisor = dp_world * minibatches * microbatch
        if prompts % divisor:
            raise ValueError(
                f"rl.prompts_per_iteration={prompts} must be divisible by {role} "
                f"dp_world * num_minibatches * microbatch_size ({divisor})"
            )

    _validate_sampling(config.rollout)
    _validate_optimizer(config.rl.optimizer, "policy")
    _validate_optimizer(config.rl.critic_optimizer, "critic")
    if type(config.rl.save_every) is not int or config.rl.save_every < 0:
        raise ValueError("rl.save_every must be a non-negative integer")
    # Repeated here rather than left to `PPOConfig.__post_init__` because `--validate` inspects the
    # UNINSTANTIATED root: a dotlist override reaches this check before any dataclass is built.
    if type(config.rl.critic_warmup_iterations) is not int or config.rl.critic_warmup_iterations < 0:
        raise ValueError(
            f"rl.critic_warmup_iterations must be a non-negative integer, got {config.rl.critic_warmup_iterations!r}"
        )

    data = config.rl.data
    validate_dataset_source(data)
    data.metadata_columns = list(
        normalize_prompt_metadata_columns(
            data.metadata_columns,
            configured_columns=(
                data.prompt_column,
                data.response_column,
                data.image_column,
                data.video_column,
                data.id_column,
            ),
        )
    )
    has_dataset = bool(data.name or data.path or data.factory)
    if has_dataset and not data.split:
        raise ValueError("dataset-backed PPO requires split")
    if has_dataset and config.rl.modality != "text" and not getattr(data, f"{config.rl.modality}_column"):
        raise ValueError(f"dataset-backed {config.rl.modality} PPO requires its media column")
    if not isinstance(config.rl.reward.terminal_kwargs, Mapping) or not isinstance(
        config.rl.reward.process_kwargs, Mapping
    ):
        raise TypeError("PPO reward kwargs must be mappings")


def make_config() -> PosttrainPPOConfig:
    return build_root(PosttrainPPOConfig, "posttrain_ppo", register_experiment_configs=False)
