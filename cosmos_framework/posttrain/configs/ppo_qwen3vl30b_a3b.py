# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native Qwen3-VL-30B-A3B policy and value-model PPO recipe."""

from __future__ import annotations

from cosmos_framework.posttrain.config.config_ppo import PosttrainPPOConfig
from cosmos_framework.posttrain.config.config_ppo import make_config as _make_root_config
from cosmos_framework.posttrain.config.model import (
    NativePolicyBuildSpec,
    NativePolicyParallelismConfig,
    Qwen3VLModelConfig,
)

QWEN3_VL_30B_A3B = "Qwen/Qwen3-VL-30B-A3B-Instruct"
QWEN3_VL_30B_A3B_REVISION = "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"

_NATIVE_POLICY_BUILDER = "cosmos_framework.posttrain.model.qwen3_vl.build_qwen3_vl_native_policy"
_NATIVE_VALUE_BUILDER = "cosmos_framework.posttrain.model.qwen3_vl.build_qwen3_vl_native_value_model"


def make_config() -> PosttrainPPOConfig:
    """Return the typed native policy/value PPO recipe."""
    config = _make_root_config()

    # The inherited Cosmos3 model, EMA, and tokenizer groups all write below ``model.config``.
    # This recipe owns ``model`` as a NativePolicyBuildSpec, so those groups must not merge a
    # generative-model tree into the typed online-policy seam during Hydra composition.
    model_owned_default_groups = {"model", "ema", "tokenizer"}
    config.defaults = [
        entry
        for entry in config.defaults
        if not (isinstance(entry, dict) and model_owned_default_groups.intersection(entry))
    ]
    config.model_backend = "i4"
    config.model = NativePolicyBuildSpec(
        builder=_NATIVE_POLICY_BUILDER,
        value_builder=_NATIVE_VALUE_BUILDER,
        model_id=QWEN3_VL_30B_A3B,
        model_config=Qwen3VLModelConfig(
            # NativePolicyBuildSpec.model_id is the single artifact identity. The builders inject
            # it here on each role; checkpoint_path may still be overridden with a storage mirror.
            name_or_path="",
            revision=QWEN3_VL_30B_A3B_REVISION,
            trust_remote_code=False,
            checkpoint_path="",
            credential_path="",
            dtype="bfloat16",
            attn_impl="sdpa",
            activation_checkpointing="full",
            fsdp_reduce_dtype="float32",
        ),
        parallelism=NativePolicyParallelismConfig(dp_replicate=1, dp_shard=-1),
        requires_collated_policy_forward=True,
    )

    config.rollout.name_or_path = QWEN3_VL_30B_A3B
    config.rollout.revision = QWEN3_VL_30B_A3B_REVISION
    config.rollout.tokenizer_revision = QWEN3_VL_30B_A3B_REVISION
    config.rollout.processor_revision = QWEN3_VL_30B_A3B_REVISION
    config.rollout.trust_remote_code = False
    config.rollout.dtype = "bfloat16"
    return config
