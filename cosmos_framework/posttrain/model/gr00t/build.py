# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Construction and input checks for the GR00T joint actor/value capability."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cosmos_framework.posttrain.embodied.sims.libero.config import LiberoConfig
    from cosmos_framework.posttrain.model.gr00t.policy import Gr00tN1d7TrainPolicy
    from cosmos_framework.posttrain.model.gr00t.serving.body import Gr00tPolicyBody
    from cosmos_framework.posttrain.model.parallel_dims import ParallelDims

from cosmos_framework.posttrain.config.model import Gr00tN1d7ModelConfig

CHECKPOINT_REVISION = "2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21"


def _validate_libero10_checkpoint(checkpoint: str) -> None:
    """Check the suite-specific source geometry without initializing PPO resources."""
    import json

    directory = Path(checkpoint)
    if directory.name != "libero_10" or directory.parent.name != CHECKPOINT_REVISION:
        raise ValueError("LIBERO-10 checkpoint must select libero_10 under the pinned snapshot revision")
    model = json.loads((directory / "config.json").read_bytes())
    if (model["action_horizon"], model["max_action_dim"], model["num_inference_timesteps"]) != (40, 132, 4):
        raise ValueError("LIBERO-10 checkpoint must define native 40x132 and four denoising steps")
    processor = json.loads((directory / "processor_config.json").read_bytes())["processor_kwargs"]["modality_configs"][
        "libero_sim"
    ]
    if processor["action"]["delta_indices"] != list(range(16)):
        raise ValueError("LIBERO-10 processor must decode sixteen actions")


def libero10_model_identity(checkpoint: str) -> dict:
    """Bind complete weight and processor sources under the reviewed LIBERO-10 geometry."""
    from cosmos_framework.posttrain.model.gr00t.source import resolve_ppo_source

    _validate_libero10_checkpoint(checkpoint)
    return {
        "repository": "nvidia/GR00T-N1.7-LIBERO",
        "revision": CHECKPOINT_REVISION,
        "subfolder": "libero_10",
        "content": resolve_ppo_source(checkpoint).identity(),
    }


def validate_joint_recipe(model: Gr00tN1d7ModelConfig, environment: LiberoConfig, *, ranks: int) -> None:
    model.__post_init__()
    environment.__post_init__()
    if not model.checkpoint_path or not environment.sim_python:
        raise ValueError("GR00T recipe requires model.checkpoint_path and environment.config.sim_python")
    if ranks < 2:
        raise ValueError("GR00T joint PPO currently requires at least two learner ranks")
    if model.execution_horizon != environment.execution_horizon or model.action_dim != 7:
        raise ValueError("GR00T action geometry differs from the bound LIBERO environment")
    if environment.suite == "libero_10":
        _validate_libero10_checkpoint(model.checkpoint_path)


def build_joint_policy(
    config: Gr00tN1d7ModelConfig, *, output: str, parallel_dims: ParallelDims
) -> Gr00tN1d7TrainPolicy:
    import torch.distributed as dist

    from cosmos_framework.posttrain.model.gr00t.policy import BF16_PRECISION, Gr00tN1d7TrainPolicy

    os.environ["GR00T_DIT_SDPA_MODE"] = "math"
    rank = dist.get_rank() if dist.is_initialized() else 0
    directory = Path(output) / "model-config" / f"policy-{rank}"
    directory.mkdir(parents=True, exist_ok=True)
    model = Gr00tN1d7TrainPolicy(
        config, device="cuda", parallel_dims=parallel_dims, save_cfg_dir=directory, precision=BF16_PRECISION, ppo=True
    )
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise ValueError("GR00T PPO backbone must remain frozen")
    return model


def build_joint_rollout(
    config: Gr00tN1d7ModelConfig, *, output: str, replica: int, evaluation_mode: str = "ode"
) -> Gr00tPolicyBody:
    from cosmos_framework.posttrain.model.gr00t.serving.body import Gr00tPolicyBody

    os.environ["GR00T_DIT_SDPA_MODE"] = "math"
    rollout = Gr00tPolicyBody(
        config,
        ppo=True,
        ppo_save_cfg_dir=str(Path(output) / "model-config" / f"rollout-{replica}"),
        transfer_dtype="bfloat16",
        verify_weight_values=True,
    )
    try:
        if config.backbone_buffer_dtype is not None:
            precision = rollout.inference_fingerprint()
            if set(precision["buffer_dtypes"].values()) != {f"torch.{config.backbone_buffer_dtype}"}:
                raise ValueError("rollout backbone buffers differ from the bound model configuration")
        rollout.evaluation_mode = evaluation_mode
        return rollout
    except BaseException:
        rollout.close()
        raise
