# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GR00T-N1.7 on the robolab/DROID sim -- the single source of the shared model contract.

The trainer and the rollout run in separate processes and cannot share the object, so both build it
from here and assert ``Gr00tN1d7ModelConfig.identity_hash()`` equality at bind time. See
``operator GR00T validation documentation``.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from cosmos_framework.posttrain.algorithm import flow_bc
from cosmos_framework.posttrain.algorithm.primitives.objective import Objective
from cosmos_framework.posttrain.config.model import OXE_DROID_TAG, Gr00tN1d7ModelConfig
from cosmos_framework.posttrain.config.rl import OptimizerConfig
from cosmos_framework.posttrain.model.gr00t.packer import collate_transitions

# The DROID-finetuned checkpoint, not the `nvidia/GR00T-N1.7-3B` base: its processor carries the
# `oxe_droid_relative_eef_relative_joint` modality contract below. Pin `revision` to an immutable
# commit before quoting numbers from a run.
GR00T_N1D7_DROID = "nvidia/GR00T-N1.7-DROID"

# From the checkpoint's processor_config.json
# (`processor_kwargs.modality_configs.oxe_droid_relative_eef_relative_joint`). The rollout body
# re-reads these from the LOADED processor at startup and refuses to run if they disagree.
DROID_ACTION_HORIZON = 40
DROID_ACTION_DIM = 17  # eef_9d(9) + gripper_position(1) + joint_position(7)
DROID_LANGUAGE_KEY = "annotation.language.language_instruction"


def make_model_config(**overrides) -> Gr00tN1d7ModelConfig:
    """Return the DROID GR00T-N1.7 model contract, with recipe-level overrides re-validated."""
    config = Gr00tN1d7ModelConfig(
        checkpoint_path=GR00T_N1D7_DROID,
        revision="",
        embodiment_tag=OXE_DROID_TAG,
        model_action_horizon=DROID_ACTION_HORIZON,
        # The supervised prefix: the useful signal is in the near horizon and the env re-plans long
        # before step 40. Anything in [execution_horizon, model_action_horizon] is valid.
        training_prefix=16,
        execution_horizon=8,  # robolab's own DROID eval re-plans every 8 steps
        # The action vector is the concatenation of the three action modalities; the body checks the
        # realised width on every call. 0 would mean "accept whatever the checkpoint emits".
        action_dim=DROID_ACTION_DIM,
        denoising_steps=4,  # config.json `num_inference_timesteps`
        # LEFT = the checkpoint's modality keys (verified at startup). RIGHT = env observation keys,
        # not yet confirmed against a running robolab env -- see operator GR00T validation documentation.
        camera_map={
            "exterior_image_1_left": "observation/exterior_image_1_left",
            "wrist_image_left": "observation/wrist_image_left",
        },
        state_map={
            "eef_9d": "observation/eef_9d",
            "gripper_position": "observation/gripper_position",
            "joint_position": "observation/joint_position",
        },
        language_key=DROID_LANGUAGE_KEY,
    )
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise AttributeError(f"unknown gr00t model config field {key!r}")
        setattr(config, key, value)
    config.__post_init__()  # re-validate: overrides can violate the horizon ordering
    return config


def make_optimizer_config() -> OptimizerConfig:
    """The official GR00T-N1.7 finetuning optimizer; the parity gate's T0 asserts these against Isaac/HF."""
    return OptimizerConfig(
        name="adamw",
        impl="",  # launch_finetune.py forces optim="adamw_torch": neither `fused` nor `foreach` is passed
        decay_groups="hf",  # Trainer.create_optimizer's [decay, no-decay] split
        lr=1e-4,  # TrainingConfig.learning_rate
        weight_decay=1e-5,  # TrainingConfig.weight_decay
        betas=[0.9, 0.999],  # HF TrainingArguments defaults; TrainingConfig does not carry them
        eps=1e-8,
        grad_clip_norm=1.0,  # TrainingConfig.max_grad_norm
    )


def make_objective(config: Gr00tN1d7ModelConfig, collator: Any) -> Objective:
    """Flow-BC over this recipe's action contract, collated by the checkpoint's own `Gr00tN1d7DataCollator`
    (read off the loaded model or processor, never constructed here: it is bound to a tokenizer repo)."""
    return flow_bc.make_objective(
        partial(
            collate_transitions, collator=collator, action_dim=config.action_dim, training_prefix=config.training_prefix
        ),
        training_prefix=config.training_prefix,
        action_dim=config.action_dim,
    )
