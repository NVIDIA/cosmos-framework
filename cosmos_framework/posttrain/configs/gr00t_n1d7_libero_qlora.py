# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native N1.7 LIBERO flow-BC QLoRA; override the model/data paths with local staged copies.

Set backbone_adaptation.mode=bf16_lora for BF16 LoRA, or nf4_frozen/bf16_frozen to train only
the native projector/action head on a frozen NF4/BF16 backbone.
"""

from cosmos_framework.posttrain.config.action_sft import make_config as base_config
from cosmos_framework.posttrain.config.gr00t_bc import Gr00tBackboneAdaptation
from cosmos_framework.posttrain.config.model import Gr00tN1d7ModelConfig
from cosmos_framework.posttrain.model.gr00t.action_sft import Gr00tActionDataConfig


def make_config():
    config = base_config()
    config.job.name = "gr00t_n1d7_libero_nf4_flow"
    config.policy_factory = "cosmos_framework.posttrain.model.gr00t.action_sft.build_policy"
    config.data_factory = "cosmos_framework.posttrain.model.gr00t.action_sft.build_dataset"
    config.objective_factory = "cosmos_framework.posttrain.model.gr00t.action_sft.build_objective"
    config.dense_policy_factory = "cosmos_framework.posttrain.model.gr00t.dense.build_policy"
    config.dense_sidecars_factory = "cosmos_framework.posttrain.model.gr00t.dense.sidecars"
    config.evaluation_callback = "cosmos_framework.posttrain.model.gr00t.action_sft.evaluate_actions"
    config.model = Gr00tN1d7ModelConfig(
        checkpoint_path="/path/to/GR00T-N1.7-LIBERO/libero_spatial",
        embodiment_tag="libero_sim",
        model_action_horizon=40,
        training_prefix=10,
        execution_horizon=8,
        action_dim=7,
        denoising_steps=4,
        camera_map={"image": "image", "wrist_image": "wrist_image"},
        state_map={key: key for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")},
        backbone_adaptation=Gr00tBackboneAdaptation(backbone_path="/path/to/Cosmos-Reason2-2B"),
    )
    config.data = Gr00tActionDataConfig(train_path="/path/to/train.jsonl", validation_path="/path/to/validation.jsonl")
    config.rl.max_steps = 100
    config.rl.seed = 17
    config.rl.evaluation_every = 25
    config.rl.optimizer.lr = 1e-5
    config.rl.optimizer.weight_decay = 1e-5
    config.rl.optimizer.eps = 1e-8
    config.rl.optimizer.grad_clip_norm = 1.0
    return config
