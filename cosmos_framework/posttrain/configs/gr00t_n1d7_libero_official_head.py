# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Resource-adjusted existing GR00T finetune control: B1/300, not upstream B64/10000."""

from cosmos_framework.posttrain.config.gr00t_official import OFFICIAL_HEAD_PROFILE, Gr00tOfficialHeadSchedule
from cosmos_framework.posttrain.configs.gr00t_n1d7_droid import make_optimizer_config
from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_qlora import make_config as matched_config
from cosmos_framework.posttrain.model.gr00t.official_head import OfficialHeadDataConfig


def make_config():
    config = matched_config()
    backbone_path = config.model.backbone_adaptation.backbone_path
    config.model.backbone_adaptation = None
    config.job.name = "gr00t_n1d7_libero_official_head_b1_300"
    config.policy_factory = "cosmos_framework.posttrain.model.gr00t.official_head.build_policy"
    config.data = OfficialHeadDataConfig(
        train_path=config.data.train_path, validation_path=config.data.validation_path, backbone_path=backbone_path
    )
    config.dense_policy_factory = config.dense_sidecars_factory = ""
    config.rl.optimizer_profile = OFFICIAL_HEAD_PROFILE
    config.rl.optimizer = make_optimizer_config()
    config.rl.max_steps = 300
    config.rl.optimizer.warmup_steps = 15
    config.rl.lr_schedule = Gr00tOfficialHeadSchedule()
    config.rl.evaluation_every = 100
    return config
