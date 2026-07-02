# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Any, List

import attrs

from cosmos_framework.trainer import ImaginaireTrainer as Trainer
from cosmos_framework.utils import config


@attrs.define(slots=False)
class DataSetting:
    """Configuration for data.

    Attributes:
        qwen_max_video_token_length: Maximum video token length.
        qwen_target_fps: Target fps for video sampling.
        text_chat_order: Order of text items in user messages.
    """

    qwen_max_video_token_length: int = 8192


@attrs.define(slots=False)
class Config(config.Config):
    data_setting: DataSetting = attrs.field(factory=DataSetting)
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"model": "mot_fsdp"},
            {"data_train": None},
            {"data_val": None},
            {"optimizer": "adamw"},
            {"scheduler": "warmup_cosine_lr"},
            {"checkpoint": "s3"},
            {"callbacks": ["basic", "optimization", "job_monitor", "generation"]},
            {"ema": "power"},
            {"tokenizer": "wan2pt2_tokenizer"},
            {"sound_tokenizer": None},  # Optional: for audio-video generation
            {"vlm_config": None},
            {"ckpt_type": "dcp"},
            {"experiment": None},
        ]
    )


def make_config() -> Config:
    c = Config(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    # Specifying values through instances of attrs
    c.job.project = "cosmos3_vfm"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = Trainer
    c.trainer.straggler_detection.enabled = False
    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 20
    c.trainer.validation_iter = 100
    c.trainer.run_validation = False
    c.trainer.callbacks = None

    c.upload_reproducible_setup = False

    from cosmos_framework.configs.base.defaults.callbacks import register_callbacks
    from cosmos_framework.configs.base.defaults.checkpointer import register_checkpoint, register_ckpt_type
    from cosmos_framework.configs.base.defaults.ema import register_ema

    # from cosmos_framework.configs.base.defaults.data import register_data
    from cosmos_framework.configs.base.defaults.model import register_model
    from cosmos_framework.configs.base.defaults.optimizer import register_optimizer, register_scheduler
    from cosmos_framework.configs.base.defaults.tokenizer import register_sound_tokenizer, register_tokenizer
    from cosmos_framework.configs.base.defaults.reasoner import register_vlm

    # Call this function to register config groups for advanced overriding. the order follows the default config groups
    # register_data()
    register_model()
    register_checkpoint()
    register_ckpt_type()
    register_optimizer()
    register_scheduler()
    register_callbacks()
    register_tokenizer()
    register_sound_tokenizer()
    register_ema()
    register_vlm()

    # Register shipped experiments explicitly. (vision_sft_nano also defines
    # vision_sft_nano_mapstyle_dataloader — the CosmosDataLoader variant — in the same module.)
    import cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano  # noqa: F401
    import cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_all_nano  # noqa: F401
    import cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_nano  # noqa: F401
    import cosmos_framework.configs.base.experiment.sft.vision_sft_nano  # noqa: F401
    import cosmos_framework.configs.base.experiment.sft.vision_sft_super  # noqa: F401

    return c
