# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from cosmos_framework.trainer import ImaginaireTrainer
from cosmos_framework.utils import log
from cosmos_framework.utils.config_helper import import_all_modules_from_package
from cosmos_framework.configs.base.vlm.defaults.callbacks import register_callbacks
from cosmos_framework.configs.base.vlm.defaults.checkpointer import register_checkpoint, register_ckpt_type
from cosmos_framework.configs.base.vlm.defaults.config import Config

from cosmos_framework.configs.base.vlm.defaults.model import register_model
from cosmos_framework.configs.base.vlm.defaults.optimizer import register_optimizer, register_scheduler
from cosmos_framework.configs.base.vlm.defaults.vlm_policy import register_vlm_policy


def make_config() -> Config:
    c = Config(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    # Specifying values through instances of attrs
    c.job.project = "cosmos_reason2"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    # Unified path: ImaginaireTrainer drives both VLM and VFM.
    c.trainer.type = ImaginaireTrainer
    c.trainer.straggler_detection.enabled = False
    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 20
    c.trainer.validation_iter = 100
    c.trainer.run_validation = False
    c.trainer.callbacks = None
    c.trainer.cudnn.benchmark = False
    c.upload_reproducible_setup = True

    # Call this function to register config groups for advanced overriding. the order follows the default config groups
    register_model()
    register_vlm_policy()
    # Register dataloader configs
    log.info("Registering optimizer, scheduler, checkpoint, ckpt type, and callbacks")
    register_optimizer()
    register_scheduler()
    register_checkpoint()
    register_ckpt_type()
    register_callbacks()
    import_all_modules_from_package("cosmos_framework.configs.base.vlm.experiment", reload=True)
    return c
