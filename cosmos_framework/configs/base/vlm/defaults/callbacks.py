# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Dataloader config options.
Based on projects/cosmos/ar/v1/configs/registry.py
"""

from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.manual_gc import ManualGarbageCollection
from cosmos_framework.utils.lazy_config import PLACEHOLDER
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.callback import LowPrecisionCallback, WandBCallback
from cosmos_framework.callbacks.dataloader_state import DataLoaderStateCallback

from cosmos_framework.callbacks.grad_clip import GradClip
from cosmos_framework.callbacks.hf_export import HFExportCallback
from cosmos_framework.callbacks.iter_speed import IterSpeed
from cosmos_framework.callbacks.learning_rate_logger import LearningRateLogger
from cosmos_framework.callbacks.log_tensor_shape import LogTensorShapeCallback
from cosmos_framework.callbacks.param_count import ParamCount
from cosmos_framework.callbacks.wandb_log import WandbCallback as WandBCallbackMultiplier
from cosmos_framework.callbacks.wandb_vis import VisualizationLoggingCallback
from cosmos_framework.configs.base.defaults.callbacks import JOB_MONITOR_CALLBACKS

# from cosmos_framework.utils.callback import NVTXCallback


def register_callbacks():
    cs = ConfigStore.instance()
    BASIC_CALLBACKS = dict(
        iter_speed=L(IterSpeed)(  # does not use model or optimizer
            every_n="${trainer.logging_iter}",
            save_s3="${upload_reproducible_setup}",
            save_s3_every_log_n=500,
            hit_thres=50,
        ),
        manual_gc=L(ManualGarbageCollection)(every_n=5),  # does not use model or optimizer
        wandb=L(WandBCallback)(),
        param_count=L(ParamCount)(  # use model
            save_s3="${upload_reproducible_setup}",
        ),
        grad_clip=L(GradClip)(clip_norm=1.0, force_finite=False),  # use model
        learning_rate_logger=L(LearningRateLogger)(every_n=10),
        low_precision=L(LowPrecisionCallback)(
            update_iter=1,
            config=PLACEHOLDER,
            trainer=PLACEHOLDER,
        ),  # reads model.precision; no extra kwarg needed

        # nvtx=L(NVTXCallback)(synchronize=True),
    )

    LOG_CALLBACKS = dict(
        wandb_10x=L(WandBCallbackMultiplier)(
            logging_iter_multipler=10,
            save_logging_iter_multipler=1,
            save_s3="${upload_reproducible_setup}",
        ),
        wandb_2x=L(WandBCallbackMultiplier)(
            logging_iter_multipler=2,
            save_logging_iter_multipler=1,
            save_s3="${upload_reproducible_setup}",
        ),
        log_tensor_shape=L(LogTensorShapeCallback)(num_log=10),
        dataloader_state=L(DataLoaderStateCallback)(
            distributor_type="${data_setting.distributor_type}",
        ),
    )

    cs.store(group="callbacks", package="trainer.callbacks", name="basic_vlm", node=BASIC_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="basic_log", node=LOG_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="job_monitor", node=JOB_MONITOR_CALLBACKS)

    DATA_VIS_CALLBACKS_QWEN = dict(
        wandb_vis=L(VisualizationLoggingCallback)(
            every_n=500,
        ),
    )
    cs.store(group="callbacks", package="trainer.callbacks", name="data_vis_qwen", node=DATA_VIS_CALLBACKS_QWEN)

    HF_EXPORT_CALLBACKS = dict(
        hf_export=L(HFExportCallback)(
            dtype="${model.config.precision}",
        ),
    )
    cs.store(group="callbacks", package="trainer.callbacks", name="hf_export", node=HF_EXPORT_CALLBACKS)
