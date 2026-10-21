# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.manual_gc import ManualGarbageCollection
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.callback import LowPrecisionCallback
from cosmos_framework.callbacks.device_monitor import DeviceMonitor
from cosmos_framework.callbacks.dmd2_metrics import (
    DMD2Metrics,  # noqa: F401 - release-only callback key
)
from cosmos_framework.callbacks.expert_heatmap import ExpertHeatmap
from cosmos_framework.callbacks.grad_clip_distillation import GradClip as GradClipCosmos3
from cosmos_framework.callbacks.heart_beat import HeartBeat
from cosmos_framework.callbacks.iter_speed import IterSpeed
from cosmos_framework.callbacks.mfu import MFUCallback
from cosmos_framework.callbacks.norm_monitor import NormMonitor
from cosmos_framework.callbacks.param_count import ParamCount
from cosmos_framework.callbacks.sequence_packing_padding import SequencePackingPadding
from cosmos_framework.callbacks.sigma_loss_analysis import SigmaLossAnalysis
from cosmos_framework.callbacks.skip_nan_step import SkipNaNStep
from cosmos_framework.callbacks.wandb_log_distillation import WandbCallback


# ============================= cosmos3 =============================
BASIC_CALLBACKS_COSMOS3 = dict(
    iter_speed=L(IterSpeed)(  # does not use model or optimizer
        every_n="${trainer.logging_iter}",
        save_s3="${upload_reproducible_setup}",
        save_s3_every_log_n=500,
        hit_thres=200,
    ),
    manual_gc=L(ManualGarbageCollection)(every_n=5),  # does not use model or optimizer
    wandb=L(WandbCallback)(
        save_s3="${upload_reproducible_setup}",
        logging_iter_multipler=1,
        save_logging_iter_multipler=10,
    ),
    wandb_10x=L(WandbCallback)(
        logging_iter_multipler=10,
        save_logging_iter_multipler=1,
        save_s3="${upload_reproducible_setup}",
    ),
    dmd2_metrics=L(DMD2Metrics)(output_path=""),
    param_count=L(ParamCount)(  # use model
        save_s3="${upload_reproducible_setup}",
    ),
    expert_heatmap=L(ExpertHeatmap)(),  # for monitoring MoE router activation
    # compile_tokenizer=L(CompileTokenizer)(enabled=False, compile_after_iterations=4, dynamic=True),
    norm_monitor=L(NormMonitor)(
        every_n=5000,
        log_stat_wandb=True,
        save_s3="${upload_reproducible_setup}",
        track_activations=True,
    ),
)

OPTIMIZATION_CALLBACKS_COSMOS3 = dict(
    skip_nan_step=L(SkipNaNStep)(max_consecutive_nan=100),
    grad_clip=L(GradClipCosmos3)(clip_norm=1.0),
    low_precision=L(LowPrecisionCallback)(update_iter=1, config=None, trainer=None),
)


MONITORING_CALLBACKS_COSMOS3 = dict(
    heart_beat=L(HeartBeat)(
        every_n=200,
        update_interval_in_minute=20,
        save_s3="${upload_reproducible_setup}",
    ),
    device_monitor=L(DeviceMonitor)(
        every_n=200,
        save_s3="${upload_reproducible_setup}",
        upload_every_n_mul=5,
    ),
    mfu=L(MFUCallback)(every_n="${trainer.logging_iter}"),
    sequence_packing_padding=L(SequencePackingPadding)(every_n="${trainer.logging_iter}"),
    sigma_loss_analysis=L(SigmaLossAnalysis)(
        every_n=5000,
        every_n_viz=5000,
        save_s3="${upload_reproducible_setup}",
    ),
)


def register_distill_callbacks() -> None:
    cs = ConfigStore.instance()
    # ============================= cosmos3 =============================
    cs.store(group="callbacks", package="trainer.callbacks", name="basic_cosmos3_distill", node=BASIC_CALLBACKS_COSMOS3)
    cs.store(
        group="callbacks",
        package="trainer.callbacks",
        name="optimization_cosmos3_distill",
        node=OPTIMIZATION_CALLBACKS_COSMOS3,
    )
    cs.store(
        group="callbacks",
        package="trainer.callbacks",
        name="job_monitor_cosmos3_distill",
        node=MONITORING_CALLBACKS_COSMOS3,
    )
    # AR sampling callback for causal training visualization
