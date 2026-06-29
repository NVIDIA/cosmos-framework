# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Dataloader config options."""

from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.manual_gc import ManualGarbageCollection
from cosmos_framework.utils.lazy_config import PLACEHOLDER
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.callback import LowPrecisionCallback, WandBCallback
from cosmos_framework.callbacks.compile_tokenizer import CompileTokenizer

from cosmos_framework.callbacks.device_monitor import DeviceMonitor
from cosmos_framework.callbacks.dit_image_sample import DiTImageSampleCallback
from cosmos_framework.callbacks.every_n_draw_sample import EveryNDrawSample
from cosmos_framework.callbacks.expert_heatmap import ExpertHeatmap
from cosmos_framework.callbacks.grad_clip import GradClip
from cosmos_framework.callbacks.heart_beat import HeartBeat
from cosmos_framework.callbacks.iter_speed import IterSpeed
from cosmos_framework.callbacks.load_pretrained import LoadPretrained
from cosmos_framework.callbacks.mfu import MFUCallback
from cosmos_framework.callbacks.moe_specialization_callback import MoESpecializationCallback
from cosmos_framework.callbacks.moe_stability_callback import MoEStabilityCallback
from cosmos_framework.callbacks.norm_monitor import NormMonitor
from cosmos_framework.callbacks.ofu import OFUCallback
from cosmos_framework.callbacks.param_count import ParamCount
from cosmos_framework.callbacks.sequence_packing_padding import SequencePackingPadding
from cosmos_framework.callbacks.sigma_loss_analysis import SigmaLossAnalysis
from cosmos_framework.callbacks.skip_nan_step import SkipNaNStep
from cosmos_framework.callbacks.termination_signal_checkpoint import TerminationSignalCheckpoint
from cosmos_framework.callbacks.training_stats import TrainingStatsCallback
from cosmos_framework.callbacks.wandb_log import WandbCallback as WandBCallbackMultiplier
from cosmos_framework.callbacks.wandb_log_eval import WandbCallback as WandBCallbackEval

BASIC_CALLBACKS = dict(
    iter_speed=L(IterSpeed)(  # does not use model or optimizer
        every_n="${trainer.logging_iter}",
        save_s3="${upload_reproducible_setup}",
        save_s3_every_log_n=500,
        hit_thres=50,
    ),
    manual_gc=L(ManualGarbageCollection)(every_n=5),  # does not use model or optimizer
    wandb=L(WandBCallback)(),
    wandb_2x=L(WandBCallbackMultiplier)(
        logging_iter_multipler=2,
        save_logging_iter_multipler=1,
        save_s3="${upload_reproducible_setup}",
    ),
    param_count=L(ParamCount)(  # use model
        save_s3="${upload_reproducible_setup}",
    ),
    wandb_val=L(WandBCallbackEval)(
        save_s3="${upload_reproducible_setup}",
    ),
    moe_stability=L(MoEStabilityCallback)(every_n=250),
    moe_specialization=L(MoESpecializationCallback)(every_n=250),
    expert_heatmap=L(ExpertHeatmap)(),
    load_pretrained=L(LoadPretrained)(),
    compile_tokenizer=L(CompileTokenizer)(enabled=False, compile_after_iterations=3),
    norm_monitor=L(NormMonitor)(
        every_n=5000,
        log_stat_wandb=True,
        save_s3="${upload_reproducible_setup}",
        track_activations=True,
    ),
    sigma_loss_analysis=L(SigmaLossAnalysis)(
        every_n=5000,
        every_n_viz=5000,
        save_s3="${upload_reproducible_setup}",
    ),
    sequence_packing_padding=L(SequencePackingPadding)(every_n="${trainer.logging_iter}"),
    mfu=L(MFUCallback)(every_n="${trainer.logging_iter}", grad_accum_iter="${trainer.grad_accum_iter}"),
    ofu=L(OFUCallback)(every_n="${trainer.logging_iter}"),
)

# LLM-only subset of BASIC_CALLBACKS.
# Drops VFM-specific callbacks:
#   - CompileTokenizer: requires model.tokenizer_vision_gen (VAE)
#   - ExpertHeatmap: requires MoE language_model with mlp_moe_gen
#   - SigmaLossAnalysis: rectified-flow specific
#   - SequencePackingPadding: VFM multi-modal packing specific
#   - NormMonitor: param filter assumes "moe_gen" params → logs nothing for dense LLM
# Drops Necessary but not supported Callbacks:
#   - MFU: @TODO
BASIC_LLM_CALLBACKS = dict(
    iter_speed=L(IterSpeed)(
        every_n="${trainer.logging_iter}",
        save_s3="${upload_reproducible_setup}",
        save_s3_every_log_n=500,
        hit_thres=50,
    ),
    manual_gc=L(ManualGarbageCollection)(every_n=5),
    wandb=L(WandBCallback)(),
    wandb_2x=L(WandBCallbackMultiplier)(
        logging_iter_multipler=2,
        save_logging_iter_multipler=1,
        save_s3="${upload_reproducible_setup}",
    ),
    param_count=L(ParamCount)(
        save_s3="${upload_reproducible_setup}",
    ),
    wandb_val=L(WandBCallbackEval)(
        save_s3="${upload_reproducible_setup}",
    ),
    ofu=L(OFUCallback)(every_n="${trainer.logging_iter}"),
)

# DiT-safe subset for LLM-backed rectified-flow image training.
BASIC_DIT_CALLBACKS = dict(BASIC_LLM_CALLBACKS)

JOB_MONITOR_CALLBACKS = dict(
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
    termination_signal_checkpoint=L(TerminationSignalCheckpoint)(
        min_save_fraction=1 / 3,
    ),
)

OPTIMIZATION_CALLBACKS = dict(
    skip_nan_step=L(SkipNaNStep)(max_consecutive_nan=100),
    grad_clip=L(GradClip)(clip_norm=1.0, track_per_modality=True),  # image/video grad-norm split
    low_precision=L(LowPrecisionCallback)(update_iter=1, config=PLACEHOLDER, trainer=PLACEHOLDER),  # use model
)

OPTIMIZATION_LLM_CALLBACKS = dict(
    skip_nan_step=L(SkipNaNStep)(max_consecutive_nan=100),
    grad_clip=L(GradClip)(clip_norm=1.0, track_per_modality=False),
    low_precision=L(LowPrecisionCallback)(update_iter=1, config=PLACEHOLDER, trainer=PLACEHOLDER),
)

# DiT reuses the same GradClip callback as LLM, without VFM image/video grad-norm split.
OPTIMIZATION_DIT_CALLBACKS = dict(
    skip_nan_step=L(SkipNaNStep)(max_consecutive_nan=100),
    grad_clip=L(GradClip)(clip_norm=1.0, track_per_modality=False),
    low_precision=L(LowPrecisionCallback)(update_iter=1, config=PLACEHOLDER, trainer=PLACEHOLDER),
)

VIZ_ONLINE_SAMPLING_CALLBACKS = dict(
    every_n_sample_reg=L(EveryNDrawSample)(
        every_n=5000,
        save_s3=True,
        do_x0_prediction=False,
    ),
    every_n_sample_ema=L(EveryNDrawSample)(
        every_n=5000,
        is_ema=True,
        save_s3=True,
        do_x0_prediction=False,
    ),
)

DIT_IMAGE_SAMPLING_CALLBACKS = dict(
    dit_image_sample_ema=L(DiTImageSampleCallback)(
        every_n=5000,
        class_ids=[0, 1, 2, 3],
        cfg_scales=[1.0, 1.25, 1.5, 2.0],
        num_steps=50,
        seed=0,
        is_ema=True,
    ),
)


def register_callbacks():
    cs = ConfigStore.instance()
    cs.store(group="callbacks", package="trainer.callbacks", name="basic", node=BASIC_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="job_monitor", node=JOB_MONITOR_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="optimization", node=OPTIMIZATION_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="optimization_llm", node=OPTIMIZATION_LLM_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="optimization_dit", node=OPTIMIZATION_DIT_CALLBACKS)
    # Online sampling generation callback
    cs.store(
        group="callbacks", package="trainer.callbacks", name="viz_online_sampling", node=VIZ_ONLINE_SAMPLING_CALLBACKS
    )
    # Register "generation" as alias for "viz_online_sampling" (expected by base config.py defaults)
    cs.store(group="callbacks", package="trainer.callbacks", name="generation", node=VIZ_ONLINE_SAMPLING_CALLBACKS)
    cs.store(
        group="callbacks", package="trainer.callbacks", name="dit_image_sampling", node=DIT_IMAGE_SAMPLING_CALLBACKS
    )

    TRAINING_STATS_CALLBACKS = dict(
        training_stats=L(TrainingStatsCallback)(
            log_freq=100,
        )
    )
    cs.store(group="callbacks", package="trainer.callbacks", name="training_stats", node=TRAINING_STATS_CALLBACKS)

    # Only for LLM training, removed callbacks that is not working for llm training
    cs.store(group="callbacks", package="trainer.callbacks", name="basic_llm", node=BASIC_LLM_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="basic_dit", node=BASIC_DIT_CALLBACKS)
