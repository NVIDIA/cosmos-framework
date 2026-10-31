# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
r"""Qwen3-VL-8B-Instruct full-parameter SFT on RoboVQA, on one GPU, through the i4-native policy builder.

A worked example with a measured before and after: one epoch over the RoboVQA training clips written by the
``prepare_robovqa.py`` script of the ``cosmos3-qwen3-vl-robovqa-sft`` skill, scored by the skill's ``eval_robovqa.py`` on
the 1,000 held-out clips before and after training. Commands, expected output and timings:
``.agents/skills/cosmos3-qwen3-vl-robovqa-sft/SKILL.md``.

Measured on one NVIDIA GB300 at seed 0, two independent runs (training is not bit-reproducible):

| | exact match | token-F1 | yes/no accuracy |
|---|---|---|---|
| Qwen3-VL-8B-Instruct, zero-shot | 0.548 | 0.676 | 0.918 |
| after this recipe (run 1 / run 2) | 0.632 / 0.620 | 0.809 / 0.798 | 0.953 / 0.946 |

Training stack, each field with its reason:
- ``model_backend='i4'`` with ``build_qwen3_vl_native_policy``: the seat the SFT guide gives Qwen3-VL
  (``docs/posttrain/guides/sft.md`` section 1), with the precision stack of ``sft_qwen3_vl_moe_native.py``: bfloat16
  storage and compute, fp32 master weights and moments inside ``OptimizerConfig(name='fusedadam')`` (apex), SDPA,
  full activation checkpointing.
- ``micro_batch_size=1``, ``grad_accum=64``, ``max_steps=125``: one epoch of the 7,998 training clips at 64 clips per
  update.
- ``lr=2e-5``, betas (0.9, 0.999), no weight decay, ``grad_clip_norm=1.0``, no warm-up, cosine to ~0 at
  ``max_steps``; every parameter trains (``train_keys`` empty).
- ``max_length=10240``: the longest training row renders 4,113 tokens with these frames, above the 4,096 default;
  the packer refuses an over-long row instead of truncating it.
- ``checkpoint_format='consolidated'``, ``checkpoint_components=['model']``, ``save_every=max_steps``: one model-only
  save after the last update, the input of the HF materializer.
- ``videos_kwargs.do_sample_frames=False``: every training row points at a lossless file that holds exactly the frames
  the evaluator decodes (the skill's ``prepare_robovqa.py frames``, checked by its ``verify``), so the processor takes
  every frame at the file's frame rate.
- Validation is off: the held-out measure is the evaluator's, run before and after training.

Set ``SFT_MODEL_PATH`` to a complete local snapshot of ``Qwen/Qwen3-VL-8B-Instruct`` (tested at revision
``0c351dd01ed87e9c1b53cbc748cba10e6187ff3b``) and ``SFT_DATA_ROOT`` to the ``--out`` directory of
``prepare_robovqa.py``. Torchrun chooses the DP world; one GPU with about 150 GB is enough (torch peak 143,566 MiB
reserved on a GB300). Override after ``--``, for example ``-- rl.seed=1 job.name=seed1``; a shorter run that should
still save sets both ``rl.max_steps`` and ``rl.save_every``.
"""

from __future__ import annotations

import os

from cosmos_framework.posttrain.config.config_sft import PosttrainSFTConfig
from cosmos_framework.posttrain.config.config_sft import make_config as make_root_config
from cosmos_framework.posttrain.config.model import (
    NativePolicyBuildSpec,
    NativePolicyParallelismConfig,
    Qwen3VLModelConfig,
)
from cosmos_framework.posttrain.config.rl import OptimizerConfig
from cosmos_framework.posttrain.config.sft import SFTDatasetConfig, SFTLRScheduleConfig, SFTValidationConfig

NATIVE_POLICY_BUILDER = "cosmos_framework.posttrain.model.qwen3_vl.build_qwen3_vl_native_policy"
# One epoch of the 7,998 training clips of prepare_robovqa.py's seed-0 split at 64 clips per update.
EPOCH_STEPS = 125


def build_config() -> PosttrainSFTConfig:
    config = make_root_config()
    config.job.project = "posttrain_sft"
    config.job.group = "qwen3_vl_8b_robovqa"
    config.job.name = "seed0"
    config.job.wandb_mode = "disabled"
    config.model_backend = "i4"
    config.model = NativePolicyBuildSpec(
        builder=NATIVE_POLICY_BUILDER,
        model_id=os.environ.get("SFT_MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct"),
        model_config=Qwen3VLModelConfig(
            # NativePolicyBuildSpec.model_id is the single artifact identity; the builder injects it.
            name_or_path="",
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
    config.rl.seed = 0
    config.rl.master_dtype = "bfloat16"
    config.rl.param_dtype = "bfloat16"
    config.rl.reduce_dtype = "float32"
    config.rl.mixed_precision_mode = "fsdp"
    config.rl.max_steps = EPOCH_STEPS
    config.rl.micro_batch_size = 1
    config.rl.grad_accum = 64
    config.rl.max_length = 10240
    config.rl.checkpoint_format = "consolidated"
    config.rl.checkpoint_components = ["model"]
    config.rl.save_every = EPOCH_STEPS
    config.rl.optimizer = OptimizerConfig(
        name="fusedadam",
        lr=2e-5,
        betas=[0.9, 0.999],
        eps=1e-8,
        weight_decay=0.0,
        grad_clip_norm=1.0,
        warmup_steps=0,
    )
    config.rl.lr_schedule = SFTLRScheduleConfig(kind="cosine", f_start=0.01, f_min=1e-9, cycle_steps=0)
    data_root = os.environ.get("SFT_DATA_ROOT", "artifacts/qwen3-vl-robovqa/data")
    config.rl.data = SFTDatasetConfig(path=os.path.join(data_root, "train.jsonl"))
    config.rl.processor_kwargs = {"videos_kwargs": {"do_sample_frames": False}}
    config.rl.validation = SFTValidationConfig(enabled=False)
    return config


def make_config() -> PosttrainSFTConfig:
    """The cosmos_framework.utils.config.load_config entry point."""
    return build_config()
