# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Minimal OPD demo: Qwen2.5-1.5B student, Qwen2.5-3B teacher, and Countdown.

This recipe exercises the basic same-token, top-k, direct-distillation path on one GPU. It is sized
as a short functional demo rather than a production-quality training schedule.

Run from the cosmos-framework repository root:

.. code-block:: bash

    PYTHONPATH=. torchrun --standalone --nproc-per-node=1 \
        -m cosmos_framework.posttrain.entrypoints.opd \
        --config cosmos_framework/posttrain/configs/opd_qwen1p5b_countdown.py
"""

from __future__ import annotations

from cosmos_framework.posttrain.config.config_opd import PosttrainOPDConfig
from cosmos_framework.posttrain.config.config_opd import make_config as _make_root_config
from cosmos_framework.posttrain.config.rl import DatasetConfig, OptimizerConfig, TeacherConfig

_STUDENT = "Qwen/Qwen2.5-1.5B-Instruct"
_TEACHER = "Qwen/Qwen2.5-3B-Instruct"
_PROMPT_COLUMN = "messages"
_REFERENCE_COLUMN = "reference"


def countdown_items(row: dict) -> dict:
    """Remove the gold assistant turn and expose Countdown's structured reference."""
    messages = row[_PROMPT_COLUMN]
    if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages):
        raise ValueError("countdown messages must be a list of chat-message dictionaries")
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    return {
        **row,
        _PROMPT_COLUMN: messages,
        _REFERENCE_COLUMN: {"nums": list(row["nums"]), "target": int(row["target"])},
    }


def build_config() -> PosttrainOPDConfig:
    """Build the one-GPU direct OPD demo config."""
    # This recipe fully populates the root, so it does not need the experiment catalogs and must
    # not import another project's optional dependencies to load (see config.py::build_root).
    cfg = _make_root_config(register_experiment_configs=False)
    cfg.job.group = "posttrain_opd_demo"
    # Stable identity is required for resume=auto to find the previous run attempt after requeue.
    cfg.job.name = "qwen1p5b_countdown"

    cfg.model_hf.name_or_path = _STUDENT
    cfg.model_hf.dtype = "bfloat16"
    cfg.model_hf.attn_impl = "eager"
    cfg.model_hf.grad_ckpt = True
    cfg.model_hf.parallelism.dp_replicate = 1
    cfg.model_hf.parallelism.dp_shard = -1

    cfg.rollout.name_or_path = _STUDENT
    cfg.rollout.tensor_parallel_size = 1
    cfg.rollout.gpu_memory_utilization = 0.2
    cfg.rollout.max_model_len = 2048
    cfg.rollout.max_tokens = 256
    cfg.rollout.temperature = 1.0
    cfg.rollout.seed = 42

    cfg.rl.teacher = TeacherConfig(
        name_or_path=_TEACHER,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.35,
        max_model_len=2048,
        dtype="bfloat16",
        input_mode="shared_vocab_multimodal",
        signal_mode="topk",
        top_k=16,
        scoring_temperature=1.0,
    )
    cfg.rl.distillation_mode = "direct"
    cfg.rl.jsd_beta = 0.5
    cfg.rl.max_steps = 10
    cfg.rl.save_every = 5
    cfg.rl.optimizer = OptimizerConfig(
        name="fusedadam",
        lr=1e-5,
        betas=[0.9, 0.95],
        eps=1e-8,
        train_keys=[],
    )
    cfg.rl.group_size = 2
    cfg.rl.batch_per_replica = 2
    cfg.rl.loss_type = "token-mean"
    cfg.rl.temperature = 1.0

    cfg.rl.reward = "countdown"
    cfg.rl.data = DatasetConfig(
        name="HuggingFaceTB/Countdown-Task-GOLD",
        subset="verified_Qwen2.5-7B-Instruct",
        split="train",
        prompt_column=_PROMPT_COLUMN,
        response_column=_REFERENCE_COLUMN,
        shuffle=False,
        items_fn="cosmos_framework.posttrain.configs.opd_qwen1p5b_countdown.countdown_items",
    )
    return cfg


def make_config() -> PosttrainOPDConfig:
    """Return the config consumed by ``entrypoints.opd --config``."""
    return build_config()
