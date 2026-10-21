# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GSPO variant of the Qwen2.5-3B GSM8K recipe.

GSPO (Group Sequence Policy Optimization, arXiv 2507.18071) keeps the GRPO topology, trainer, and
clipped objective and changes the granularity of the importance ratio: one ratio per response,
``exp(mean_t(log pi_theta - log pi_old))``, shared by every token of that response, so the epsilon
clip admits or drops whole responses. The shared GRPO entry consumes it as resolved config.

Every value ``apply_gspo_knobs`` sets is required by the sequence-level ratio or by the paper:

- ``importance_sampling_level="sequence"`` selects the ratio granularity on the clipped surrogate.
- ``lower_bound_ratio=0``: GSPO defines no dual clip; the config rejects any other value.
- ``epsilon_low/high=3e-4/4e-4`` (paper sec. 5.1): a response-mean log-ratio is about two orders of
  magnitude smaller than a token's, so the token-level 0.2 would never bind (the config warns).
- ``loss_type="seq-mean-token-mean"``: the paper's objective averages over responses with equal
  weight; ``token-mean`` would weight responses by length.
- ``mini_batch_per_replica`` so that ``n_mini > 1``: the ratio is identically 1 unless the batch
  drives several optimizer steps against a frozen pi_old (the paper uses four mini-batches). The base
  GSM8K recipe runs ``n_mini == 1``, where GSPO is token-level GRPO in value and gradient.
- ``certified_deterministic_forward=True``: the pi_old ahead pass at ``n_mini > 1`` requires it;
  Qwen2.5-3B reports ``attention_dropout=0.0`` and no other stochastic layer.

The paper (sec. 5.2) reports a response clip fraction about two orders of magnitude above GRPO's
token clip fraction, so a high ``train/seq_clip_frac`` is the expected reading, not a defect.

Run from the cosmos-framework repository root:

.. code-block:: bash

    PYTHONPATH=. python -m cosmos_framework.posttrain.entrypoints.grpo \
        --config cosmos_framework/posttrain/configs/gspo_qwen3b_gsm8k.py
"""

from __future__ import annotations

from cosmos_framework.posttrain.config.config_grpo import PosttrainGRPOConfig
from cosmos_framework.posttrain.config.rl import GRPOConfig
from cosmos_framework.posttrain.configs.grpo_qwen3b_gsm8k import build_config as _build_grpo_config

GSPO_EPSILON_LOW = 3e-4
GSPO_EPSILON_HIGH = 4e-4


def apply_gspo_knobs(rl: GRPOConfig, *, mini_batch_per_replica: int) -> None:
    """Turn a clipped GRPO ``rl`` section into the paper's GSPO objective.

    One place for the knob set, so every GSPO recipe selects the same objective and a reader can see
    at a glance what differs from GRPO. ``mini_batch_per_replica`` is a required argument because
    the off-policy cadence is recipe-specific, while everything else follows from the paper.
    """
    rl.grpo_surrogate = "clipped"
    rl.importance_sampling_level = "sequence"
    rl.lower_bound_ratio = 0.0
    rl.epsilon_low = GSPO_EPSILON_LOW
    rl.epsilon_high = GSPO_EPSILON_HIGH
    rl.loss_type = "seq-mean-token-mean"
    rl.mini_batch_per_replica = mini_batch_per_replica
    rl.certified_deterministic_forward = True


def build_config() -> PosttrainGRPOConfig:
    """Build the GSPO GSM8K variant with distinct run provenance."""
    cfg = _build_grpo_config()
    cfg.algorithm = "gspo"
    cfg.job.name = "qwen3b_bf16_gspo_seed42"
    # Four optimizer steps per collected batch of 128 (paper sec. 5.1 cadence). The base recipe sets
    # no micro batch, so each mini-batch is one forward of 16 sequences per rank against the base
    # recipe's 64.
    apply_gspo_knobs(cfg.rl, mini_batch_per_replica=32)
    return cfg


def make_config() -> PosttrainGRPOConfig:
    """Return the config consumed by the shared GRPO entrypoint."""
    return build_config()
