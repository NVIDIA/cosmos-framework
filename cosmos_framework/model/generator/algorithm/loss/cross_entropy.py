# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CE loss for VLM training.

Ported from cosmos_rl.policy.trainer.llm_trainer.sft_trainer.async_safe_ce
(packages/cosmos-rl/cosmos_rl/policy/trainer/llm_trainer/sft_trainer.py).

The reduction formula must match async_safe_ce exactly to preserve loss parity
between cosmos-rl and this module — with two deliberate departures described
below: the group the normalizer spans, and the dropped context-parallel path.

The reduction:

    Sum CE loss / (global_n_valid_tokens + 1e-8) × (num_dp_workers × scaling).

The ×num_dp_workers compensates for FSDP's gradient averaging across DP ranks,
ensuring the effective gradient equals the gradient of the global mean loss even
with unbalanced per-rank token counts.
Reference: async_safe_ce:97-109 in the source file above.

Both losses reduce their normalizer (valid-token count for plain CE, the
exponent-weighted sample weight for weighted CE) over the WHOLE WORLD, taking no
group argument. Every rank must hold a different sample for that to be the count the
formula wants, which holds for VLM: ``ParallelDims`` pins
``dp_replicate * dp_shard == world_size``, so the data-parallel mesh is the world, and
the axes that would put the same sample on several ranks are rejected — cp/cfgp by
``VLMModel``, tp/pp by not existing. Revisit this if any of that changes.

async_safe_ce instead reduces over the dp_shard sub-group, which under HSDP
normalizes each replicate group separately; that agrees with the global token mean
only when those groups carry equal token counts.

Neither loss takes a cp_group: reducing across context-parallel segments is out of
scope here.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from cosmos_framework.utils.generator.input_probe import maybe_dump_loss_reduction
from cosmos_framework.utils.generator.reasoner.constant import IGNORE_INDEX


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_scaling_factor: float = 1.0,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Next-token-prediction CE loss, normalized over the world's valid tokens.

    Matches the behavior of cosmos_rl.policy.trainer.llm_trainer.sft_trainer.async_safe_ce
    with the TORCH_CROSS_ENTROPY backend (F.cross_entropy with float32 cast), minus its
    context-parallel path (see module docstring).

    Args:
        logits: (B, T, V) float tensor — raw model output before softmax.
        labels: (B, T) long tensor — ground-truth token ids.
                Positions equal to ignore_index are excluded from the loss.
        loss_scaling_factor: scalar multiplied into the returned loss.
        ignore_index: label value to exclude (defaults to ``IGNORE_INDEX``, -100).

    Returns:
        Scalar loss tensor.
    """
    # Shift for next-token prediction: predict token[t+1] using hidden state[t].
    # logits[:, :-1] aligns with labels[:, 1:].
    # Reference: async_safe_ce:63-73 (output[:, :-1], target[:, 1:])
    shifted_logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))
    shifted_labels = labels[:, 1:].contiguous().view(-1)

    # Per-token loss, then normalize over the global valid-token count.
    # Reference: async_safe_ce:89-109
    per_token_loss = F.cross_entropy(
        shifted_logits.float(),
        shifted_labels,
        ignore_index=ignore_index,
        reduction="none",
    )
    n_valid_tokens = (shifted_labels != ignore_index).sum()
    num_dp_workers = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(n_valid_tokens, op=dist.ReduceOp.SUM)
        num_dp_workers = dist.get_world_size()

    loss = per_token_loss.sum() / (n_valid_tokens + 1e-8) * (num_dp_workers * loss_scaling_factor)
    return loss


def weighted_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    exponent: float,
    loss_scaling_factor: float = 1.0,
    ignore_index: int = IGNORE_INDEX,
    probe_step: int | None = None,
    probe_tag: str | None = None,
) -> torch.Tensor:
    """Next-token-prediction CE loss interpolated between per-token and per-sample reductions.

    Matches ``cosmos_rl.policy.trainer.llm_trainer.sft_trainer.async_safe_weighted_ce``
    for the non-packed, non-CP VLM path.

    Args:
        logits: [B,T,V] float tensor, raw model output before softmax.
        labels: [B,T] long tensor, ground-truth token ids.
        exponent: 0 gives per-token loss, 1 gives per-sample loss, values in
            between interpolate by valid-token-count weight.
        loss_scaling_factor: scalar multiplied into the returned loss.
        ignore_index: label value to exclude.

    Returns:
        Scalar loss tensor.
    """
    batch_size = labels.shape[0]
    shifted_logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))  # [B*(T-1),V]
    shifted_labels = labels[:, 1:].contiguous().view(-1)  # [B*(T-1)]

    per_token_loss = F.cross_entropy(
        shifted_logits.float(),  # [B*(T-1),V]
        shifted_labels,  # [B*(T-1)]
        ignore_index=ignore_index,
        reduction="none",
    ).view(batch_size, -1)  # [B,T-1]
    valid_mask = (shifted_labels.view(batch_size, -1) != ignore_index).float()  # [B,T-1]
    valid_counts = valid_mask.sum(dim=1)  # [B]
    has_valid = (valid_counts > 0).float()  # [B]

    sample_losses = (per_token_loss * valid_mask).sum(dim=1) / valid_counts.clamp(min=1).pow(exponent)  # [B]
    local_loss_sum = (sample_losses * has_valid).sum()  # []
    local_exp_weight_sum = (valid_counts.pow(1 - exponent) * has_valid).sum()  # []
    local_exp_weight_sum_before = local_exp_weight_sum.detach().clone()  # []

    num_dp_workers = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local_exp_weight_sum, op=dist.ReduceOp.SUM)  # local_exp_weight_sum: []
        num_dp_workers = dist.get_world_size()

    loss = local_loss_sum / local_exp_weight_sum.clamp(min=1) * (num_dp_workers * loss_scaling_factor)  # []

    maybe_dump_loss_reduction(
        step=probe_step,
        tag=probe_tag,
        valid_counts=valid_counts,
        local_loss_sum=local_loss_sum,
        denominator_before=local_exp_weight_sum_before,
        denominator_after=local_exp_weight_sum,
        final_loss=loss,
        exponent=exponent,
        loss_scaling_factor=loss_scaling_factor,
        world_size=num_dp_workers,
    )
    return loss
