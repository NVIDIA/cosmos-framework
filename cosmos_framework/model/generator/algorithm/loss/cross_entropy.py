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

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from cosmos_framework.utils.generator.input_probe import maybe_dump_loss_reduction
from cosmos_framework.utils.generator.reasoner.constant import IGNORE_INDEX


@dataclass(frozen=True)
class LossStatistics:
    """Detached local numerators and denominators for exact validation aggregation."""

    objective_numerator: torch.Tensor
    objective_denominator: torch.Tensor
    global_objective_denominator: torch.Tensor
    token_ce_sum: torch.Tensor
    valid_token_count: torch.Tensor


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_scaling_factor: float = 1.0,
    ignore_index: int = IGNORE_INDEX,
    cu_seqlens: torch.Tensor | None = None,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, LossStatistics]:
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
        cu_seqlens: accepted for call-site parity with ``weighted_cross_entropy_loss`` and IGNORED.
            Standard CE is packing-invariant: it is a global per-token mean, and the collate already
            boundary-masks the cross-sample next-token pairs, so no segment metadata is needed.

    Returns:
        Scalar loss tensor.
    """
    del cu_seqlens  # packing-invariant; see docstring
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
    local_token_ce_sum = per_token_loss.sum()  # []
    local_n_valid_tokens = (shifted_labels != ignore_index).sum()  # []
    n_valid_tokens = local_n_valid_tokens.detach().clone()  # []
    num_dp_workers = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(n_valid_tokens, op=dist.ReduceOp.SUM)
        num_dp_workers = dist.get_world_size()

    loss = local_token_ce_sum / (n_valid_tokens + 1e-8) * (num_dp_workers * loss_scaling_factor)
    if return_stats:
        return loss, LossStatistics(
            objective_numerator=(local_token_ce_sum * loss_scaling_factor).detach(),
            objective_denominator=local_n_valid_tokens.detach(),
            global_objective_denominator=n_valid_tokens.detach(),
            token_ce_sum=local_token_ce_sum.detach(),
            valid_token_count=local_n_valid_tokens.detach(),
        )
    return loss


def weighted_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    exponent: float,
    loss_scaling_factor: float = 1.0,
    ignore_index: int = IGNORE_INDEX,
    probe_step: int | None = None,
    probe_tag: str | None = None,
    cu_seq_lens: torch.Tensor | None = None,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, LossStatistics]:
    """Segment-aware weighted next-token CE for padded and true-packed layouts.

    Padded rows define logical samples directly. For one true-packed row, ``cu_seq_lens``
    recovers the logical sample owning each shifted target. The objective is therefore
    invariant to the physical layout for every exponent. The denominator is normalized
    over the whole world, matching :func:`cross_entropy_loss` and the VLM parallelism
    invariant documented at module level.
    """
    batch_size, sequence_length, vocab_size = logits.shape
    shifted_logits = logits[:, :-1].contiguous().view(-1, vocab_size)  # [N,V]
    shifted_labels = labels[:, 1:].contiguous().view(-1)  # [N]
    per_token_loss = F.cross_entropy(
        shifted_logits.float(),
        shifted_labels,
        ignore_index=ignore_index,
        reduction="none",
    )  # [N]
    valid = shifted_labels != ignore_index  # [N]

    if cu_seq_lens is None:
        num_samples = batch_size
        sample_ids = torch.arange(batch_size, device=labels.device).repeat_interleave(sequence_length - 1)  # [N]
    else:
        if batch_size != 1:
            raise ValueError(f"cu_seq_lens requires one packed row, got batch_size={batch_size}")
        if cu_seq_lens.ndim != 1 or cu_seq_lens.numel() < 2:
            raise ValueError(f"cu_seq_lens must have shape [num_segments+1], got {tuple(cu_seq_lens.shape)}")
        cumulative_lengths = cu_seq_lens.to(device=labels.device, dtype=torch.long)  # [K+1]
        num_samples = cumulative_lengths.numel() - 1
        target_positions = torch.arange(1, sequence_length, device=labels.device)  # [N]
        sample_ids = (torch.searchsorted(cumulative_lengths, target_positions, right=True) - 1).clamp_(
            0, num_samples - 1
        )  # [N]

    valid_float = valid.to(torch.float32)  # [N]
    valid_counts = torch.zeros(num_samples, dtype=torch.float32, device=labels.device).scatter_add(
        0, sample_ids, valid_float
    )  # [K]
    loss_sums = torch.zeros(num_samples, dtype=per_token_loss.dtype, device=labels.device).scatter_add(
        0, sample_ids, per_token_loss * valid_float.to(per_token_loss.dtype)
    )  # [K]
    has_valid = valid_counts > 0  # [K]
    safe_counts = valid_counts.clamp(min=1)  # [K]
    per_sample_terms = loss_sums / safe_counts.to(loss_sums.dtype).pow(exponent)  # [K]
    local_loss_sum = torch.where(has_valid, per_sample_terms, torch.zeros_like(per_sample_terms)).sum()  # []
    normalizer_terms = safe_counts.pow(1 - exponent)  # [K]
    local_normalizer = torch.where(has_valid, normalizer_terms, torch.zeros_like(normalizer_terms)).sum()  # []
    local_normalizer_before = local_normalizer.detach().clone()  # []

    num_dp_workers = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local_normalizer, op=dist.ReduceOp.SUM)
        num_dp_workers = dist.get_world_size()

    loss = local_loss_sum / local_normalizer.clamp(min=1) * (num_dp_workers * loss_scaling_factor)  # []
    maybe_dump_loss_reduction(
        step=probe_step,
        tag=probe_tag,
        valid_counts=valid_counts,
        local_loss_sum=local_loss_sum,
        denominator_before=local_normalizer_before,
        denominator_after=local_normalizer,
        final_loss=loss,
        exponent=exponent,
        loss_scaling_factor=loss_scaling_factor,
        world_size=num_dp_workers,
    )
    if return_stats:
        return loss, LossStatistics(
            objective_numerator=(local_loss_sum * loss_scaling_factor).detach(),
            objective_denominator=local_normalizer_before,
            global_objective_denominator=local_normalizer.detach(),
            token_ce_sum=per_token_loss.sum().detach(),
            valid_token_count=valid.sum().detach(),
        )
    return loss
