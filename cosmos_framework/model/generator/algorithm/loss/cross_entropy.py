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
formula wants on the logits-based path: ``ParallelDims`` pins
``dp_replicate * dp_shard == world_size``, so the data-parallel mesh is the world.
CP is accepted only by the separately normalized fused path described below;
CFGP remains unsupported, and TP/PP are not implemented here.

async_safe_ce instead reduces over the dp_shard sub-group, which under HSDP
normalizes each replicate group separately; that agrees with the global token mean
only when those groups carry equal token counts.

The two logits-based losses remain CP-disabled. The fused weighted loss additionally
accepts a context_parallel_group and globally shifted, locally sliced labels; its
per-sample counts are reduced over CP before applying the sample weighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint

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
    per_sample_objective_numerator: torch.Tensor
    per_sample_objective_denominator: torch.Tensor
    per_sample_token_ce_sum: torch.Tensor
    per_sample_valid_token_count: torch.Tensor


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
    batch_size = labels.shape[0]
    shifted_logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))  # [B*(T-1),V]
    shifted_labels = labels[:, 1:].contiguous().view(-1)  # [B*(T-1)]

    # Per-token loss, then normalize over the global valid-token count.
    # Reference: async_safe_ce:89-109
    per_token_loss = F.cross_entropy(
        shifted_logits.float(),
        shifted_labels,
        ignore_index=ignore_index,
        reduction="none",
    )
    per_token_loss_2d = per_token_loss.view(batch_size, -1)  # [B,T-1]
    valid_2d = (shifted_labels != ignore_index).view(batch_size, -1)  # [B,T-1]
    per_sample_token_ce_sum = (per_token_loss_2d * valid_2d).sum(dim=1)  # [B]
    per_sample_valid_token_count = valid_2d.sum(dim=1)  # [B]
    local_token_ce_sum = per_sample_token_ce_sum.sum()  # []
    local_n_valid_tokens = per_sample_valid_token_count.sum()  # []
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
            per_sample_objective_numerator=(per_sample_token_ce_sum * loss_scaling_factor).detach(),
            per_sample_objective_denominator=per_sample_valid_token_count.detach(),
            per_sample_token_ce_sum=per_sample_token_ce_sum.detach(),
            per_sample_valid_token_count=per_sample_valid_token_count.detach(),
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
            per_sample_objective_numerator=(
                torch.where(has_valid, per_sample_terms, torch.zeros_like(per_sample_terms)) * loss_scaling_factor
            ).detach(),
            per_sample_objective_denominator=torch.where(
                has_valid, normalizer_terms, torch.zeros_like(normalizer_terms)
            ).detach(),
            per_sample_token_ce_sum=loss_sums.detach(),
            per_sample_valid_token_count=valid_counts.detach().to(torch.long),
        )
    return loss


def weighted_ce_denominator_from_labels(
    labels: torch.Tensor,
    *,
    exponent: float,
    cu_seq_lens: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Count the weighted objective mass from full, unsharded CPU labels.

    This is label-only lookahead, before CP shifts/slices the sequence. Packed
    boundaries retain per-conversation weighting; ignored-only segments add zero.
    The caller handles CP replication and the WORLD reduction.
    """
    if labels.device.type != "cpu" or labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError("Accumulation lookahead requires full CPU labels [B,T], T >= 2")
    if not 0.0 <= exponent <= 1.0:
        raise ValueError("Weighted CE exponent must be in [0,1]")
    valid = labels != ignore_index
    valid[:, 0] = False  # Match the global next-token shift, without changing labels.
    if cu_seq_lens is None:
        counts = valid.sum(dim=1).float()
    else:
        if (
            labels.shape[0] != 1
            or cu_seq_lens.device.type != "cpu"
            or cu_seq_lens.dtype not in (torch.int32, torch.int64)
            or cu_seq_lens.ndim != 1
            or cu_seq_lens.numel() < 2
            or cu_seq_lens[0] != 0
            or cu_seq_lens[-1] != labels.shape[1]
            or not bool((cu_seq_lens[1:] > cu_seq_lens[:-1]).all())
        ):
            raise ValueError("Packed lookahead boundaries must cover one complete CPU row")
        prefix = torch.cat((torch.zeros(1, dtype=torch.long), valid[0].cumsum(0)))
        counts = (prefix[cu_seq_lens[1:]] - prefix[cu_seq_lens[:-1]]).float()
    return torch.where(counts > 0, counts.clamp(min=1).pow(1 - exponent), 0.0).sum()


def _fused_linear_ce_sum(
    hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, ignore_index: int
) -> torch.Tensor:
    """Use Liger's scalar reduction; reduction='none' has incompatible backward semantics."""
    from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy

    return liger_fused_linear_cross_entropy(
        hidden, weight, labels, ignore_index=ignore_index, reduction="sum", accum_dtype=torch.float32
    )


class _FP32GradientAnchor(torch.autograd.Function):
    """Carry a matrix-shaped FP32 gradient without copying the forward weight.

    Only shape, dtype and the autograd edge are used by the compute-weight
    aliases below. The expanded scalar's values never enter the loss kernel.
    """

    @staticmethod
    def forward(ctx: Any, weight: torch.Tensor) -> torch.Tensor:
        ctx.weight_dtype = weight.dtype
        return weight.new_zeros((), dtype=torch.float32).expand_as(weight)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> torch.Tensor:
        return gradient.to(ctx.weight_dtype)


class _ComputeWeightWithFP32Gradient(torch.autograd.Function):
    """Alias compute storage while returning each head contribution to one FP32 node.

    All conversations share the original compute weight. A distinct autograd edge
    per conversation promotes its gradient before the shared FP32 fan-in; a
    single cast then returns the sum to the original gathered weight's dtype.
    """

    @staticmethod
    def forward(ctx: Any, accumulation_weight: torch.Tensor, compute_weight: torch.Tensor) -> torch.Tensor:
        del ctx, accumulation_weight
        return compute_weight

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        del ctx
        return gradient.float(), None


def fused_weighted_cross_entropy_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    exponent: float,
    ignore_index: int = IGNORE_INDEX,
    context_parallel_group: dist.ProcessGroup | None = None,
    cu_seq_lens: torch.Tensor | None = None,
    recompute: bool = False,
    accumulate_head_grad_fp32: bool = False,
) -> tuple[torch.Tensor, LossStatistics]:
    """Compute the unified weighted objective without allocating [B,T,vocabulary] logits.

    Called inside the FSDP model forward, while the lm_head weight is gathered.
    Each sample uses a scalar fused CE call. The denominator spans the unified
    trainer's whole DP world, including HSDP replicas. With CP, labels have
    already been shifted globally and sliced with the inputs.
    Sample weights use CP-global valid counts; each shard contributes 1/CP of
    the sample's denominator so FSDP's WORLD average retains the same objective.
    Optional recomputation avoids keeping Liger's dense weight-gradient buffer
    for every conversation alive until backward; the objective is unchanged.
    Optional FP32 accumulation adds the native per-conversation head gradients
    before casting once to the gathered weight dtype. CE compute and hidden
    gradients keep their original precision, and compute-weight storage is shared.
    """
    if hidden_states.ndim != 3 or labels.shape != hidden_states.shape[:2]:
        raise ValueError("Fused CE requires aligned hidden states [B,T,H] and labels [B,T]")
    if hidden_states.shape[0] < 1 or hidden_states.shape[1] < 2:
        raise ValueError("Fused CE needs at least one sample with two tokens")
    if not 0.0 <= exponent <= 1.0:
        raise ValueError("Weighted CE exponent must be in [0,1]")
    if isinstance(lm_head_weight, DTensor):
        raise ValueError("Fused CE must execute inside the FSDP root with its gathered lm_head weight")
    weight = lm_head_weight.to(hidden_states.dtype)
    targets = labels[:, 1:].contiguous() if context_parallel_group is None else labels.contiguous()
    hidden = hidden_states[:, :-1] if context_parallel_group is None else hidden_states
    if cu_seq_lens is None:
        segments = [(hidden[index], targets[index]) for index in range(labels.shape[0])]
    else:
        if labels.shape[0] != 1 or cu_seq_lens.ndim != 1 or cu_seq_lens.numel() < 2:
            raise ValueError("Packed fused CE requires one row and cumulative conversation boundaries")
        boundaries = cu_seq_lens.tolist()
        cp_size = 1 if context_parallel_group is None else dist.get_world_size(context_parallel_group)
        full_length = labels.shape[1] * cp_size
        if (
            boundaries[0] != 0
            or boundaries[-1] != full_length
            or any(left >= right for left, right in zip(boundaries[:-1], boundaries[1:], strict=True))
        ):
            raise ValueError("Packed boundaries must cover the complete physical sequence")
        # Ownership follows the globally shifted target. Conversation-start
        # labels are ignored by the pack collator, so no cross-conversation
        # prediction contributes to either CE or the per-conversation counts.
        target_start = 1
        if context_parallel_group is not None:
            target_start += dist.get_rank(context_parallel_group) * labels.shape[1]
        segments = []
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
            start = min(max(left - target_start, 0), targets.shape[1])
            end = min(max(right - target_start, 0), targets.shape[1])
            segments.append((hidden[0, start:end], targets[0, start:end]))
    counts = torch.stack([(target != ignore_index).sum() for _, target in segments])
    sample_counts = counts.detach().to(torch.float32)
    cp_size = 1
    if context_parallel_group is not None:
        dist.all_reduce(sample_counts, group=context_parallel_group)
        cp_size = dist.get_world_size(context_parallel_group)
    losses = []
    accumulation_weight: torch.Tensor | None = None
    promote_head_grad = (
        accumulate_head_grad_fp32
        and torch.is_grad_enabled()
        and lm_head_weight.requires_grad
        and hidden_states.dtype in (torch.bfloat16, torch.float16)
        and len(segments) > 1
    )
    for index, (sample_hidden, sample_targets) in enumerate(segments):
        if bool(counts[index] > 0):
            sample_weight = weight
            if promote_head_grad:
                if accumulation_weight is None:
                    accumulation_weight = _FP32GradientAnchor.apply(lm_head_weight)
                sample_weight = _ComputeWeightWithFP32Gradient.apply(accumulation_weight, weight.detach())
            args = (sample_hidden.contiguous(), sample_weight, sample_targets.contiguous(), ignore_index)
            if recompute and torch.is_grad_enabled():
                losses.append(checkpoint(_fused_linear_ce_sum, *args, use_reentrant=False))
            else:
                losses.append(_fused_linear_ce_sum(*args))
        else:
            # Empty slices keep both branches in autograd without reducing the large weight matrix.
            losses.append(sample_hidden[:0].sum() + weight[:0].sum())
    loss_sums = torch.stack(losses).float()
    has_valid = sample_counts > 0
    sample_numerators = loss_sums / sample_counts.clamp(min=1).pow(exponent)
    sample_denominators = torch.where(has_valid, sample_counts.clamp(min=1).pow(1 - exponent), 0.0) / cp_size
    numerator = sample_numerators.sum()
    denominator = sample_denominators.sum()
    global_denominator = denominator.detach().clone()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    loss = numerator / global_denominator.clamp(min=1) * world_size
    return loss, LossStatistics(
        objective_numerator=numerator.detach(),
        objective_denominator=denominator.detach(),
        global_objective_denominator=global_denominator.detach(),
        token_ce_sum=loss_sums.sum().detach(),
        valid_token_count=counts.sum().detach(),
        per_sample_objective_numerator=sample_numerators.detach(),
        per_sample_objective_denominator=sample_denominators.detach(),
        per_sample_token_ce_sum=loss_sums.detach(),
        per_sample_valid_token_count=counts.detach().to(torch.long),
    )
