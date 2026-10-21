# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Train-vs-rollout logprob gap diagnostics.

The rollout engine records the logprob of each sampled token as its own forward computed it; the
training forward recomputes the same quantity with different kernels, parallelism, and (for MoE)
possibly different expert routing. The per-token gap |log π_train − log π_rollout| is the standard
train/infer-consistency instrument: the published MoE-GRPO collapse signature is a RISING gap from
routing flips (GSPO §5.3; Rollout Routing Replay §3), and a large gap is what makes the rollout
logprobs unusable as π_old. The engine's values are raw logprobs (temperature 1, before any
sampling warp), so the training side is compared at temperature 1 regardless of the policy-loss
temperature; otherwise the metric would carry a recipe-dependent floor of the same order as the
threshold it watches.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Gap above which a token counts as inconsistent between the two forwards, in nats. Fixed rather
# than configurable: the public metric name `logprob_gap_gt0p5_frac` spells it out.
LOGPROB_GAP_THRESHOLD_NATS = 0.5
# Positions per log-softmax chunk; bounds the fp32 [chunk, V] temporary to a few hundred MB at a
# 150k vocabulary instead of a second full [B, L, V] copy.
_LOGPROB_CHUNK_POSITIONS = 512


def selected_token_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    masks: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:  # logits: [B,L,V], input_ids/masks: [B,L], returns [N]
    """Temperature-1 logprobs of the next-token targets at the loss-masked positions.

    Same next-token alignment and row-major order as the loss path (``masks[:, :-1]`` selects the
    positions whose prediction is a completion token, ``input_ids[:, 1:]`` supplies the targets), so
    the result pairs elementwise with the objective's ``current`` tensor. Detached and chunked over
    selected positions only.
    """
    with torch.no_grad():
        rows, positions = masks[:, :-1].nonzero(as_tuple=True)  # [N], [N]
        targets = input_ids[rows, positions + 1]  # [N]
        chunks: list[torch.Tensor] = []
        for start in range(0, rows.numel(), _LOGPROB_CHUNK_POSITIONS):
            stop = start + _LOGPROB_CHUNK_POSITIONS
            chunk = logits[rows[start:stop], positions[start:stop]].to(dtype)  # [n,V]
            chosen = chunk.gather(-1, targets[start:stop].unsqueeze(-1)).squeeze(-1)  # [n]
            chunks.append(chosen - torch.logsumexp(chunk, dim=-1))  # [n]
        if not chunks:
            return torch.empty(0, dtype=dtype, device=logits.device)  # [0]
        return torch.cat(chunks)  # [N]


@dataclass(frozen=True)
class LogprobGapStats:
    """Rank-local gap statistics over the tokens that carried a finite rollout logprob.

    ``token_count`` is the aggregation weight: the mean and the threshold fraction compose exactly
    under token weighting, the p99 only approximately (a token-weighted mean of per-rank p99s).
    """

    mean: float
    p99: float
    gt_threshold_frac: float
    token_count: int


NO_LOGPROB_GAP = LogprobGapStats(mean=0.0, p99=0.0, gt_threshold_frac=0.0, token_count=0)


def logprob_gap_stats(
    policy_logprobs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
) -> LogprobGapStats:  # policy_logprobs/rollout_logprobs: [N]
    """Summarize |policy − rollout| over the finite rollout entries; zeros with count 0 when none."""
    if policy_logprobs.shape != rollout_logprobs.shape:
        raise ValueError(
            f"logprob gap needs one rollout logprob per policy logprob, got {tuple(rollout_logprobs.shape)} "
            f"vs {tuple(policy_logprobs.shape)}"
        )
    with torch.no_grad():
        rollout = rollout_logprobs.to(torch.float32)  # [N]
        finite = torch.isfinite(rollout)  # [N]
        gap = (policy_logprobs.to(torch.float32) - rollout).abs()[finite]  # [M]
        if gap.numel() == 0:
            return NO_LOGPROB_GAP
        return LogprobGapStats(
            mean=float(gap.mean()),
            p99=float(torch.quantile(gap, 0.99)),
            gt_threshold_frac=float((gap > LOGPROB_GAP_THRESHOLD_NATS).float().mean()),
            token_count=int(gap.numel()),
        )
