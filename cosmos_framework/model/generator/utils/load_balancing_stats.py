# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import NamedTuple

import attrs
import torch


@attrs.define(slots=False)
class LBLConfig:
    """Load-balancing loss configuration shared by training and router-statistics collection."""

    # For load balancing loss computation.
    # - "local": Use the fraction of tokens routed to each expert only for the local rank.
    # - "global": Use the fraction of tokens routed to each expert across all ranks.
    # - "sample": Compute the loss per original sample in a packed sequence, then average the sample losses.
    method: str = "local"

    # Coefficients for the load balancing loss.
    # - "und": Coefficient for the load balancing loss for the "und" pathway.
    # - "gen": Coefficient for the load balancing loss for the "gen" pathway.
    coeff_und: float | None = None
    coeff_gen: float | None = None


class LBLMetadata(NamedTuple):
    """Sufficient router statistics for token- or sample-level load balancing."""

    num_tokens_per_expert: torch.Tensor
    num_tokens: torch.Tensor
    mean_router_prob_per_expert: torch.Tensor
    # Number of experts selected per token. Shape is [1] per layer, or
    # [num_layers, 1] after stacking across layers.
    top_k: torch.Tensor
    sample_num_tokens_per_expert: torch.Tensor | None = None
    sample_num_tokens: torch.Tensor | None = None
    sample_router_prob_sum_per_expert: torch.Tensor | None = None


def compute_sample_lbl_stats(
    routing_probabilities: torch.Tensor,  # [N,E]
    expert_indices: torch.Tensor,  # [N,K]
    sample_ids: torch.Tensor,  # [N]
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate router statistics independently for each sample in a packed sequence."""
    assert sample_ids.shape == (routing_probabilities.shape[0],)
    num_experts = routing_probabilities.shape[-1]
    num_buckets = num_samples + 1

    flat_expert_indices = sample_ids.unsqueeze(1) * num_experts + expert_indices  # [N,K]
    flat_counts = torch.zeros(
        num_buckets * num_experts,
        dtype=torch.int64,
        device=expert_indices.device,
    )  # [(B+1)*E]
    flat_counts = flat_counts.scatter_add(
        0,
        flat_expert_indices.reshape(-1),
        torch.ones_like(flat_expert_indices, dtype=torch.int64).reshape(-1),
    )  # [(B+1)*E]
    sample_num_tokens_per_expert = flat_counts.reshape(num_buckets, num_experts)[:num_samples]  # [B,E]

    router_prob_sum_per_bucket = routing_probabilities.new_zeros(
        num_buckets,
        num_experts,
    )  # [B+1,E]
    router_prob_sum_per_bucket = router_prob_sum_per_bucket.index_add(
        0,
        sample_ids,
        routing_probabilities,
    )  # [B+1,E]
    sample_router_prob_sum_per_expert = router_prob_sum_per_bucket[:num_samples]  # [B,E]

    num_tokens_per_bucket = torch.zeros(num_buckets, dtype=torch.int64, device=sample_ids.device)  # [B+1]
    num_tokens_per_bucket = num_tokens_per_bucket.scatter_add(
        0,
        sample_ids,
        torch.ones_like(sample_ids, dtype=torch.int64),
    )  # [B+1]
    sample_num_tokens = num_tokens_per_bucket[:num_samples].unsqueeze(-1)  # [B,1]
    return sample_num_tokens_per_expert, sample_num_tokens, sample_router_prob_sum_per_expert
