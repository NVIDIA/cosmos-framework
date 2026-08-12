# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch
from torch.distributed.tensor import DTensor, Partial
from torch.distributed.tensor.device_mesh import DeviceMesh

from cosmos_framework.model.generator.utils.load_balancing_stats import LBLMetadata


def _sum_over_mesh(local_tensor: torch.Tensor, device_mesh: DeviceMesh) -> torch.Tensor:
    """Sum a local tensor over a device mesh while preserving autograd."""
    return DTensor.from_local(
        local_tensor,
        device_mesh=device_mesh,
        placements=[Partial()] * device_mesh.ndim,
    ).full_tensor()


def _scale_rank_sample_mean(
    local_mean_per_layer: torch.Tensor,  # [num_layers]
    local_sample_count_per_layer: torch.Tensor,  # [num_layers]
    device_mesh: DeviceMesh | None,
) -> torch.Tensor:
    """Scale rank-local sample means so FSDP averaging produces a global sample mean."""
    if device_mesh is None or device_mesh.size() == 1:
        return local_mean_per_layer

    global_sample_count_per_layer = _sum_over_mesh(
        local_sample_count_per_layer,
        device_mesh,
    )  # [num_layers]
    # FSDP averages gradients over the mesh, so pre-multiply by the mesh size.
    # This is analogous to OmniMoTModel._sample_level_loss_scale.
    pre_fsdp_sample_scale = (
        device_mesh.size() * local_sample_count_per_layer / global_sample_count_per_layer.clamp_min(1)
    )  # [num_layers]
    return local_mean_per_layer * pre_fsdp_sample_scale  # [num_layers]


def _aggregate_sample_lbl_stats(
    lbl_metadata: LBLMetadata,
    context_parallel_mesh: DeviceMesh | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sample LBL statistics aggregated over sequence-sharded CP.

    Returns:
        Expert assignment counts ``[num_layers,num_samples,num_experts]``,
        token counts ``[num_layers,num_samples,1]``, and router probability
        sums ``[num_layers,num_samples,num_experts]``.
    """
    local_sample_num_tokens_per_expert = (
        lbl_metadata.sample_num_tokens_per_expert
    )  # [num_layers,num_samples,num_experts] or None
    local_sample_num_tokens = lbl_metadata.sample_num_tokens  # [num_layers,num_samples,1] or None
    local_sample_router_prob_sum_per_expert = (
        lbl_metadata.sample_router_prob_sum_per_expert
    )  # [num_layers,num_samples,num_experts] or None
    if (
        local_sample_num_tokens_per_expert is None
        or local_sample_num_tokens is None
        or local_sample_router_prob_sum_per_expert is None
    ):
        raise ValueError("Sample-level load balancing requires per-sample router metadata.")

    if context_parallel_mesh is None or context_parallel_mesh.size() == 1:
        return (
            local_sample_num_tokens_per_expert,
            local_sample_num_tokens,
            local_sample_router_prob_sum_per_expert,
        )

    # Keep CP collectives outside torch-compiled decoder layers. Compile can
    # reorder collectives and cause deadlocks.
    local_counts_and_tokens = torch.cat(
        [local_sample_num_tokens_per_expert, local_sample_num_tokens],
        dim=-1,
    )  # [num_layers,num_samples,num_experts+1]
    cp_counts_and_tokens = _sum_over_mesh(
        local_counts_and_tokens,
        context_parallel_mesh,
    )  # [num_layers,num_samples,num_experts+1]
    cp_sample_num_tokens_per_expert = cp_counts_and_tokens[..., :-1]  # [num_layers,num_samples,num_experts]
    cp_sample_num_tokens = cp_counts_and_tokens[..., -1:]  # [num_layers,num_samples,1]
    cp_sample_router_prob_sum_per_expert = _sum_over_mesh(
        local_sample_router_prob_sum_per_expert,
        context_parallel_mesh,
    )  # [num_layers,num_samples,num_experts]
    return cp_sample_num_tokens_per_expert, cp_sample_num_tokens, cp_sample_router_prob_sum_per_expert


def compute_load_balancing_loss(
    lbl_metadata: LBLMetadata | None,
    coeff: float | None,
    method: str,
    device_mesh: DeviceMesh | None,
    context_parallel_mesh: DeviceMesh | None = None,
) -> torch.Tensor | None:
    """
    Compute the load balancing loss. We compute the load balancing loss
    for each layer, and then average the loss across all layers.

    For computing the load balancing loss for each layer, we can either
    use the fraction of tokens routed to each expert for this rank ("local" method), or
    use the fraction of tokens routed to each expert across all ranks ("global" method), or
    average independently computed losses for each sample in a packed sequence
    ("sample" method).

    Args:
        lbl_metadata: The load balancing metadata. Contains the following tensors
            - num_tokens_per_expert: [num_layers, num_experts] - The number of
              tokens routed to each expert for this rank for each layer.
            - num_tokens: [num_layers, 1] - The total number of tokens in the
              batch for each layer.
            - mean_router_prob_per_expert: [num_layers, num_experts] - The average
              probability of routing to each expert for this rank for each layer.
            - top_k: [num_layers, 1] - Experts selected per token for each layer.
        coeff: The coefficient for the load balancing loss.
        method: The method for the load balancing loss. Can be "local", "global", or "sample".
        device_mesh: The data-parallel mesh. Used to aggregate statistics for
            the "global" method and valid sample counts for the "sample" method.
        context_parallel_mesh: Sequence-sharded context-parallel mesh. Sample statistics
            are summed over this mesh before computing the loss.

    Returns:
        The load balancing loss. None if lbl_metadata is None or coeff is None.
    """
    if lbl_metadata is None or coeff is None:
        return None
    assert method in ["local", "global", "sample"], "Invalid method"

    if method == "sample":
        sample_num_tokens_per_expert, sample_num_tokens, sample_router_prob_sum_per_expert = (
            _aggregate_sample_lbl_stats(
                lbl_metadata,
                context_parallel_mesh,
            )
        )  # [num_layers,num_samples,num_experts], [num_layers,num_samples,1], [num_layers,num_samples,num_experts]

        num_experts = sample_num_tokens_per_expert.shape[-1]
        top_k = lbl_metadata.top_k.float().unsqueeze(1)  # [num_layers,1,1]
        valid_samples = sample_num_tokens > 0  # [num_layers,num_samples,1]
        safe_num_tokens = sample_num_tokens.clamp_min(1).float()  # [num_layers,num_samples,1]
        # Normalize by per-layer top_k so sum_i f_i = 1 (each token routes to top_k experts).
        mean_assignments_per_expert = sample_num_tokens_per_expert.float() / (
            safe_num_tokens * top_k
        )  # [num_layers,num_samples,num_experts]
        mean_router_prob_per_expert = (
            sample_router_prob_sum_per_expert / safe_num_tokens
        )  # [num_layers,num_samples,num_experts]
        loss_per_sample = (
            torch.sum(mean_assignments_per_expert * mean_router_prob_per_expert, dim=-1) * num_experts
        )  # [num_layers,num_samples]
        # Average over valid samples within each layer, then mean over layers.
        # Empty sample slots (padding or zero-token CP shards) are excluded.
        # NOTE: Equal sample weighting assumes broadly comparable token counts. Generation samples at supported
        # resolutions and frame counts are expected to be long enough for stable estimates. If short samples produce
        # noisy router gradients, add a minimum-token eligibility mask.
        valid_sample_mask = valid_samples.squeeze(-1).float()  # [num_layers,num_samples]
        local_sample_count_per_layer = valid_sample_mask.sum(dim=-1)  # [num_layers]
        local_loss_sum_per_layer = (loss_per_sample * valid_sample_mask).sum(dim=-1)  # [num_layers]
        local_mean_per_layer = local_loss_sum_per_layer / local_sample_count_per_layer.clamp_min(1)  # [num_layers]
        loss_per_layer = _scale_rank_sample_mean(
            local_mean_per_layer,
            local_sample_count_per_layer,
            device_mesh,
        )  # [num_layers]
        lbl = torch.mean(loss_per_layer)  # []
        return lbl * coeff

    num_tokens_per_expert = lbl_metadata.num_tokens_per_expert  # [num_layers,num_experts]
    num_experts = num_tokens_per_expert.shape[-1]
    num_tokens = lbl_metadata.num_tokens  # [num_layers,1]
    mean_router_prob_per_expert = lbl_metadata.mean_router_prob_per_expert  # [num_layers,num_experts]
    top_k = lbl_metadata.top_k  # [num_layers,1]

    if method == "global":
        # Note that these collectives must be executed outside a torch compiled region
        # since torch compile could reorder the collectives and cause deadlocks.
        assert device_mesh is not None, "MoE models require multiple GPUs."

        num_tokens_per_expert = DTensor.from_local(
            num_tokens_per_expert,
            device_mesh=device_mesh,
            placements=[Partial()] * device_mesh.ndim,
        ).full_tensor()  # [num_layers,num_experts]
        num_tokens = DTensor.from_local(
            num_tokens,
            device_mesh=device_mesh,
            placements=[Partial()] * device_mesh.ndim,
        ).full_tensor()  # [num_layers,1]

    # Normalize by per-layer top_k so sum_i f_i = 1 (each token routes to top_k experts).
    mean_tokens_per_expert = num_tokens_per_expert.float() / (
        num_tokens.float() * top_k.float()
    )  # [num_layers,num_experts]

    lbl = torch.mean(torch.sum(mean_tokens_per_expert * mean_router_prob_per_expert, dim=-1) * num_experts)  # []
    return lbl * coeff
