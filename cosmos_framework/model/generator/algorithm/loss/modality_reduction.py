# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Global sample means for the supervised terms of one training microbatch.

The objective is a fixed weighted sum of image, video, action, sound, LiDAR
and radar sample means. A sample can supervise several modalities, but owns at most one
entry in each modality's denominator. Multiple supervised items in that entry
are averaged first. This prevents extra camera/control items, missing audio, or
the number of tokens in a sample from implicitly changing its loss weight.

These are timestep-weighted training losses. The unweighted per-instance values
returned for logging by ``compute_flow_matching_loss`` are not suitable inputs.
The trainer retains its existing gradient-accumulation averaging: normalization
here is per global microbatch, not across an optimizer's accumulation window.
MoE auxiliary losses are added by the caller after this supervised reduction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.distributed as dist

from cosmos_framework.model.generator.algorithm.loss.flow_matching import FlowMatchingLossItems

MODALITIES: tuple[str, ...] = ("image", "video", "action", "sound", "lidar", "radar")


@dataclass(frozen=True)
class ModalityLossItems:
    """Differentiable item losses and their ownership in a single modality.

    ``valid`` excludes fully conditioned items and items with no valid supervised
    elements. An enabled but locally absent modality should supply a zero loss
    connected to its network predictions with ``valid=False``; its parameters
    then participate in backward without adding a sample to the denominator.
    Sample IDs only need to be unique within the local batch, and may be sparse.
    """

    weighted_losses: torch.Tensor  # [N_items]
    sample_ids: torch.Tensor  # [N_items]
    valid: torch.Tensor  # [N_items]

    def sample_sum_and_count(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Average supervised items per sample, then return a sum and sample count."""
        losses = self.weighted_losses  # [N_items]
        if losses.ndim != 1 or self.sample_ids.shape != losses.shape or self.valid.shape != losses.shape:
            raise ValueError("Item losses, sample IDs and validity must be aligned one-dimensional tensors.")
        if self.sample_ids.dtype != torch.long or self.valid.dtype != torch.bool:
            raise ValueError("Sample IDs must be int64 and item validity must be boolean.")
        if self.sample_ids.device != losses.device or self.valid.device != losses.device:
            raise ValueError("Losses, sample IDs and validity must share a device.")
        if losses.numel() == 0:
            return losses.sum(), losses.new_zeros(())  # [], []

        sample_ids, inverse = torch.unique(self.sample_ids, sorted=True, return_inverse=True)  # [B], [N_items]
        valid_losses = torch.where(self.valid, losses, torch.zeros_like(losses))  # [N_items]
        sample_sums = losses.new_zeros(sample_ids.numel()).scatter_add(0, inverse, valid_losses)  # [B]
        item_counts = losses.new_zeros(sample_ids.numel()).scatter_add(0, inverse, self.valid.to(losses.dtype))  # [B]
        sample_means = sample_sums / item_counts.clamp(min=1)  # [B]
        sample_count = item_counts.gt(0).sum().to(losses.dtype)  # []
        return sample_means.sum(), sample_count  # [], []


def build_modality_loss_items(
    collectors: Mapping[str, FlowMatchingLossItems],
    predictions: Mapping[str, list[torch.Tensor]],
    sample_ids: Mapping[str, list[int]],
    *,
    vision_is_image: list[bool] | None,
    is_image_batch: bool,
) -> dict[str, ModalityLossItems]:
    """Associate weighted flow losses with logical owners and image/video identity.

    Absent enabled streams retain the network's zero-weighted probes, so all
    enabled heads remain connected to backward even on a rank without that data.
    """
    result: dict[str, ModalityLossItems] = {}
    for name, collector in collectors.items():
        losses, valid = collector.weighted_losses, collector.valid  # [N_items] or None
        if losses is None or valid is None:
            losses = torch.stack([prediction.sum() * 0.0 for prediction in predictions[f"preds_{name}"]])  # [N_probe]
            valid = torch.zeros_like(losses, dtype=torch.bool)  # [N_probe]
        if name not in sample_ids:
            raise ValueError(f"Missing logical sample ownership for {name}.")
        owners = sample_ids[name]
        # An absent stream has no owners but carries one graph-connected probe.
        owner_ids = owners if owners else [0] * losses.numel()
        if len(owner_ids) != losses.numel():
            raise ValueError(f"{name} has {losses.numel()} item losses but {len(owner_ids)} owners.")
        owner_tensor = torch.tensor(owner_ids, dtype=torch.long, device=losses.device)  # [N_items]
        if name == "vision":
            flags = vision_is_image if vision_is_image else [is_image_batch] * losses.numel()
            if len(flags) != losses.numel():
                raise ValueError("Vision image identities must align with item losses.")
            image_mask = torch.tensor(flags, dtype=torch.bool, device=losses.device)  # [N_items]
            for modality, mask in (("image", image_mask), ("video", ~image_mask)):
                result[modality] = ModalityLossItems(losses[mask], owner_tensor[mask], valid[mask])
        else:
            result[name] = ModalityLossItems(losses, owner_tensor, valid)
    return result


def reduce_global_modality_means(
    items: Mapping[str, ModalityLossItems],
    weights: Mapping[str, float],
    *,
    group: dist.ProcessGroup | None = None,
    gradient_average_size: int = 1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the weighted global supervised objective and unscaled modality means.

    Every rank reduces the same six sums/counts, even when it has no local data
    for a modality. Differentiable local sums are multiplied by the actual
    FSDP/DDP gradient-averaging group size and divided by global sample counts.
    Detached global sums supply consistent logged values without changing that
    backward computation. CP copies must participate through the same averaging
    group as the model: equal replication of sums and counts cancels, rather than
    giving each copy a new share of the objective.
    Sharded CP hidden states additionally require summed gather gradients
    (``correct_cp_gradients=True``); counting replicas alone cannot correct the
    backbone gradient without incorrectly rescaling replicated output heads.
    Disabling correction is allowed for ablations and retains legacy backbone
    gradient scaling while preserving the forward loss values.

    Fixed configured weights are summed, never normalized over present modalities.
    A globally absent modality contributes a graph-connected zero and keeps its
    weight; it does not transfer that weight to other modalities. This deliberately
    differs from both legacy rank-level and homogeneous sample-level averaging.
    """
    unknown = (set(items) | set(weights)) - set(MODALITIES)
    if unknown:
        raise ValueError(f"Unknown supervised modalities: {sorted(unknown)}")
    if not items:
        raise ValueError("At least one modality must provide losses or graph-connected dummy predictions.")
    if gradient_average_size < 1:
        raise ValueError("gradient_average_size must be positive.")
    if gradient_average_size > 1:
        if not dist.is_initialized() or dist.get_world_size(group) != gradient_average_size:
            raise ValueError("The reduction group must match the model's gradient-averaging group size.")

    reference = next(iter(items.values())).weighted_losses  # [N_items]
    local_sums: list[torch.Tensor] = []
    local_counts: list[torch.Tensor] = []
    for name in MODALITIES:
        if name in items:
            local_sum, local_count = items[name].sample_sum_and_count()  # [], []
        else:
            local_sum = reference.sum() * 0.0  # []
            local_count = reference.new_zeros(())  # []
        local_sums.append(local_sum)
        local_counts.append(local_count)

    sums = torch.stack(local_sums)  # [6]
    counts = torch.stack(local_counts)  # [6]
    # Counts are logical contributing samples, not tokens or flattened item counts.
    # FP64 keeps the small collective stable for large distributed sample totals.
    global_stats = torch.stack((sums.detach().double(), counts.detach().double()))  # [2,6]
    if gradient_average_size > 1:
        dist.all_reduce(global_stats, group=group)

    total_loss = sums.sum() * 0.0  # []
    means: dict[str, torch.Tensor] = {}
    for index, name in enumerate(MODALITIES):
        denominator = global_stats[1, index].clamp(min=1).to(sums.dtype)  # []
        backward_mean = sums[index] * gradient_average_size / denominator  # []
        logged_mean = (global_stats[0, index] / global_stats[1, index].clamp(min=1)).to(sums.dtype)  # []
        mean = logged_mean + (backward_mean - backward_mean.detach())  # []
        means[name] = mean
        total_loss = total_loss + float(weights.get(name, 0.0)) * mean  # []
    return total_loss, means
