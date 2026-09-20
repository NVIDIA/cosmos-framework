# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared LiDAR input and output normalization.

Both V0 and V1 encode metric range, unit intensity, and a sensor mask through
:func:`metric_lidar_to_network`. V0 then drops intensity before its 2-channel
network; V1 keeps all three channels. Both come back out through
:func:`network_lidar_to_metric_clip`, so a decoded clip carries the same channels
in the same units whichever version produced it.

This module composes the three-channel layout and decides validity; the per-channel
scaling it composes lives in ``preprocessing`` so that the encode side, the decode
side and the visualizers cannot drift onto separate copies of the same affine.
"""

from __future__ import annotations

import torch

from cosmos_framework.model.generator.tokenizers.lidar.postprocessing import (
    DEFAULT_VALIDITY_THRESHOLD,
    validity_mask,
)
from cosmos_framework.model.generator.tokenizers.lidar.preprocessing import (
    INVALID_NORMALIZED_VALUE,
    network_intensity_to_unit,
    network_range_to_metric,
    normalize_range_map,
    unit_intensity_to_network,
)


def metric_lidar_to_network(
    video: torch.Tensor,
    *,
    min_range: float | torch.Tensor,
    max_range: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert metric range, unit intensity, and sensor mask to network space."""
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(f"Expected metric LiDAR [B,3,T,H,W], got {tuple(video.shape)}")
    range_m = video[:, :1]
    intensity = video[:, 1:2]
    sensor_valid = validity_mask(video[:, 2:3])
    min_value = torch.as_tensor(min_range, dtype=range_m.dtype, device=range_m.device)
    max_value = torch.as_tensor(max_range, dtype=range_m.dtype, device=range_m.device)
    if bool((max_value <= min_value).item()):
        raise ValueError(f"max_range must exceed min_range, got {min_value.item()} and {max_value.item()}")
    valid = sensor_valid & (range_m >= min_value) & (range_m <= max_value)
    normalized_range = normalize_range_map(range_m, min_range=min_value, max_range=max_value)
    normalized_intensity = unit_intensity_to_network(intensity)
    invalid_fill = torch.full_like(normalized_range, INVALID_NORMALIZED_VALUE)
    normalized_range = torch.where(valid, normalized_range, invalid_fill)
    normalized_intensity = torch.where(valid, normalized_intensity, invalid_fill)
    valid_float = valid.to(dtype=video.dtype)
    return torch.cat((normalized_range, normalized_intensity, valid_float), dim=1), valid_float


def network_lidar_to_metric(
    reconstruction: torch.Tensor,
    validity: torch.Tensor,
    *,
    min_range: float | torch.Tensor,
    max_range: float | torch.Tensor,
    apply_validity_mask: bool = True,
    validity_threshold: float = DEFAULT_VALIDITY_THRESHOLD,
) -> torch.Tensor:
    """Convert normalized range/intensity reconstructions to metric public output.

    ``validity_threshold`` is the probability above which a predicted ray is
    kept. Raising it trades recall for precision: a sparser sweep holding only
    the returns the decoder is confident about.
    """
    if reconstruction.ndim != 5 or reconstruction.shape[1] < 2:
        raise ValueError(f"Expected reconstruction [B,2+,T,H,W], got {tuple(reconstruction.shape)}")
    min_value = torch.as_tensor(min_range, dtype=reconstruction.dtype, device=reconstruction.device)
    max_value = torch.as_tensor(max_range, dtype=reconstruction.dtype, device=reconstruction.device)
    range_m = network_range_to_metric(reconstruction[:, :1], min_range=min_value, max_range=max_value)
    intensity = network_intensity_to_unit(reconstruction[:, 1:2])
    if apply_validity_mask:
        valid = validity_mask(validity, threshold=validity_threshold)
        range_m = torch.where(valid, range_m, torch.zeros_like(range_m))
        intensity = torch.where(valid, intensity, torch.zeros_like(intensity))
    return torch.cat((range_m, intensity), dim=1)


def network_lidar_to_metric_clip(
    normalized: torch.Tensor,
    validity: torch.Tensor,
    *,
    min_range: float | torch.Tensor,
    max_range: float | torch.Tensor,
    apply_validity_mask: bool = True,
    validity_threshold: float = DEFAULT_VALIDITY_THRESHOLD,
) -> torch.Tensor:
    """The public decode payload, shared by V0 and V1: ``[B,3,T,H,W]`` in metric units.

    Channel 0 is range in metres, channel 1 unit intensity, channel 2 the resolved
    ``{0, 1}`` mask -- the same layout, in the same units, that the dataloader hands
    to encode. Both tokenizers return this, so a consumer never has to ask which
    version produced a clip in order to know what its numbers mean.

    V0's frozen network predicts no intensity. It passes the absent fill through
    channel 1 rather than special-casing the layout, and the inverse below maps
    that fill to zero intensity.
    """
    metric = network_lidar_to_metric(
        normalized,
        validity,
        min_range=min_range,
        max_range=max_range,
        apply_validity_mask=False,
        validity_threshold=validity_threshold,
    )  # [B,2,T,H,W]
    if apply_validity_mask:
        valid = validity_mask(validity, threshold=validity_threshold)
        metric = torch.where(valid, metric, torch.zeros_like(metric))
        channel = valid.to(dtype=validity.dtype)
    else:
        channel = validity
    return torch.cat((metric, channel), dim=1)  # [B,3,T,H,W]
