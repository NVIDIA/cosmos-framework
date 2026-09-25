# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

import cosmos_framework.model.generator.tokenizers.lidar.normalization as normalization
from cosmos_framework.model.generator.tokenizers.lidar.checkpoint import parse_lidar_checkpoint_stats, parse_lidar_stats
from cosmos_framework.model.generator.tokenizers.lidar.normalization import (
    metric_lidar_to_network,
    network_lidar_to_metric,
    network_lidar_to_metric_clip,
)
from cosmos_framework.model.generator.tokenizers.lidar.preprocessing import (
    network_intensity_to_unit,
    network_range_to_metric,
    normalize_range_map,
    unit_intensity_to_network,
    unnormalize_range_map,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_channel_scaling_round_trips_through_the_shared_primitives() -> None:
    """The forward and inverse affines invert each other inside the clamped domain."""
    metric = torch.tensor([[[5.0, 42.5, 100.0]]])
    unit = torch.tensor([[[0.0, 0.25, 1.0]]])

    normalized_range = normalize_range_map(metric, min_range=5.0, max_range=100.0)
    normalized_intensity = unit_intensity_to_network(unit)

    torch.testing.assert_close(normalized_range.flatten(), torch.tensor([-1.0, -0.21052632, 1.0]))
    torch.testing.assert_close(normalized_intensity.flatten(), torch.tensor([-1.0, -0.5, 1.0]))
    torch.testing.assert_close(
        network_range_to_metric(normalized_range, min_range=5.0, max_range=100.0),
        metric,
    )
    torch.testing.assert_close(network_intensity_to_unit(normalized_intensity), unit)


def test_composed_normalization_matches_the_primitives_it_delegates_to() -> None:
    """``metric_lidar_to_network`` and its inverse must not re-derive the scaling.

    Pinning the composed three-channel path against the per-channel primitives is what
    keeps the encode side, the decode side and the visualizers on one affine.
    """
    video = torch.tensor([[[[[7.5, 60.0]]], [[[0.25, 0.75]]], [[[1.0, 1.0]]]]])

    normalized, valid = metric_lidar_to_network(video, min_range=5.0, max_range=100.0)

    torch.testing.assert_close(normalized[:, :1], normalize_range_map(video[:, :1], min_range=5.0, max_range=100.0))
    torch.testing.assert_close(normalized[:, 1:2], unit_intensity_to_network(video[:, 1:2]))

    metric = network_lidar_to_metric(normalized, valid, min_range=5.0, max_range=100.0)

    torch.testing.assert_close(
        metric[:, :1], network_range_to_metric(normalized[:, :1], min_range=5.0, max_range=100.0)
    )
    torch.testing.assert_close(metric[:, 1:2], network_intensity_to_unit(normalized[:, 1:2]))


def test_prediction_overshoot_is_clamped_for_output_but_masked_for_display() -> None:
    """The two inverse-range callers disagree on overshoot, and that is deliberate.

    A decoded public sweep clamps back to the range envelope, while the display
    helper leaves the value alone and lets the derived mask drop it, so neither
    path silently adopts the other's handling.
    """
    overshoot = torch.tensor([[[1.5]]])

    torch.testing.assert_close(
        network_range_to_metric(overshoot, min_range=5.0, max_range=100.0),
        torch.tensor([[[100.0]]]),
    )
    unclamped, mask = unnormalize_range_map(overshoot, min_range=5.0, max_range=100.0)
    torch.testing.assert_close(unclamped, torch.tensor([[[0.0]]]))
    assert not bool(mask.any())


def test_metric_lidar_round_trip_and_range_validity() -> None:
    video = torch.tensor([[[[[0.5, 1.0, 50.5, 101.0]]], [[[0.2, 0.3, 0.4, 0.5]]], [[[1.0, 1.0, 1.0, 1.0]]]]])

    normalized, valid = metric_lidar_to_network(video, min_range=1.0, max_range=100.0)
    metric = network_lidar_to_metric(
        normalized,
        valid,
        min_range=1.0,
        max_range=100.0,
        apply_validity_mask=True,
    )

    torch.testing.assert_close(valid.flatten(), torch.tensor([0.0, 1.0, 1.0, 0.0]))
    torch.testing.assert_close(metric[0, 0].flatten(), torch.tensor([0.0, 1.0, 50.5, 0.0]))
    torch.testing.assert_close(metric[0, 1].flatten(), torch.tensor([0.0, 0.3, 0.4, 0.0]))


def test_network_lidar_to_metric_honors_a_raised_validity_threshold() -> None:
    """Raising the cut trades recall for precision on the predicted rays."""
    reconstruction = torch.zeros((1, 3, 1, 1, 3))
    validity = torch.tensor([[[[[0.4, 0.7, 0.99]]]]])

    default_cut = network_lidar_to_metric(reconstruction, validity, min_range=1.0, max_range=101.0)
    strict_cut = network_lidar_to_metric(
        reconstruction,
        validity,
        min_range=1.0,
        max_range=101.0,
        validity_threshold=0.9,
    )

    torch.testing.assert_close(default_cut[0, 0].flatten(), torch.tensor([0.0, 51.0, 51.0]))
    torch.testing.assert_close(strict_cut[0, 0].flatten(), torch.tensor([0.0, 0.0, 51.0]))


def test_metric_clip_resolves_validity_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    original = normalization.validity_mask

    def counting_validity_mask(validity: torch.Tensor, *, threshold: float) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(validity, threshold=threshold)

    monkeypatch.setattr(normalization, "validity_mask", counting_validity_mask)
    reconstruction = torch.zeros((1, 2, 1, 1, 2))
    validity = torch.tensor([[[[[0.4, 0.8]]]]])

    decoded = network_lidar_to_metric_clip(reconstruction, validity, min_range=1.0, max_range=101.0)

    assert calls == 1
    torch.testing.assert_close(decoded[0, 2].flatten(), torch.tensor([0.0, 1.0]))


def test_lidar_stats_preserve_bounds_and_accept_legacy_payloads() -> None:
    mean, std, min_range, max_range = parse_lidar_stats(
        {"mean": torch.zeros(2), "std": torch.ones(2), "min_range": 1.0, "max_range": 105.0}
    )
    assert mean.shape == std.shape == (2,)
    assert (min_range, max_range) == (1.0, 105.0)

    _, _, legacy_min, legacy_max = parse_lidar_stats((torch.zeros(2), torch.ones(2)))
    assert legacy_min is legacy_max is None


def test_lidar_stats_can_be_read_from_training_checkpoint() -> None:
    mean, std, min_range, max_range = parse_lidar_checkpoint_stats(
        {
            "model": {
                "latent_mean": torch.tensor([1.0, 2.0]),
                "latent_std": torch.tensor([3.0, 4.0]),
                "min_range": torch.tensor(1.0),
                "max_range": torch.tensor(105.0),
            }
        }
    )

    torch.testing.assert_close(mean, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(std, torch.tensor([3.0, 4.0]))
    assert (min_range, max_range) == (1.0, 105.0)

    _, _, min_range, max_range = parse_lidar_checkpoint_stats(
        {
            "model": {
                "latent_mean": torch.tensor([1.0, 2.0]),
                "latent_std": torch.tensor([3.0, 4.0]),
            }
        }
    )
    assert min_range is max_range is None


