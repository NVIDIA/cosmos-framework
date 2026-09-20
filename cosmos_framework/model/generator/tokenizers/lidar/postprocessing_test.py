# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.model.generator.tokenizers.lidar.postprocessing import (
    DEFAULT_VALIDITY_THRESHOLD,
    RangeSmoothing,
    as_lidar_clip,
    prepare_range_for_display,
    smooth_metric_range,
    validate_validity_threshold,
    validity_from_mask_logits,
    validity_from_reconstruction,
    validity_from_video,
    validity_mask,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, 1.5])
def test_validity_threshold_rejects_degenerate_cuts(threshold: float) -> None:
    """Zero keeps every ray and one drops every ray, so both erase the prediction."""
    with pytest.raises(ValueError, match="validity_threshold"):
        validate_validity_threshold(threshold)


def test_validity_threshold_accepts_the_sweep_points_and_normalizes_type() -> None:
    """The cut is read from config, so it arrives as whatever OmegaConf held."""
    assert validate_validity_threshold(DEFAULT_VALIDITY_THRESHOLD) == 0.5
    assert validate_validity_threshold(0.7) == pytest.approx(0.7)
    assert validate_validity_threshold(0.99) == pytest.approx(0.99)
    assert isinstance(validate_validity_threshold("0.7"), float)  # type: ignore[arg-type]


def test_as_lidar_clip_peels_a_singleton_batch() -> None:
    clip = torch.zeros((1, 3, 2, 4, 8))
    assert as_lidar_clip(clip).shape == (3, 2, 4, 8)


def test_prepare_range_for_display_treats_v0_with_empty_intensity_as_v0() -> None:
    clip = torch.full((1, 1, 1, 2), -1.0)
    clip[0, 0, 0, 1] = 0.0
    with_empty_intensity = torch.cat((clip, torch.zeros_like(clip)), dim=0)
    normalized, valid, lo, hi = prepare_range_for_display(with_empty_intensity)
    assert valid is None
    assert (lo, hi) == (5.0, 100.0)
    torch.testing.assert_close(normalized, clip[0])


def test_validity_from_mask_logits_sigmoids_then_cuts() -> None:
    """Training reconstructions carry logits; this is the only place they are cut."""
    logits = torch.tensor([0.25, 4.0, -4.0, 0.0])
    assert validity_from_mask_logits(logits).tolist() == [True, True, False, True]


def test_validity_mask_reads_a_resolved_channel_without_sigmoid() -> None:
    assert validity_mask(torch.tensor([1.0, 0.0, 0.56, 0.44])).tolist() == [True, False, True, False]


def test_validity_from_video_reads_a_resolved_mask_channel() -> None:
    video = torch.zeros((1, 3, 1, 1, 2))
    video[0, 2, 0, 0, 0] = 1.0
    assert validity_from_video(video).flatten().tolist() == [True, False]


def test_validity_from_reconstruction_sigmoids_training_logits() -> None:
    recon = torch.zeros((1, 3, 1, 1, 2))
    recon[0, 2, 0, 0, 0] = 0.25
    recon[0, 2, 0, 0, 1] = -4.0
    assert validity_from_reconstruction(recon).flatten().tolist() == [True, False]


def test_validity_from_reconstruction_does_not_sigmoid_an_explicit_probability() -> None:
    recon = torch.zeros((1, 3, 1, 1, 1))
    recon[0, 2] = 4.0
    probability = torch.tensor([0.44]).reshape(1, 1, 1, 1, 1)
    assert bool(validity_from_reconstruction(recon, probability).item()) is False


def test_prepare_range_for_display_leaves_v0_on_its_sentinel() -> None:
    clip = torch.full((1, 1, 1, 2), -1.0)
    clip[0, 0, 0, 1] = 0.0
    normalized, valid, lo, hi = prepare_range_for_display(clip)
    assert valid is None
    assert (lo, hi) == (5.0, 100.0)
    torch.testing.assert_close(normalized, clip[0])


def test_prepare_range_for_display_reads_a_resolved_v1_mask() -> None:
    """Channel 2 is ``{0, 1}`` on a dataloader clip and on a decoded one alike.

    The tokenizer resolves its own cut, so display only reads the answer back.
    """
    lidar = torch.zeros((3, 1, 1, 3))
    lidar[0] = 50.0
    lidar[2, :, :, 0] = 1.0
    lidar[2, :, :, 1] = 0.0
    lidar[2, :, :, 2] = 1.0

    _, valid, lo, hi = prepare_range_for_display(lidar)

    assert (lo, hi) == (5.0, 100.0)
    assert valid is not None
    assert valid.flatten().tolist() == [True, False, True]


def test_prepare_range_for_display_reads_an_unmasked_decode_as_probability() -> None:
    """``apply_mask=False`` leaves the probability in channel 2, not a logit.

    The half cut then reproduces what the tokenizer's default would have done,
    so an unmasked decode still renders sensibly without a threshold argument.
    """
    lidar = torch.zeros((3, 1, 1, 2))
    lidar[0] = 50.0
    lidar[2, :, :, 0] = 0.56
    lidar[2, :, :, 1] = 0.44

    _, valid, _, _ = prepare_range_for_display(lidar)

    assert valid is not None
    assert valid.flatten().tolist() == [True, False]


def test_prepare_range_for_display_drops_zeroed_range_even_when_the_mask_is_set() -> None:
    """A raised tokenizer threshold zeroes range; a stale mask must not resurrect it."""
    lidar = torch.zeros((3, 1, 1, 2))
    lidar[2] = 1.0
    lidar[0, :, :, 0] = 50.0
    lidar[0, :, :, 1] = 0.0

    _, valid, _, _ = prepare_range_for_display(lidar)

    assert valid is not None
    assert valid.flatten().tolist() == [True, False]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"azimuth_kernel_size": 2},
        {"azimuth_kernel_size": 1},
        {"elevation_kernel_size": 0},
        {"elevation_kernel_size": 2},
        {"spatial_sigma": 0.0},
        {"range_sigma_m": 0.0},
        {"range_sigma_ratio": -0.1},
        # Above what the default window's four neighbours can contribute, so no
        # ray could clear it and the filter would drop the sweep.
        {"min_neighbor_support": 5.0},
        {"strength": 1.5},
    ],
)
def test_range_smoothing_rejects_a_window_that_cannot_estimate_a_surface(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        RangeSmoothing(**kwargs)  # type: ignore[arg-type]


def test_smoothing_reads_one_beam_unless_asked_for_more() -> None:
    """The default window is a row, because the other beam is at another distance.

    Down a column each sample is a different laser at a different elevation, so
    on ground the beams above and below are legitimately at another distance --
    close enough to weigh, which is what makes them harmful rather than merely
    useless. Here they sit 0.3 m out, well inside the 1.1 m bandwidth at this
    range, and a row-only window leaves the middle beam exactly where it was.
    Widening the window is the knob that reaches them again.
    """
    ranges = torch.full((1, 3, 8), 30.3)
    ranges[0, 1, :] = 30.0
    valid = torch.ones_like(ranges, dtype=torch.bool)

    per_row, _ = smooth_metric_range(ranges, valid)
    across_beams, _ = smooth_metric_range(
        ranges, valid, smoothing=RangeSmoothing(elevation_kernel_size=3, min_neighbor_support=2.0)
    )

    torch.testing.assert_close(per_row[0, 1], torch.full((8,), 30.0))
    assert across_beams[0, 1].min().item() > 30.15


def test_smooth_metric_range_pulls_a_jittered_wall_flat() -> None:
    """Per-ray jitter on a flat surface is what sprays the BEV along each ray."""
    flat = torch.full((1, 5, 8), 30.0)
    jittered = flat.clone()
    jittered[0, 2, 3] = 30.4
    jittered[0, 1, 5] = 29.6
    valid = torch.ones_like(flat, dtype=torch.bool)

    smoothed, kept = smooth_metric_range(jittered, valid)

    assert bool(kept.all())
    # A weighted mean keeps a fraction of the ray's own vote, so it lands near the
    # surface rather than exactly on it. One row is four neighbours rather than a
    # square's twenty-four, so that fraction is larger and a lone spike keeps more
    # of itself: a third of this 0.4 m one survives where a twentieth used to.
    # Real jitter is on every ray at once, where averaging fewer but better
    # neighbours still wins; see the measurements on RangeSmoothing.
    torch.testing.assert_close(smoothed, flat, atol=0.15, rtol=0.0)


def test_smooth_metric_range_drops_a_flying_pixel_instead_of_averaging_it_in() -> None:
    """A ray far off its neighbours spans a depth gap, and weighting alone cannot fix it.

    Its neighbours are many range sigma away, so they weigh nothing and the filter
    would hand the ray straight back. Only the support rule removes it.
    """
    ranges = torch.full((1, 5, 8), 30.0)
    ranges[0, 2, 3] = 70.0
    valid = torch.ones_like(ranges, dtype=torch.bool)

    smoothed, kept = smooth_metric_range(ranges, valid)

    assert not bool(kept[0, 2, 3])
    assert bool(kept.sum() == kept.numel() - 1)
    # A dropped ray keeps its measured range so nothing invents geometry behind the mask.
    assert smoothed[0, 2, 3] == pytest.approx(70.0)


def test_smooth_metric_range_widens_the_bandwidth_with_distance() -> None:
    """Adjacent beams on the same ground diverge with range, so a fixed bandwidth over-drops far."""
    smoothing = RangeSmoothing(range_sigma_m=0.2, range_sigma_ratio=0.03)
    near = torch.full((1, 5, 8), 10.0)
    far = torch.full((1, 5, 8), 80.0)
    near[0, 2, 3] += 2.0
    far[0, 2, 3] += 2.0
    valid = torch.ones_like(near, dtype=torch.bool)

    _, near_kept = smooth_metric_range(near, valid, smoothing=smoothing)
    _, far_kept = smooth_metric_range(far, valid, smoothing=smoothing)

    assert not bool(near_kept[0, 2, 3])
    assert bool(far_kept[0, 2, 3])


def test_smooth_metric_range_drops_isolated_speckle() -> None:
    """One lit ray in an empty window is noise, not a surface worth keeping."""
    ranges = torch.zeros((1, 5, 8))
    valid = torch.zeros_like(ranges, dtype=torch.bool)
    ranges[0, 2, 3] = 30.0
    valid[0, 2, 3] = True

    _, kept = smooth_metric_range(ranges, valid)

    assert not bool(kept.any())


def test_smooth_metric_range_ignores_dropped_neighbors() -> None:
    """Masked rays carry no distance, so giving them weight would pull the estimate down."""
    ranges = torch.full((1, 3, 8), 30.0)
    ranges[0, :, 4] = 0.0
    valid = torch.ones_like(ranges, dtype=torch.bool)
    valid[0, :, 4] = False

    smoothed, _ = smooth_metric_range(ranges, valid)

    torch.testing.assert_close(smoothed[0, 1, 3], torch.tensor(30.0))
    torch.testing.assert_close(smoothed[0, 1, 5], torch.tensor(30.0))


def test_smooth_metric_range_wraps_across_the_azimuth_seam() -> None:
    """The canvas spans a full turn, so its first and last columns are neighbours."""
    ranges = torch.full((1, 3, 8), 30.0)
    ranges[0, 1, 0] = 30.4
    valid = torch.ones_like(ranges, dtype=torch.bool)

    seam_only = ranges.clone()
    seam_only[0, :, 1:-1] = 0.0
    seam_valid = valid.clone()
    seam_valid[0, :, 1:-1] = False

    smoothed, _ = smooth_metric_range(ranges, valid)
    wrapped, wrapped_kept = smooth_metric_range(seam_only, seam_valid)

    torch.testing.assert_close(smoothed[0, 1, 0], torch.tensor(30.0), atol=0.15, rtol=0.0)
    # The column-zero window has no support without the wrap, so only the wrap keeps this
    # ray. Its pull is weaker than above because the wrapped column is all the support
    # there is, and the ray's own vote is a larger share of a smaller total.
    assert bool(wrapped_kept[0, 1, 0])
    torch.testing.assert_close(wrapped[0, 1, 0], torch.tensor(30.0), atol=0.3, rtol=0.0)


def test_smooth_metric_range_strength_interpolates_toward_the_filtered_estimate() -> None:
    ranges = torch.full((1, 3, 8), 30.0)
    ranges[0, 1, 4] = 30.4
    valid = torch.ones_like(ranges, dtype=torch.bool)

    full, _ = smooth_metric_range(ranges, valid)
    half, _ = smooth_metric_range(ranges, valid, smoothing=RangeSmoothing(strength=0.5))

    assert full[0, 1, 4].item() < 30.4
    assert half[0, 1, 4].item() == pytest.approx(0.5 * (30.4 + full[0, 1, 4].item()))


def test_prepare_range_for_display_smooths_before_it_normalizes() -> None:
    """Unprojection reads this output, so the filter has to land on the metric range."""
    lidar = torch.full((3, 1, 3, 8), 30.0)
    lidar[1] = 0.0
    lidar[2] = 1.0
    lidar[0, 0, 1, 4] = 30.4

    normalized, valid, lo, hi = prepare_range_for_display(lidar, smoothing=RangeSmoothing())

    assert valid is not None and bool(valid.all())
    metric = (normalized + 1.0) * 0.5 * (hi - lo) + lo
    torch.testing.assert_close(metric, torch.full_like(metric, 30.0), atol=0.15, rtol=0.0)
