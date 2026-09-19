# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""Decode-side LiDAR postprocessing shared by the V0 and V1 tokenizers.

This module owns the validity cut: sigmoid of mask logits, then a probability
threshold. Tokenizers apply it in decode; tokenizer-training reconstructions
are still raw logits, so visualization asks this module for a keep-mask rather
than reimplementing the cut. Callbacks and renderers do not sigmoid or threshold.

A resolved clip -- dataloader ``{0, 1}``, or decode after the cut -- is read
with the same half-cut and is never sigmoided again. Range-map preparation on
the encode side belongs to ``preprocessing``.

This module also owns the range smoothing a generated clip is run through before
it is unprojected to XYZ, and the single switch that turns it on; see
:class:`RangeSmoothing` and ``DEFAULT_RANGE_SMOOTHING``.
"""

from __future__ import annotations

import attrs
import torch

from cosmos_framework.model.generator.tokenizers.lidar.preprocessing import (
    INVALID_NORMALIZED_VALUE,
    MAX_RANGE_METERS,
    MIN_RANGE_METERS,
)

DEFAULT_VALIDITY_THRESHOLD = 0.5
# The sentinel a range-only clip drops rays at is the fill encode writes, so read it
# from there: a sentinel detector that disagreed with the fill would keep dropped rays.
DEFAULT_MASK_INPUT_MIN_VALUE = INVALID_NORMALIZED_VALUE
DEFAULT_MASK_INPUT_EPS = 0.02


@attrs.frozen
class RangeSmoothing:
    """Validity-aware bilateral filter for a metric range map, run before unprojection.

    A generated range map carries per-ray error that is invisible in the range
    view -- neighbouring rays differ by a few tenths of a metre either way -- but
    unprojection turns each ray into a point at its own distance, so that error
    becomes scatter along the ray direction and reads as a thick, noisy surface
    in bird's-eye view. Filtering the range before the unprojection is what
    removes it; nothing downstream of XYZ can tell scatter from structure.

    Each ray is replaced by a weighted mean of its window, where a neighbour's
    weight is a spatial Gaussian in pixels times a range Gaussian in metres. The
    range term is what keeps a depth discontinuity intact: rays across the gap
    fall many sigma away and contribute nothing, so a foreground edge is
    averaged along itself rather than smeared into the background behind it.
    Unlike a median, the estimate is continuous in the samples, so it removes
    sub-bin jitter instead of quantizing every ray onto whichever neighbour
    happens to sit in the middle.

    That window is one row wide by default, and the two axes are not
    interchangeable. Along a row the samples are one laser sweeping azimuth, so
    a smooth surface is nearly iso-range and the neighbours estimate it. Down a
    column each sample is a different beam at a different elevation, and on the
    ground plane -- most of a sweep -- that is a steep true range gradient, so
    those neighbours disagree because the surface really is at another distance.
    Averaging them in trades noise for distortion. Measured over 15 MADS sweeps,
    filtering ground truth with the 5x5 window this shipped with moves a
    surviving ray 0.098 m, against 0.067 m for one row; a 5x1 column-only
    control accounts for 0.079 m of that, which is where it comes from. On truth
    plus 0.2 m of per-ray jitter the row leaves 0.125 m against the square's
    0.133 m, and on 0.2 m of spatially correlated error -- the shape a decoder's
    error actually has -- the square leaves 0.165 m where not filtering at all
    leaves 0.160 m, while the row leaves 0.158 m. Set ``elevation_kernel_size``
    above 1 to average beams together again.

    The range bandwidth is ``range_sigma_m + range_sigma_ratio * range``,
    because a fixed bandwidth is the wrong shape for this data: two rays a
    degree apart that land on the same sloped ground are centimetres apart at
    10 m and metres apart at 80 m, so a bandwidth tight enough to clean the near
    field would treat the whole far field as unrelated surfaces.

    Weighting alone cannot remove a bad ray -- a flying pixel between a
    foreground edge and the background agrees with itself perfectly and would
    keep its own value. So a ray is dropped unless the weight its neighbours
    contribute, ``min_neighbor_support``, says a local surface corroborates it.
    That one rule covers both the flying pixel, whose neighbours are all too far
    in range to weigh anything, and isolated speckle, which has no valid
    neighbours to begin with. A row corroborates nearly as well as a square did:
    over the same sweeps with 1% of rays displaced, one row catches 97.6% of
    them against the square's 97.9%. The default asks for 19% of what a full
    window could contribute, which is the share the 5x5 default asked for, so
    the bar tracks the window rather than the shape.

    ``strength`` interpolates between the measured range (0.0) and the filtered
    estimate (1.0), so a run can keep some of the original relief.
    """

    azimuth_kernel_size: int = 5
    elevation_kernel_size: int = 1
    spatial_sigma: float = 1.5
    range_sigma_m: float = 0.2
    range_sigma_ratio: float = 0.03
    min_neighbor_support: float = 0.45
    strength: float = 1.0

    @property
    def window_size(self) -> int:
        """Rays in one window, the ray itself included."""
        return self.azimuth_kernel_size * self.elevation_kernel_size

    def __attrs_post_init__(self) -> None:
        if self.azimuth_kernel_size < 3 or self.azimuth_kernel_size % 2 == 0:
            raise ValueError(f"azimuth_kernel_size must be an odd size of at least 3, got {self.azimuth_kernel_size}")
        # One is the per-row default rather than a degenerate case; anything
        # wider still has to be centred on the ray it filters.
        if self.elevation_kernel_size < 1 or self.elevation_kernel_size % 2 == 0:
            raise ValueError(f"elevation_kernel_size must be odd and at least 1, got {self.elevation_kernel_size}")
        if self.spatial_sigma <= 0.0:
            raise ValueError(f"spatial_sigma must be positive, got {self.spatial_sigma}")
        if self.range_sigma_m <= 0.0:
            raise ValueError(f"range_sigma_m must be positive, got {self.range_sigma_m}")
        if self.range_sigma_ratio < 0.0:
            raise ValueError(f"range_sigma_ratio must be non-negative, got {self.range_sigma_ratio}")
        if not 0.0 <= self.min_neighbor_support <= self.window_size - 1:
            raise ValueError(
                f"min_neighbor_support must lie in [0, {self.window_size - 1}], got {self.min_neighbor_support}"
            )
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must lie in [0, 1], got {self.strength}")


# The one switch for smoothing: the saved generated range map and every BEV unprojection
# read this constant, so setting it to None goes back to raw generations everywhere. The
# saved artifact is filtered because its consumers unproject it, which is where the error
# shows: a generated sweep's per-ray range error is too small to see in the range view,
# which stays raw so a reconstruction can be judged on what the tokenizer produced.
# Measured over 48 sweeps of a generated V0 clip, the 5x5 window this shipped with took
# the disagreement between a ray and its neighbours from 0.73 m to 0.26 m while moving a
# surviving ray 0.15 m, and dropped the 4.6% of rays no local surface corroborated. Those
# figures describe the square, not this default; what replaced it is measured against
# ground truth in :class:`RangeSmoothing`, which is the comparison that showed the square
# was spending more of that movement on distorting real structure than on noise.
DEFAULT_RANGE_SMOOTHING: RangeSmoothing | None = RangeSmoothing()


def _pad_range_window(value: torch.Tensor, *, pad_elevation: int, pad_azimuth: int) -> torch.Tensor:
    """Pad ``[N,1,H,W]`` for a neighbourhood read: azimuth wraps, elevation does not."""
    padded = torch.nn.functional.pad(value, (pad_azimuth, pad_azimuth, 0, 0), mode="circular")
    if pad_elevation == 0:
        return padded
    return torch.nn.functional.pad(padded, (0, 0, pad_elevation, pad_elevation), mode="replicate")


def _spatial_weights(
    elevation: int, azimuth: int, sigma: float, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Gaussian falloff over an ``elevation x azimuth`` window, flattened to one axis."""
    rows = torch.arange(elevation, dtype=dtype, device=device) - elevation // 2
    columns = torch.arange(azimuth, dtype=dtype, device=device) - azimuth // 2
    squared = rows.reshape(-1, 1) ** 2 + columns.reshape(1, -1) ** 2
    return torch.exp(-squared / (2.0 * sigma**2)).reshape(-1)


def _smooth_range_sweeps(
    ranges: torch.Tensor,
    keep: torch.Tensor,
    *,
    smoothing: RangeSmoothing,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter a ``[N,1,H,W]`` block of sweeps; see :func:`smooth_metric_range`."""
    elevation = smoothing.elevation_kernel_size
    azimuth = smoothing.azimuth_kernel_size
    window = smoothing.window_size

    def neighbourhood(value: torch.Tensor) -> torch.Tensor:
        """Read each ray's window as a trailing axis."""
        return (
            _pad_range_window(value, pad_elevation=elevation // 2, pad_azimuth=azimuth // 2)
            .unfold(2, elevation, 1)
            .unfold(3, azimuth, 1)
            .reshape(*ranges.shape, window)
        )

    # A dropped ray carries no distance, so it is zeroed here and then zero-weighted
    # below; the same mechanism excludes a pad column that wrapped in over the seam.
    windows = neighbourhood(torch.where(keep, ranges, torch.zeros_like(ranges)))
    window_keep = neighbourhood(keep.to(ranges.dtype))

    sigma = (smoothing.range_sigma_m + smoothing.range_sigma_ratio * ranges.abs()).unsqueeze(-1)
    similarity = torch.exp(-0.5 * ((windows - ranges.unsqueeze(-1)) / sigma) ** 2)
    spatial = _spatial_weights(elevation, azimuth, smoothing.spatial_sigma, dtype=ranges.dtype, device=ranges.device)
    weights = window_keep * spatial * similarity

    total = weights.sum(dim=-1)
    filtered = (weights * windows).sum(dim=-1) / total.clamp_min(torch.finfo(ranges.dtype).tiny)

    # The ray's own vote is excluded from the support: a flying pixel agrees with
    # itself perfectly, and it is the neighbours that have to corroborate it.
    support = total - weights[..., window // 2]
    kept = keep & (support >= smoothing.min_neighbor_support)

    smoothed = torch.lerp(ranges, filtered, smoothing.strength)
    return torch.where(kept, smoothed, ranges), kept


# A window holds one float per ray in it and the filter keeps a few tensors that
# shape at once, so a 400-sweep clip at full azimuth would allocate tens of gigabytes
# in one shot -- on the decode device, where that is fatal rather than merely slow.
# Sweeps are filtered independently, so slicing them into blocks of roughly this many
# window elements bounds the peak without changing the result.
_WINDOW_ELEMENT_BUDGET = 16_000_000


def smooth_metric_range(
    metric_range: torch.Tensor,
    valid: torch.Tensor,
    *,
    smoothing: RangeSmoothing = RangeSmoothing(),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilateral-filter a metric range map ``[...,H,W]``, returning ranges and validity.

    Only kept rays carry weight, and a dropped ray keeps its input range so no
    distance is invented behind the mask. Azimuth wraps,
    because the canvas spans a full turn with ``azimuth_endpoint=False`` and its
    two edge columns are therefore neighbours on the sensor; elevation does not,
    so a window that spans beams edge-pads the rows instead. The default window
    is one row and reads no other beam at all.
    """
    if metric_range.shape != valid.shape:
        raise ValueError(f"Expected matching range and validity shapes, got {metric_range.shape} and {valid.shape}")
    if metric_range.ndim < 2:
        raise ValueError(f"Expected a range map with [...,H,W] axes, got {tuple(metric_range.shape)}")

    height, width = metric_range.shape[-2:]
    if width < smoothing.azimuth_kernel_size:
        raise ValueError(f"Azimuth width {width} is narrower than the {smoothing.azimuth_kernel_size}-column window")

    leading = metric_range.shape[:-2]
    ranges = metric_range.reshape(-1, 1, height, width).float()
    keep = valid.reshape(-1, 1, height, width)

    block = max(1, _WINDOW_ELEMENT_BUDGET // (height * width * smoothing.window_size))
    smoothed_blocks: list[torch.Tensor] = []
    kept_blocks: list[torch.Tensor] = []
    for start in range(0, ranges.shape[0], block):
        smoothed_block, kept_block = _smooth_range_sweeps(
            ranges[start : start + block],
            keep[start : start + block],
            smoothing=smoothing,
        )
        smoothed_blocks.append(smoothed_block)
        kept_blocks.append(kept_block)

    smoothed = torch.cat(smoothed_blocks) if len(smoothed_blocks) > 1 else smoothed_blocks[0]
    kept = torch.cat(kept_blocks) if len(kept_blocks) > 1 else kept_blocks[0]
    return (
        smoothed.reshape(*leading, height, width).to(dtype=metric_range.dtype),
        kept.reshape(*leading, height, width),
    )


def validate_validity_threshold(threshold: float) -> float:
    """Check that a validity probability cut lies strictly inside ``(0, 1)``.

    Zero would keep every ray and one would drop every ray, so both ends erase
    the prediction the cut is meant to read.
    """
    value = float(threshold)
    if not 0.0 < value < 1.0:
        raise ValueError(f"validity_threshold must lie in (0, 1), got {threshold}")
    return value


def validity_probability(mask_logits: torch.Tensor) -> torch.Tensor:
    """Convert mask logits to a per-ray keep probability."""
    return torch.sigmoid(mask_logits)


def validity_mask(
    probability: torch.Tensor,
    *,
    threshold: float = DEFAULT_VALIDITY_THRESHOLD,
) -> torch.Tensor:
    """Boolean keep-mask from a probability (or already-resolved ``{0, 1}``) channel."""
    return probability >= validate_validity_threshold(threshold)


def resolved_validity_channel(
    validity: torch.Tensor,
    *,
    should_mask: bool,
    threshold: float = DEFAULT_VALIDITY_THRESHOLD,
) -> torch.Tensor:
    """Channel 2 of a decoded clip, never raw logits.

    A dataloader clip carries a ``{0, 1}`` sensor mask in this slot, so a
    decoded clip resolves the cut here as well and every consumer reads the
    two the same way. Leaving logits here is what previously pushed the
    sigmoid out into the display path, where it drifted out of step with the
    cut applied to range and intensity.

    With ``should_mask=False`` there is no cut to resolve, so the probability
    passes through and a reader's own ``> 0.5`` reproduces the default.
    """
    if not should_mask:
        return validity
    return validity_mask(validity, threshold=threshold).to(dtype=validity.dtype)


def validity_from_mask_logits(
    mask_logits: torch.Tensor,
    *,
    threshold: float = DEFAULT_VALIDITY_THRESHOLD,
) -> torch.Tensor:
    """Sigmoid then cut. The only path that may see raw mask logits."""
    return validity_mask(validity_probability(mask_logits), threshold=threshold)


def validity_from_range_sentinel(
    range_channel: torch.Tensor,
    *,
    mask_input_min_value: float = DEFAULT_MASK_INPUT_MIN_VALUE,
    mask_input_eps: float = DEFAULT_MASK_INPUT_EPS,
) -> torch.Tensor:
    """Keep rays whose normalized range sits strictly above the invalid fill."""
    finite = torch.isfinite(range_channel)
    return finite & (range_channel > (float(mask_input_min_value) + float(mask_input_eps)))


def validity_from_video(
    video: torch.Tensor,
    *,
    mask_input_min_value: float = DEFAULT_MASK_INPUT_MIN_VALUE,
    mask_input_eps: float = DEFAULT_MASK_INPUT_EPS,
) -> torch.Tensor:
    """Boolean keep-mask for a dataloader or encoded clip ``[B,C,T,H,W]``.

    Three-channel clips carry an already-resolved ``{0, 1}`` mask; one-channel
    V0 clips drop rays at the range sentinel.
    """
    if video.shape[1] >= 3:
        return validity_mask(video[:, 2:3])
    return validity_from_range_sentinel(
        video[:, :1],
        mask_input_min_value=mask_input_min_value,
        mask_input_eps=mask_input_eps,
    )


def validity_from_reconstruction(
    reconstruction: torch.Tensor,
    validity: torch.Tensor | None = None,
    *,
    mask_input_min_value: float = DEFAULT_MASK_INPUT_MIN_VALUE,
    mask_input_eps: float = DEFAULT_MASK_INPUT_EPS,
) -> torch.Tensor:
    """Boolean keep-mask for a tokenizer training reconstruction ``[B,C,T,H,W]``.

    ``reconstruction`` must be the raw network output, whose channel 2 is a mask
    logit, because that channel goes through :func:`validity_from_mask_logits`.
    A clip that came out of ``decode()`` is not such an input: channel 2 there is
    already resolved to ``{0, 1}`` (or to a probability under ``apply_mask=False``),
    and sigmoid of either always clears the cut, so every ray would read as kept.
    Use :func:`validity_from_video` for a decoded clip, or pass the mask through
    ``validity`` here, which is taken as a probability and only cut.

    One-channel V0 reconstructions carry no mask channel and fall back to the
    range sentinel.
    """
    if validity is not None:
        return validity_mask(validity)
    if reconstruction.shape[1] >= 3:
        return validity_from_mask_logits(reconstruction[:, 2:3])
    return validity_from_range_sentinel(
        reconstruction[:, :1],
        mask_input_min_value=mask_input_min_value,
        mask_input_eps=mask_input_eps,
    )


def as_lidar_clip(value: torch.Tensor, *, name: str = "LiDAR") -> torch.Tensor:
    """Peel a packed ``[1,C,T,H,W]`` batch down to ``[C,T,H,W]`` with ``C`` in ``{1, 2, 3}``.

    ``C=1`` is legacy V0 range-only. ``C=2`` is V0 range plus an empty intensity
    channel. ``C=3`` is V1 metric range, intensity, and mask.
    """
    sample = value.detach().float().cpu()
    if sample.ndim == 5:
        if sample.shape[0] != 1:
            raise ValueError(f"Expected one LiDAR sample, got {name} shape {tuple(sample.shape)}")
        sample = sample[0]
    if sample.ndim != 4 or sample.shape[0] not in (1, 2, 3):
        raise ValueError(f"Expected {name} shape [1|2|3,T,H,W] or [1,1|2|3,T,H,W], got {tuple(sample.shape)}")
    return sample


def prepare_range_for_display(
    lidar: torch.Tensor,
    *,
    min_range: float | None = None,
    max_range: float | None = None,
    smoothing: RangeSmoothing | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, float, float]:
    """Turn a V0 or V1 clip into normalized range, validity, and colour bounds.

    Both versions colorize over ``[MIN_RANGE_METERS, MAX_RANGE_METERS]`` (5--100 m),
    the Drive-Dreams span the V0 tokenizer and V1 transfer recipes use.

    A clip with fewer than three channels is still in the network's normalized space
    with dropped rays at ``-1``, so validity is left to the renderer to read off that
    sentinel. That covers the legacy single-channel V0 prepare and a tokenizer-training
    reconstruction; channel 1, when present, is an empty intensity slot, ignored here.

    A three-channel clip is metric range, unit intensity, and a mask -- what the
    dataloader prepares and what both tokenizers' ``decode()`` returns. Channel 2 is
    ``{0, 1}`` on a dataloader clip and on a decoded one alike, so a plain half cut
    reads it;
    when a caller decodes with ``apply_mask=False`` the channel holds the
    probability instead and the same half cut reproduces the default. A ray is
    kept only when its metric range is also positive, because the tokenizer
    zeroes the range of a dropped ray and renormalizing that zero would paint a
    centimetre ghost at the near end of the colormap.

    ``smoothing`` filters the metric range against the resolved mask before
    normalizing, so a caller that unprojects the result gets the filtered
    geometry. It needs both channels and is therefore ignored on a clip with
    fewer than three, where the mask is left to the renderer's sentinel read.
    """
    clip = as_lidar_clip(lidar, name="LiDAR range")
    lo = float(MIN_RANGE_METERS if min_range is None else min_range)
    hi = float(MAX_RANGE_METERS if max_range is None else max_range)
    if hi <= lo:
        raise ValueError(f"max_range must exceed min_range, got {lo} and {hi}")
    if clip.shape[0] < 3:
        return clip[0], None, lo, hi

    metric = clip[0]
    valid = (metric > 0.0) & validity_mask(clip[2])
    if smoothing is not None:
        metric, valid = smooth_metric_range(metric, valid, smoothing=smoothing)
    normalized = 2.0 * (metric - lo) / (hi - lo) - 1.0
    return normalized, valid, lo, hi
