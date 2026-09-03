# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
FLOP calculator for Qwen3VL dynamic batching.

This module computes theoretical FLOPs for Qwen3VL samples to enable
FLOP-based batching (instead of token-based batching).

Key insight: Runtime scales linearly with FLOPs based on fitted curve from benchmarks.
"""

import os
from typing import Any

import torch

from cosmos_framework.tools.flops.qwen3_vl import compute_qwen3vl_flops_from_config, compute_vision_encoder_flops
from cosmos_framework.data.generator.packing_iterable_dataset import round_up_to_multiple


class FlopCalculator:
    """Calculate theoretical FLOPs for Qwen3VL samples."""

    def __init__(
        self,
        config: Any,
        batch_multiplier: float = 3.0,
        fitted_slope: float = 5.078355e-12,  # ms/FLOP from fitted curve
        fitted_intercept: float = 133.88,  # ms from fitted curve
        is_causal: bool | None = None,
        calibration_is_causal: bool | None = None,
        exact_visual_flops: bool | None = None,
        true_packing: bool | None = None,
        calibration_true_packing: bool | None = None,
    ) -> None:
        """
        Initialize FLOP calculator.

        Args:
            config: Qwen3VLConfig object or dict with model parameters
            batch_multiplier: Multiplier for forward+backward pass (default: 3.0)
                             forward = 1x, backward = 2x, total = 3x
            fitted_slope: Slope from runtime_ms vs flops fitted curve (ms/FLOP)
            fitted_intercept: Intercept from fitted curve (ms)
            is_causal: Whether to count the text decoder's S^2 attention terms causally
                (half work). The shipped ``fitted_slope`` / ``fitted_intercept`` were
                calibrated against ``is_causal=False`` (the bidirectional upper bound), so
                flipping this WITHOUT refitting makes the runtime estimator underestimate
                per-sample work and the batcher overfill (cost model). Defaults (None) resolve from
                ``COSMOS_PACK_IS_CAUSAL`` then to ``False`` to preserve the calibrated default;
                the validation harness flips it (paired with a slope/intercept refit) per arm.
            calibration_is_causal: Whether ``fitted_slope``/``fitted_intercept`` were fit with
                causal FLOP accounting. Defaults to False for the shipped legacy calibration and
                must match ``is_causal`` to prevent silent over-admission.
            exact_visual_flops: When True, ``compute_batch_flops`` sums the vision-encoder FLOPs
                over each sample's ACTUAL patch count instead of averaging then truncating with
                ``int()`` (cost model: the truncated average drops fractional visual tokens and biases the
                vision term low). Defaults (None) resolve from ``COSMOS_PACK_EXACT_VISUAL_FLOPS``
                then to ``False`` to preserve the calibrated default.
            true_packing: When True, ``compute_batch_flops`` models a true-packed row (samples
                concatenated, no inter-sample padding, block-diagonal causal attention) as the sum
                of logical-segment work plus one shared FP8 alignment-tail segment, instead of the
                padded model ``full_flops(max_len) * batch_size``. This accounts for linear work over
                ``round16(sum(L_i))`` and attention work over ``sum(L_i^2) + tail^2`` with no
                cross-segment terms. Without this, the padded FLOP gate grows ~linearly in batch size
                and would cap true packing at the padded sample count, hiding the throughput win.
                Defaults (None) resolve from ``COSMOS_PACK_TRUE_PACKING`` then to ``False``.
            calibration_true_packing: Layout used to fit ``fitted_slope`` / ``fitted_intercept``.
                Must match ``true_packing``. Defaults (None) resolve from
                ``COSMOS_PACK_CALIBRATION_TRUE_PACKING`` then to ``False`` so the shipped padded
                calibration cannot silently size true-packed batches.

        Fitted curve from benchmarks (R² = 0.9460):
            runtime_ms = 5.078355e-12 * flops + 133.88
        """
        self.config = config
        self.batch_multiplier = batch_multiplier
        self.fitted_slope = fitted_slope
        self.fitted_intercept = fitted_intercept
        self.is_causal: bool = (
            os.environ.get("COSMOS_PACK_IS_CAUSAL", "0") == "1" if is_causal is None else bool(is_causal)
        )
        self.calibration_is_causal: bool = (
            os.environ.get("COSMOS_PACK_CALIBRATION_IS_CAUSAL", "0") == "1"
            if calibration_is_causal is None
            else bool(calibration_is_causal)
        )
        if self.calibration_is_causal != self.is_causal:
            raise ValueError(
                "calibration_is_causal must match is_causal; causal FLOP accounting requires a paired fitted calibration"
            )
        self.exact_visual_flops: bool = (
            os.environ.get("COSMOS_PACK_EXACT_VISUAL_FLOPS", "0") == "1"
            if exact_visual_flops is None
            else bool(exact_visual_flops)
        )
        self.true_packing: bool = (
            os.environ.get("COSMOS_PACK_TRUE_PACKING", "0") == "1" if true_packing is None else bool(true_packing)
        )
        self.calibration_true_packing: bool = (
            os.environ.get("COSMOS_PACK_CALIBRATION_TRUE_PACKING", "0") == "1"
            if calibration_true_packing is None
            else bool(calibration_true_packing)
        )
        if self.true_packing and not self.exact_visual_flops:
            raise ValueError("true-packing FLOP admission requires exact_visual_flops=True")
        if self.true_packing != self.calibration_true_packing:
            raise ValueError(
                "calibration_true_packing must match true_packing; do not reuse padded runtime fits "
                "for the block-diagonal/shared-tail layout"
            )

        # Extract config parameters
        if hasattr(config, "vision_config"):
            self.spatial_merge_size = config.vision_config.spatial_merge_size
        elif isinstance(config, dict):
            self.spatial_merge_size = config.get("spatial_merge_size", 2)
        else:
            self.spatial_merge_size = 2

    def _grid_segment_patch_counts(self, grid_thw: Any) -> tuple[int, ...]:
        """Return per-temporal-segment patch counts matching Qwen vision-varlen boundaries."""
        if isinstance(grid_thw, torch.Tensor):
            grid = grid_thw.reshape(-1, 3)  # [N_media,3]
            temporal = grid[:, 0].to(torch.int64)  # [N_media]
            spatial = (grid[:, 1] * grid[:, 2]).to(torch.int64)  # [N_media]
            segment_patches = torch.repeat_interleave(spatial, temporal)  # [N_segments]
            return tuple(int(value) for value in segment_patches.cpu().tolist())  # [N_segments]

        import numpy as np

        grid = np.asarray(grid_thw).reshape(-1, 3)  # [N_media,3]
        counts: list[int] = []
        for temporal, height, width in grid:
            counts.extend([int(height * width)] * int(temporal))
        return tuple(counts)

    def get_vision_segment_patches(self, sample: dict[str, Any]) -> tuple[int, ...]:
        """Collect image and video patch counts without collapsing independent attention segments."""
        counts: list[int] = []
        if "image_grid_thw" in sample:
            counts.extend(self._grid_segment_patch_counts(sample["image_grid_thw"]))
        elif "pixel_values" in sample:
            counts.append(int(sample["pixel_values"].shape[0]))

        if "video_grid_thw" in sample:
            counts.extend(self._grid_segment_patch_counts(sample["video_grid_thw"]))
        elif "pixel_values_videos" in sample:
            counts.append(int(sample["pixel_values_videos"].shape[0]))
        return tuple(counts)

    def get_num_visual_tokens(self, sample: dict[str, Any]) -> int:
        """
        Extract number of visual tokens from sample.

        Args:
            sample: Data sample containing visual information

        Returns:
            Number of visual tokens (after spatial merging)
        """
        merge_area = self.spatial_merge_size**2
        return sum(num_patches // merge_area for num_patches in self.get_vision_segment_patches(sample))

    def get_num_patches(self, sample: dict[str, Any]) -> int:
        """
        Extract number of patches from sample.

        Args:
            sample: Data sample containing visual information

        Returns:
            Number of patches (before spatial merging)
        """
        return sum(self.get_vision_segment_patches(sample))

    def sample_stats(self, sample: dict[str, Any]) -> tuple[int, int, tuple[int, ...]]:
        """Return ``(total_tokens, visual_tokens, segment_patches)`` for one sample.

        Pulled out so callers (the dynamic batcher) can cache these per pooled sample and avoid
        recomputing the tensor reductions in ``get_num_visual_tokens`` / ``get_num_patches`` for
        every candidate on every admit (stats cache).
        """
        segment_patches = self.get_vision_segment_patches(sample)
        merge_area = self.spatial_merge_size**2
        visual_tokens = sum(num_patches // merge_area for num_patches in segment_patches)
        return (len(sample["input_ids"]), visual_tokens, segment_patches)

    def compute_vision_flops(self, num_patches: int) -> int:
        """Compute only the vision-tower FLOPs for one sample without rebuilding full-model statistics."""
        vision_config = self.config.vision_config
        return compute_vision_encoder_flops(
            num_patches=num_patches,
            vision_hidden_size=vision_config.hidden_size,
            vision_intermediate_size=vision_config.intermediate_size,
            vision_num_heads=vision_config.num_heads,
            num_vision_layers=vision_config.depth,
            out_hidden_size=self.config.text_config.hidden_size,
            patch_size=getattr(vision_config, "patch_size", 16),
            temporal_patch_size=getattr(vision_config, "temporal_patch_size", 2),
            spatial_merge_size=vision_config.spatial_merge_size,
            in_channels=getattr(vision_config, "in_channels", 3),
        )

    def compute_single_sample_flops(self, sample: dict[str, Any]) -> float:
        """
        Compute FLOPs for single sample (batch_size=1).

        Args:
            sample: Data sample

        Returns:
            Total FLOPs for forward + backward pass
        """

        if self.exact_visual_flops:
            return self.compute_batch_flops(stats=[self.sample_stats(sample)])

        total_tokens = round_up_to_multiple(len(sample["input_ids"]))
        num_visual_tokens = self.get_num_visual_tokens(sample)
        num_patches = self.get_num_patches(sample)

        result = compute_qwen3vl_flops_from_config(
            self.config,
            total_tokens=total_tokens,
            visual_tokens=num_visual_tokens,
            num_patches=num_patches,
            is_causal=self.is_causal,
        )

        # Return total FLOPs including forward + backward
        return result["total_flops"] * self.batch_multiplier

    def compute_batch_flops(
        self,
        samples: list[dict[str, Any]] | None = None,
        stats: list[tuple[int, int, tuple[int, ...]]] | None = None,
    ) -> float:
        """
        Compute FLOPs for a batch of samples.

        Key insight (padded path): In a padded batch, all samples are padded to max sequence
        length, so the text decoder / LM-head cost scales with ``max(sequence_lengths)``; the
        vision encoder runs per sample on its own (unpadded) patches.

        Key insight (true-packing path): a true-packed row concatenates the samples with no
        inter-sample padding and uses block-diagonal causal attention. Its cost is the sum of each
        logical segment plus one shared FP8 alignment-tail segment: linear work over
        ``round16(sum(L_i))`` and attention over ``sum(L_i^2) + tail^2``. See ``true_packing`` in
        ``__init__`` and true packing.

        Args:
            samples: list of samples in batch. Optional if ``stats`` is supplied.
            stats: optional precomputed ``[(total_tokens, visual_tokens, segment_patches), ...]`` (one
                per sample) so the caller can reuse a per-sample cache (stats cache) instead of re-running
                the tensor reductions here.

        Returns:
            Total FLOPs for forward + backward pass on this batch
        """
        if stats is None:
            if not samples:
                return 0.0
            stats = [self.sample_stats(s) for s in samples]
        if not stats:
            return 0.0

        if self.true_packing:
            # The decoder sees one row but block-diagonal attention makes its work the sum of the
            # logical segments. The FP8 alignment tail is its own inert segment in the collate, so
            # include it explicitly instead of rounding every sample independently.
            packed_flops = 0.0
            for total_tokens, visual_tokens, segment_patches in stats:
                text_result = compute_qwen3vl_flops_from_config(
                    self.config,
                    total_tokens=total_tokens,
                    visual_tokens=visual_tokens,
                    num_patches=None,
                    is_causal=self.is_causal,
                )
                packed_flops += text_result["total_flops"]
                packed_flops += sum(self.compute_vision_flops(num_patches) for num_patches in segment_patches)

            tail_tokens = round_up_to_multiple(sum(total_tokens for total_tokens, _, _ in stats)) - sum(
                total_tokens for total_tokens, _, _ in stats
            )
            if tail_tokens:
                tail_result = compute_qwen3vl_flops_from_config(
                    self.config,
                    total_tokens=tail_tokens,
                    visual_tokens=0,
                    num_patches=None,
                    is_causal=self.is_causal,
                )
                packed_flops += tail_result["total_flops"]
            return packed_flops * self.batch_multiplier

        # Find max sequence length (determines padding for the decoder / LM head).
        max_total_tokens = round_up_to_multiple(max(t for t, _, _ in stats))
        total_visual_tokens = sum(v for _, v, _ in stats)
        total_num_patches = sum(sum(segment_patches) for _, _, segment_patches in stats)
        batch_size = len(stats)
        avg_visual_tokens = total_visual_tokens / batch_size
        avg_num_patches = total_num_patches / batch_size

        if self.exact_visual_flops:
            # Cost model: decoder + LM head + embeddings at the padded length (num_patches=None drops the
            # vision term), times batch_size -> the true padded text cost. The vision encoder is
            # the SUM of each sample's real patch work (no average, no int() truncation).
            text_result = compute_qwen3vl_flops_from_config(
                self.config,
                total_tokens=max_total_tokens,
                visual_tokens=int(avg_visual_tokens),
                num_patches=None,
                is_causal=self.is_causal,
            )
            vision_flops = 0.0
            for _, _, segment_patches in stats:
                vision_flops += sum(self.compute_vision_flops(num_patches) for num_patches in segment_patches)
            return (text_result["total_flops"] * batch_size + vision_flops) * self.batch_multiplier

        # Legacy (calibrated) path: full model FLOPs at (max_total_tokens, truncated-avg visual),
        # scaled by batch size and the forward+backward multiplier.
        result = compute_qwen3vl_flops_from_config(
            self.config,
            total_tokens=max_total_tokens,
            visual_tokens=int(avg_visual_tokens),
            num_patches=int(avg_num_patches),
            is_causal=self.is_causal,
        )
        return result["total_flops"] * batch_size * self.batch_multiplier

    def estimate_runtime_ms(self, flops: float) -> float:
        """
        Estimate runtime in milliseconds based on fitted curve.

        Args:
            flops: Theoretical FLOPs

        Returns:
            Estimated runtime in milliseconds

        Fitted curve (R² = 0.9460):
            runtime_ms = 5.078355e-12 * flops + 133.88
        """
        return self.fitted_slope * flops + self.fitted_intercept

    def compute_max_flops_for_runtime(self, target_runtime_seconds: float) -> float:
        """
        Compute maximum FLOPs for a target runtime.

        Args:
            target_runtime_seconds: Target runtime in seconds

        Returns:
            Maximum FLOPs to stay within runtime budget

        Solves: target_runtime_ms = fitted_slope * max_flops + fitted_intercept
                max_flops = (target_runtime_ms - fitted_intercept) / fitted_slope
        """
        target_runtime_ms = target_runtime_seconds * 1000
        max_flops = (target_runtime_ms - self.fitted_intercept) / self.fitted_slope
        return max_flops
