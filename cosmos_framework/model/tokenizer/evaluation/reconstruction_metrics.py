# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Metric computation for tokenizer evaluation.

This module provides metrics for evaluating tokenizer quality:
    - PSNRMetric: Peak signal-to-noise ratio (using torchmetrics)
    - SSIMMetric: Structural similarity index (using torchmetrics)
    - LPIPSMetric: Learned perceptual image patch similarity
    - TokenizerMetric: Composite metric that includes codebook usage via compute_codebook_usage
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

# Import torchmetrics for SSIM and LPIPS
try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False

# Standard batch keys
INPUT_KEY = "inputs"  # [0, 1] range for PSNR/SSIM
RECON_KEY = "reconstructions"  # [0, 1] range for PSNR/SSIM
INPUT_RAW_KEY = "inputs_raw"  # [-1, 1] range for LPIPS
RECON_RAW_KEY = "reconstructions_raw"  # [-1, 1] range for LPIPS


class TokenizerMetric(nn.Module):
    """Composite metric module for tokenizer evaluation.

    Combines multiple metrics and computes them in a single forward pass.

    Args:
        compute_psnr: Whether to compute PSNR.
        compute_ssim: Whether to compute SSIM.
        compute_lpips: Whether to compute LPIPS.
        compute_code_usage: Whether to compute codebook usage.
    """

    def __init__(
        self,
        compute_psnr: bool = True,
        compute_ssim: bool = True,
        compute_lpips: bool = False,
        compute_code_usage: bool = False,
        num_codes: int = 65536,
    ) -> None:
        super().__init__()
        self.compute_psnr = compute_psnr
        self.compute_ssim = compute_ssim
        self.compute_lpips = compute_lpips
        self.compute_code_usage = compute_code_usage
        self.num_codes = num_codes

        if compute_psnr:
            self.psnr = PSNRMetric()
        if compute_ssim:
            self.ssim = SSIMMetric()
        if compute_lpips:
            self.lpips = LPIPSMetric()

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        iteration: int,
    ) -> dict[str, Any]:
        """Compute all enabled metrics.

        Args:
            inputs: Input batch with original images/videos. Should contain:
                - "inputs": [0, 1] range for PSNR/SSIM
                - "inputs_raw": [-1, 1] range for LPIPS
            output_batch: Output batch with reconstructions. Should contain:
                - "reconstructions": [0, 1] range for PSNR/SSIM
                - "reconstructions_raw": [-1, 1] range for LPIPS
            iteration: Current iteration.

        Returns:
            Dictionary of metric values. PSNR/SSIM/LPIPS return dicts with 'sum' and 'count'
            for proper distributed averaging.
        """
        metrics = {}

        # [0, 1] range data for PSNR/SSIM
        original = inputs.get(INPUT_KEY)
        recon = output_batch.get(RECON_KEY)

        # [-1, 1] range data for LPIPS
        original_raw = inputs.get(INPUT_RAW_KEY)
        recon_raw = output_batch.get(RECON_RAW_KEY)

        if original is None or recon is None:
            return metrics

        if self.compute_psnr:
            metrics["psnr"] = self.psnr(original, recon)

        if self.compute_ssim:
            metrics["ssim"] = self.ssim(original, recon)

        if self.compute_lpips:
            # Use [-1, 1] range data for LPIPS
            # Fall back to converting [0, 1] to [-1, 1] if raw data not available
            if original_raw is not None and recon_raw is not None:
                metrics["lpips"] = self.lpips(original_raw, recon_raw)
            else:
                # Convert [0, 1] to [-1, 1] if raw data not provided
                original_lpips = original * 2.0 - 1.0
                recon_lpips = recon * 2.0 - 1.0
                metrics["lpips"] = self.lpips(original_lpips, recon_lpips)

        if self.compute_code_usage:
            quant_info = output_batch.get("quant_info")
            if quant_info is not None:
                indices = quant_info.get("indices")
                if indices is not None:
                    from cosmos_framework.model.tokenizer.evaluation.metric import compute_codebook_usage

                    code_stats = compute_codebook_usage(indices, self.num_codes)
                    metrics["code_perplexity"] = code_stats["perplexity"]
                    metrics["code_active_ratio"] = code_stats["active_ratio"]
                    metrics["code_active_count"] = code_stats["active_codes"]

        return metrics


class PSNRMetric(nn.Module):
    """Peak Signal-to-Noise Ratio metric.

    Computes PSNR between original and reconstructed images.
    Expects inputs in [0, 1] range (already normalized by caller).

    Uses per-sample MSE calculation on uint8 [0, 255] range:
    - Convert [0, 1] float to [0, 255] uint8
    - Compute MSE per sample on uint8 values (average over C, H, W dimensions)
    - Compute PSNR per sample with max_val=255
    - Return dict with sum and count for proper distributed averaging
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
        """Compute PSNR between original and reconstructed tensors.

        Args:
            original: Original tensor in [0, 1] range. Shape: (B, C, H, W) or (B, T, C, H, W).
            reconstructed: Reconstructed tensor in [0, 1] range.

        Returns:
            Dict with 'sum' (sum of per-sample PSNRs) and 'count' (number of samples)
            for proper distributed averaging.
        """
        # Handle video format by flattening batch and time dimensions
        if original.dim() == 5:  # (B, T, C, H, W)
            b, t, c, h, w = original.shape
            original = original.reshape(b * t, c, h, w)
            reconstructed = reconstructed.reshape(b * t, c, h, w)

        # Convert to uint8 [0, 255] range
        original_uint8 = (original.clamp(0, 1) * 255).byte()
        reconstructed_uint8 = (reconstructed.clamp(0, 1) * 255).byte()

        # Compute per-sample MSE on uint8 values (as float for precision)
        mse = torch.mean((original_uint8.float() - reconstructed_uint8.float()) ** 2, dim=[1, 2, 3])  # (B,)

        # Handle zero MSE (identical images) - use max PSNR of 100 dB
        max_psnr = 100.0
        mse = torch.where(
            mse == 0,
            torch.tensor(10.0 ** (-max_psnr / 10.0) * 255.0 * 255.0, device=mse.device, dtype=mse.dtype),
            mse,
        )

        # Compute PSNR per sample with max_val=255
        psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))

        # Return sum and count for proper distributed averaging
        return {"sum": psnr.sum().item(), "count": psnr.shape[0]}


class SSIMMetric(nn.Module):
    """Structural Similarity Index metric.

    Uses torchmetrics for SSIM computation.
    Expects inputs in [0, 1] range (already normalized by caller).
    """

    def __init__(self) -> None:
        super().__init__()
        if HAS_TORCHMETRICS:
            # data_range=1.0 for [0, 1] normalized images
            self._ssim_metric = StructuralSimilarityIndexMeasure(
                data_range=1.0,
                sync_on_compute=False,
                dist_sync_on_step=False,
            )
        else:
            self._ssim_metric = None

    def forward(self, original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
        """Compute SSIM between original and reconstructed tensors.

        Args:
            original: Original tensor in [0, 1] range. Shape: (B, C, H, W) or (B, T, C, H, W).
            reconstructed: Reconstructed tensor in [0, 1] range.

        Returns:
            Dict with 'sum' (sum of per-sample SSIMs) and 'count' (number of samples)
            for proper distributed averaging.
        """
        if not HAS_TORCHMETRICS or self._ssim_metric is None:
            return {"sum": 0.0, "count": 0}

        # Handle video by flattening temporal dimension
        if original.dim() == 5:  # B, T, C, H, W
            b, t, c, h, w = original.shape
            original = original.reshape(b * t, c, h, w)
            reconstructed = reconstructed.reshape(b * t, c, h, w)

        # Clamp to [0, 1] range and convert to float32 for SSIM computation
        original = original.clamp(0, 1).float()
        reconstructed = reconstructed.clamp(0, 1).float()

        batch_size = original.shape[0]

        # Move metric to correct device
        self._ssim_metric = self._ssim_metric.to(original.device)

        # Reset metric state before computing to avoid accumulation from previous calls
        self._ssim_metric.reset()

        # Compute SSIM for each sample individually to get per-sample values
        # We need to reset between samples to avoid state accumulation
        ssim_sum = 0.0
        for i in range(batch_size):
            orig_i = original[i : i + 1]
            recon_i = reconstructed[i : i + 1]
            # Update with single sample
            self._ssim_metric.update(recon_i, orig_i)
            # Compute returns the value for accumulated samples (just 1 here)
            ssim_val = self._ssim_metric.compute()
            ssim_sum += ssim_val.item()
            # Reset for next sample to avoid accumulation
            self._ssim_metric.reset()

        # Return sum and count for proper distributed averaging
        return {"sum": ssim_sum, "count": batch_size}


class LPIPSMetric(nn.Module):
    """Learned Perceptual Image Patch Similarity metric.

    Uses torchmetrics LPIPS with VGG backbone.
    Expects inputs in [-1, 1] range for LPIPS computation.
    Note: The forward() method expects [-1, 1] range directly (no conversion needed).
    """

    def __init__(self, net_type: str = "vgg") -> None:
        super().__init__()
        if HAS_TORCHMETRICS:
            # LPIPS expects inputs in [-1, 1] range
            self._lpips_metric = LearnedPerceptualImagePatchSimilarity(
                net_type=net_type,
                sync_on_compute=False,
                dist_sync_on_step=False,
            )
        else:
            self._lpips_metric = None

    def forward(self, original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
        """Compute LPIPS between original and reconstructed tensors.

        Args:
            original: Original tensor in [-1, 1] range. Shape: (B, C, H, W) or (B, T, C, H, W).
            reconstructed: Reconstructed tensor in [-1, 1] range.

        Returns:
            Dict with 'sum' (sum of per-sample LPIPS) and 'count' (number of samples)
            for proper distributed averaging.
        """
        if not HAS_TORCHMETRICS or self._lpips_metric is None:
            return {"sum": 0.0, "count": 0}

        # Handle video by flattening temporal dimension
        if original.dim() == 5:  # B, T, C, H, W
            b, t, c, h, w = original.shape
            original = original.reshape(b * t, c, h, w)
            reconstructed = reconstructed.reshape(b * t, c, h, w)

        # LPIPS expects [-1, 1] range - clamp and convert to float32
        original_lpips = original.clamp(-1.0, 1.0).float()
        reconstructed_lpips = reconstructed.clamp(-1.0, 1.0).float()

        batch_size = original.shape[0]

        # Move metric to correct device
        self._lpips_metric = self._lpips_metric.to(original.device)

        # Reset metric state before computing to avoid accumulation from previous calls
        self._lpips_metric.reset()

        # Compute LPIPS for each sample individually
        lpips_sum = 0.0
        for i in range(batch_size):
            orig_i = original_lpips[i : i + 1]
            recon_i = reconstructed_lpips[i : i + 1]
            # Update with single sample
            self._lpips_metric.update(recon_i, orig_i)
            # Compute returns the value for accumulated samples (just 1 here)
            lpips_val = self._lpips_metric.compute()
            lpips_sum += lpips_val.item()
            # Reset for next sample to avoid accumulation
            self._lpips_metric.reset()

        # Return sum and count for proper distributed averaging
        return {"sum": lpips_sum, "count": batch_size}


def calculate_psnr(
    original: torch.Tensor | list[torch.Tensor],
    reconstructed: torch.Tensor | list[torch.Tensor],
) -> torch.Tensor:
    """Calculate PSNR between two tensors or lists of tensors.

    This is a standalone function for use in evaluation and training logging.
    Expects inputs already in [0, 1] range. Converts to uint8 [0, 255] internally.

    Supports multiple input formats:
    - Lists of tensors (variable-size images from sparse_to_img_list)
    - 5D tensors (B, T, C, H, W) for video
    - 4D tensors (B, C, H, W) for batched images
    - 3D tensors (C, H, W) for single images

    Args:
        original: Original image(s) in [0, 1] range. Can be tensor or list of tensors.
        reconstructed: Reconstructed image(s) in [0, 1] range. Must match original format.

    Returns:
        PSNR value as a tensor (scalar, for distributed gathering).
    """
    # Handle lists of tensors (from sparse_to_img_list)
    if isinstance(original, list) and isinstance(reconstructed, list):
        if len(original) != len(reconstructed):
            raise ValueError(f"Image lists must have the same length. Got {len(original)} and {len(reconstructed)}")

        psnr_values = []
        for orig, rec in zip(original, reconstructed):
            psnr_values.append(calculate_psnr(orig, rec))

        # Average PSNR across all images
        return sum(psnr_values) / len(psnr_values)

    # At this point, both should be tensors
    if original.shape != reconstructed.shape:
        raise ValueError(f"Images must have the same shape. Got {original.shape} and {reconstructed.shape}")

    # Handle 3D tensor (C, H, W) - add batch dimension
    if original.dim() == 3:
        original = original.unsqueeze(0)
        reconstructed = reconstructed.unsqueeze(0)

    # Handle 5D tensor (B, T, C, H, W) - flatten batch and time
    if original.dim() == 5:
        b, t = original.shape[:2]
        original = original.reshape(b * t, *original.shape[2:])
        reconstructed = reconstructed.reshape(b * t, *reconstructed.shape[2:])

    # Now we have 4D tensors (B, C, H, W)
    # Convert to uint8 [0, 255] range
    original_uint8 = (original.detach().clamp(0, 1) * 255).byte()
    reconstructed_uint8 = (reconstructed.detach().clamp(0, 1) * 255).byte()

    # Compute MSE per sample on uint8 values
    mse = torch.mean((original_uint8.float() - reconstructed_uint8.float()) ** 2, dim=[1, 2, 3])

    # Handle zero MSE (identical images) - cap at 100 dB
    max_psnr = 100.0
    mse = torch.where(
        mse == 0,
        torch.tensor(10.0 ** (-max_psnr / 10.0) * 255.0 * 255.0, device=mse.device, dtype=mse.dtype),
        mse,
    )

    # Compute PSNR with max_val=255
    psnr = 20 * torch.log10(torch.tensor(255.0, device=mse.device, dtype=mse.dtype)) - 10 * torch.log10(mse)

    # Return mean PSNR
    return psnr.mean()


class Rank0FIDMetric(nn.Module):
    """FID metric that runs only on rank 0 to avoid distributed sync issues.

    Uses torchmetrics FrechetInceptionDistance internally but only computes
    on rank 0's data to avoid NCCL collective operation mismatches caused by
    torchmetrics/torch-fidelity's internal distributed synchronization.

    Note: FID is computed only on rank 0's portion of the data (1/world_size),
    which may be less representative than full dataset FID, but avoids
    distributed synchronization issues.

    Usage:
        fid = Rank0FIDMetric(rank=rank).to(device)

        # During evaluation loop (only rank 0 updates)
        for batch in dataloader:
            fid.update(real_images, fake_images)

        # Compute FID (only rank 0 has valid result)
        if rank == 0:
            fid_value = fid.compute()
    """

    def __init__(self, rank: int = 0, feature_dim: int = 2048) -> None:
        super().__init__()
        self.rank = rank
        self.feature_dim = feature_dim
        self._fid_metric = None

        # Only initialize FID metric on rank 0
        if self.rank == 0:
            try:
                from torchmetrics.image.fid import FrechetInceptionDistance

                # normalize=True means input is [0, 1] float, not uint8
                self._fid_metric = FrechetInceptionDistance(
                    feature=feature_dim,
                    normalize=True,
                    sync_on_compute=False,
                    dist_sync_on_step=False,
                )
            except ImportError:
                pass

    @torch.no_grad()
    def update(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> None:
        """Update FID statistics with a batch of real and fake images.

        Only updates on rank 0.

        Args:
            real_images: Real images in [0, 1] range, shape (B, C, H, W) or (B, T, C, H, W)
            fake_images: Fake/reconstructed images in [0, 1] range
        """
        if self.rank != 0 or self._fid_metric is None:
            return

        # Handle video format by flattening batch and time dimensions
        if real_images.dim() == 5:  # (B, T, C, H, W)
            real_images = real_images.reshape(-1, *real_images.shape[2:])
            fake_images = fake_images.reshape(-1, *fake_images.shape[2:])

        # Move metric to same device as images
        device = real_images.device
        self._fid_metric = self._fid_metric.to(device)

        # torchmetrics FID update
        self._fid_metric.update(real_images, real=True)
        self._fid_metric.update(fake_images, real=False)

    def compute(self) -> torch.Tensor:
        """Compute FID from accumulated statistics.

        Only valid on rank 0.

        Returns:
            FID value as a scalar tensor (inf if not rank 0 or metric unavailable)
        """
        if self.rank != 0 or self._fid_metric is None:
            return torch.tensor(float("inf"))

        return self._fid_metric.compute()

    def reset(self) -> None:
        """Reset accumulated statistics."""
        if self._fid_metric is not None:
            self._fid_metric.reset()




__all__ = [
    "TokenizerMetric",
    "PSNRMetric",
    "SSIMMetric",
    "LPIPSMetric",
    "Rank0FIDMetric",
    "calculate_psnr",
]
