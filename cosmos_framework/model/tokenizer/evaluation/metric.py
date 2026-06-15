# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Standalone metric computation functions.

This module provides metric computation for tokenizer evaluation:
    - compute_psnr: Peak signal-to-noise ratio
    - compute_fid: Frechet Inception Distance (using torchmetrics)
    - compute_imagenet_accuracy: ImageNet zero-shot classification accuracy
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any

import torch
from loguru import logger as logging


def compute_psnr(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    max_value: float = 1.0,
) -> float:
    """Compute PSNR between original and reconstructed tensors.

    Args:
        original: Original tensor in [0, 1] range (or [0, max_value]).
            Shape: (B, C, H, W) or (C, H, W), or list of tensors.
        reconstructed: Reconstructed tensor in same range.
        max_value: Maximum value of the pixels (1.0 for normalized images).

    Returns:
        PSNR value in dB.
    """
    # Handle list inputs
    if isinstance(original, list) and isinstance(reconstructed, list):
        if len(original) != len(reconstructed):
            raise ValueError(f"Image lists must have same length. Got {len(original)} and {len(reconstructed)}")
        psnr_values = [compute_psnr(orig, recon, max_value) for orig, recon in zip(original, reconstructed)]
        return sum(psnr_values) / len(psnr_values)

    # Ensure same shape
    if original.shape != reconstructed.shape:
        raise ValueError(f"Images must have same shape. Got {original.shape} and {reconstructed.shape}")

    # Add batch dimension if not present
    if len(original.shape) == 3:
        original = original.unsqueeze(0)
        reconstructed = reconstructed.unsqueeze(0)

    # Calculate MSE per image in batch (average over C, H, W)
    mse = torch.mean(
        (original.detach() - reconstructed.detach()) ** 2,
        dim=[1, 2, 3],
    )

    # Handle identical images (return 100.0 dB as maximum)
    if torch.any(mse == 0):
        max_psnr = 100.0
        mse = torch.where(
            mse == 0,
            torch.tensor(10.0 ** (-max_psnr / 10.0), device=mse.device),
            mse,
        )

    # Calculate PSNR
    psnr = 20 * torch.log10(torch.tensor(max_value, device=mse.device) / torch.sqrt(mse))

    # Return mean PSNR if batch size > 1
    return psnr.mean().item() if psnr.shape[0] > 1 else psnr[0].item()


@torch.no_grad()
def compute_imagenet_accuracy(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    labels: torch.Tensor,
    logit_scale: float = 100.0,
    logit_bias: float | None = None,
    top_k: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """Compute ImageNet zero-shot classification accuracy.

    Args:
        image_features: Image features of shape (N, D), L2-normalized.
        text_features: Text features for class templates of shape (num_classes, D), L2-normalized.
        labels: Ground truth labels of shape (N,).
        logit_scale: Logit scaling factor.
        logit_bias: Optional logit bias.
        top_k: Tuple of k values for top-k accuracy.

    Returns:
        Dictionary with top-k accuracies.
    """
    # Compute logits
    logits = logit_scale * image_features @ text_features.T
    if logit_bias is not None:
        logits = logits + logit_bias

    # Compute top-k accuracy
    results = {}
    for k in top_k:
        _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(labels.view(-1, 1).expand_as(pred))
        accuracy = correct.float().sum() / labels.numel()
        results[f"top{k}_acc"] = accuracy.item() * 100

    return results


@torch.no_grad()
def compute_codebook_usage(
    indices: torch.Tensor,
    num_codes: int = 65536,
) -> dict[str, float]:
    """Compute codebook usage statistics.

    Args:
        indices: Quantized code indices.
        num_codes: Total number of codes in codebook.

    Returns:
        Dictionary with usage statistics.
    """
    # Flatten indices
    flat_indices = indices.flatten().long()

    # Handle empty indices
    if flat_indices.numel() == 0:
        return {
            "perplexity": 0.0,
            "active_codes": 0,
            "active_ratio": 0.0,
        }

    # Gather indices across all GPUs for accurate codebook usage
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        # Gather sizes first (indices may have different lengths per GPU)
        local_size = torch.tensor([flat_indices.numel()], device=flat_indices.device)
        sizes = [torch.zeros(1, dtype=torch.long, device=flat_indices.device) for _ in range(world_size)]
        torch.distributed.all_gather(sizes, local_size)
        max_size = max(s.item() for s in sizes)

        # Pad indices to max_size for gathering
        padded = torch.zeros(max_size, dtype=flat_indices.dtype, device=flat_indices.device)
        padded[: flat_indices.numel()] = flat_indices
        gathered = [
            torch.zeros(max_size, dtype=flat_indices.dtype, device=flat_indices.device) for _ in range(world_size)
        ]
        torch.distributed.all_gather(gathered, padded)

        # Concatenate only valid indices from each GPU
        all_indices = []
        for i, g in enumerate(gathered):
            all_indices.append(g[: sizes[i].item()])
        flat_indices = torch.cat(all_indices)

    # Compute code histogram
    histogram = torch.bincount(flat_indices, minlength=num_codes).float()
    total = histogram.sum()
    if total == 0:
        return {
            "perplexity": 0.0,
            "active_codes": 0,
            "active_ratio": 0.0,
        }
    histogram_norm = histogram / total

    # Compute perplexity (exponential of entropy)
    log_probs = torch.log(histogram_norm + 1e-10)
    entropy = -torch.sum(histogram_norm * log_probs)
    perplexity = torch.exp(entropy)

    # Compute active code ratio
    active_codes = (histogram > 0).sum()
    active_ratio = active_codes.float() / num_codes

    return {
        "perplexity": perplexity.item(),
        "active_codes": active_codes.item(),
        "active_ratio": active_ratio.item() * 100,
    }


class FIDComputer:
    """Compute Frechet Inception Distance between two sets of images.

    Uses torchmetrics.image.fid.FrechetInceptionDistance for computation.

    This wrapper is lazy and distributed-aware:
    - It delays Inception construction until first use.
    - It coordinates feature extractor initialization so only one local rank per
      node performs the initial weight download/cache population.
    - It can use a pre-downloaded weight file via ``feature_extractor_weights_path``
      or the ``TOKENIZER_FID_WEIGHTS_PATH`` environment variable.
    - In distributed mode, it reduces FID sufficient statistics explicitly
      instead of relying on torchmetrics internal process-group synchronization.
    """

    def __init__(
        self,
        device: str = "cuda",
        feature: int | torch.nn.Module = 2048,
        normalize: bool = True,
        sync_on_compute: bool = True,
        dist_sync_on_step: bool = False,
        feature_extractor_weights_path: str | None = None,
    ) -> None:
        """Initialize FID computer.

        Args:
            device: Device for computation.
            feature: InceptionV3 feature dimension (2048 for final pool), or a
                custom feature extractor module for testing/specialized use.
            normalize: Whether to normalize input images to [0, 1].
            sync_on_compute: Whether to synchronize metric state on compute.
            dist_sync_on_step: Whether to synchronize on each update step.
            feature_extractor_weights_path: Optional local path to torch-fidelity
                Inception weights. Falls back to TOKENIZER_FID_WEIGHTS_PATH env var.
        """
        self.device = device
        self._metric = None
        self._normalize = normalize
        self._feature = feature
        self._sync_on_compute = sync_on_compute
        self._dist_sync_on_step = dist_sync_on_step
        self._feature_extractor_weights_path = feature_extractor_weights_path or os.environ.get(
            "TOKENIZER_FID_WEIGHTS_PATH"
        )
        self._initialized = False

    @staticmethod
    def _is_local_leader() -> bool:
        """Return True for the first process on the current node."""
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            return int(local_rank) == 0
        if torch.cuda.is_available():
            return torch.cuda.current_device() == 0
        return True

    def _autocast_context(self) -> Any:
        """Return an autocast context appropriate for the current device."""
        device_type = torch.device(self.device).type
        if device_type == "cuda":
            return torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
        return nullcontext()

    def _ensure_initialized(self) -> bool:
        """Lazily initialize the FID metric."""
        if self._initialized:
            return self._metric is not None

        self._initialized = True
        init_exception: Exception | None = None
        should_coordinate = False
        is_local_leader = True
        try:
            import torch.distributed as dist
            import torchmetrics.image.fid

            is_distributed = dist.is_available() and dist.is_initialized()
            should_coordinate = self._feature_extractor_weights_path is None and is_distributed
            is_local_leader = self._is_local_leader()
            if should_coordinate and not is_local_leader:
                dist.barrier()

            self._metric = torchmetrics.image.fid.FrechetInceptionDistance(
                feature=self._feature,
                normalize=self._normalize,
                sync_on_compute=self._sync_on_compute if not is_distributed else False,
                dist_sync_on_step=self._dist_sync_on_step if not is_distributed else False,
                feature_extractor_weights_path=self._feature_extractor_weights_path,
            )
            self._metric.to(self.device)
            logging.info(
                f"Initialized FID metric with feature={self._feature}, "
                f"normalize={self._normalize}, "
                f"weights_path={self._feature_extractor_weights_path or '<download>'}"
            )
        except ImportError:
            logging.warning("torchmetrics not available for FID computation")
            self._metric = None
        except Exception as e:
            init_exception = e
            self._metric = None
        finally:
            if should_coordinate and is_local_leader:
                import torch.distributed as dist

                dist.barrier()

        if init_exception is not None:
            logging.warning(
                f"FID metric initialization failed: {init_exception}. "
                "Pre-cache the torch-fidelity Inception weights or set TOKENIZER_FID_WEIGHTS_PATH."
            )

        return self._metric is not None

    def to(self, device: str | torch.device) -> "FIDComputer":
        """Move the underlying metric to a device.

        FID accumulates statistics in float64 internally for numerical
        stability, so callers should not use this wrapper to change dtypes.
        """
        self.device = str(device)
        if self._metric is not None:
            self._metric.to(device)
        return self

    def reset(self) -> None:
        """Reset accumulated features."""
        if self._metric is not None:
            self._metric.reset()

    @torch.no_grad()
    def update(
        self,
        images: torch.Tensor,
        real: bool = True,
    ) -> None:
        """Update with a batch of images.

        Separate calls for real and fake images.

        Args:
            images: Images in [0, 1] range, shape (B, C, H, W).
            real: Whether these are real images (True) or fake/reconstructed (False).
        """
        if not self._ensure_initialized():
            return

        # Handle video tensors (B, C, T, H, W) -> (B*T, C, H, W)
        if images.ndim == 5:
            images = images.reshape(-1, *images.shape[-3:])

        if self._normalize and images.dtype == torch.uint8:
            images = images.float() / 255.0

        if self._normalize:
            images = images.float().to(self.device)
        else:
            images = images.to(self.device)

        # Update with autocast disabled for numerical stability
        with self._autocast_context():
            self._metric.update(images, real=real)

    def _get_reduced_states(self) -> dict[str, torch.Tensor]:
        """Return local or globally reduced FID sufficient statistics."""
        states = {
            "real_features_sum": self._metric.real_features_sum.detach().clone(),
            "real_features_cov_sum": self._metric.real_features_cov_sum.detach().clone(),
            "real_features_num_samples": self._metric.real_features_num_samples.detach().clone(),
            "fake_features_sum": self._metric.fake_features_sum.detach().clone(),
            "fake_features_cov_sum": self._metric.fake_features_cov_sum.detach().clone(),
            "fake_features_num_samples": self._metric.fake_features_num_samples.detach().clone(),
        }

        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            for value in states.values():
                dist.all_reduce(value, op=dist.ReduceOp.SUM)

        return states

    @staticmethod
    def _compute_fid_from_states(states: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute FID from sufficient statistics."""
        real_sum = states["real_features_sum"]
        real_cov_sum = states["real_features_cov_sum"]
        fake_sum = states["fake_features_sum"]
        fake_cov_sum = states["fake_features_cov_sum"]

        real_num = states["real_features_num_samples"].to(real_sum.dtype)
        fake_num = states["fake_features_num_samples"].to(fake_sum.dtype)

        if real_num.item() < 2 or fake_num.item() < 2:
            raise RuntimeError(
                "More than one sample is required for both the real and fake distributions to compute FID"
            )

        mean_real = (real_sum / real_num).unsqueeze(0)
        mean_fake = (fake_sum / fake_num).unsqueeze(0)

        cov_real_num = real_cov_sum - real_num * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (real_num - 1)
        cov_fake_num = fake_cov_sum - fake_num * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (fake_num - 1)

        diff = (mean_real.squeeze(0) - mean_fake.squeeze(0)).square().sum(dim=-1)
        trace = cov_real.trace() + cov_fake.trace()
        covmean = torch.linalg.eigvals(cov_real @ cov_fake).sqrt().real.sum(dim=-1)
        return diff + trace - 2 * covmean

    def compute(self) -> float:
        """Compute FID from accumulated features.

        Returns:
            FID value (lower is better).
        """
        if not self._ensure_initialized():
            return float("nan")

        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                states = self._get_reduced_states()
                fid_value = self._compute_fid_from_states(states)
            else:
                with self._autocast_context():
                    fid_value = self._metric.compute()
            return fid_value.item()
        except Exception as e:
            logging.warning(f"FID computation failed: {e}")
            return float("nan")


__all__ = [
    "compute_psnr",
    "compute_imagenet_accuracy",
    "compute_codebook_usage",
    "FIDComputer",
]
