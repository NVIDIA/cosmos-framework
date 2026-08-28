# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Protocol, cast

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import wandb

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed, misc
from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.easy_io import easy_io


def _get_sigma_bins(n: int = 10) -> np.ndarray:
    """Get ``n`` uniformly spaced bin edges covering the rectified-flow sigma range [0, 1]."""
    return np.linspace(0.0, 1.0, n + 1)


class _SupportsIsImageBatch(Protocol):
    """VFM models that can tell image batches apart from video batches.

    ``ImaginaireModel`` does not declare ``is_image_batch``, so the attribute
    otherwise resolves through ``torch.nn.Module.__getattr__``.
    """

    def is_image_batch(self, data_batch: dict[str, torch.Tensor]) -> bool: ...


class _SigmaLossCache:
    """A fixed-size queue for caching sigma and loss tensors.

    Stores sigma/loss pairs on CPU.
    When the total number of elements exceeds queue_size, the oldest entries
    are automatically removed to maintain the size limit.

    Args:
        queue_size: Maximum number of elements to store in the cache.
    """

    def __init__(self, queue_size: int = 2000) -> None:
        self.queue_size = queue_size
        self.reset()

    def reset(self) -> None:
        self.sigma_list: list[torch.Tensor] = []
        self.loss_list: list[torch.Tensor] = []
        self._total_elements: int = 0

    def add(self, sigma: torch.Tensor, loss: torch.Tensor) -> None:
        # Sigma and loss are paired element-wise downstream, and ``_all_gather_arrays`` sizes
        # its padded buffers from the SIGMA length alone. A shorter loss tensor is therefore
        # zero-filled out to the sigma length and those zeros are gathered and paired against
        # real sigmas as though they were losses -- silent corruption of every statistic and
        # plot downstream, not a crash. Raise rather than assert, so ``python -O`` cannot
        # strip the check; one ``numel()`` comparison per step costs nothing.
        if sigma.numel() != loss.numel():
            raise ValueError(
                f"sigma and loss must hold the same number of elements, got {sigma.numel()} and {loss.numel()}"
            )

        # Convert to bf16 and store on CPU
        sigma_cpu = sigma.detach().cpu().to(torch.bfloat16)
        loss_cpu = loss.detach().cpu().to(torch.bfloat16)

        self.sigma_list.append(sigma_cpu)
        self.loss_list.append(loss_cpu)
        self._total_elements += sigma_cpu.numel()

        # Remove oldest elements if queue exceeds max size
        while self._total_elements > self.queue_size and len(self.sigma_list) > 1:
            removed_sigma = self.sigma_list.pop(0)
            self.loss_list.pop(0)
            self._total_elements -= removed_sigma.numel()

    def get_arrays(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.sigma_list:
            return torch.tensor([], dtype=torch.bfloat16), torch.tensor([], dtype=torch.bfloat16)

        sigma_arr = torch.cat(self.sigma_list, dim=0)  # [N_total]  (concatenated across cached batches)
        loss_arr = torch.cat(self.loss_list, dim=0)  # [N_total]

        return sigma_arr, loss_arr


class SigmaLossAnalysis(Callback):
    """Analyze the relationship between sigma (noise level) and flow matching loss.

    This callback tracks per-instance flow matching losses at different sigma values
    during training. It maintains separate caches for image and video batches,
    periodically aggregates statistics across all distributed ranks, and logs
    the results to wandb.

    The analysis helps understand how well the model learns to denoise at different
    noise levels, which is useful for diagnosing training dynamics in flow matching
    models.

    Args:
        every_n: Log statistics every N iterations.
        every_n_viz: Create visualization plots every N iterations (must be multiple of every_n).
        save_s3: If True, save raw data to S3 for offline analysis.
    """

    def __init__(
        self,
        every_n: int = 1,
        every_n_viz: int = 1,
        save_s3: bool = False,
    ) -> None:
        super().__init__()
        self.save_s3 = save_s3
        # Raise rather than assert, so ``python -O`` cannot strip these: they validate a
        # user-supplied config, and the period check below divides by ``every_n``.
        if every_n < 1 or every_n_viz < 1:
            raise ValueError(
                f"every_n and every_n_viz must be >= 1 in sigma_loss_analysis callback, "
                f"got every_n={every_n} and every_n_viz={every_n_viz}"
            )
        self.every_n = every_n
        if every_n_viz % every_n != 0:
            raise ValueError(
                f"every_n_viz must be a multiple of every_n in sigma_loss_analysis callback, "
                f"got every_n_viz={every_n_viz} and every_n={every_n}"
            )
        self.every_n_viz = every_n_viz
        self.name = self.__class__.__name__

        self.image_cache = _SigmaLossCache(queue_size=2000)
        self.video_cache = _SigmaLossCache(queue_size=2000)

    @staticmethod
    def _create_analysis_plots(
        sigma_arr: torch.Tensor,
        loss_arr: torch.Tensor,  # [N]  # [N]
    ) -> wandb.Image | None:
        if len(sigma_arr) == 0:
            return None

        # Convert to numpy for plotting
        sigma_np = sigma_arr.cpu().float().numpy()
        loss_np = loss_arr.cpu().float().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        sigma_bins = _get_sigma_bins(10)

        y_tick_min, y_tick_max = 0, 1.0
        # 2D histogram over uniform sigma bins and a fixed [0,1] loss range
        loss_bins = np.linspace(y_tick_min, y_tick_max, 20)

        counts, _, _ = np.histogram2d(sigma_np, loss_np, bins=(sigma_bins, loss_bins))
        if counts.max() < 0.1:
            plt.close(fig)
            return None

        # Plot heatmap with log-scale colormap. vmax is floored above vmin so a single-sample
        # bin does not collapse LogNorm to a degenerate vmin == vmax range.
        im = ax1.imshow(
            counts.T,
            origin="lower",
            aspect="auto",
            extent=[sigma_bins[0], sigma_bins[-1], y_tick_min, y_tick_max],
            norm=matplotlib.colors.LogNorm(vmin=1, vmax=max(float(counts.max()), 2.0)),
        )
        plt.colorbar(im, ax=ax1)

        # Set fixed loss ticks from 0 to 1
        yticks = np.linspace(y_tick_min, y_tick_max, 6)
        ax1.set_yticks(yticks)
        ax1.set_yticklabels([f"{y:.1f}" for y in yticks])

        ax1.set_xlabel("Sigma")
        ax1.set_ylabel("Loss")
        title = "Sigma vs Loss Distribution"
        ax1.set_title(title)

        # Sigma histogram with loss statistics per bin
        hist_counts, _ = np.histogram(sigma_np, bins=sigma_bins)
        bin_indices = np.digitize(sigma_np, sigma_bins) - 1

        # Calculate statistics per bin
        n_bins = len(sigma_bins) - 1
        means = np.zeros(n_bins)
        stds = np.zeros(n_bins)
        for i in range(n_bins):
            bin_mask = bin_indices == i
            if bin_mask.any():
                means[i] = loss_np[bin_mask].mean()
                stds[i] = loss_np[bin_mask].std()
            else:
                means[i] = np.nan
                stds[i] = np.nan

        # Plot histogram
        bin_centers = (sigma_bins[:-1] + sigma_bins[1:]) / 2
        ax2.bar(bin_centers, hist_counts, width=np.diff(sigma_bins), alpha=0.3, align="center")

        # Plot loss statistics on twin axis
        ax2_twin = ax2.twinx()
        valid_mask = ~np.isnan(means)
        ax2_twin.errorbar(
            bin_centers[valid_mask], means[valid_mask], yerr=stds[valid_mask], color="red", fmt="o-", alpha=0.5
        )

        ax2.set_xlabel("Sigma")
        ax2.set_ylabel("Count")
        ax2_twin.set_ylabel("Loss (mean ± std)")
        title = "Sigma Distribution with Loss Statistics"
        ax2.set_title(title)

        # Add grid for better readability
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)

        sigma_labels = [f"{val:.2f}" for val in sigma_bins]
        ax1.set_xticks(sigma_bins[1:-1])  # Skip boundary bins
        ax1.set_xticklabels(sigma_labels[1:-1], rotation=45)
        ax1.set_xscale("linear")
        ax2.set_xticks(sigma_bins[1:-1])
        ax2.set_xticklabels(sigma_labels[1:-1], rotation=45)
        ax2.set_xscale("linear")

        plt.tight_layout()
        fig_img = wandb.Image(fig)
        plt.close(fig)

        return fig_img

    @staticmethod
    def _process_stats(sigma: torch.Tensor, loss: torch.Tensor) -> dict[str, float]:
        """Calculate summary statistics for sigma and loss distributions.

        Args:
            sigma: Tensor of sigma (noise level) values.
            loss: Tensor of corresponding loss values.

        Returns:
            Dictionary containing:
                - sigma_log_mean: Mean of log(sigma). Log-space is used since sigma spans
                    multiple orders of magnitude, a standard practice on flow matching / EDM models.
                - sigma_log_std: Standard deviation of log(sigma).
                - loss_mean: Average loss across all samples.
                - loss_std: Standard deviation of loss, measuring spread.
                - loss_min: Minimum loss value observed.
                - loss_max: Maximum loss value observed.
                - loss_median: Median (50th percentile) loss, robust to outliers.
                - loss_q1: First quartile (25th percentile) of loss.
                - loss_q3: Third quartile (75th percentile) of loss.
        """
        return {
            "sigma_log_mean": float(sigma.log().mean()),
            "sigma_log_std": float(sigma.log().std()),
            "loss_mean": float(loss.mean()),
            "loss_std": float(loss.std()),
            "loss_min": float(loss.min()),
            "loss_max": float(loss.max()),
            "loss_median": float(loss.median()),
            "loss_q1": float(torch.quantile(loss.float(), 0.25)),
            "loss_q3": float(torch.quantile(loss.float(), 0.75)),
        }

    @staticmethod
    def _all_gather_arrays(
        local_sigma: torch.Tensor, local_loss: torch.Tensor, world_size: int
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """All-gather variable-length sigma/loss pairs across the world.

        Returns the concatenated arrays on rank0, and ``(None, None)`` both on the other ranks
        and when no rank cached anything. Every rank must reach the collectives below, so
        callers cannot skip this on ranks whose own cache happens to be empty.
        """
        # Gather sizes first
        local_size = torch.tensor([len(local_sigma)], dtype=torch.long, device="cuda")  # [1]
        sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(sizes, local_size)
        size_list = [int(s.item()) for s in sizes]

        # Gather data
        max_size = max(size_list)
        if max_size == 0:
            return None, None

        # Move to GPU for gathering
        padded_sigma = torch.zeros(max_size, dtype=torch.bfloat16, device="cuda")  # [max_size]
        padded_loss = torch.zeros(max_size, dtype=torch.bfloat16, device="cuda")  # [max_size]

        if len(local_sigma) > 0:
            padded_sigma[: len(local_sigma)] = local_sigma.cuda()
            padded_loss[: len(local_loss)] = local_loss.cuda()

        all_sigma = [torch.zeros_like(padded_sigma) for _ in range(world_size)]
        all_loss = [torch.zeros_like(padded_loss) for _ in range(world_size)]

        dist.all_gather(all_sigma, padded_sigma)
        dist.all_gather(all_loss, padded_loss)

        if not distributed.is_rank0():
            return None, None

        # Combine data from all ranks, dropping the zero padding each rank contributed
        valid_sigma = [sigma[:size] for sigma, size in zip(all_sigma, size_list) if size > 0]
        valid_loss = [loss[:size] for loss, size in zip(all_loss, size_list) if size > 0]
        if not valid_sigma:
            return None, None

        return torch.cat(valid_sigma), torch.cat(valid_loss)  # [N_total] each (across all ranks)

    def _gather_and_save(
        self, cache: _SigmaLossCache, iteration: int, prefix: str, log_viz: bool = True
    ) -> dict[str, int | float | wandb.Image]:
        # int from total_samples, float from _process_stats, wandb.Image from the plot.
        info: dict[str, int | float | wandb.Image] = {}

        # Gather data from all ranks
        local_sigma, local_loss = cache.get_arrays()
        world_size = distributed.get_world_size()

        if world_size > 1:
            sigma_arr, loss_arr = self._all_gather_arrays(local_sigma, local_loss, world_size)
        elif len(local_sigma) > 0:
            # Single process: the local cache already holds everything, and rank0 is this rank.
            sigma_arr, loss_arr = local_sigma, local_loss
        else:
            sigma_arr, loss_arr = None, None

        if sigma_arr is not None and loss_arr is not None:
            # Overall statistics
            info[f"{prefix}/total_samples"] = sigma_arr.shape[0]

            # Calculate statistics
            stats = self._process_stats(sigma_arr, loss_arr)
            info.update({f"{prefix}/{k}": v for k, v in stats.items()})

            # Create visualization
            if log_viz:
                fig_img = self._create_analysis_plots(sigma_arr, loss_arr)
                if fig_img is not None:
                    info[f"{prefix}/distribution_plot"] = fig_img

            if self.save_s3:
                save_data = {
                    "sigma": sigma_arr.cpu(),
                    "loss": loss_arr.cpu(),
                    "stats": {k: v for k, v in info.items() if not isinstance(v, wandb.Image)},
                }
                easy_io.dump(
                    save_data,
                    f"s3://rundir/{self.name}/{prefix}_Iter{iteration:09d}.pkl",
                )

        cache.reset()
        return info

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        sigma = output_batch["sigma"]
        fm_loss_vision_per_instance = output_batch["flow_matching_loss_vision_per_instance"]

        # sigma is [B] (base), [B,1] (TF), or [B,T_max] (DF); reduce to [B] for logging.
        # Raise rather than assert, so ``python -O`` cannot strip the check: with it gone, a
        # 3-D sigma survives the ``mean(dim=-1)`` below as [B,T] instead of [B] and reaches
        # ``_SigmaLossCache.add``, which rejects it only for having a different element count
        # than the loss -- a confusing numel error one frame away from the real problem,
        # which is the shape arriving here.
        if sigma.ndim > 2:
            raise ValueError(f"Sigma should be [B], [B,1] or [B,T_max], got shape {tuple(sigma.shape)}")
        if sigma.ndim == 2:
            sigma = sigma.mean(dim=-1)  # [B]  (reduced from [B,T_max] or [B,1])

        if cast(_SupportsIsImageBatch, model).is_image_batch(data_batch):
            self.image_cache.add(sigma, fm_loss_vision_per_instance)
        else:
            self.video_cache.add(sigma, fm_loss_vision_per_instance)

        if iteration % self.every_n == 0:
            info = {}

            with misc.timer("sigma_loss_analysis"):
                log_viz = iteration % self.every_n_viz == 0
                # Every rank must call _gather_and_save for both caches, even when its own
                # cache is empty: it collectively all_gathers cache sizes across the world,
                # so a rank skipping the call here (e.g. because it packed only image or only
                # video samples this window) leaves the other ranks' all_gather hanging until
                # the NCCL watchdog times out.
                info.update(self._gather_and_save(self.image_cache, iteration, "sigma_loss_image", log_viz=log_viz))
                info.update(self._gather_and_save(self.video_cache, iteration, "sigma_loss_video", log_viz=log_viz))

                if distributed.is_rank0() and info and wandb.run:
                    wandb.log(info, step=iteration)
