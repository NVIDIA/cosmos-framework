# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""Framework model wrapper for LiDAR TransformerVAE training."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import attrs
import torch
import torch.distributed as dist

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.model.generator.tokenizers.lidar.checkpoint import load_model_checkpoint
from cosmos_framework.model.generator.tokenizers.lidar.geometry import pandar128_ray_directions
from cosmos_framework.model.generator.tokenizers.lidar.lidar_tokenizer_v1 import (
    DEFAULT_LATENT_CH,
    DEFAULT_NETWORK_CONFIG,
)
from cosmos_framework.model.generator.tokenizers.lidar.network.transformer_vae import TransformerVAE
from cosmos_framework.model.generator.tokenizers.lidar.normalization import metric_lidar_to_network
from cosmos_framework.model.generator.tokenizers.lidar.preprocessing import (
    crop_range_video_width,
    network_range_to_metric,
)
from cosmos_framework.model.generator.tokenizers.lidar.range_projection import LidarRangeProjectionConfig
from cosmos_framework.model.tokenizer.lidar.loss import LidarVAELoss
from cosmos_framework.utils.config import PowerEMAConfig
from cosmos_framework.utils.easy_io import easy_io
from cosmos_framework.utils.ema import PowerEMATracker
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import instantiate


@attrs.define(slots=False)
class LidarTokenizerModelConfig:
    range_projection: LidarRangeProjectionConfig = attrs.Factory(LidarRangeProjectionConfig)
    network: dict[str, Any] = attrs.Factory(lambda: dict(DEFAULT_NETWORK_CONFIG))
    loss: Any = attrs.Factory(lambda: L(LidarVAELoss)(recon_weight=1.0, kl_weight=1e-6, validity_weight=0.1))
    freeze_encoder: bool = False
    # Optional published tokenizer to start the network from. Applied at
    # construction, so a real resume loaded by the trainer afterwards wins.
    init_from_checkpoint: str | None = None
    latent_stats_ema_momentum: float = 0.999
    precision: str = "float16"
    ema: PowerEMAConfig = attrs.Factory(PowerEMAConfig)


class LidarTokenizerModel(ImaginaireModel):
    """Trainable LiDAR tokenizer: TransformerVAE + LidarVAELoss."""

    config: LidarTokenizerModelConfig
    ema: PowerEMATracker | None
    range_projection: LidarRangeProjectionConfig
    semantic_width: int
    network: TransformerVAE
    latent_stats_ema_momentum: float
    _autocast_dtype: torch.dtype | None

    def __init__(self, config: LidarTokenizerModelConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        if config is None:
            config = LidarTokenizerModelConfig()
        elif isinstance(config, Mapping):
            config_values = dict(config)
            range_projection = config_values.get("range_projection")
            if isinstance(range_projection, Mapping):
                config_values["range_projection"] = LidarRangeProjectionConfig.from_dict(range_projection)
            config = LidarTokenizerModelConfig(**config_values)
        self.config = config
        self.ema = None
        self.range_projection = config.range_projection
        self.semantic_width = self.range_projection.semantic_width

        network_kwargs = dict(config.network)
        network_kwargs["freeze_encoder"] = bool(config.freeze_encoder)
        self.network = TransformerVAE(**network_kwargs)
        if config.init_from_checkpoint:
            load_model_checkpoint(
                self.network,
                config.init_from_checkpoint,
                backend_args=None,
            )
        self.loss_fn: LidarVAELoss = config.loss if isinstance(config.loss, LidarVAELoss) else instantiate(config.loss)

        z_dim = int(network_kwargs.get("z_dim", DEFAULT_LATENT_CH))
        self.latent_stats_ema_momentum = float(config.latent_stats_ema_momentum)
        self.register_buffer("latent_mean", torch.zeros(1, z_dim, 1, 1, 1), persistent=True)  # [1,C,1,1,1]
        self.register_buffer("latent_std", torch.ones(1, z_dim, 1, 1, 1), persistent=True)  # [1,C,1,1,1]
        self.register_buffer("_latent_stats_initialized", torch.tensor(False, dtype=torch.bool), persistent=True)  # []
        min_range = torch.tensor(self.range_projection.min_range_m)  # []
        self.register_buffer("min_range", min_range, persistent=True)
        max_range = torch.tensor(self.range_projection.max_range_m)  # []
        self.register_buffer("max_range", max_range, persistent=True)
        if self.loss_fn.needs_points:
            ray_directions = torch.from_numpy(
                pandar128_ray_directions(range_projection=self.range_projection)
            )  # [H,W,3]
            self.register_buffer("ray_directions", ray_directions[None, None], persistent=False)  # [1,1,H,W,3]

        precision = str(config.precision).lower()
        if precision in {"bf16", "bfloat16"}:
            self._autocast_dtype = torch.bfloat16
        elif precision in {"fp16", "float16"}:
            self._autocast_dtype = torch.float16
        else:
            self._autocast_dtype = None

        # Build the tracker after network initialization: it clones the current
        # trainable parameters once, so its initial values must match the warm
        # start. The training loop updates the tracker after each optimizer step.
        #
        # Power EMA grows its averaging window with training age. A fixed decay
        # holds a constant window and retains its initial seed by beta**iteration;
        # the relative-width parameter instead keeps smoothing proportional to
        # progress. At relative width 0.1 the effective window is approximately
        # iteration / 7.9, with seed weight (iteration + 1)**-7.9.
        if config.ema.enabled:
            self.ema = PowerEMATracker(self, s=config.ema.s)

    def init_optimizer_scheduler(
        self, optimizer_config: Any, scheduler_config: Any
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        # Accept either LazyCall configs or plain dicts with lr.
        if hasattr(optimizer_config, "_target_") or (
            isinstance(optimizer_config, Mapping) and "_target_" in dict(optimizer_config)
        ):
            return super().init_optimizer_scheduler(optimizer_config, scheduler_config)

        opt_dict = dict(optimizer_config) if isinstance(optimizer_config, Mapping) else {}
        lr = float(opt_dict.get("lr", 1e-4))
        betas = tuple(opt_dict.get("betas", (0.9, 0.999)))
        weight_decay = float(opt_dict.get("weight_decay", 0.0))
        eps = float(opt_dict.get("eps", 1e-8))
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)

        sched_dict = dict(scheduler_config) if isinstance(scheduler_config, Mapping) else {}
        warmup = int(sched_dict.get("warmup_steps", 100))
        total = int(sched_dict.get("total_steps", 10_000))
        end_lr = float(sched_dict.get("end_lr", lr * 0.1))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step + 1) / float(max(warmup, 1))
            progress = min(1.0, float(step - warmup) / float(max(total - warmup, 1)))
            return (end_lr / lr) + (1.0 - end_lr / lr) * (1.0 - progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return optimizer, scheduler

    @torch.no_grad()
    def _update_latent_stats_ema(self, latent: torch.Tensor) -> None:
        if self.latent_stats_ema_momentum <= 0:
            return
        reduce_dims = (0, 2, 3, 4)
        latent_float = latent.float()  # [B,C,T,H,W]
        latent_sum = latent_float.sum(dim=reduce_dims, keepdim=True)  # [1,C,1,1,1]
        latent_squared_sum = latent_float.square().sum(dim=reduce_dims, keepdim=True)  # [1,C,1,1,1]
        sample_count = torch.tensor(
            latent_float.numel() // latent_float.shape[1],
            dtype=torch.float32,
            device=latent.device,
        )  # []
        if dist.is_available() and dist.is_initialized():
            moments = torch.stack((latent_sum, latent_squared_sum))  # [2,1,C,1,1,1]
            dist.all_reduce(moments, op=dist.ReduceOp.SUM)
            dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
            latent_sum, latent_squared_sum = moments.unbind(0)  # each [1,C,1,1,1]
        batch_mean = latent_sum / sample_count  # [1,C,1,1,1]
        batch_var = (latent_squared_sum / sample_count - batch_mean.square()).clamp_min(0.0)  # [1,C,1,1,1]
        batch_std = (batch_var + 1e-6).sqrt()  # [1,C,1,1,1]
        if not bool(self._latent_stats_initialized.item()):
            self.latent_mean.copy_(batch_mean)  # [1,C,1,1,1]
            self.latent_std.copy_(batch_std)  # [1,C,1,1,1]
            self._latent_stats_initialized.fill_(True)  # []
            return
        m = self.latent_stats_ema_momentum
        self.latent_mean.mul_(m).add_((1.0 - m) * batch_mean)  # [1,C,1,1,1]
        self.latent_std.mul_(m).add_((1.0 - m) * batch_std)  # [1,C,1,1,1]

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Load latent-stat initialization state, including legacy checkpoints."""
        state_dict.setdefault(f"{prefix}min_range", self.min_range.detach().clone())  # []
        state_dict.setdefault(f"{prefix}max_range", self.max_range.detach().clone())  # []
        initialized_key = f"{prefix}_latent_stats_initialized"
        if initialized_key not in state_dict:
            mean_key = f"{prefix}latent_mean"
            std_key = f"{prefix}latent_std"
            # Older checkpoints predate the persistent flag. If their latent
            # buffers are present, preserve them as an initialized EMA instead
            # of replacing them with statistics from the first resumed batch.
            if mean_key in state_dict and std_key in state_dict:
                state_dict[initialized_key] = torch.tensor(True, dtype=torch.bool)  # []
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _prepare_batch(self, data_batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video = data_batch["video"]  # [B,3,T,H,W]
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"Expected video [B,3,T,H,W], got {tuple(video.shape)}")
        target = data_batch.get("target", video[:, :2])  # [B,2,T,H,Ws]
        valid_mask = data_batch.get("valid_mask", video[:, 2:3])  # [B,1,T,H,Ws]
        device = next(self.parameters()).device
        video = video.to(device=device, non_blocking=True)  # [B,3,T,H,W]
        target = target.to(device=device, non_blocking=True)  # [B,2,T,H,Ws]
        valid_mask = valid_mask.to(device=device, non_blocking=True)  # [B,1,T,H,Ws]
        normalized_video, _ = metric_lidar_to_network(
            video,
            min_range=self.min_range,
            max_range=self.max_range,
        )  # [B,3,T,H,W], [B,1,T,H,W]
        normalized_target, normalized_valid = metric_lidar_to_network(
            torch.cat((target, valid_mask), dim=1),
            min_range=self.min_range,
            max_range=self.max_range,
        )  # [B,3,T,H,Ws], [B,1,T,H,Ws]
        return normalized_video, normalized_target[:, :2], normalized_valid  # [B,3,T,H,W], [B,2,T,H,Ws], [B,1,T,H,Ws]

    def _range_to_points(self, normalized_range: torch.Tensor) -> torch.Tensor:
        """Unproject a normalized range channel ``[B,1,T,H,W]`` to metric XYZ ``[B,T,H,W,3]``.

        Rays outside the sensor's metric band collapse to the origin. A
        prediction that overshoots is first clamped back to the band edge and
        then dropped, so it cannot drag the point loss toward a range the
        tokenizer could never have encoded. The same rule applies to the target,
        which parks its dropped rays at the near edge.

        Held in float32 regardless of the autocast dtype: at 100 m, float16
        steps in ~6 cm, which is the scale of the residual this term exists to
        resolve.
        """
        metric = network_range_to_metric(
            normalized_range[:, 0].float(),  # [B,T,H,W]
            min_range=self.min_range,
            max_range=self.max_range,
        )  # [B,T,H,W]
        inside = (metric > self.min_range) & (metric < self.max_range)  # [B,T,H,W]
        depth = torch.where(inside, metric, torch.zeros_like(metric))  # [B,T,H,W]
        return self.ray_directions * depth[..., None]  # [B,T,H,W,3]

    def training_step(self, data_batch: dict[str, Any], iteration: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        del iteration
        video, target, valid_mask = self._prepare_batch(data_batch)  # [B,3,T,H,W], [B,2,T,H,Ws], [B,1,T,H,Ws]
        device_type = video.device.type
        autocast_enabled = self._autocast_dtype is not None and device_type == "cuda"
        with torch.autocast(device_type=device_type, dtype=self._autocast_dtype, enabled=autocast_enabled):
            outputs = self.network(video)  # reconstructions: [B,3,T,H,W]; latent: [B,C,T,h,w]
            reconstructions = crop_range_video_width(
                outputs["reconstructions"],
                semantic_width=self.semantic_width,
            )  # [B,3,T,H,Ws]
            validity_logits = outputs.get("validity_logits")  # [B,1,T,H,W] or None
            if validity_logits is not None:
                validity_logits = crop_range_video_width(
                    validity_logits,
                    semantic_width=self.semantic_width,
                )  # [B,1,T,H,Ws]
            recon_points: torch.Tensor | None = None
            target_points: torch.Tensor | None = None
            if self.loss_fn.needs_points:
                recon_points = self._range_to_points(reconstructions[:, :1])  # [B,T,H,W,3]
                target_points = self._range_to_points(target[:, :1])  # [B,T,H,W,3]
            loss, metrics = self.loss_fn(
                reconstructions=reconstructions,
                target=target,
                posteriors=outputs["posteriors"],
                valid_mask=valid_mask,
                validity_logits=validity_logits,
                recon_points=recon_points,
                target_points=target_points,
            )  # loss: []; metrics: dict[str,scalar tensor]
        if self.network.training:
            self._update_latent_stats_ema(outputs["latent"].detach())  # [B,C,T,h,w]
        # Report error per measured-valid ray independently of the objective's reduction.
        range_error = (reconstructions[:, :1].detach() - target[:, :1].detach()).abs()  # [B,1,T,H,Ws]
        measured_valid = valid_mask.detach()  # [B,1,T,H,Ws]
        range_mae_meters = (
            (range_error * measured_valid).sum()
            / measured_valid.sum().clamp_min(1.0)
            * (self.max_range - self.min_range)
            * 0.5
        )  # []
        loss_dict = {
            "train-video_rec/reconstruction_loss": metrics["loss_recon"],
            "train-video_rec/range_reconstruction_loss": metrics["loss_range_recon"],
            "train-video_rec/intensity_reconstruction_loss": metrics["loss_intensity_recon"],
            "train-video_rec/range_mae_meters": range_mae_meters,
            "train-video_rec/kl_loss": metrics["loss_kl"],
            "train-video_rec/validity_loss": metrics["loss_validity"],
            "train-video_rec/point_loss": metrics["loss_point"],
            "train-video_rec/gradient_loss": metrics["loss_gradient"],
        }
        output_batch = {
            "inputs": crop_range_video_width(video, semantic_width=self.semantic_width).detach(),  # [B,3,T,H,Ws]
            "video": video.detach(),  # [B,3,T,H,W]
            "reconstructions": reconstructions.detach(),  # [B,3,T,H,Ws]
            "latent": outputs["latent"].detach(),  # [B,C,T,h,w]
            "task_type": "video_rec",
            "loss_dict": loss_dict,
            **metrics,
        }
        return output_batch, loss

    @torch.no_grad()
    def validation_step(
        self, data_batch: dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        was_training = self.network.training
        self.network.eval()
        try:
            return self.training_step(data_batch, iteration)  # loss: []; output tensors: batch-dependent
        finally:
            self.network.train(was_training)

    @torch.inference_mode()
    def reconstruct(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        normalized, _ = metric_lidar_to_network(
            video.to(device=next(self.parameters()).device),
            min_range=self.min_range,
            max_range=self.max_range,
        )  # [B,3,T,H,W], [B,1,T,H,W]
        return self.network(normalized)  # reconstructions: [B,3,T,H,W]; latent: [B,C,T,h,w]

    def export_latent_stats(self, path: str | Path) -> None:
        payload = {
            "mean": self.latent_mean.detach().float().cpu().reshape(-1),  # [C]
            "std": self.latent_std.detach().float().cpu().reshape(-1),  # [C]
            "min_range": float(self.min_range.item()),
            "max_range": float(self.max_range.item()),
        }
        if "://" in str(path):
            easy_io.dump(payload, str(path), file_format="pt")
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, path)

    def forward(self, data_batch: dict[str, Any], iteration: int = 0) -> tuple[dict[str, Any], torch.Tensor]:
        """Enter the loss graph through DDP.forward for gradient synchronization."""
        return self.training_step(data_batch, iteration)  # loss: []; output tensors: batch-dependent
