# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""LiDAR V1.2 continuation recipe; paths are supplied by the caller."""

from dataclasses import asdict, dataclass
from typing import Any

from cosmos_framework.model.generator.tokenizers.lidar.lidar_tokenizer_v1 import DEFAULT_NETWORK_CONFIG
from cosmos_framework.model.generator.tokenizers.lidar.range_projection import V1P2_TRANSFER_RANGE_PROJECTION
from cosmos_framework.model.tokenizer.lidar.loss import LidarVAELoss
from cosmos_framework.model.tokenizer.lidar.model import LidarTokenizerModelConfig
from cosmos_framework.utils.config import PowerEMAConfig


@dataclass(frozen=True)
class LidarV1P2Recipe:
    batch_size: int = 2
    num_source_frames: int = 8
    max_steps: int = 30_000
    save_every: int = 5_000
    learning_rate: float = 1e-5
    end_lr: float = 1e-6
    warmup_steps: int = 1_000
    max_grad_norm: float = 1.0
    seed: int = 42

    def model_config(self, checkpoint: str | None = None) -> LidarTokenizerModelConfig:
        network = dict(DEFAULT_NETWORK_CONFIG)
        network.update(
            resolution=[128, 1808],
            patch_size=[2, 2],
            window_size=[5, 45],
            depths=[3, 3, 3, 3],
            num_heads=[4, 4, 8, 8],
            dilation=[1, 1, 1, 1],
            temporal_downsample=[False, False, False],
            temporal_upsample=[False, False, False],
            gradient_checkpointing=True,
        )
        return LidarTokenizerModelConfig(
            range_projection=V1P2_TRANSFER_RANGE_PROJECTION,
            network=network,
            loss=LidarVAELoss(
                recon_weight=2.0,
                range_weight=1.0,
                intensity_weight=1.0,
                kl_weight=1e-6,
                validity_weight=1.0,
                point_weight=1.0,
                gradient_weight=1.0,
                recon_reduction="masked_mean_all_rays",
            ),
            precision="float32",
            ema=PowerEMAConfig(enabled=True, s=0.1),
            init_from_checkpoint=checkpoint,
        )

    def optimizer_config(self) -> dict[str, Any]:
        return {"lr": self.learning_rate, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}

    def scheduler_config(self) -> dict[str, Any]:
        return {"warmup_steps": self.warmup_steps, "total_steps": self.max_steps, "end_lr": self.end_lr}

    def manifest(self) -> dict[str, Any]:
        config = self.model_config()
        return {
            "recipe": "lidar_v1p2",
            "training": asdict(self),
            "network": config.network,
            "range_projection": config.range_projection.to_dict(),
            "precision": config.precision,
            "ema_relative_width": config.ema.s,
            "latent_stats_momentum": config.latent_stats_ema_momentum,
            "loss": {
                "recon": 2.0,
                "range": 1.0,
                "intensity": 1.0,
                "validity": 1.0,
                "point": 1.0,
                "gradient": 1.0,
                "gradient_term_count": 2,
                "kl": 1e-6,
                "reduction": "masked_mean_all_rays",
            },
            "optimizer": self.optimizer_config(),
            "scheduler": self.scheduler_config(),
        }
