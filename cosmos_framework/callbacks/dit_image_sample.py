# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fixed class-conditioned image sampling callback for DiT training."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torchvision
import wandb

from cosmos_framework.callbacks.every_n import EveryN
from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.trainer import ImaginaireTrainer
from cosmos_framework.utils import distributed


class DiTImageSampleCallback(EveryN):
    """Generate fixed ImageNet class samples through ``model.generate_image``."""

    def __init__(
        self,
        every_n: int = 5000,
        class_ids: list[int] | None = None,
        cfg_scales: list[float] | None = None,
        num_steps: int = 50,
        seed: int = 0,
        is_ema: bool = True,
        run_at_start: bool = False,
    ) -> None:
        super().__init__(every_n=every_n, run_at_start=run_at_start)
        self.class_ids = class_ids or [0, 1, 2, 3]
        self.cfg_scales = cfg_scales or [1.0, 1.25, 1.5, 2.0]
        self.num_steps = num_steps
        self.seed = seed
        self.is_ema = is_ema
        self.rank = distributed.get_rank()

    @torch.no_grad()
    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        del trainer, data_batch, output_batch, loss

        if not hasattr(model, "generate_image"):
            raise AttributeError("DiTImageSampleCallback requires model.generate_image().")
        if self.is_ema and not model.config.ema.enabled:
            return

        was_training = model.training
        context: Any = model.ema_scope("dit_image_sample") if self.is_ema else nullcontext()
        generated_rows: list[torch.Tensor] = []
        seed_list = [self.seed + sample_idx for sample_idx in range(len(self.class_ids))]
        try:
            with context:
                for cfg_scale in self.cfg_scales:
                    images = model.generate_image(
                        class_ids=self.class_ids,
                        num_steps=self.num_steps,
                        cfg_scale=cfg_scale,
                        seed=seed_list,
                    )  # [B,3,H,W]
                    if self.rank == 0:
                        generated_rows.append(images.detach().float().cpu())  # [B,3,H,W]
        finally:
            if was_training:
                model.train()

        if self.rank != 0 or wandb.run is None or not generated_rows:
            return

        grid_images = torch.cat(generated_rows, dim=0)  # [R*B,3,H,W]
        grid = torchvision.utils.make_grid(grid_images, nrow=len(self.class_ids), padding=2, normalize=False)  # [3,H,W]
        grid_np = grid.clamp(0.0, 1.0).permute(1, 2, 0).numpy()  # [H,W,3]
        tag = "ema" if self.is_ema else "reg"
        caption = f"classes={self.class_ids}, cfg={self.cfg_scales}, steps={self.num_steps}, seed={self.seed}, {tag}"
        wandb.log(
            {f"dit_image_sample/{tag}": wandb.Image(grid_np, caption=caption)},
            step=iteration,
        )
