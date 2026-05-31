# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import time

import torch
import wandb
from torch import Tensor

from cosmos_framework.callbacks.every_n import EveryN
from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.trainer import ImaginaireTrainer
from cosmos_framework.utils import log
from cosmos_framework.utils.distributed import rank0_only
from cosmos_framework.utils.easy_io import easy_io


class IterSpeed(EveryN):
    """
    Args:
        hit_thres (int): Number of iterations to wait before logging.
        save_s3 (bool): Whether to save to S3.
        save_s3_every_log_n (int): Save to S3 every n log iterations, which means save_s3_every_log_n n * every_n global iterations.
    """

    def __init__(self, *args, hit_thres: int = 5, save_s3: bool = True, save_s3_every_log_n: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.time = None
        self.hit_counter = 0
        self.hit_thres = hit_thres
        self.save_s3 = save_s3
        self.save_s3_every_log_n = save_s3_every_log_n
        self.name = self.__class__.__name__
        self.last_hit_time = time.time()

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if self.hit_counter < self.hit_thres:
            log.info(
                f"Iteration {iteration}: "
                f"Hit counter: {self.hit_counter + 1}/{self.hit_thres} | "
                f"Loss: {loss.detach().item():.4f} | "
                f"Time: {time.time() - self.last_hit_time:.2f}s",
                rank0_only=False,
            )
            self.hit_counter += 1
            self.last_hit_time = time.time()
            #! useful for large scale training and avoid oom crash in the first two iterations!!!
            torch.cuda.synchronize()
            return
        super().on_training_step_end(model, data_batch, output_batch, loss, iteration)

    @rank0_only
    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, Tensor],
        output_batch: dict[str, Tensor],
        loss: Tensor,
        iteration: int,
    ) -> None:
        if self.time is None:
            self.time = time.time()
            return
        cur_time = time.time()
        iter_speed = (cur_time - self.time) / self.every_n / self.step_size

        log.info(
            f"{iteration} : iter_speed {iter_speed:.2f} seconds per iteration | Loss: {loss.detach().item():.4f}",
            rank0_only=False,
        )

        per_sample_batch_counter = dict()
        if hasattr(model, "is_image_batch"):
            is_image_batch = model.is_image_batch(data_batch)
            if is_image_batch:
                image_batch_size = len(data_batch[model.input_image_key])
                per_sample_batch_counter["image_batch_size"] = image_batch_size
            else:
                video_batch_size = len(data_batch[model.input_video_key])
                per_sample_batch_counter["video_batch_size"] = video_batch_size

        if wandb.run:
            sample_counter = getattr(trainer, "sample_counter", iteration)
            wandb.log(
                {
                    "timer/iter_speed": iter_speed,
                    "sample_counter": sample_counter,
                }
                | per_sample_batch_counter,
                step=iteration,
            )
        self.time = cur_time
        if self.save_s3:
            if iteration % (self.save_s3_every_log_n * self.every_n) == 0:
                easy_io.dump(
                    {
                        "iter_speed": iter_speed,
                        "iteration": iteration,
                    },
                    f"s3://rundir/{self.name}/iter_{iteration:09d}.yaml",
                )
