# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_fd_id_libero_posttrain`` — cell 2: ``action_fd_libero_posttrain`` with forward and inverse
dynamics mixed 50/50 per visit and video frame rates mixed 5/10/20 fps (17 frames at source stride 4/2/1,
dense 20 Hz actions of length 64/32/16, sequence-plan "Case C"). Same data, held-out split, token budget
(37000 x 16 per step), seed, optimizer, scheduler and checkpointing as cell 1; the three dataset deltas are
``mode``, ``video_fps_choices`` and ``mix_seed``. Conventions are derived from the framework (no local field
guide): per-visit uniform FD/ID draw (WAM excluded; the shipped ``mode="joint"`` would be 1/3 FD, 1/3 ID,
1/3 WAM), per-visit uniform fps, draws from ``random.Random(f"{mix_seed}:{epoch}:{idx}")``.

Expect the console ``Loss`` / ``train/loss_avg`` to read several times higher than cell 1 (the 10x-weighted
action loss now contributes; ID items add zeros to the vision mean): compare per-modality losses, which
``ModalityLossConsole`` prints, and the evaluation metrics, not the totals.
"""

import copy
from typing import Any

import torch
from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.every_n import EveryN
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_fd_libero_posttrain import (
    action_fd_libero_posttrain,
)
from cosmos_framework.utils import log
from cosmos_framework.utils.lazy_config import LazyCall as L


class ModalityLossConsole(EveryN):
    """Print per-modality flow-matching losses (mean over the last ``every_n`` optimizer steps) and the
    realised mode / video-fps mix of the last micro-batch, so mixed-mode runs stay interpretable with
    W&B disabled."""

    def __init__(self, every_n: int = 10) -> None:
        super().__init__(every_n=every_n, step_size=1, barrier_after_run=False)
        self._sums: dict[str, torch.Tensor] = {}
        self._count = 0
        self._mode_counts: dict[str, int] = {}
        self._fps_counts: dict[int, int] = {}

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0) -> None:
        for key, value in output_batch.items():
            if "loss" in key and "per_instance" not in key and isinstance(value, torch.Tensor) and value.numel() == 1:
                self._sums[key] = self._sums.get(key, 0.0) + value.detach().float()
        self._count += 1
        modes = data_batch.get("mode")
        if isinstance(modes, (list, tuple)):
            for m in modes:
                self._mode_counts[str(m)] = self._mode_counts.get(str(m), 0) + 1
        fps = data_batch.get("conditioning_fps")
        if isinstance(fps, torch.Tensor) and fps.numel() > 0:
            for v in fps.flatten().tolist():
                self._fps_counts[int(round(v))] = self._fps_counts.get(int(round(v)), 0) + 1
        super().on_training_step_end(model, data_batch, output_batch, loss, iteration=iteration)

    def every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration: int) -> None:
        if self._count == 0:
            return
        parts = [f"{k}={float(v) / self._count:.5f}" for k, v in sorted(self._sums.items())]
        n_modes = sum(self._mode_counts.values()) or 1
        mode_str = " ".join(f"{k}={v / n_modes:.2f}" for k, v in sorted(self._mode_counts.items()))
        n_fps = sum(self._fps_counts.values()) or 1
        fps_str = " ".join(f"{k}fps={v / n_fps:.2f}" for k, v in sorted(self._fps_counts.items()))
        log.info(
            f"modality_loss (mean of last {self._count} steps, iteration {iteration}): {' '.join(parts)} | "
            f"last-microbatch share: {mode_str} | {fps_str}"
        )
        self._sums, self._count, self._mode_counts, self._fps_counts = {}, 0, {}, {}


action_fd_id_libero_posttrain: Any = copy.deepcopy(action_fd_libero_posttrain)
action_fd_id_libero_posttrain.job.name = "${now:%Y-%m-%d_%H-%M-%S}_action_fd_id_libero_posttrain"
action_fd_id_libero_posttrain.dataloader_train.dataset_name = "action_libero_fd_id"
_ds = action_fd_id_libero_posttrain.dataloader_train.dataloader.datasets.libero.dataset
_ds.mode = (
    "forward_dynamics+inverse_dynamics"  # per-visit uniform FD/ID (WAM excluded); NOT 'joint' (1/3 each of FD/ID/WAM)
)
_ds.video_fps_choices = [
    5,
    10,
    20,
]  # per-visit uniform video fps; 17 frames at stride 4/2/1, dense 20 Hz actions (64/32/16)
_ds.mix_seed = 42  # per-visit RNG: random.Random(f"{mix_seed}:{epoch}:{idx}"); override per relaunch segment
action_fd_id_libero_posttrain.trainer.callbacks["modality_loss_console"] = L(ModalityLossConsole)(every_n=10)

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="action_fd_id_libero_posttrain",
    node=action_fd_id_libero_posttrain,
)
