# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from dataclasses import dataclass
from typing import cast

import torch
import wandb
from torch.nn.utils.clip_grad import clip_grad_norm_

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils.callback import Callback

__all__: tuple[str, ...] = ("GradClip",)


@torch.jit.script
def _fused_nan_to_num(params: list[torch.Tensor]) -> None:
    for param in params:
        torch.nan_to_num(param, nan=0.0, posinf=0.0, neginf=0.0, out=param)


def _to_scalar(value: torch.Tensor | float) -> float:
    """Convert tensor or DTensor to Python float for logging; avoids DTensor collectives."""
    if not isinstance(value, torch.Tensor):
        return float(value)
    to_local = getattr(value, "to_local", None)
    t = cast(torch.Tensor, to_local() if callable(to_local) else value)
    return t.detach().float().item()


def _replicate_norm(value: torch.Tensor | float) -> torch.Tensor | float:
    """Materialize a DTensor norm partial before logging it as a global scalar."""
    full_tensor = getattr(value, "full_tensor", None)
    if callable(full_tensor):
        return cast(torch.Tensor, full_tensor())  # []
    return value


def _clip_scale(total_norm: torch.Tensor | float, clip_norm: float) -> float:
    """Return the coefficient used by PyTorch gradient clipping."""
    if isinstance(total_norm, torch.Tensor):
        scale = torch.clamp(clip_norm / (total_norm + 1e-6), max=1.0)  # []
        return _to_scalar(scale)
    return min(1.0, clip_norm / (float(total_norm) + 1e-6))


def _phase_optimizer_parameters(
    model: torch.nn.Module, optimizer: object, iteration: int
) -> tuple[str, list[torch.nn.Parameter]]:
    assert hasattr(model, "get_optimizer_key"), (
        f"{type(model).__name__} missing get_optimizer_key — GradClipCosmos3 requires a PhaseOptimizer-compatible model"
    )
    assert hasattr(optimizer, "parameters_for_key"), (
        f"{type(optimizer).__name__} missing parameters_for_key — GradClipCosmos3 requires PhaseOptimizer"
    )
    key = str(model.get_optimizer_key(iteration))
    return key, list(optimizer.parameters_for_key(key))


@dataclass
class _MagnitudeRecord:
    state: float = 0
    iter_count: int = 0

    def reset(self) -> None:
        self.state = 0
        self.iter_count = 0

    def update(self, cur_state: torch.Tensor | float) -> None:
        self.state += _to_scalar(cur_state)
        self.iter_count += 1

    def get_stat(self) -> float:
        if self.iter_count > 0:
            avg_state = self.state / self.iter_count
        else:
            avg_state = 0.0
        self.reset()
        return avg_state


class GradClip(Callback):
    """
    This callback is used to clip the gradient norm of the model.
    It also logs the average gradient norm of the model to wandb.
    """

    def __init__(self, clip_norm: float = 1.0, force_finite: bool = True) -> None:
        self.clip_norm: float = clip_norm
        self.force_finite: bool = force_finite

        self.img_mag_log: _MagnitudeRecord = _MagnitudeRecord()
        self.video_mag_log: _MagnitudeRecord = _MagnitudeRecord()
        self.phase_grad_norm_logs: dict[str, _MagnitudeRecord] = {}
        self.phase_clip_scale_logs: dict[str, _MagnitudeRecord] = {}
        self._cur_state: _MagnitudeRecord | None = None

    def on_training_step_start(
        self, model: torch.nn.Module, data_batch: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        model._distillation_parity_grad_metrics = {}
        if model.is_image_batch(data_batch):
            self._cur_state = self.img_mag_log
        else:
            self._cur_state = self.video_mag_log

    def on_before_optimizer_step(
        self,
        model: ImaginaireModel,
        optimizer: object,
        scheduler: object,
        grad_scaler: object,
        iteration: int = 0,
    ) -> None:
        del scheduler
        optimizer_key, clip_params = _phase_optimizer_parameters(model, optimizer, iteration)
        if self.force_finite:
            params = [param.grad for param in clip_params if param.grad is not None]
            _fused_nan_to_num(params)

        clip_norm = 1e12 if not getattr(model.config, "grad_clip", True) else self.clip_norm
        total_norm = _replicate_norm(clip_grad_norm_(clip_params, clip_norm))  # []
        grad_norm = _to_scalar(total_norm)
        clip_scale = _clip_scale(total_norm, clip_norm)
        grad_metrics: dict[str, float] | None = getattr(model, "_distillation_parity_grad_metrics", None)
        if grad_metrics is None:
            grad_metrics = {}
            model._distillation_parity_grad_metrics = grad_metrics
        grad_metrics.update(
            {
                f"clip_grad_norm/{optimizer_key}_selected_preclip": grad_norm,
                f"clip_grad_norm/{optimizer_key}_selected_clip_scale": clip_scale,
                f"clip_grad_norm/{optimizer_key}_selected_clip_norm": clip_norm,
            }
        )
        self.phase_grad_norm_logs.setdefault(optimizer_key, _MagnitudeRecord()).update(grad_norm)
        self.phase_clip_scale_logs.setdefault(optimizer_key, _MagnitudeRecord()).update(clip_scale)

        self._cur_state.update(total_norm)  # type: ignore[union-attr]
        if iteration % self.config.trainer.logging_iter == 0:
            avg_img_mag, avg_video_mag = self.img_mag_log.get_stat(), self.video_mag_log.get_stat()
            metrics = {
                "clip_grad_norm/image": avg_img_mag,
                "clip_grad_norm/video": avg_video_mag,
                "iteration": iteration,
            }
            for key, record in self.phase_grad_norm_logs.items():
                if record.iter_count > 0:
                    metrics[f"clip_grad_norm/{key}_selected_preclip"] = record.get_stat()
            for key, record in self.phase_clip_scale_logs.items():
                if record.iter_count > 0:
                    metrics[f"clip_grad_norm/{key}_selected_clip_scale"] = record.get_stat()
            if wandb.run:
                wandb.log(metrics, step=iteration)
