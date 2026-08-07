# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""TAO-compatible lifecycle and metric logging for Cosmos Framework training."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback


def _to_json_value(value: Any) -> Any:
    """Recursively convert tensors and array-like values to JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.numel() == 1:
            return value.item()
        return value.cpu().tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except (TypeError, ValueError):
            pass
    return value


class _TAOStatusWriter:
    """Use TAO Core when available, with a compatible JSON-lines fallback."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        self._tao_logger = None
        self._tao_status = None
        self._tao_verbosity = None

        try:
            from nvidia_tao_core.loggers.logging import Status, StatusLogger, Verbosity
        except ImportError:
            log.warning(f"nvidia_tao_core is not installed; writing TAO-compatible JSON records directly to {filename}")
        else:
            self._tao_status = Status
            self._tao_verbosity = Verbosity
            self._tao_logger = StatusLogger(
                filename=filename,
                is_master=True,
                verbosity=Verbosity.INFO,
                append=True,
            )

    def write(
        self,
        *,
        status: str,
        message: str,
        data: dict[str, Any] | None = None,
        kpi: dict[str, Any] | None = None,
        verbosity: str = "INFO",
    ) -> None:
        data = _to_json_value(data or {})
        kpi = _to_json_value(kpi or {})

        if self._tao_logger is not None:
            status_level = getattr(self._tao_status, status)
            verbosity_level = getattr(self._tao_verbosity, verbosity)
            self._tao_logger.kpi = kpi
            self._tao_logger.write(
                data=data,
                status_level=status_level,
                verbosity_level=verbosity_level,
                message=message,
            )
            return

        now = datetime.now()
        payload: dict[str, Any] = {
            **data,
            "date": f"{now.month}/{now.day}/{now.year}",
            "time": f"{now.hour}:{now.minute}:{now.second}",
            "status": status,
            "verbosity": verbosity,
            "message": message,
        }
        if kpi:
            payload["kpi"] = kpi
        with open(self.filename, "a", encoding="utf-8") as status_file:
            status_file.write(json.dumps(payload, default=str) + "\n")


def write_early_failure(error: BaseException) -> bool:
    """Write a terminal TAO record before the callback/config exists.

    The orchestration layer supplies ``TAO_STATUS_FILE`` for direct launches,
    or the normal TAO job/result variables. No implicit host path is used.
    """
    status_path = os.environ.get("TAO_STATUS_FILE")
    if not status_path:
        job_id = os.environ.get("TAO_JOB_ID")
        results_root = os.environ.get("TAO_RESULTS_ROOT")
        if job_id and results_root:
            status_path = os.path.join(results_root, job_id, "status.json")
    if not status_path:
        api_job_id = os.environ.get("TAO_API_JOB_ID")
        results_root = os.environ.get("TAO_API_RESULTS_DIR")
        if api_job_id and results_root:
            status_path = os.path.join(results_root, api_job_id, "status.json")
    if not status_path:
        return False
    _TAOStatusWriter(status_path).write(
        status="FAILURE",
        verbosity="ERROR",
        message=f"Cosmos Framework training failed before callback initialization: {error}",
        data={"phase": "preflight_or_initialization", "error_type": type(error).__name__},
    )
    return True


class TAOStatusCallback(Callback):
    """Write TAO lifecycle, training, and validation records from rank zero.

    The output path is resolved in this order:

    1. ``status_file_path`` when explicitly configured.
    2. ``$TAO_RESULTS_ROOT/$TAO_JOB_ID/status.json`` (TAO SDK).
    3. ``$TAO_API_RESULTS_DIR/$TAO_API_JOB_ID/status.json`` (TAO API).
    4. ``<job.path_local>/status.json`` for direct launches.
    """

    def __init__(
        self,
        enabled: bool = False,
        status_file_path: str | None = None,
        experiment_name: str = "",
        logging_interval: int = 1,
        validation_heartbeat_interval: int = 1,
    ) -> None:
        if logging_interval < 1:
            raise ValueError("logging_interval must be >= 1")
        if validation_heartbeat_interval < 1:
            raise ValueError("validation_heartbeat_interval must be >= 1")
        self.enabled = enabled
        self.status_file_path = status_file_path
        self.experiment_name = experiment_name
        self.logging_interval = logging_interval
        self.validation_heartbeat_interval = validation_heartbeat_interval
        self._writer: _TAOStatusWriter | None = None
        self._train_start_time = 0.0
        self._step_start_time = 0.0
        self._validation_batches = 0
        self._validation_loss_numerator = 0.0
        self._validation_loss_denominator = 0
        self._train_loss_numerator = 0.0
        self._train_loss_denominator = 0
        self.last_training_loss: float | None = None
        self.last_validation_loss: float | None = None

    def _is_rank_zero(self) -> bool:
        if dist.is_available() and dist.is_initialized():
            return distributed.is_rank0()
        return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0

    def _component_name(self) -> str:
        if self.experiment_name:
            return self.experiment_name
        return getattr(self.config.job, "name", "Cosmos Framework SFT")

    def _resolve_status_file(self) -> str:
        if self.status_file_path:
            return self.status_file_path

        job_id = os.environ.get("TAO_JOB_ID")
        if job_id:
            results_root = os.environ.get("TAO_RESULTS_ROOT")
            if not results_root:
                raise RuntimeError("TAO_RESULTS_ROOT is required when TAO_JOB_ID is set")
            return os.path.join(results_root, job_id, "status.json")

        api_job_id = os.environ.get("TAO_API_JOB_ID")
        if api_job_id:
            results_root = os.environ.get("TAO_API_RESULTS_DIR")
            if not results_root:
                raise RuntimeError("TAO_API_RESULTS_DIR is required when TAO_API_JOB_ID is set")
            return os.path.join(results_root, api_job_id, "status.json")

        return os.path.join(self.config.job.path_local, "status.json")

    def _get_writer(self) -> _TAOStatusWriter | None:
        if not self.enabled or not self._is_rank_zero():
            return None
        if self._writer is None:
            self._writer = _TAOStatusWriter(self._resolve_status_file())
        return self._writer

    def _progress_data(self, iteration: int, seconds_per_step: float | None = None) -> dict[str, Any]:
        trainer = getattr(self, "trainer", None)
        max_step = int(getattr(trainer, "max_iterations", self.config.trainer.max_iter))
        if seconds_per_step is None:
            elapsed = max(time.monotonic() - self._train_start_time, 0.0)
            seconds_per_step = elapsed / max(iteration, 1)
        eta_seconds = max(max_step - iteration, 0) * seconds_per_step
        data = {
            "component": self._component_name(),
            "step": iteration,
            "max_step": max_step,
            "time_per_step": str(timedelta(seconds=seconds_per_step)),
            "eta": str(timedelta(seconds=eta_seconds)),
        }
        steps_per_epoch = getattr(trainer, "steps_per_epoch", None)
        num_epochs = getattr(trainer, "num_epochs", None)
        if steps_per_epoch and num_epochs:
            completed_epochs = iteration // steps_per_epoch
            if iteration == 0:
                epoch = 1
                step_in_epoch = 0
            elif iteration % steps_per_epoch == 0:
                epoch = min(completed_epochs, num_epochs)
                step_in_epoch = steps_per_epoch
            else:
                epoch = min(completed_epochs + 1, num_epochs)
                step_in_epoch = iteration % steps_per_epoch
            data.update(
                {
                    "epoch": epoch,
                    "max_epoch": num_epochs,
                    "completed_epochs": min(completed_epochs, num_epochs),
                    "step_in_epoch": step_in_epoch,
                    "steps_per_epoch": steps_per_epoch,
                    "time_per_epoch": str(timedelta(seconds=seconds_per_step * steps_per_epoch)),
                }
            )
        return data

    def _epoch_label(self, iteration: int) -> str:
        progress = self._progress_data(iteration, seconds_per_step=0.0)
        if "epoch" not in progress:
            return f"training step {iteration}/{progress['max_step']}"
        return f"epoch {progress['epoch']}/{progress['max_epoch']}"

    @staticmethod
    def _local_token_stats(
        loss: torch.Tensor,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        numerator = output_batch.get("loss_numerator")
        denominator = output_batch.get("loss_denominator")
        if numerator is not None and denominator is not None:
            return numerator.detach(), denominator.detach().to(dtype=torch.long)

        # Backward-compatible fallback for non-VLM models. It is deliberately
        # sample-weighted and is never used by the Cosmos3 VLM path, which
        # always emits exact token statistics.
        sample_count = next(
            (
                int(value.shape[0])
                for value in data_batch.values()
                if isinstance(value, torch.Tensor) and value.ndim > 0
            ),
            1,
        )
        return (
            loss.detach() * sample_count,
            torch.tensor(sample_count, device=loss.device, dtype=torch.long),
        )

    @classmethod
    def _global_token_average(
        cls,
        loss: torch.Tensor,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
    ) -> tuple[float, float, int]:
        numerator, denominator = cls._local_token_stats(loss, data_batch, output_batch)
        numerator = numerator.clone()
        denominator = denominator.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(numerator, op=dist.ReduceOp.SUM)
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
        global_denominator = int(denominator.item())
        global_numerator = float(numerator.item())
        return global_numerator / max(global_denominator, 1), global_numerator, global_denominator

    @staticmethod
    def _reduce_accumulator(numerator: float, denominator: int) -> tuple[float, int]:
        device = "cuda" if dist.is_available() and dist.is_initialized() else "cpu"
        values = torch.tensor([numerator, float(denominator)], dtype=torch.float64, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        return float(values[0].item()), int(values[1].item())

    def on_train_start(self, model: Any, iteration: int = 0) -> None:
        self._train_start_time = time.monotonic()
        self._train_loss_numerator = 0.0
        self._train_loss_denominator = 0
        writer = self._get_writer()
        if writer is not None:
            writer.write(
                status="STARTED",
                message=f"Starting {self._component_name()} training",
                data={
                    **self._progress_data(iteration, seconds_per_step=0.0),
                    "parameter_summary": getattr(model, "parameter_summary", None),
                },
            )
            log.info(f"TAO status will be logged to {writer.filename}")

    def on_training_step_start(self, model: Any, data: dict[str, Any], iteration: int = 0) -> None:
        self._step_start_time = time.monotonic()

    def on_training_step_batch_end(
        self,
        model: Any,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        numerator, denominator = self._local_token_stats(loss, data_batch, output_batch)
        self._train_loss_numerator += float(numerator.item())
        self._train_loss_denominator += int(denominator.item())

    def on_training_step_end(
        self,
        model: Any,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        interval = int(self.config.trainer.logging_iter) * self.logging_interval
        if iteration % interval != 0:
            return

        average_loss, numerator, denominator = self._global_token_average(loss, data_batch, output_batch)
        writer = self._get_writer()
        if writer is None:
            return
        seconds_per_step = max(time.monotonic() - self._step_start_time, 0.0)
        kpi = {
            "train/step_loss": average_loss,
            "train/step_loss_numerator": numerator,
            "train/step_loss_denominator": denominator,
        }
        progress = self._progress_data(iteration, seconds_per_step=seconds_per_step)
        if "epoch" in progress:
            message = (
                f"Training epoch {progress['epoch']}/{progress['max_epoch']}, "
                f"step {progress['step_in_epoch']}/{progress['steps_per_epoch']} "
                f"(global step {iteration}/{progress['max_step']}) - Loss: {average_loss:.6f}"
            )
        else:
            message = f"Training step {iteration}/{progress['max_step']} - Loss: {average_loss:.6f}"
        writer.write(
            status="RUNNING",
            message=message,
            data=progress,
            kpi=kpi,
        )

    def on_validation_start(self, model: Any, dataloader_val: Any, iteration: int = 0) -> None:
        self._validation_batches = 0
        self._validation_loss_numerator = 0.0
        self._validation_loss_denominator = 0
        writer = self._get_writer()
        if writer is not None:
            writer.write(
                status="RUNNING",
                message=f"Starting validation for {self._epoch_label(iteration)}",
                data={**self._progress_data(iteration), "phase": "validation_starting"},
            )
            log.info(f"Starting validation for {self._epoch_label(iteration)}")

    def on_validation_step_end(
        self,
        model: Any,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        local_numerator, local_denominator = self._local_token_stats(loss, data_batch, output_batch)
        average_loss, _, _ = self._global_token_average(loss, data_batch, output_batch)
        self._validation_batches += 1
        self._validation_loss_numerator += float(local_numerator.item())
        self._validation_loss_denominator += int(local_denominator.item())

        if self._validation_batches % self.validation_heartbeat_interval != 0:
            return
        writer = self._get_writer()
        max_validation_batches = getattr(self.config.trainer, "max_val_iter", None)
        batch_progress = (
            f"{self._validation_batches}/{max_validation_batches}"
            if max_validation_batches is not None
            else str(self._validation_batches)
        )
        if writer is not None:
            writer.write(
                status="RUNNING",
                message=(
                    f"Validation {self._epoch_label(iteration)}, batch {batch_progress} - Loss: {average_loss:.6f}"
                ),
                data={
                    **self._progress_data(iteration),
                    "phase": "validation_batch_complete",
                    "validation_batch": self._validation_batches,
                    "max_validation_batches": max_validation_batches,
                },
                kpi={"val/batch_loss": average_loss},
            )
            log.info(f"Validation {self._epoch_label(iteration)}, batch {batch_progress} - Loss: {average_loss:.6f}")

    def on_validation_end(self, model: Any, iteration: int = 0) -> None:
        numerator, denominator = self._reduce_accumulator(
            self._validation_loss_numerator, self._validation_loss_denominator
        )
        if denominator == 0:
            log.warning("TAO validation logging saw zero samples; no val/loss record was written")
            return

        self.last_validation_loss = numerator / denominator
        writer = self._get_writer()
        if writer is not None:
            writer.write(
                status="RUNNING",
                message=(
                    f"Validation complete for {self._epoch_label(iteration)} - Loss: {self.last_validation_loss:.6f}"
                ),
                data={
                    **self._progress_data(iteration),
                    "phase": "validation_complete",
                    "validation_batches": self._validation_batches,
                    "validation_loss_numerator": numerator,
                    "validation_valid_label_count": denominator,
                },
                kpi={
                    "val/loss": self.last_validation_loss,
                    "val/avg_loss": self.last_validation_loss,
                    "val/loss_numerator": numerator,
                    "val/valid_label_count": denominator,
                },
            )
        log.info(f"Validation loss ({self._epoch_label(iteration)}): {self.last_validation_loss:.6f}")

    def on_save_checkpoint_success(self, iteration: int = 0, elapsed_time: float = 0) -> None:
        checkpoint_path = None
        checkpoint_root = Path(self.config.job.path_local) / "checkpoints"
        latest = checkpoint_root / "latest_checkpoint.txt"
        if latest.is_file():
            checkpoint_name = latest.read_text(encoding="utf-8").strip()
            if checkpoint_name:
                checkpoint_path = str((checkpoint_root / checkpoint_name).resolve())
        writer = self._get_writer()
        if writer is not None:
            writer.write(
                status="RUNNING",
                message=f"Checkpoint saved successfully at step {iteration}",
                data={
                    **self._progress_data(iteration),
                    "phase": "checkpoint_complete",
                    "checkpoint_iteration": iteration,
                    "checkpoint_elapsed_seconds": elapsed_time,
                    "checkpoint_path": checkpoint_path,
                },
            )

    def on_train_end(self, model: Any, iteration: int = 0) -> None:
        numerator, denominator = self._reduce_accumulator(
            self._train_loss_numerator, self._train_loss_denominator
        )
        if denominator == 0:
            raise RuntimeError("TAO metric collection observed zero valid training labels")
        self.last_training_loss = numerator / denominator
        writer = self._get_writer()
        if writer is not None:
            writer.write(
                status="RUNNING",
                message=f"Training complete - token-weighted loss: {self.last_training_loss:.6f}",
                data={
                    **self._progress_data(iteration),
                    "phase": "training_complete",
                    "train_loss_numerator": numerator,
                    "train_valid_label_count": denominator,
                },
                kpi={
                    "train/avg_loss": self.last_training_loss,
                    "train/loss_numerator": numerator,
                    "train/valid_label_count": denominator,
                },
            )

    def on_app_end(self) -> None:
        writer = self._get_writer()
        if writer is not None:
            writer.write(
                status="SUCCESS",
                message=f"{self._component_name()} training completed successfully",
                data=self._progress_data(
                    int(getattr(getattr(self, "trainer", None), "max_iterations", self.config.trainer.max_iter))
                ),
                kpi={
                    **(
                        {"train/avg_loss": self.last_training_loss}
                        if self.last_training_loss is not None
                        else {}
                    ),
                    **(
                        {"val/loss": self.last_validation_loss, "val/avg_loss": self.last_validation_loss}
                        if self.last_validation_loss is not None
                        else {}
                    ),
                },
            )

    def on_exception(self, error: BaseException) -> None:
        writer = self._get_writer()
        if writer is not None:
            writer.write(
                status="FAILURE",
                verbosity="ERROR",
                message=f"{self._component_name()} training failed: {error}",
                data={"component": self._component_name(), "error_type": type(error).__name__},
            )
