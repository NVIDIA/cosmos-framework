# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Dependency-free JSON-lines lifecycle and metric records for workflow actions."""

from __future__ import annotations

import contextvars
import json
import os
import threading
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from pathlib import Path


class Status(str, Enum):
    STARTED = "STARTED"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class Verbosity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class StatusLogger:
    def __init__(self, filename=None, *, is_master=True, **_):
        self.filename = Path(filename) if filename else None
        self.is_master = is_master
        self.kpi = {}
        self._lock = threading.Lock()

    def write(self, *, status_level=Status.RUNNING, message="", verbosity_level=Verbosity.INFO, data=None):
        if self.filename is None or not self.is_master:
            return
        now = datetime.now(timezone.utc)
        payload = {
            **(data or {}),
            "date": now.strftime("%m/%d/%Y"),
            "time": now.strftime("%H:%M:%S"),
            "status": status_level,
            "verbosity": verbosity_level,
            "message": message,
        }
        if self.kpi:
            payload["kpi"] = dict(self.kpi)
        with self._lock:
            self.filename.parent.mkdir(parents=True, exist_ok=True)
            with self.filename.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, default=str) + "\n")


_logger = contextvars.ContextVar("cosmos_workflow_status", default=None)


def get_status_logger():
    return _logger.get() or StatusLogger()


def set_status_logger(logger):
    _logger.set(logger)


def log_workflow_status(data, step=None, component_name="Cosmos", max_steps=None, current_epoch=None, max_epochs=None):
    logger = get_status_logger()
    logger.kpi = dict(data)
    progress = {"component": component_name}
    if current_epoch is not None:
        progress.update(epoch=current_epoch, max_epoch=max_epochs)
    elif step is not None:
        progress.update(step=step, max_step=max_steps)
    logger.write(data=progress, message=f"{component_name} in progress")


def monitor_status(name="Cosmos", mode="train", results_dir=None, verbosity=None):
    """Publish terminal success only when the action returns; preserve failures."""

    def decorate(runner):
        @wraps(runner)
        def run(*args, **kwargs):
            directory = results_dir
            if directory is None and args:
                directory = getattr(args[0], "results_dir", None)
                config_path = getattr(args[0], "config", None)
                if not directory and isinstance(config_path, (str, Path)):
                    import toml

                    directory = toml.load(config_path).get("results_dir")
            filename = os.environ.get("COSMOS_STATUS_FILE")
            if not filename:
                filename = str(Path(directory or os.environ.get("COSMOS_RESULTS_DIR", "./results")) / "status.json")
            rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
            logger = StatusLogger(filename, is_master=rank == 0)
            token = _logger.set(logger)
            try:
                logger.write(status_level=Status.STARTED, message=f"Starting {name} {mode}")
                result = runner(*args, **kwargs)
                logger.write(status_level=Status.SUCCESS, message=f"{name} {mode} completed successfully")
                return result
            except BaseException as error:
                logger.write(
                    status_level=Status.FAILURE,
                    verbosity_level=Verbosity.ERROR,
                    message=f"{name} {mode} failed: {error}",
                )
                raise
            finally:
                _logger.reset(token)

        return run

    return decorate
