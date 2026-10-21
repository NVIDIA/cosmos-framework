# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Accepted-job and attempt journals for checkpoint-driven evaluation operations."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass

from cosmos_framework.posttrain.checkpoint.io import atomic_write_bytes


@dataclass(frozen=True)
class AcceptedJob:
    """Scheduler-accepted job and the number of accepted attempts including it."""

    job_id: str
    attempts: int


@dataclass(frozen=True)
class AttemptJournal:
    """One-file-per-work-key counter that advances only after scheduler acceptance."""

    root: str

    def path(self, key: Sequence[str]) -> str:
        return os.path.join(self.root, "_".join(map(str, key)))

    def count(self, key: Sequence[str]) -> int:
        try:
            with open(self.path(key), encoding="utf-8") as handle:
                return int(handle.read().strip() or 0)
        except (OSError, ValueError):
            return 0

    def record_accepted(self, key: Sequence[str]) -> int:
        os.makedirs(self.root, exist_ok=True)
        attempts = self.count(key) + 1
        atomic_write_bytes(self.path(key), str(attempts).encode())
        return attempts


def read_accepted_job(path: str) -> AcceptedJob | None:
    """Read the current JSON record; a legacy bare job id counts as the first accepted attempt."""
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read().strip()
    except OSError:
        return None
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    if isinstance(payload, dict) and "job" in payload:
        return AcceptedJob(str(payload["job"]), int(payload.get("attempts", 1)))
    return AcceptedJob(text, 1) if text else None


def write_accepted_job(path: str, job_id: str, attempts: int) -> None:
    """Atomically record a scheduler-accepted job."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = json.dumps({"job": str(job_id), "attempts": int(attempts)}).encode()
    atomic_write_bytes(path, payload)
