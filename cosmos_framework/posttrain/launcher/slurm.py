# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Small dependency-free Slurm adapter shared by launch and evaluation control paths."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from typing import Any

ACTIVE_JOB_STATES = frozenset(
    {
        "PENDING",
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "SUSPENDED",
        "REQUEUED",
        "REQUEUE_HOLD",
        "RESIZING",
        "STAGE_OUT",
        "SIGNALING",
    }
)


def parse_duration(value: str) -> int | None:
    """Return Slurm duration seconds, or None for UNLIMITED/INFINITE/zero.

    A day prefix makes the first clock field hours; otherwise one/two fields
    mean minutes/minutes:seconds. Slurm permits overflowing clock components.
    """
    if value.upper() in ("UNLIMITED", "INFINITE"):
        return None
    if not re.fullmatch(r"(?:[0-9]+-)?[0-9]+(?::[0-9]+){0,2}", value):
        raise ValueError(f"invalid Slurm duration {value!r}")
    days, separator, clock = value.partition("-")
    parts = [int(item) for item in (clock if separator else days).split(":")]
    if separator:
        parts += [0] * (3 - len(parts))
        seconds = int(days) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    else:
        parts = [0] * (3 - len(parts)) + parts
        if ":" not in value:
            seconds = parts[2] * 60
        else:
            seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
    return seconds or None


def normalize_state(raw: str) -> str:
    """Normalize one Slurm State field, including truncation suffixes and cancellation details."""
    words = raw.split()
    return words[0].upper().rstrip("+") if words else "UNKNOWN"


def parse_job_id(stdout: str) -> str | None:
    """Parse sbatch --parsable output or the conventional Submitted batch job line."""
    first = stdout.strip().splitlines()[0].strip() if stdout.strip() else ""
    parsable = first.split(";", 1)[0]
    if parsable.isdigit():
        return parsable
    match = re.search(r"\bSubmitted batch job\s+(\d+)\b", stdout)
    return match.group(1) if match else None


def parse_accounting(stdout: str) -> tuple[str, int]:
    """Parse primary JobID|State|ExitCode output, ignoring step rows."""
    for line in stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 3 or "." in parts[0]:
            continue
        code_text = parts[2].split(":", 1)[0]
        return normalize_state(parts[1]), int(code_text) if code_text.isdigit() else -1
    return "UNKNOWN", -1


def job_state(
    job_id: str,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str | None:
    """Read one job's normalized state; unknown accounting is not a terminal failure."""
    try:
        result = run(
            ["sacct", "-j", str(job_id), "-X", "-n", "-o", "State"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    state = normalize_state(result.stdout)
    return None if state == "UNKNOWN" else state


def job_activity(
    job_id: str,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[str, str] | None:
    """Return an ACTIVE/TERMINAL classification and normalized state, or None when accounting is unreadable."""
    state = job_state(job_id, run=run)
    if state is None:
        return None
    return ("ACTIVE" if state in ACTIVE_JOB_STATES else "TERMINAL", state)


def submit(
    command: Sequence[str],
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout: int = 180,
    **kwargs: Any,
) -> tuple[str | None, subprocess.CompletedProcess[str]]:
    """Submit one parsable command and return its accepted job id plus the process result."""
    result = run(list(command), capture_output=True, text=True, timeout=timeout, **kwargs)
    return (parse_job_id(result.stdout) if result.returncode == 0 else None), result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Query the shared posttrain Slurm state adapter.")
    parser.add_argument("action", choices=("classify",))
    parser.add_argument("job_id")
    args = parser.parse_args(argv)
    activity = job_activity(args.job_id)
    if activity is None:
        print(f"Slurm state unavailable for job {args.job_id}", file=sys.stderr)
        return 2
    print(*activity)
    return 0


if __name__ == "__main__":
    sys.exit(main())
