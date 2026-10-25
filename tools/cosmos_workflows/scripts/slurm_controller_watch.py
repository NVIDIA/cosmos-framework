#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent, restartable startup observer. Never submits, requeues or cancels.

Run on an approved service host with shared storage and native SLURM tools,
outside the CPU allocations it observes. Supervise this process independently
of an interactive shell/chat (see the accompanying user-service template).
"""

import argparse
import fcntl
import math
import os
import re
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from slurm_controller_startup import atomic, observe, read


def registry(root):
    value = read(root / "registry.json")
    if value.get("uid") != os.getuid() or not math.isfinite(value.get("deadline", float("inf"))):
        raise ValueError("Observer registry requires an owner and finite absolute deadline")
    if not isinstance(value.get("jobs"), dict):
        raise ValueError("Invalid observer job registry")
    for job, entry in value["jobs"].items():
        if (
            not re.fullmatch(r"[1-9][0-9]*", job)
            or not Path(entry["attempt_base"]).is_absolute()
            or type(entry["max_allocations"]) is not int
            or entry["max_allocations"] < 1
        ):
            raise ValueError("Invalid observer job identity or allocation cap")
    return value


def ready(root, max_age=90):
    value = registry(root)
    state = read(root / "watch.json")
    if (
        state.get("state") != "WATCHING"
        or state.get("uid") != value["uid"]
        or state.get("deadline") != value["deadline"]
        or not 0 <= time.time() - state.get("updated_at", 0) < max_age
        or time.time() >= value["deadline"]
    ):
        raise ValueError("Persistent startup observer is not ready; do not submit controllers")
    # A recent heartbeat alone is insufficient after the observer has exited.
    with (root / "watch.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return state
        raise ValueError("Startup observer is not running")


def register(root, job_id, attempt_base, max_allocations):
    """Append an acknowledged native ID; do not adopt unknown submissions here."""
    ready(root)
    with (root / "registry.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        value = registry(root)
        entry = {"attempt_base": str(attempt_base.absolute()), "max_allocations": max_allocations}
        if not re.fullmatch(r"[1-9][0-9]*", job_id) or max_allocations < 1:
            raise ValueError("Native ID and positive allocation cap required")
        if job_id in value["jobs"] and value["jobs"][job_id] != entry:
            raise ValueError("Observer identity cannot be replaced")
        value["jobs"][job_id] = entry
        atomic(root / "registry.json", value)


def tick(job_id, entry):
    try:
        result = subprocess.run(
            ["scontrol", "show", "job", "-o", job_id],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        fields = dict(re.findall(r"(?:^|\s)(\w+)=([^\s]+)", result.stdout))
        if fields.get("JobId") != job_id or not fields.get("UserId", "").endswith(f"({os.getuid()})"):
            raise ValueError("Native observer ownership mismatch")
        generation = int(fields["Restarts"])
        if not 0 <= generation < entry["max_allocations"]:
            raise ValueError("Native restart exceeds the approved allocation cap")
        if fields.get("JobState") != "RUNNING":
            return {"state": "OBSERVING", "native_state": fields.get("JobState"), "generation": generation}
        attempt = Path(entry["attempt_base"]) / f"{job_id}-g{generation}"
        return {"state": "OBSERVING", "generation": generation, **observe(attempt, job_id)}
    except (OSError, subprocess.SubprocessError) as exc:
        # Scheduler/shared-storage outages do not kill all observers or authorize
        # a replacement. Do not persist exception text, stderr or environment.
        return {"state": "WAITING", "error_type": type(exc).__name__, "errno": getattr(exc, "errno", None)}
    except (ValueError, KeyError, TypeError) as exc:
        return {"state": "BLOCKED", "error_type": type(exc).__name__}


def watch(root, poll_seconds=2):
    value = registry(root)
    saved = read(root / "watch.json")
    if saved and (saved.get("uid") != value["uid"] or saved.get("deadline") != value["deadline"]):
        raise ValueError("Observer identity/deadline changed")
    known = saved.get("registered", {})
    next_check = {}
    observations = saved.get("observations", {})
    stopping = False

    def stop(*unused):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    with (root / "watch.lock").open("a") as handle, ThreadPoolExecutor(max_workers=4) as pool:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not stopping and time.time() < value["deadline"]:
            current = registry(root)
            if current["deadline"] != value["deadline"] or any(current["jobs"].get(k) != v for k, v in known.items()):
                raise ValueError("Registered observer identities/deadline must be immutable")
            known = current["jobs"]
            futures = {
                job: pool.submit(tick, job, entry)
                for job, entry in known.items()
                if time.time() >= next_check.get(job, 0)
            }
            for job, future in futures.items():
                observations[job] = future.result()
                observations[job]["observed_at"] = time.time()
                # Fast only during the handshake; do not hammer the scheduler
                # for hours after the brain starts or while it is queued.
                phase = observations[job].get("phase")
                delay = poll_seconds if phase in {"NOT_ENTERED", "LOCKED", "CLAIMED"} else 30
                next_check[job] = time.time() + delay
            saved = dict(
                uid=value["uid"],
                deadline=value["deadline"],
                registered=known,
                state="WATCHING",
                updated_at=time.time(),
                observations=observations,
            )
            atomic(root / "watch.json", saved)
            wake = min(value["deadline"], time.time() + poll_seconds)
            while not stopping and time.time() < wake:
                time.sleep(max(0, min(0.2, wake - time.time())))
        saved.update(state="STOPPED" if stopping else "EXPIRED", updated_at=time.time())
        atomic(root / "watch.json", saved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verb", choices=("init", "watch", "ready", "register"))
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--job-id")
    parser.add_argument("--attempt-base", type=Path)
    parser.add_argument("--max-allocations", type=int)
    options = parser.parse_args()
    root = options.state_dir.absolute()
    if options.verb == "init":
        if options.deadline is None or not math.isfinite(options.deadline) or options.deadline <= time.time():
            parser.error("init needs a future finite absolute --deadline")
        root.mkdir(parents=True, exist_ok=False)
        atomic(root / "registry.json", dict(uid=os.getuid(), deadline=options.deadline, jobs={}))
    elif options.verb == "watch":
        watch(root)
    elif options.verb == "ready":
        ready(root)
    else:
        if not options.job_id or options.attempt_base is None or options.max_allocations is None:
            parser.error("register needs --job-id, --attempt-base and --max-allocations")
        register(root, options.job_id, options.attempt_base, options.max_allocations)


if __name__ == "__main__":
    main()
