#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stdlib-only CPU-controller bootstrap and cross-node observer.

The external submitter must drive ``observe`` while watching native SLURM
status, including failures before this interpreter starts. Each CPU allocation
uses a fresh attempt directory. This helper never submits or retries a job.
"""

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path


def atomic(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def identity(job_id):
    if not job_id.isdigit():
        raise ValueError("A native SLURM job ID is required")
    return {"job_id": job_id, "uid": os.getuid(), "host": socket.gethostname()}


def observe(root, job_id):
    """Idempotent, bounded observer; never starts the controller or a GPU job."""
    observer = identity(job_id)
    state = read(root / "startup.json")
    if not state:
        return {"phase": "NOT_ENTERED"}
    if state.get("job_id") != job_id or state.get("uid") != observer["uid"]:
        raise ValueError("Bootstrap identity mismatch")
    if state["host"] == observer["host"]:
        raise ValueError("The observer must run on a different node")
    phase = state["phase"]
    if phase == "LOCKED":
        with (root / "probe.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                atomic(root / "observer.json", {**observer, "lock_blocked": True})
            else:
                # The owner may just have released the lock. Never manufacture
                # a failure or overwrite an acknowledgement from a stale read.
                if not read(root / "observer.json"):
                    raise ValueError("Cross-node lock exclusion failed")
    elif phase == "CLAIMED":
        with (root / "probe.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            claim = read(root / "probe-claim.json")
            if claim != {"job_id": job_id, "uid": observer["uid"], "state": "claimed"}:
                raise ValueError("Cross-node claim visibility failed")
            atomic(root / "verified.json", {**observer, "claim_preserved": True})
    return {"phase": phase, "job_id": job_id}


def mark_fanout(root):
    """Called BEFORE any algorithm process, submission, or paid request."""
    owner = identity(os.environ["SLURM_JOB_ID"])
    with (root / "fanout.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(root / "startup.json")
        if any(state.get(key) != owner[key] for key in owner) or state.get("phase") != "STARTED":
            raise ValueError("Controller has not passed bootstrap")
        if time.time() >= state["deadline"]:
            raise ValueError("Approved deadline expired")
        if (root / "closed.json").exists():
            raise ValueError("This CPU startup attempt was fenced before replacement")
        with (root / "fanout.json").open("x") as stream:
            json.dump({**owner, "started_at": time.time()}, stream)
            stream.flush()
            os.fsync(stream.fileno())


def close_before_retry(root):
    """Fence a terminal pre-fan-out attempt before allocating a replacement."""
    with (root / "fanout.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "fanout.json").exists():
            raise ValueError("Fan-out already claimed; startup retry forbidden")
        atomic(root / "closed.json", {"closed_at": time.time(), "uid": os.getuid()})


def retry_allowed(native_state, root):
    """Conservative policy, only AFTER matching native terminal observation.

    Caller must also check the shared workspace for any experiment/submission,
    preserve the original deadline, and enforce its approved CPU attempt cap.
    Missing/corrupt evidence, cancellation and application errors never retry.
    """
    if (root / "fanout.json").exists():
        return False
    state = read(root / "startup.json")
    if native_state in {"NODE_FAIL", "BOOT_FAIL", "PREEMPTED"}:
        return True
    return native_state == "FAILED" and state.get("error_kind") == "observer_timeout"


def run(root, command, deadline, probe_seconds, minimum_run_seconds):
    root.mkdir(parents=True, exist_ok=True)
    owner = identity(os.environ["SLURM_JOB_ID"])
    # Exclusive, persistent claim: native requeues/duplicate invocations cannot
    # replay even when the first process died before its first status receipt.
    with (root / "entered.json").open("x") as stream:
        json.dump(owner, stream)
        stream.flush()
        os.fsync(stream.fileno())
    state = {**owner, "deadline": deadline, "started_at": time.time()}
    child = None
    stop_requested = False
    error_kind = "bootstrap_error"

    def save(phase, **fields):
        state.update(phase=phase, updated_at=time.time(), **fields)
        atomic(root / "startup.json", state)

    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, stop)

    def await_receipt(name, field, until):
        nonlocal error_kind
        while time.time() < until and not stop_requested:
            receipt = read(root / name)
            if receipt:
                if (
                    receipt.get("job_id") != owner["job_id"]
                    or receipt.get("uid") != owner["uid"]
                    or receipt.get("host") == owner["host"]
                    or receipt.get(field) is not True
                ):
                    raise ValueError("Invalid observer receipt")
                return receipt
            time.sleep(0.2)
        error_kind = "interrupted" if stop_requested else "observer_timeout"
        raise TimeoutError("Observer did not finish within the startup budget")

    try:
        save("ENTERED")
        if time.time() + minimum_run_seconds >= deadline:
            error_kind = "insufficient_time"
            raise TimeoutError("Insufficient approved time")
        until = min(time.time() + probe_seconds, deadline - minimum_run_seconds)
        with (root / "probe.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            save("LOCKED")
            observer = await_receipt("observer.json", "lock_blocked", until)
        with (root / "probe.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            atomic(root / "probe-claim.json", {"job_id": owner["job_id"], "uid": owner["uid"], "state": "claimed"})
        save("CLAIMED")
        verified = await_receipt("verified.json", "claim_preserved", until)
        if verified["host"] != observer["host"]:
            raise ValueError("Observer changed nodes during the check")
        if stop_requested or time.time() + minimum_run_seconds >= deadline:
            error_kind = "insufficient_time"
            raise TimeoutError("Insufficient approved time")
        save("STARTED", observer_host=observer["host"], controller_started_at=time.time())
        environment = {**os.environ, "AUTOML_STARTUP_ATTEMPT": str(root)}
        child = subprocess.Popen(command, env=environment, start_new_session=True)
        stopping_at = None
        while child.poll() is None:
            if (stop_requested or time.time() >= deadline - 180) and stopping_at is None:
                stopping_at = time.monotonic()
                os.killpg(child.pid, signal.SIGTERM)
            if stopping_at is not None and time.monotonic() - stopping_at >= 150:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            time.sleep(1)
        code = child.returncode
        save("COMPLETE" if code == 0 else "ERROR", exit_code=code, error_kind=None if code == 0 else "controller_exit")
        return 0 if code == 0 else 1
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=150)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        save("ERROR", error_kind=error_kind, error_type=type(exc).__name__)
        return 75 if error_kind == "observer_timeout" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verb", choices=("run", "observe", "fanout", "status"))
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--job-id")
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--probe-seconds", type=int, default=600)
    parser.add_argument("--minimum-run-seconds", type=int, default=14400)
    options, command = parser.parse_known_args()
    root = options.attempt_dir.absolute()
    if options.verb == "run":
        if command[:1] == ["--"]:
            command = command[1:]
        if not command or options.deadline is None or options.probe_seconds <= 0 or options.minimum_run_seconds < 0:
            parser.error("run requires a command, absolute deadline and valid time budgets")
        raise SystemExit(run(root, command, options.deadline, options.probe_seconds, options.minimum_run_seconds))
    if command:
        parser.error("Unexpected arguments")
    if options.verb == "observe":
        if not options.job_id:
            parser.error("observe requires --job-id")
        print(json.dumps(observe(root, options.job_id)))
    elif options.verb == "fanout":
        mark_fanout(root)
    else:
        print(json.dumps(read(root / "startup.json")))


if __name__ == "__main__":
    main()
