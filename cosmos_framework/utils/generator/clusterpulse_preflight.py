# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Check ClusterPulse GPU and in-process NVTX providers before distributed training."""

import fcntl
import os
import socket
import subprocess
import time
from pathlib import Path


def emit_nvtx_probe() -> None:
    # Import only on the opt-in training path; ordinary launches keep their import order.
    import torch

    torch.cuda.set_device(int(os.environ["SLURM_LOCALID"]))
    torch.cuda.init()
    with torch.cuda.nvtx.range(f"clusterpulse_provider_probe_rank_{os.environ['SLURM_PROCID']}"):
        torch.cuda.synchronize()


def run_health_check(checker: Path, *, process: bool) -> None:
    env = dict(os.environ)
    # The observer uses the same compat UMD but does not need workload injection.
    env.pop("LD_PRELOAD", None)
    env["NVAON_LOG_ENABLE_STDERR"] = "1"
    command = [str(checker), "--service-port", os.environ["NVAON_SERVICE_PORT"], "--skip-pm"]
    timeout = 10 if process else 60
    command.extend(["--timeout", str(timeout)])
    if process:
        command.extend(["--process-pid", str(os.getpid())])
    try:
        subprocess.run(command, env=env, check=True, timeout=timeout + 5)
    except (OSError, subprocess.SubprocessError) as error:
        scope = "workload NVTX" if process else "GPU trace"
        raise RuntimeError(f"ClusterPulse {scope} provider readiness failed before training") from error


def prepare_clusterpulse_providers() -> None:
    checker_path = os.environ.get("COSMOS_CLUSTERPULSE_HEALTH_CHECK")
    if not checker_path:
        return
    checker = Path(checker_path)
    if not checker.is_absolute() or not checker.is_file() or not os.access(checker, os.X_OK):
        raise RuntimeError(f"Missing executable ClusterPulse health checker: {checker}")
    run_health_check(checker, process=False)
    # The lock is per node and releases automatically when its process exits.
    # It serializes initialization without storing readiness or creating a process group.
    lock_path = Path(os.environ["SLURM_LOG_DIR"]) / f"clusterpulse-provider-init-{socket.gethostname()}.lock"
    deadline = time.monotonic() + 60
    with lock_path.open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Timed out waiting for node-local ClusterPulse provider initialization"
                    ) from None
                time.sleep(0.1)
        try:
            emit_nvtx_probe()
            run_health_check(checker, process=True)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    print(f"ClusterPulse providers verified: rank={os.environ['SLURM_PROCID']} pid={os.getpid()}", flush=True)
