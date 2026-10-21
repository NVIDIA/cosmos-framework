# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Python multinode launcher: submit a self-contained slurm (sbatch) job that starts a Ray cluster
across >=2 nodes (reusing utils/launch_multinode.sh for the in-allocation bringup) and runs a
target command on the head. By default, poll to completion and return the exit code + log;
--submit-only returns after scheduler acceptance instead.

Run this BY FILE PATH on the slurm submit host. The launcher is pure-stdlib (no torch/ray), but a
login node typically has sbatch and NOT torch, so `python -m ...posttrain.utils.launch_multinode`
would import the posttrain package __init__ -> torch -> ModuleNotFoundError. The `--run` command is
arbitrary and executes INSIDE the allocation (where torch/ray exist), so it may itself be `-m`.

Env-specific values resolve arg -> env -> default (see resolve_config); the container image has NO
default (set POSTTRAIN_CONTAINER_IMAGE or pass --container-image). Example:

    python3 cosmos_framework/posttrain/utils/launch_multinode.py \
      --nodes 2 --gpus-per-node 8 \
      --run "python -m cosmos_framework.posttrain.entrypoints.grpo \\
             --config cosmos_framework/posttrain/configs/grpo_qwen3b_gsm8k.py"

Full tutorial (launcher->entry contract, plugging in your own training entry, caveats):
operator-provided multinode documentation
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

try:
    from cosmos_framework.posttrain.launcher.slurm import parse_accounting, parse_duration, parse_job_id
    from cosmos_framework.posttrain.utils.log import logger
except ModuleNotFoundError:  # run BY FILE PATH without PYTHONPATH: bootstrap the repo root
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
    from cosmos_framework.posttrain.launcher.slurm import parse_accounting, parse_duration, parse_job_id
    from cosmos_framework.posttrain.utils.log import logger
_TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "TIMEOUT",
    "CANCELLED",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "BOOT_FAIL",
    "DEADLINE",
    "PREEMPTED",
}


@dataclass
class LaunchConfig:
    container_image: str
    container_mounts: str
    account: str
    partition: str
    repo_root: str


@dataclass
class SlurmRunResult:
    job_id: str
    state: str
    exit_code: int
    log_path: str

    @property
    def ok(self) -> bool:
        # A job is successful ONLY if it COMPLETED with exit 0. A CANCELLED/PREEMPTED job can report
        # ExitCode 0:<signal> → exit_code 0, so state must be checked too.
        return self.state == "COMPLETED" and self.exit_code == 0


def check_partition_time(partition: str, time_limit: str) -> str:
    """Read the live partition limit before submitting an allocation."""
    result = subprocess.run(
        ["scontrol", "show", "partition", partition, "--oneliner"], text=True, capture_output=True, check=True
    )
    fields = dict(word.split("=", 1) for word in result.stdout.split() if "=" in word)
    limit = fields["MaxTime"]

    requested, maximum = parse_duration(time_limit), parse_duration(limit)
    if maximum is not None and (requested is None or (requested + 59) // 60 > (maximum + 59) // 60):
        raise ValueError(f"requested time {time_limit} exceeds {partition} MaxTime={limit}")
    return limit


def _default_repo_root() -> str:
    # utils -> posttrain -> cosmos_framework -> <repo root>
    return str(Path(__file__).resolve().parents[3])


def _pick(arg, env_names, default):
    if arg is not None:  # explicit arg wins (even ""); callers pass None for "not provided"
        return arg
    for e in env_names:
        v = os.environ.get(e)
        if v:
            return v
    return default


#: Characters the kernel octal-escapes in /proc/self/mountinfo paths.
_MOUNTINFO_ESCAPES = {"\\": "\\134", " ": "\\040", "\t": "\\011", "\n": "\\012"}


def _mountinfo_path(path: str) -> str:
    """Spell ``path`` the way /proc/self/mountinfo lists it."""
    return "".join(_MOUNTINFO_ESCAPES.get(char, char) for char in path)


def _mount_probe(container_mounts: str) -> str:
    """A command that succeeds only in a container namespace holding every declared mount.

    ``container_mounts`` uses pyxis syntax, ``src[:dst[:flags]]`` entries joined by commas. A path
    that already exists in the image (a baked-in workdir, say) proves nothing, so each destination
    is looked up in the mount table instead. Destinations are compared in the table's escaped
    spelling and passed through the environment, which awk does not escape-process as it does
    ``-v`` assignments.
    """
    targets: list[str] = []
    for entry in container_mounts.split(","):
        fields = entry.strip().split(":")
        if not fields[0]:
            continue
        targets.append(os.path.normpath(fields[1] if len(fields) > 1 and fields[1] else fields[0]))
    if not targets:
        return "true"
    checks = " && ".join(
        f"T={shlex.quote(_mountinfo_path(target))} "
        """awk 'BEGIN { t = ENVIRON["T"] } $5 == t { f = 1 } END { exit !f }' /proc/self/mountinfo"""
        for target in dict.fromkeys(targets)
    )
    return f"bash -c {shlex.quote(checks)}"


def resolve_config(
    *, container_image=None, container_mounts=None, account=None, partition=None, repo_root=None
) -> LaunchConfig:
    """Resolve env-specific config (arg -> env -> default). container_image is required."""
    image = _pick(container_image, ["POSTTRAIN_CONTAINER_IMAGE"], None)
    if not image:
        raise ValueError("no container image: set POSTTRAIN_CONTAINER_IMAGE or pass --container-image")
    return LaunchConfig(
        container_image=image,
        container_mounts=_pick(container_mounts, ["POSTTRAIN_CONTAINER_MOUNTS"], "/lustre:/lustre"),
        account=_pick(account, ["POSTTRAIN_SLURM_ACCOUNT", "SBATCH_ACCOUNT"], ""),
        partition=_pick(partition, ["POSTTRAIN_SLURM_PARTITION"], "interactive"),
        repo_root=_pick(repo_root, ["POSTTRAIN_REPO_ROOT"], _default_repo_root()),
    )


def build_sbatch_script(
    *,
    run: str,
    nodes: int,
    gpus_per_node: int,
    account: str,
    partition: str,
    time_limit: str,
    container_image: str,
    container_mounts: str,
    container_name: str,
    repo_root: str,
    ray_port: int,
    log_path: str,
    pip_install: bool = True,
    mode: str = "ray",
    venv: str = "",
    wckey: str = "",
    ray_runtime_env: str = "",
    preflight_cmd: str = "",
    preflight_timeout_s: int = 120,
    signal_seconds: int = 600,
    container_remap_root: bool = False,
    driver_capabilities: str = "",
    no_requeue: bool = False,
    container_workdir: str | None = None,
    env: dict[str, str] | None = None,
    job_name: str = "",
    qos: str = "",
    cpus_per_task: int | None = None,
) -> str:
    """Return the sbatch script text. The container bring-up is identical for both modes; only the
    in-allocation launcher differs — launch_multinode.sh (Ray cluster) or launch_multinode_torchrun.sh
    (one torchrun per node, c10d rendezvous). RUN_CMD is passed through to it. Pure function (the
    unit-test seam)."""
    if type(signal_seconds) is not int or signal_seconds <= 0:
        raise ValueError("signal_seconds must be a positive integer")
    if container_workdir is not None and not Path(container_workdir).is_absolute():
        raise ValueError("container_workdir must be an absolute in-image path")
    directives = []
    for key, value in (("job-name", job_name), ("qos", qos)):
        if value:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
                raise ValueError(f"{key} must contain only letters, digits, '_', '.', or '-'")
            directives.append(f"#SBATCH --{key}={value}")
    if cpus_per_task is not None:
        if type(cpus_per_task) is not int or cpus_per_task <= 0:
            raise ValueError("cpus_per_task must be a positive integer")
        directives.append(f"#SBATCH --cpus-per-task={cpus_per_task}")
    exports = []
    for key, value in (env or {}).items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or not isinstance(value, str) or "\0" in value:
            raise ValueError(f"invalid environment entry: {key!r}")
        exports.append(f"export {key}={shlex.quote(value)}")
    extra_directives = "\n".join(directives)
    env_exports = "\n".join(exports)
    root_flag = " --container-remap-root" if container_remap_root else ""
    driver_export = (
        f"export NVIDIA_DRIVER_CAPABILITIES={shlex.quote(driver_capabilities)}" if driver_capabilities else ""
    )
    requeue_line = "#SBATCH --no-requeue" if no_requeue else ""
    preflight_export = (
        f"export PREFLIGHT_CMD={shlex.quote(preflight_cmd)} PREFLIGHT_TIMEOUT_SECONDS={preflight_timeout_s}"
    )
    effective_wckey = wckey or "p1"
    if partition == "batch_long" and effective_wckey != "p0":
        raise ValueError("batch_long requires wckey p0 on GCP; omitted wckey is treated as p1")

    launcher_name = "launch_multinode_torchrun.sh" if mode == "torchrun" else "launch_multinode.sh"
    bash_launcher = f"{repo_root}/cosmos_framework/posttrain/utils/{launcher_name}"
    # Quote every interpolated shell word (paths may contain spaces on other clusters).
    cn = shlex.quote(container_name)
    img = shlex.quote(container_image)
    mounts = shlex.quote(container_mounts)
    root = shlex.quote(repo_root)
    workdir = shlex.quote(container_workdir if container_workdir is not None else repo_root)
    mount_probe = _mount_probe(container_mounts)
    launcher_q = shlex.quote(bash_launcher)
    sig_dir = os.path.dirname(log_path) or "."
    # venv (an `activate` script path inside the image) is sourced for the pip step here AND by
    # launch_multinode.sh's run() for every in-container command — one python/ray everywhere.
    venv_prefix = f"source {shlex.quote(venv)} && " if venv else ""
    pip_step = (
        f"if ! srun --overlap --ntasks-per-node=1 --container-name {cn} \\\n"
        f"  bash -c \"{venv_prefix}${{POSTTRAIN_RUNTIME_ENV:+$POSTTRAIN_RUNTIME_ENV; }}pip install --no-cache-dir 'ray[default]==2.56.1' transformers accelerate\"; then\n"
        '  echo "ERROR: multinode dependency installation failed" >&2\n'
        "  exit 1\n"
        "fi\n"
        if pip_install
        else ""
    )
    venv_export = f" VENV_ACTIVATE={shlex.quote(venv)}" if venv else ""
    runtime_export = f" RAY_RUNTIME_ENV={shlex.quote(ray_runtime_env)}" if ray_runtime_env else ""
    # Cluster priority tag (e.g. wckey=p0): without it, batch-partition jobs may be preempted
    # (and requeued) repeatedly while higher-priority work sweeps the nodes.
    wckey_line = f"#SBATCH --wckey={wckey}\n" if wckey else ""
    return f"""#!/usr/bin/env bash
#SBATCH -N {nodes}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH -A {account}
#SBATCH -p {partition}
#SBATCH --time={time_limit}
#SBATCH -o {log_path}
#SBATCH --signal=B:SIGUSR1@{signal_seconds}
{requeue_line}
{wckey_line}{extra_directives}
set -uo pipefail
umask 077
# Apply literal environment values inside the container, after image/venv defaults.
export POSTTRAIN_RUNTIME_ENV={shlex.quote(env_exports)}
{driver_export}
{preflight_export}
# Preemption/timeout hook: Slurm delivers SIGUSR1 to THIS batch shell {signal_seconds}s before the
# limit (and on preemption) — pyxis containers never see it. The trap drops a sentinel on
# the shared FS; the training loop polls it each step (control/termination.py) and saves
# an emergency checkpoint, then exits cleanly.
export SLURM_LOG_DIR="{sig_dir}/sig_${{SLURM_JOB_ID}}"
mkdir -p "$SLURM_LOG_DIR"
trap 'touch "$SLURM_LOG_DIR/SIGUSR1_RECEIVED"' USR1
# Two-step container bring-up per node. Both steps are needed and the ORDER matters — this exact
# shape was arrived at empirically (see the two failure modes below).
#
# (1) Import the image ONCE per node, BLOCKING, with NO concurrent srun. Backgrounding the image
#     import and letting the readiness probe run against it makes pyxis extract the same squashfs
#     twice -> "failed to create container filesystem ... File exists", leaving a half-built,
#     unmounted container. A blocking foreground import is race-free.
#     Even so, a multi-minute extraction (large images) can intermittently fail the same way when
#     slurm re-fires the launch on one node (duplicate enroot create -> "File exists"), so retry
#     after removing that node's partial rootfs; nodes that imported cleanly just reuse theirs.
import_ok=0
for attempt in 1 2 3; do
  if srun --ntasks-per-node=1 --container-image {img} --container-mounts {mounts}{root_flag} \\
       --container-name {cn} --container-workdir {workdir} true; then import_ok=1; break; fi
  echo "WARN: container import attempt $attempt failed; removing partial rootfs and retrying" >&2
  srun --ntasks-per-node=1 bash -c "enroot remove -f pyxis_\\${{SLURM_JOB_ID}}_{cn} >/dev/null 2>&1; true"
done
[ "$import_ok" = 1 ] || {{ echo "ERROR: container image import failed on one or more nodes" >&2; exit 1; }}
# (2) Start a long-lived instance that HOLDS THE MOUNTS. --container-image is repeated but the rootfs
#     {cn} already exists from step 1, so pyxis REUSES it (no second extract, no race) — and the mount
#     namespace stays live as long as this step, so every later `--overlap --container-name` attach
#     (probe, pip, the Ray launcher's run_in_container) inherits /lustre. A mount-less keep-alive here
#     instead drops the mounts the moment step 1 exits, so attaches then fail with `cd: .../imaginaire4:
#     No such file or directory`.
srun --overlap --ntasks-per-node=1 --container-image {img} --container-mounts {mounts}{root_flag} \\
  --container-name {cn} --container-workdir {workdir} sleep infinity &
# WAIT until the container is attachable WITH its mounts on all nodes before the launcher's
# `--overlap` steps run. The probe is attach-only (no --container-image) so it never starts a
# competing import. An attach that lands before a node's keep-alive is up still succeeds, in a
# mount-less container, so the probe checks the mount table for every declared mount: probing
# `true` let Ray start on a node without the repo (`No module named 'projects'` at actor creation,
# 2026-09-30, job 725539).
ready=0
for i in $(seq 1 60); do
  if srun --overlap --ntasks-per-node=1 --container-name {cn} {mount_probe} >/dev/null 2>&1; then ready=1; break; fi
  sleep 5
done
[ "$ready" = 1 ] || {{ echo "ERROR: container not ready with its mounts on all nodes" >&2; exit 1; }}
{pip_step}export CONTAINER_NAME={cn} GPUS_PER_NODE={gpus_per_node} RAY_PORT={ray_port} REPO_ROOT={root} CONTAINER_WORKDIR={workdir}{venv_export}{runtime_export}
export RUN_CMD={shlex.quote(run)}
# Run the launcher in the BACKGROUND and `wait`: bash defers traps while a foreground
# child runs, but `wait` is interruptible — the first wait returns when SIGUSR1 fires
# (sentinel written), the second collects the child after its emergency save.
bash {launcher_q} &
_lp=$!
wait $_lp
wait $_lp
"""


def parse_sacct(sacct_stdout: str) -> tuple[str, int]:
    """Parse `sacct -j <id> -o JobID,State,ExitCode -n -P` into (state, exit_code) for the PRIMARY
    job row (JobID without a '.step' suffix). Empty/unknown -> ("UNKNOWN", -1)."""
    return parse_accounting(sacct_stdout)


def submit_multinode(
    run: str,
    nodes: int = 2,
    gpus_per_node: int = 8,
    *,
    account: str | None = None,
    partition: str | None = None,
    time_limit: str = "1:00:00",
    container_image: str | None = None,
    container_mounts: str | None = None,
    container_name: str = "posttrain_mn",
    repo_root: str | None = None,
    ray_port: int = 6379,
    poll_interval: float = 15.0,
    timeout_s: float = 3600.0,
    log_dir: str = "/tmp",
    pip_install: bool = True,
    mode: str = "ray",
    venv: str | None = None,
    wckey: str | None = None,
    ray_runtime_env: str | None = None,
    preflight_cmd: str = "",
    preflight_timeout_s: int = 120,
    signal_seconds: int = 600,
    container_remap_root: bool = False,
    driver_capabilities: str = "",
    no_requeue: bool = False,
    dry_run: bool = False,
    container_workdir: str | None = None,
    env: dict[str, str] | None = None,
    job_name: str = "",
    qos: str = "",
    cpus_per_task: int | None = None,
    submit_only: bool = False,
) -> SlurmRunResult | None:
    """Write and submit an sbatch script; optionally return without polling or cancellation."""
    import subprocess
    import tempfile
    import time

    cfg = resolve_config(
        container_image=container_image,
        container_mounts=container_mounts,
        account=account,
        partition=partition,
        repo_root=repo_root,
    )
    log_path = os.path.join(log_dir, "posttrain_mn_%j.out")  # %j -> slurm job id
    script = build_sbatch_script(
        run=run,
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        account=cfg.account,
        partition=cfg.partition,
        time_limit=time_limit,
        container_image=cfg.container_image,
        container_mounts=cfg.container_mounts,
        container_name=container_name,
        repo_root=cfg.repo_root,
        ray_port=ray_port,
        log_path=log_path,
        pip_install=pip_install,
        mode=mode,
        venv=_pick(venv, ["POSTTRAIN_VENV"], ""),
        wckey=_pick(wckey, ["POSTTRAIN_SLURM_WCKEY"], ""),
        ray_runtime_env=_pick(ray_runtime_env, ["POSTTRAIN_RAY_RUNTIME_ENV"], ""),
        preflight_cmd=preflight_cmd,
        preflight_timeout_s=preflight_timeout_s,
        signal_seconds=signal_seconds,
        container_remap_root=container_remap_root,
        driver_capabilities=driver_capabilities,
        no_requeue=no_requeue,
        container_workdir=container_workdir,
        env=env,
        job_name=job_name,
        qos=qos,
        cpus_per_task=cpus_per_task,
    )
    os.makedirs(log_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".sbatch", dir=log_dir, delete=False) as f:
        f.write(script)
        script_path = f.name
    if dry_run:
        print(script)
        logger.info("dry-run script=%s", script_path)
        return None
    check_partition_time(cfg.partition, time_limit)
    # sbatch environment variables override script directives; explicit launcher values must win.
    command = ["sbatch", "--parsable"]
    for key, value in (("job-name", job_name), ("qos", qos), ("cpus-per-task", cpus_per_task)):
        if value:
            command.append(f"--{key}={value}")
    proc = subprocess.run([*command, script_path], capture_output=True, text=True, check=True)
    job_id = parse_job_id(proc.stdout)
    if job_id is None:
        raise RuntimeError(f"sbatch returned no numeric job id: stdout={proc.stdout!r} stderr={proc.stderr!r}")
    resolved_log = log_path.replace("%j", job_id)
    logger.info("submitted Slurm job %s; script=%s log=%s", job_id, script_path, resolved_log)
    Path(log_dir, "submission.json").write_text(
        json.dumps({"job_id": job_id, "script": script_path, "log_path": resolved_log}, indent=4) + "\n"
    )

    if submit_only:
        return SlurmRunResult(job_id, "SUBMITTED", 0, resolved_log)

    deadline = time.monotonic() + timeout_s
    while True:
        r = subprocess.run(
            ["sacct", "-j", job_id, "-o", "JobID,State,ExitCode", "-n", "-P"],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:  # surface sacct failures instead of silently polling to timeout; still retry
            logger.warning(f"sacct rc={r.returncode} for job {job_id}: {r.stderr.strip()!r}")
        state, code = parse_sacct(r.stdout)  # empty/not-yet-registered -> UNKNOWN (keep polling)
        if state in _TERMINAL_STATES:
            return SlurmRunResult(job_id, state, code, resolved_log)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(poll_interval, remaining))  # never overshoot the deadline

    cancel = subprocess.run(["scancel", job_id], capture_output=True, text=True)
    if cancel.returncode != 0:
        logger.warning(f"scancel {job_id} failed (rc={cancel.returncode}): {cancel.stderr!r}")
    return SlurmRunResult(job_id, "TIMEOUT", -1, resolved_log)


def _launch_config_defaults(parser: argparse.ArgumentParser, path: str) -> None:
    """A selected deployment file supplies CLI defaults, ahead of ambient environment values."""
    import math

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate launch configuration key {key!r}")
            result[key] = value
        return result

    try:
        values = json.loads(Path(path).read_text(), object_pairs_hook=unique_object)
    except (OSError, ValueError) as error:
        parser.error(f"--launch-config {path}: {error}")
    if not isinstance(values, dict):
        parser.error("--launch-config must contain a JSON object")
    actions = {
        action.dest: action for action in parser._actions if action.dest not in {"help", "launch_config", "overrides"}
    }
    for key, value in values.items():
        if key not in actions:
            parser.error(f"--launch-config: unknown key {key!r}")
        action = actions[key]
        if key == "env":
            if not isinstance(value, dict) or any(
                not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) or not isinstance(v, str) for k, v in value.items()
            ):
                parser.error("--launch-config: env requires valid variable names and string values")
            values[key] = [f"{k}={v}" for k, v in value.items()]
            continue
        expected = (
            bool if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)) else action.type or str
        )
        valid = type(value) is expected
        if expected is float:
            valid = type(value) in (int, float) and math.isfinite(value)
        if not valid:
            parser.error(f"--launch-config: {key} requires {expected.__name__}, got {type(value).__name__}")
        if action.choices is not None and value not in action.choices:
            parser.error(f"--launch-config: {key} must be one of {action.choices}")
    parser.set_defaults(**values)


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Submit a multinode Ray or torchrun posttrain job via sbatch.")
    p.add_argument("--launch-config", help="JSON defaults for these CLI arguments; explicit CLI values win")
    p.add_argument("--run", help="command to run on the head once Ray is up")
    p.add_argument("--nodes", type=int, default=2)
    p.add_argument("--gpus-per-node", type=int, default=8)
    p.add_argument("--account", default=None)
    p.add_argument("--partition", default=None)
    p.add_argument("--time", default="1:00:00")
    p.add_argument("--container-image", default=None)
    p.add_argument("--container-mounts", default=None)
    p.add_argument("--repo-root", default=None)
    p.add_argument("--container-workdir", default=None, help="in-image code root; defaults to --repo-root")
    p.add_argument("--container-name", default="posttrain_mn")
    p.add_argument("--ray-port", type=int, default=6379)
    p.add_argument("--poll-interval", type=float, default=15.0, help="seconds between sacct polls")
    p.add_argument("--timeout", type=float, default=3600.0, help="max seconds to wait before scancel")
    p.add_argument("--log-dir", default=None, help="shared log dir; defaults to <output-root>/logs/<job-name> or /tmp")
    p.add_argument("--output-root", help="IMAGINAIRE_OUTPUT_ROOT; relative paths resolve against the checkout")
    p.add_argument("--job-name", default="", help="Slurm name and training job.name override (training payloads only)")
    p.add_argument("--qos", default="", help="Slurm QOS")
    p.add_argument("--cpus-per-task", type=int, help="Slurm CPUs per task")
    p.add_argument(
        "--env", action="append", default=[], metavar="KEY=VALUE", help="literal runtime environment; repeatable"
    )
    p.add_argument("--submit-only", action="store_true", help="return after sbatch acceptance; do not poll or cancel")
    p.add_argument(
        "--wait", dest="submit_only", action="store_false", help="wait for completion, overriding submit_only"
    )
    p.add_argument("--no-pip", action="store_true", help="skip pip-installing deps in the job")
    p.add_argument(
        "--mode",
        choices=["ray", "torchrun"],
        default="ray",
        help="ray: Ray cluster, driver joins via RAY_ADDRESS. torchrun: one torchrun per node, "
        "c10d rendezvous at the head (--run is the torchrun payload, e.g. '-m ...entrypoints.dpo -- ...').",
    )
    p.add_argument(
        "--venv",
        default=None,
        help="in-image venv activate script (e.g. /opt/venv/cosmos_rl/bin/activate), sourced for "
        "Ray daemons, pip step, and the driver alike; env: POSTTRAIN_VENV",
    )
    p.add_argument(
        "--wckey",
        default=None,
        help="slurm --wckey priority tag (e.g. p0) to resist preemption; env: POSTTRAIN_SLURM_WCKEY",
    )
    p.add_argument(
        "--ray-runtime-env",
        default=None,
        help="trusted shell fragment exported before every Ray daemon and driver; env: POSTTRAIN_RAY_RUNTIME_ENV",
    )
    p.add_argument(
        "--until-iter",
        type=int,
        default=0,
        help="window RELAY: resubmit (up to --max-rounds) until <--watch-dir>'s latest "
        "checkpoint reaches this iteration — for time-capped partitions (batch=4h) where "
        "one job cannot finish; the payload's rl.resume=auto continues each window. "
        "0 = single submission (default; batch_long usually needs no relay).",
    )
    p.add_argument(
        "--watch-dir",
        default=None,
        help="the experiment NAME dir (<output_root>/<project>/<group>/<name>) whose newest "
        "run-attempt's checkpoints are checked between rounds; required with --until-iter",
    )
    p.add_argument("--max-rounds", type=int, default=20, help="relay round cap for --until-iter")
    p.add_argument("--preflight-cmd", default="", help="command run on every node before starting Ray")
    p.add_argument("--preflight-timeout-seconds", type=int, default=120)
    p.add_argument("--signal-seconds", type=int, default=600)
    p.add_argument("--container-remap-root", action="store_true")
    p.add_argument("--driver-capabilities", default="")
    p.add_argument("--no-requeue", action="store_true")
    p.add_argument(
        "--dry-run", action="store_true", help="write and print the script without allocation or runtime checks"
    )
    p.add_argument("overrides", nargs=argparse.REMAINDER, help="training dotlist overrides after --")
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--launch-config")
    selected, _ = config_parser.parse_known_args(argv)
    if selected.launch_config:
        _launch_config_defaults(p, selected.launch_config)
    a = p.parse_args(argv)
    if not a.run:
        p.error("--run is required, either on the command line or in --launch-config")
    if a.submit_only and a.until_iter:
        p.error("--submit-only cannot be combined with relay --until-iter")
    if a.overrides and a.overrides[0] != "--":
        p.error("training overrides must follow --")
    overrides = a.overrides[1:]
    if a.job_name:
        if any(item.startswith("job.name=") for item in overrides) or any(
            item.startswith("job.name=") for item in shlex.split(a.run)
        ):
            p.error("use --job-name or job.name in the payload, not both")
        overrides = [*overrides, f"job.name={a.job_name}"]
    if overrides:
        if "--" not in shlex.split(a.run):
            a.run += " --"
        a.run += " " + shlex.join(overrides)
    env = {}
    for item in a.env:
        key, separator, value = item.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            p.error("--env requires KEY=VALUE with a valid environment variable name")
        env[key] = value
    if a.output_root:
        root = Path(a.repo_root or os.environ.get("POSTTRAIN_REPO_ROOT") or _default_repo_root()).resolve()
        a.output_root = str((root / a.output_root).resolve())
        env["IMAGINAIRE_OUTPUT_ROOT"] = a.output_root
    if a.log_dir is None:
        a.log_dir = str(Path(a.output_root) / "logs" / a.job_name) if a.output_root else "/tmp"
    if a.dry_run and a.until_iter:
        p.error("--dry-run cannot be combined with relay --until-iter")
    if a.until_iter and not a.watch_dir:
        p.error("--until-iter requires --watch-dir")

    def _reached() -> bool:
        from cosmos_framework.posttrain.checkpoint import latest_dir, resolve_run_ts

        ts = resolve_run_ts(a.watch_dir, "auto")  # adopts the newest attempt; legacy -> ""
        latest = latest_dir(os.path.join(a.watch_dir, ts, "checkpoints"))
        if latest is None:
            return False
        digits = "".join(c for c in os.path.basename(latest) if c.isdigit())
        return bool(digits) and int(digits) >= a.until_iter

    rounds = max(1, a.max_rounds) if a.until_iter else 1
    for r_i in range(rounds):
        if a.until_iter and _reached():
            logger.info(f"relay: target iter {a.until_iter} already reached; done")
            raise SystemExit(0)
        if a.until_iter:
            logger.info(f"relay round {r_i + 1}/{rounds}")
        res = submit_multinode(
            a.run,
            nodes=a.nodes,
            gpus_per_node=a.gpus_per_node,
            account=a.account,
            partition=a.partition,
            time_limit=a.time,
            container_image=a.container_image,
            container_mounts=a.container_mounts,
            container_name=a.container_name,
            repo_root=a.repo_root,
            ray_port=a.ray_port,
            poll_interval=a.poll_interval,
            timeout_s=a.timeout,
            log_dir=a.log_dir,
            pip_install=not a.no_pip,
            mode=a.mode,
            venv=a.venv,
            wckey=a.wckey,
            ray_runtime_env=a.ray_runtime_env,
            preflight_cmd=a.preflight_cmd,
            preflight_timeout_s=a.preflight_timeout_seconds,
            signal_seconds=a.signal_seconds,
            container_remap_root=a.container_remap_root,
            driver_capabilities=a.driver_capabilities,
            no_requeue=a.no_requeue,
            dry_run=a.dry_run,
            container_workdir=a.container_workdir,
            env=env,
            job_name=a.job_name,
            qos=a.qos,
            cpus_per_task=a.cpus_per_task,
            submit_only=a.submit_only,
        )
        if a.dry_run:
            return
        logger.info(f"job {res.job_id} state={res.state} exit={res.exit_code} log={res.log_path}")
        if a.submit_only:
            return
        if not a.until_iter:
            # Exit nonzero unless the job COMPLETED with 0 (a CANCELLED job's ExitCode can be 0:<signal>).
            raise SystemExit(0 if res.ok else (res.exit_code if res.exit_code > 0 else 1))
    raise SystemExit(0 if _reached() else 1)


if __name__ == "__main__":
    main()
