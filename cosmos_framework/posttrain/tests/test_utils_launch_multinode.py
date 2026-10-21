# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unit tests for the Python multinode sbatch launcher (pure helpers; no slurm/GPU)."""

from __future__ import annotations

import json
import pathlib
import shlex
import subprocess

import pytest

from cosmos_framework.posttrain.utils import launch_multinode
from cosmos_framework.posttrain.utils.launch_multinode import (
    _default_repo_root,
    _mount_probe,
    build_sbatch_script,
    parse_sacct,
    resolve_config,
)


@pytest.mark.L0
@pytest.mark.CPU
def test_default_repo_root_is_the_checkout_containing_this_module():
    module = pathlib.Path(launch_multinode.__file__).resolve()
    assert pathlib.Path(_default_repo_root()) / "cosmos_framework" / "posttrain" / "utils" / module.name == module


@pytest.mark.L0
@pytest.mark.CPU
def test_checkpoint_discovery_on_submit_host_needs_only_stdlib(tmp_path):
    import subprocess
    import sys

    script = """
import sys
sys.path.insert(0, sys.argv[1])
from cosmos_framework.posttrain.checkpoint import latest_dir, resolve_run_ts
assert latest_dir(sys.argv[2]) is None
timestamp = resolve_run_ts(sys.argv[2], 'auto')
assert len(timestamp) == 14 and timestamp.isdigit()
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, str(pathlib.Path(__file__).resolve().parents[4]), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.L0
@pytest.mark.CPU
def test_launch_file_supplies_defaults_and_cli_overrides_them(tmp_path, monkeypatch):
    from cosmos_framework.posttrain.utils import launch_multinode as launcher

    profile = tmp_path / "launch.json"
    profile.write_text(
        json.dumps(
            {
                "run": "python -m application",
                "nodes": 2,
                "gpus_per_node": 4,
                "account": "profile-account",
                "container_image": "/<runtime-image>",
                "container_workdir": "/workspace/cosmos-framework",
                "dry_run": True,
                "no_pip": True,
            }
        )
    )
    monkeypatch.setenv("POSTTRAIN_SLURM_ACCOUNT", "ambient-account")
    calls = []
    monkeypatch.setattr(launcher, "submit_multinode", lambda run, **kw: calls.append((run, kw)))
    launcher.main(["--launch-config", str(profile), "--nodes", "1", "--run", "python -m override"])
    [(run, arguments)] = calls
    assert run == "python -m override"
    assert arguments["nodes"] == 1 and arguments["gpus_per_node"] == 4
    assert arguments["account"] == "profile-account" and not arguments["pip_install"]
    assert arguments["container_workdir"] == "/workspace/cosmos-framework"
    assert arguments["dry_run"]


@pytest.mark.L0
@pytest.mark.CPU
def test_invalid_launch_file_rejects_before_submission(tmp_path, monkeypatch):
    from cosmos_framework.posttrain.utils import launch_multinode as launcher

    profile = tmp_path / "launch.json"
    profile.write_text('{"run":"x","gpus_per_nod":4}')
    monkeypatch.setattr(launcher, "submit_multinode", lambda *a, **kw: pytest.fail("invalid config submitted"))
    with pytest.raises(SystemExit) as error:
        launcher.main(["--launch-config", str(profile)])
    assert error.value.code == 2


@pytest.mark.L0
@pytest.mark.CPU
def test_resolve_config_precedence(monkeypatch):
    monkeypatch.setenv("POSTTRAIN_CONTAINER_IMAGE", "/<runtime-image>")
    monkeypatch.setenv("POSTTRAIN_SLURM_ACCOUNT", "envacct")
    c = resolve_config()  # env
    assert c.container_image == "/<runtime-image>"
    assert c.account == "envacct"
    assert c.partition == "interactive"  # default
    assert c.container_mounts == "/lustre:/lustre"  # default
    c2 = resolve_config(container_image="/<runtime-image>", account="argacct")  # arg wins over env
    assert c2.container_image == "/<runtime-image>"
    assert c2.account == "argacct"


@pytest.mark.L0
@pytest.mark.CPU
def test_resolve_config_requires_container_image(monkeypatch):
    monkeypatch.delenv("POSTTRAIN_CONTAINER_IMAGE", raising=False)
    with pytest.raises(ValueError):
        resolve_config()


@pytest.mark.L0
@pytest.mark.CPU
def test_build_sbatch_script_contains_config_and_quotes_run():
    s = build_sbatch_script(
        run="python -m foo --x 1",
        nodes=2,
        gpus_per_node=8,
        account="acct",
        partition="part",
        time_limit="1:00:00",
        container_image="/<runtime-image>",
        container_mounts="/lustre:/lustre",
        container_name="cn",
        repo_root="/repo",
        ray_port=6379,
        log_path="/tmp/j_%j.out",
        ray_runtime_env="export PYTHONPATH=/repo; export UCX_TLS=rc_mlx5,tcp",
    )
    assert "#SBATCH -N 2" in s
    assert "#SBATCH --gpus-per-node=8" in s
    assert "#SBATCH -A acct" in s and "#SBATCH -p part" in s
    assert "#SBATCH -o /tmp/j_%j.out" in s
    assert "/<runtime-image>" in s and "--container-mounts /lustre:/lustre" in s
    # Two-step bring-up invariants (each maps to a real multinode failure we hit):
    # 1) the IMPORT is a BLOCKING, non-overlap step ending in `true` — a backgrounded import raced by
    #    the readiness probe makes pyxis double-extract the squashfs -> "File exists".
    assert "srun --ntasks-per-node=1 --container-image '/<runtime-image>' --container-mounts /lustre:/lustre" in s
    assert "--container-name cn --container-workdir /repo true" in s
    # 2) the long-lived keep-alive REUSES the rootfs and carries the mounts (sleep infinity), else
    #    attaches lose /lustre and hit `cd: .../imaginaire4: No such file or directory`.
    assert (
        "srun --overlap --ntasks-per-node=1 --container-image '/<runtime-image>' --container-mounts /lustre:/lustre"
        in s
    )
    assert "--container-name cn --container-workdir /repo sleep infinity &" in s
    # 3) readiness requires every declared mount in each node's mount table: an attach before a node's
    #    keep-alive is up succeeds in a mount-less container, and Ray then starts there without the repo.
    assert (
        f"srun --overlap --ntasks-per-node=1 --container-name cn {_mount_probe('/lustre:/lustre')} >/dev/null 2>&1"
    ) in s
    assert "RUN_CMD='python -m foo --x 1'" in s  # shlex.quote
    assert "GPUS_PER_NODE=8" in s
    assert "RAY_RUNTIME_ENV='export PYTHONPATH=/repo; export UCX_TLS=rc_mlx5,tcp'" in s
    assert "/repo/cosmos_framework/posttrain/utils/launch_multinode.sh" in s
    assert "pip install" in s
    assert "if ! srun" in s and "multinode dependency installation failed" in s
    assert "/dev/null" not in s.split("pip install", 1)[1].split("fi", 1)[0]


@pytest.mark.L0
@pytest.mark.CPU
def test_mount_probe_requires_every_declared_target_in_the_mount_table(tmp_path: pathlib.Path) -> None:
    """A path baked into the image passes `test -d` without its mount; only the mount table is proof."""
    assert _mount_probe("") == "true"
    probe = _mount_probe("/lustre:/lustre,/path/to/home-resource,/data/,/lustre")
    assert probe.count("/proc/self/mountinfo") == 3
    for target in ("T=/lustre ", "T=/path/to/home-resource ", "T=/data "):
        assert target in probe

    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text("36 35 98:0 / / rw - overlay overlay rw\n37 36 0:52 /share /lustre rw - lustre x rw\n")
    present = shlex.split(_mount_probe("/lustre"))[2].replace("/proc/self/mountinfo", str(mountinfo))
    missing = shlex.split(_mount_probe("/lustre,/data"))[2].replace("/proc/self/mountinfo", str(mountinfo))
    assert subprocess.run(["bash", "-c", present], check=False).returncode == 0
    assert subprocess.run(["bash", "-c", missing], check=False).returncode != 0


@pytest.mark.L0
@pytest.mark.CPU
def test_mount_probe_matches_mount_points_the_kernel_octal_escapes(tmp_path: pathlib.Path) -> None:
    """mountinfo spells space, tab, newline and backslash in paths as \\040, \\011, \\012 and \\134."""
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "36 35 98:0 / / rw - overlay overlay rw\n"
        "37 36 0:52 /share /data\\040set rw - lustre x rw\n"
        "38 36 0:53 /share /tab\\011dir\\134x rw - lustre x rw\n"
    )

    def passes(container_mounts: str) -> bool:
        checks = shlex.split(_mount_probe(container_mounts))[2].replace("/proc/self/mountinfo", str(mountinfo))
        return subprocess.run(["bash", "-c", checks], check=False).returncode == 0

    assert passes("/scratch:/data set")
    assert passes("/scratch:/tab\tdir\\x,/scratch:/data set:ro")
    assert not passes("/scratch:/data\\040set")
    assert not passes("/scratch:/data")


@pytest.mark.L0
@pytest.mark.CPU
def test_build_sbatch_script_rejects_batch_long_without_p0() -> None:
    with pytest.raises(ValueError, match="batch_long requires wckey p0"):
        build_sbatch_script(
            run="-m cosmos_framework.posttrain.entrypoints.dpo",
            nodes=2,
            gpus_per_node=4,
            account="cosmos_base_training",
            partition="batch_long",
            time_limit="12:00:00",
            container_image="/containers/<runtime-image>",
            container_mounts="/lustre:/lustre",
            container_name="cosmos",
            repo_root="/repo",
            ray_port=6379,
            log_path="/logs/job.out",
        )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.L0
@pytest.mark.CPU
def test_build_sbatch_script_has_preemption_hook():
    """The batch script must forward Slurm's SIGUSR1 as the shared-FS sentinel the
    training loop polls (pyxis containers never receive the signal directly), and run
    the launcher backgrounded + wait so the trap can fire mid-run."""
    from cosmos_framework.posttrain.utils.launch_multinode import build_sbatch_script

    script = build_sbatch_script(
        run="python -m x",
        nodes=2,
        gpus_per_node=4,
        account="a",
        partition="p",
        time_limit="4:00:00",
        container_image="<runtime-image>",
        container_mounts="/lustre:/lustre",
        container_name="c",
        repo_root="/repo",
        ray_port=1,
        log_path="/logs/out_%j.log",
    )
    assert "--signal=B:SIGUSR1@600" in script
    assert "SIGUSR1_RECEIVED" in script and "SLURM_LOG_DIR" in script
    assert "wait $_lp" in script  # backgrounded launcher, trap-interruptible


def test_build_sbatch_script_no_pip():
    s = build_sbatch_script(
        run="x",
        nodes=2,
        gpus_per_node=8,
        account="a",
        partition="p",
        time_limit="1:00:00",
        container_image="/i",
        container_mounts="/m",
        container_name="cn",
        repo_root="/r",
        ray_port=6379,
        log_path="/l",
        pip_install=False,
    )
    assert "pip install" not in s


@pytest.mark.L0
@pytest.mark.CPU
def test_ray_head_readiness_is_bounded_and_checks_background_failure() -> None:
    launcher = pathlib.Path(__file__).resolve().parents[1] / "utils" / "launch_multinode.sh"
    source = launcher.read_text()
    assert "RAY_READY_ATTEMPTS" in source and "RAY_READY_TIMEOUT_SECONDS" in source
    assert 'kill -0 "$head_pid"' in source
    assert 'timeout "$seconds" srun' in source
    assert 'run_timed "$head" "$RAY_READY_TIMEOUT_SECONDS"' in source
    assert "Ray head was not ready" in source


@pytest.mark.L0
@pytest.mark.CPU
def test_runtime_preflight_runs_on_every_node_before_ray() -> None:
    launcher = pathlib.Path(__file__).resolve().parents[1] / "utils" / "launch_multinode.sh"
    source = launcher.read_text()
    preflight_at = source.index('if [ -n "$PREFLIGHT_CMD" ]')
    ray_start_at = source.index('run "$head" "ray start', preflight_at)
    assert "for node in $nodes" in source[preflight_at:ray_start_at]
    assert 'run_timed "$node" "$PREFLIGHT_TIMEOUT_SECONDS" "$PREFLIGHT_CMD"' in source
    assert preflight_at < ray_start_at


@pytest.mark.L0
@pytest.mark.CPU
def test_container_commands_default_every_compile_cache_node_local() -> None:
    """vLLM re-points inductor under VLLM_CACHE_ROOT, so inductor and Triton defaults alone left
    engines on different nodes compiling into one shared ~/.cache/vllm (FXGraphCacheMiss)."""
    launcher = pathlib.Path(__file__).resolve().parents[1] / "utils" / "launch_multinode.sh"
    source = launcher.read_text()
    for wrapper in ("run() {", "run_timed() {"):
        start = source.index(wrapper)
        body = source[start : source.index("\n}\n", start)]
        for default in (
            r"TORCHINDUCTOR_CACHE_DIR=\"\${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_\$USER}\"",
            r"TRITON_CACHE_DIR=\"\${TRITON_CACHE_DIR:-/tmp/triton_\$USER}\"",
            r"VLLM_CACHE_ROOT=\"\${VLLM_CACHE_ROOT:-/tmp/vllm_\$USER}\"",
        ):
            assert default in body, f"{wrapper} lacks the node-local default {default}"
        # The run's own environment comes after the defaults, so it can still pick another location.
        assert body.index("VLLM_CACHE_ROOT=") < body.index("RAY_RUNTIME_ENV")


@pytest.mark.L0
@pytest.mark.CPU
def test_parse_sacct_picks_primary_row():
    out = "42|COMPLETED|0:0\n42.batch|COMPLETED|0:0\n42.extern|COMPLETED|0:0\n"
    assert parse_sacct(out) == ("COMPLETED", 0)
    assert parse_sacct("7|FAILED|1:0\n7.batch|FAILED|1:0\n") == ("FAILED", 1)
    assert parse_sacct("7|CANCELLED by 5|0:15\n") == ("CANCELLED", 0)  # code is exit field, not signal
    assert parse_sacct("42|RUNNING|0:0\n") == ("RUNNING", 0)  # non-terminal; caller checks state, not code
    assert parse_sacct("42.batch|COMPLETED|0:0\n") == ("UNKNOWN", -1)  # only step rows ⇒ unknown
    assert parse_sacct("garbage line\n\n") == ("UNKNOWN", -1)  # malformed
    assert parse_sacct("") == ("UNKNOWN", -1)


@pytest.mark.L0
@pytest.mark.CPU
def test_submit_multinode_flow_mocked(monkeypatch, tmp_path):
    import subprocess as sp
    import types

    from cosmos_framework.posttrain.utils import launch_multinode as lm

    monkeypatch.setenv("POSTTRAIN_CONTAINER_IMAGE", "/<runtime-image>")
    calls = []
    sacct_seq = iter(["", "12345|RUNNING|0:0\n", "12345|COMPLETED|0:0\n"])  # empty, running, done

    def fake_run(cmd, **kw):
        calls.append(list(cmd))
        if cmd[0] == "scontrol":
            return types.SimpleNamespace(stdout="PartitionName=batch MaxTime=04:00:00", stderr="", returncode=0)
        if cmd[0] == "sbatch":
            return types.SimpleNamespace(stdout="12345\n", stderr="", returncode=0)
        if cmd[0] == "sacct":
            return types.SimpleNamespace(stdout=next(sacct_seq), stderr="", returncode=0)
        return types.SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(sp, "run", fake_run)
    res = lm.submit_multinode(run="echo hi", nodes=2, gpus_per_node=8, poll_interval=0, log_dir=str(tmp_path))
    assert res.job_id == "12345"
    assert res.ok  # COMPLETED + exit 0
    assert "12345" in res.log_path  # %j substituted
    assert any(c[0] == "sbatch" for c in calls)
    assert sum(1 for c in calls if c[0] == "sacct") >= 3  # polled past the empty + RUNNING rows


@pytest.mark.L0
@pytest.mark.CPU
def test_submit_multinode_missing_image_raises(monkeypatch):
    from cosmos_framework.posttrain.utils import launch_multinode as lm

    monkeypatch.delenv("POSTTRAIN_CONTAINER_IMAGE", raising=False)
    with pytest.raises(ValueError):
        lm.submit_multinode(run="echo hi")  # fails before any sbatch


@pytest.mark.L2
@pytest.mark.GPU
def test_multinode_launcher_runs_value_check():
    """Opt-in: submit a REAL 2-node slurm job that runs the multinode exact-value weight-sync test,
    and assert it passed. Needs a slurm cluster with sbatch + POSTTRAIN_CONTAINER_IMAGE. Never CI."""
    import os
    import shutil

    if os.environ.get("POSTTRAIN_MULTINODE") != "1" or not shutil.which("sbatch"):
        pytest.skip("opt-in: set POSTTRAIN_MULTINODE=1 on a slurm cluster (sbatch + POSTTRAIN_CONTAINER_IMAGE)")

    from cosmos_framework.posttrain.utils.launch_multinode import submit_multinode

    # Mesh size is cluster-specific: default to 2x8 but let a 4-GPU-per-node cluster (or any other)
    # override via env. The value test only needs the mesh to span >=2 nodes (see detect_mesh).
    nodes = int(os.environ.get("POSTTRAIN_MULTINODE_NODES", "2"))
    gpus_per_node = int(os.environ.get("POSTTRAIN_MULTINODE_GPUS_PER_NODE", "8"))
    res = submit_multinode(
        run=(
            "python -m pytest cosmos_framework/posttrain/tests/test_control_grpo_topologies.py"
            "::test_weight_sync_exact_values_multinode --L1 -p no:cacheprovider"
        ),
        nodes=nodes,
        gpus_per_node=gpus_per_node,
    )
    assert res.ok, f"multinode value test failed: state={res.state} exit={res.exit_code} log={res.log_path}"


@pytest.mark.L0
@pytest.mark.CPU
def test_main_passes_all_cli_flags(monkeypatch):
    from cosmos_framework.posttrain.utils import launch_multinode as lm

    captured = {}

    def fake_submit(run, **kw):
        captured["run"] = run
        captured.update(kw)
        return lm.SlurmRunResult("1", "COMPLETED", 0, "/l")

    monkeypatch.setattr(lm, "submit_multinode", fake_submit)
    with pytest.raises(SystemExit) as e:
        lm.main(
            [
                "--run",
                "echo hi",
                "--nodes",
                "4",
                "--gpus-per-node",
                "4",
                "--account",
                "acct",
                "--partition",
                "batch",
                "--time",
                "2:00:00",
                "--container-image",
                "/i",
                "--container-mounts",
                "/d:/d",
                "--container-name",
                "cn",
                "--ray-port",
                "7000",
                "--poll-interval",
                "3",
                "--timeout",
                "120",
                "--log-dir",
                "/logs",
                "--no-pip",
                "--preflight-cmd",
                "model-probe && environment-probe",
                "--preflight-timeout-seconds",
                "900",
                "--signal-seconds",
                "1800",
                "--driver-capabilities",
                "all",
                "--container-remap-root",
                "--no-requeue",
            ]
        )
    assert e.value.code == 0  # COMPLETED -> exit 0
    assert captured["run"] == "echo hi"
    assert captured["nodes"] == 4 and captured["gpus_per_node"] == 4
    assert captured["account"] == "acct" and captured["partition"] == "batch"
    assert captured["time_limit"] == "2:00:00"
    assert captured["container_image"] == "/i" and captured["container_mounts"] == "/d:/d"
    assert captured["container_name"] == "cn" and captured["ray_port"] == 7000
    assert captured["poll_interval"] == 3.0 and captured["timeout_s"] == 120.0
    assert captured["log_dir"] == "/logs"
    assert captured["pip_install"] is False
    assert captured["no_requeue"] and captured["container_remap_root"]
    assert captured["preflight_cmd"] == "model-probe && environment-probe"
    assert captured["preflight_timeout_s"] == 900 and captured["signal_seconds"] == 1800
    assert captured["driver_capabilities"] == "all"


@pytest.mark.L0
@pytest.mark.CPU
def test_build_sbatch_script_mode_selects_launcher():
    # mode picks the in-allocation launcher; container bring-up + RUN_CMD are shared.
    common = dict(
        run="-m foo -- x=1",
        nodes=2,
        gpus_per_node=8,
        account="a",
        partition="p",
        time_limit="1:00:00",
        container_image="/<runtime-image>",
        container_mounts="/m",
        container_name="cn",
        repo_root="/r",
        container_workdir="/workspace/cosmos-framework",
        ray_port=6379,
        log_path="/l",
    )
    ray = build_sbatch_script(**common, mode="ray")
    tr = build_sbatch_script(**common, mode="torchrun")
    assert "utils/launch_multinode.sh" in ray and "launch_multinode_torchrun.sh" not in ray
    assert "utils/launch_multinode_torchrun.sh" in tr
    assert "RUN_CMD='-m foo -- x=1'" in ray and "RUN_CMD='-m foo -- x=1'" in tr  # shared passthrough
    for script in (ray, tr):
        assert "--container-workdir /workspace/cosmos-framework" in script
        assert "REPO_ROOT=/r CONTAINER_WORKDIR=/workspace/cosmos-framework" in script
