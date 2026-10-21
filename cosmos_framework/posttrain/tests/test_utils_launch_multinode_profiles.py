# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Deployment profile and submission contracts; no training imports or Slurm allocation."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from cosmos_framework.posttrain.utils import launch_multinode as launcher

pytestmark = [pytest.mark.L0, pytest.mark.CPU]
POSTTRAIN = Path(__file__).resolve().parents[1]


def test_profile_is_self_contained_and_cli_overrides_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    submit = Mock()
    monkeypatch.setattr(launcher, "submit_multinode", submit)
    monkeypatch.setenv("OMP_NUM_THREADS", "99")
    launcher.main(
        [
            "--launch-config",
            str(POSTTRAIN / "configs/sft_qwen3p8_27b_obj1k_gcp.json"),
            "--repo-root",
            str(tmp_path),
            "--job-name",
            "new_experiment",
            "--env",
            "OMP_NUM_THREADS=2",
            "--env",
            "SFT_S3_CREDENTIALS=/shared/my credentials.json",
            "--dry-run",
            "--",
            "rl.optimizer.lr=5e-6",
        ]
    )
    arguments = submit.call_args.kwargs
    assert arguments["repo_root"] == str(tmp_path)
    assert arguments["env"]["OMP_NUM_THREADS"] == "2"
    assert arguments["env"]["SFT_S3_CREDENTIALS"] == "/shared/my credentials.json"
    assert arguments["env"]["SFT_MODEL_PATH"].endswith("/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
    assert arguments["env"]["IMAGINAIRE_OUTPUT_ROOT"] == str(tmp_path / "outputs/sft_obj1k")
    assert arguments["log_dir"] == str(tmp_path / "outputs/sft_obj1k/logs/new_experiment")
    assert arguments["job_name"] == "new_experiment"
    assert arguments["qos"] == "normal" and arguments["cpus_per_task"] == 32
    assert arguments["venv"] == "/opt/venv/posttrain/bin/activate"
    assert arguments["submit_only"] and not arguments["pip_install"]
    tokens = shlex.split(submit.call_args.args[0])
    assert tokens[-3:] == ["--", "rl.optimizer.lr=5e-6", "job.name=new_experiment"]
    assert ".ene/scratch" not in str(submit.call_args)


def test_wait_overrides_profile_submit_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    submit = Mock()
    monkeypatch.setattr(launcher, "submit_multinode", submit)
    launcher.main(
        [
            "--launch-config",
            str(POSTTRAIN / "configs/sft_qwen3p8_27b_obj1k_gcp.json"),
            "--wait",
            "--dry-run",
            "--output-root",
            str(tmp_path),
        ]
    )
    assert not submit.call_args.kwargs["submit_only"]


@pytest.mark.parametrize(
    "extra",
    [
        ["--env", "NOT-VALID=x"],
        ["--env", "MISSING_VALUE"],
        ["--job-name", "x", "--", "job.name=y"],
        ["--submit-only", "--until-iter", "2", "--watch-dir", "/shared/watch"],
    ],
)
def test_invalid_cli_never_submits(extra: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    submit = Mock()
    monkeypatch.setattr(launcher, "submit_multinode", submit)
    with pytest.raises(SystemExit) as error:
        launcher.main(["--run", "-m application", *extra])
    assert error.value.code == 2
    submit.assert_not_called()


@pytest.mark.parametrize("env", [{"FLAG": True}, ["FLAG=value"], {"BAD=KEY": "value"}])
def test_profile_env_requires_string_mapping(env: object, tmp_path: Path) -> None:
    profile = tmp_path / "invalid.json"
    profile.write_text(json.dumps({"run": "x", "env": env}))
    with pytest.raises(SystemExit) as error:
        launcher.main(["--launch-config", str(profile)])
    assert error.value.code == 2


def test_overrides_extend_existing_separator(monkeypatch: pytest.MonkeyPatch) -> None:
    submit = Mock()
    monkeypatch.setattr(launcher, "submit_multinode", submit)
    launcher.main(["--run", "-m application -- rl.max_steps=2", "--dry-run", "--", "job.name=two words"])
    assert shlex.split(submit.call_args.args[0]) == ["-m", "application", "--", "rl.max_steps=2", "job.name=two words"]


def test_submit_only_records_acceptance_without_polling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = Mock(
        side_effect=[
            subprocess.CompletedProcess([], 0, "PartitionName=batch MaxTime=04:00:00", ""),
            subprocess.CompletedProcess([], 0, "12345\n", ""),
        ]
    )
    monkeypatch.setattr(subprocess, "run", run)
    result = launcher.submit_multinode(
        "echo training",
        container_image="/<runtime-image>",
        log_dir=str(tmp_path),
        submit_only=True,
        job_name="experiment",
        qos="normal",
        cpus_per_task=32,
    )
    assert result is not None and result.job_id == "12345" and result.state == "SUBMITTED"
    assert not result.ok  # scheduler acceptance is not training completion
    assert [call.args[0][0] for call in run.call_args_list] == ["scontrol", "sbatch"]
    assert run.call_args.args[0][1:-1] == ["--parsable", "--job-name=experiment", "--qos=normal", "--cpus-per-task=32"]
    record = json.loads((tmp_path / "submission.json").read_text())
    assert record["job_id"] == result.job_id and record["log_path"] == result.log_path
    script = Path(record["script"]).read_text()
    assert "#SBATCH --job-name=experiment" in script
    assert "#SBATCH --qos=normal" in script and "#SBATCH --cpus-per-task=32" in script


def test_dry_run_by_file_needs_only_stdlib(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(POSTTRAIN / "utils/launch_multinode.py"),
            "--launch-config",
            str(POSTTRAIN / "configs/sft_qwen3p8_27b_obj1k_gcp.json"),
            "--output-root",
            str(tmp_path),
            "--job-name",
            "dry-run",
            "--dry-run",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    syntax = subprocess.run(["bash", "-n"], input=result.stdout, text=True, capture_output=True, timeout=30)
    assert syntax.returncode == 0, syntax.stderr
    assert "#SBATCH --job-name=dry-run" in result.stdout
    assert not (tmp_path / "logs/dry-run/submission.json").exists()


@pytest.mark.parametrize("mode", ["torchrun", "ray", "ray-timed"])
def test_runtime_env_survives_container_and_venv_defaults(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise container command construction with mocked scheduler/executable boundaries."""
    venv = tmp_path / "activate"
    venv.write_text("export EXAMPLE=venv-default\n")
    value = "literal 'quote' $HOME $(touch should-not-exist)\nsecond line"
    script = launcher.build_sbatch_script(
        run="-m training",
        nodes=2,
        gpus_per_node=4,
        account="acct",
        partition="batch",
        time_limit="1:00:00",
        container_image="/<runtime-image>",
        container_mounts="/shared:/shared",
        container_name="test",
        repo_root=str(tmp_path),
        ray_port=29500,
        log_path=str(tmp_path / "log_%j.out"),
        env={"EXAMPLE": value},
        pip_install=False,
    )
    # The generated prefix passes exports as literal data, not evaluated shell substitutions.
    prefix = script.split("import_ok=0", 1)[0]
    functions = r"""
scontrol() { printf 'node1\nnode2\n'; }
getent() { printf '127.0.0.1 STREAM node1\n'; }
srun() {
  while [ "$1" != bash ]; do shift; done
  EXAMPLE=image-default "$@"
}
torchrun() { printf '%s' "$EXAMPLE"; test -d "$PYTORCH_KERNEL_CACHE_PATH"; }
timeout() { shift; "$@"; }
export -f scontrol getent srun torchrun timeout
"""
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node[1-2]")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("RUN_CMD", "-m training")
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("VENV_ACTIVATE", str(venv))
    monkeypatch.setenv("PYTORCH_KERNEL_CACHE_PATH", str(tmp_path / "kernels"))
    if mode == "torchrun":
        payload = f"bash {shlex.quote(str(POSTTRAIN / 'utils/launch_multinode_torchrun.sh'))}"
    else:
        # Load the actual Ray command wrappers, stopping before cluster startup.
        payload = (POSTTRAIN / "utils/launch_multinode.sh").read_text().split('if [ -n "$PREFLIGHT_CMD" ]', 1)[0]
        command = 'run_timed "$head" 10' if mode == "ray-timed" else 'run "$head"'
        payload += command + ' \'printf "%s" "$EXAMPLE"\'\n'
    result = subprocess.run(
        ["bash"],
        input=prefix + functions + payload,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == value
    assert not (tmp_path / "should-not-exist").exists()
