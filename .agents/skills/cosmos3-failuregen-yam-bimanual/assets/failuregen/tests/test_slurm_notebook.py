# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Validate the Slurm/Pyxis handoff without allocating a node or starting Jupyter."""

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/slurm_notebook.sbatch"


def setup_job(tmp_path):
    root = tmp_path / "shared workspace"
    cookbook = root / "repos/cosmos-framework/.agents/skills/cosmos3-failuregen-yam-bimanual/assets/failuregen"
    work = root / "outputs/failuregen-work"
    run_dir = work / "training/cosmos3_yam/fd_id/yam_eval_eef_gb200_v6"
    files = [
        cookbook / "src/failuregen/workflow.py",
        cookbook / "scripts/slurm_notebook.sbatch",
        root / "repos/cosmos-framework/cosmos_framework/inference/args.py",
        work / "dataset/meta/info.json",
        work / "dataset/failuregen_annotations.json",
        work / "prepared/manifest.json",
        work / "prepared/action_stats.json",
        run_dir / "config.yaml",
        run_dir / "yam_training_contract.json",
        run_dir / "checkpoints/iter_000001243/model/.metadata",
    ]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    srun = bin_dir / "srun"
    srun.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    srun.chmod(0o755)
    env = {
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "ROOT": str(root),
        "SLURM_JOB_ID": "123",
        "SLURM_NNODES": "1",
        "CONTAINER_IMAGE": "registry#cosmos:cluster",
        "COOKBOOK": str(cookbook),
        "CAPTURE": str(tmp_path / "arguments.json"),
    }
    return root, work, env


def test_launch_keeps_shared_paths_and_four_gpu_allocation(tmp_path):
    root, work, env = setup_job(tmp_path)
    result = subprocess.run(["bash", str(SCRIPT)], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    args = json.loads(Path(env["CAPTURE"]).read_text())
    assert "--nodes=1" in args and "--ntasks=1" in args
    assert f"--container-mounts={root}:{root}" in args
    assert f"FAILUREGEN_PREPARED={work}/prepared" in args
    assert "JUPYTER_IP=127.0.0.1" in args
    assert args[-1] == "--in-container"
    assert not any("CUDA_VISIBLE_DEVICES=0" == arg for arg in args)


def test_missing_inputs_fail_before_starting_container(tmp_path):
    root, _, env = setup_job(tmp_path)
    (root / "repos/cosmos-framework/cosmos_framework/inference/args.py").unlink()
    result = subprocess.run(["bash", str(SCRIPT)], env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "Missing input" in result.stderr
    assert not Path(env["CAPTURE"]).exists()


def test_login_node_launch_without_allocation_is_rejected(tmp_path):
    _, _, env = setup_job(tmp_path)
    del env["SLURM_JOB_ID"]
    result = subprocess.run(["bash", str(SCRIPT)], env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "Run inside a Slurm allocation" in result.stderr
    assert not Path(env["CAPTURE"]).exists()


def test_remote_network_bind_requires_explicit_override(tmp_path):
    _, _, env = setup_job(tmp_path)
    env.update(JUPYTER_IP="0.0.0.0", JUPYTER_PORT="8890")
    result = subprocess.run(["bash", str(SCRIPT)], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    args = json.loads(Path(env["CAPTURE"]).read_text())
    assert "JUPYTER_IP=0.0.0.0" in args and "JUPYTER_PORT=8890" in args
