# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render + security tests for the single-node SLURM sbatch template.

A vendored launch template exists so the exact #SBATCH directives, the
sidecar-cred security control, and `--requeue` are faithful rather than authored
freehand per run. These tests assert a rendered instance is valid bash, leaves
no unsubstituted markers, never inlines a secret, and carries the load-bearing
directives.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "templates/slurm/singlenode.sbatch.tmpl"
sys.path.insert(0, str(REPO / "scripts"))
import redact_secrets  # noqa: E402

BASE = {
    "JOB_NAME": "cosmos3-train-a1b2c3",
    "NUM_GPUS": "1",
    "CPUS_PER_TASK": "16",
    "TIME": "04:00:00",
    "LOG_DIR": "/lustre/fsw/portfolios/edgeai/users/me/results/cosmos3-train-a1b2c3/slurm-logs",
    "SBATCH_EXTRA": "#SBATCH --account=edgeai\n#SBATCH --partition=polar,polar3",
    "ENV_FILE": "",
    "EXTRA_ENV": "",
    "IMAGE": "/lustre/fsw/sqsh/cosmos-framework.sqsh",
    "CONTAINER_MOUNTS": "/lustre",
    "COMMAND": "python -m cosmos_framework.scripts.train --sft-toml /lustre/fsw/.../specs/cosmos3-train-a1b2c3/spec.yaml",
}


def render(overrides=None):
    values = {**BASE, **(overrides or {})}
    text = TEMPLATE.read_text()
    for key, val in values.items():
        text = text.replace(f"@@{key}@@", val)
    return text


def bash_syntax_ok(text):
    return subprocess.run(["bash", "-n", "/dev/stdin"], input=text, text=True, capture_output=True).returncode == 0


def test_all_markers_substituted():
    rendered = render()
    leftover = re.findall(r"@@[A-Z_]+@@", rendered)
    assert leftover == [], f"unsubstituted markers: {leftover}"


def test_rendered_is_valid_bash():
    assert bash_syntax_ok(render())


def test_env_file_present_case_is_valid_bash():
    rendered = render({"ENV_FILE": "/lustre/.../job_cosmos3-train-a1b2c3.env"})
    assert bash_syntax_ok(rendered)
    assert "trap 'shred -u" in rendered  # sidecar shredded on exit
    assert 'source "/lustre' in rendered


def test_load_bearing_directives_present():
    rendered = render()
    for needle in (
        "#SBATCH --requeue",
        "#SBATCH --nodes=1",
        "#SBATCH --gres=gpu:1",
        "srun --container-image=",
        "--container-mounts=/lustre",
        "#SBATCH --account=edgeai",
    ):
        assert needle in rendered, f"missing: {needle}"


def test_no_cross_node_rendezvous_in_singlenode():
    # single-node must NOT set the multi-node rendezvous env (that is M7)
    rendered = render()
    assert "MASTER_ADDR" not in rendered
    assert "WORLD_SIZE" not in rendered


def test_secrets_never_inlined_lints_clean():
    # the sidecar pattern means the rendered script carries no literal creds,
    # even if the agent (wrongly) put a secret-shaped value in EXTRA_ENV we catch it
    rendered = render()
    assert rendered.count("export ") >= 1  # NCCL_DEBUG etc. are fine
    assert redact_secrets.scan(rendered) == []  # no literal credential

    # a deliberately bad render (inline secret) MUST be caught by the lint gate
    bad = render({"EXTRA_ENV": "export NGC_KEY=nvapi-LEAKEDsecret1234567890"})
    assert redact_secrets.scan(bad), "lint must flag an inlined credential"


def test_extra_env_nccl_knob_renders():
    rendered = render({"EXTRA_ENV": "export NCCL_P2P_DISABLE=1"})
    assert bash_syntax_ok(rendered)
    assert "NCCL_P2P_DISABLE=1" in rendered
