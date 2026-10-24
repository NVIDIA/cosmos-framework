# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the real batch shell with a recording child, never a controller."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BATCH = Path(__file__).resolve().parents[4] / "examples/automl/slurm_controller.sbatch"


@pytest.fixture
def launch(tmp_path):
    interpreter = tmp_path / "record-python"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps({'argv': sys.argv[1:], 'pid': os.getpid(),\n"
        "        'attempt': os.environ.get('AUTOML_STARTUP_ATTEMPT')}) + '\\n')\n"
        "if sys.argv[1] != '-m':\n"
        "    sys.exit(int(os.environ.get('HELPER_EXIT', '0')))\n"
    )
    interpreter.chmod(0o755)
    count = 0

    def run(**overrides):
        nonlocal count
        count += 1
        calls = tmp_path / f"calls-{count}.jsonl"
        env = {
            "PATH": os.environ["PATH"],
            "CALLS": str(calls),
            "AUTOML_CONTROLLER_PYTHON": str(interpreter),
            "AUTOML_MANIFEST": "/shared/path with spaces/search.yaml",
            "AUTOML_WORKSPACE": "/shared/runs",
            "AUTOML_CONTROLLER_MAX_ALLOCATIONS": "3",
            **overrides,
        }
        process = subprocess.Popen(["bash", str(BATCH)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            _, stderr = process.communicate(timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        entries = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
        return process.returncode, stderr.decode(), entries, process.pid

    return run


def expected_cli(workflow="sft", action="run", cap="3"):
    # Original #288 argv for default action, including exact argument ordering.
    return [
        "-m",
        "cosmos_framework.scripts.automl",
        *(["prompts"] if workflow == "prompts" else []),
        action,
        "--manifest",
        "/shared/path with spaces/search.yaml",
        "--workspace",
        "/shared/runs",
        "--approve",
        "--controller-max-allocations",
        cap,
        "--controller-allocation-seconds",
        "3600",
        "--controller-grace-seconds",
        "300",
    ]


@pytest.mark.parametrize("workflow", ["sft", "prompts"])
@pytest.mark.parametrize("settings", [{}, {"AUTOML_STARTUP_HELPER": ""}, {"AUTOML_STARTUP_BASE": "/ignored"}])
def test_unsupervised_default_preserves_original_argv_and_exec(launch, workflow, settings):
    code, stderr, entries, pid = launch(AUTOML_WORKFLOW=workflow, **settings)
    assert (code, stderr) == (0, "")
    assert entries == [{"argv": expected_cli(workflow), "pid": pid, "attempt": None}]


@pytest.mark.parametrize("cap", ["1", "3", "0003", "10", "9" * 25])
def test_allocation_count_boundary_includes_initial_allocation(launch, cap):
    # Same mathematical boundary as automl_core.controller.Session.run:
    # generation >= max_allocations is exhausted; generation starts at zero.
    for generation in [0, int(cap) - 1, int(cap), int(cap) + 1]:
        code, stderr, entries, _ = launch(AUTOML_CONTROLLER_MAX_ALLOCATIONS=cap, SLURM_RESTART_COUNT=str(generation))
        if generation < int(cap):
            assert (code, stderr) == (0, "")
            assert entries[0]["argv"] == expected_cli(cap=cap)
        else:
            assert code == 2 and "allocation cap exhausted" in stderr
            assert entries == []


@pytest.mark.parametrize("cap", ["0", "000", "-1", "1+1", "x"])
def test_invalid_cap_fails_before_child(launch, cap):
    code, stderr, entries, _ = launch(AUTOML_CONTROLLER_MAX_ALLOCATIONS=cap)
    assert code == 2 and "positive decimal integer" in stderr
    assert entries == []


@pytest.mark.parametrize("generation", ["-1", "1+1", "x"])
def test_invalid_generation_fails_before_child(launch, generation):
    code, stderr, entries, _ = launch(SLURM_RESTART_COUNT=generation)
    assert code == 2 and "nonnegative decimal integer" in stderr
    assert entries == []


@pytest.fixture
def supervised():
    return {
        "AUTOML_STARTUP_HELPER": "/shared/probe.py",
        "AUTOML_STARTUP_BASE": "/shared/startup",
        "AUTOML_DEADLINE": "2000000000",
        "SLURM_JOB_ID": "123",
    }


def test_supervised_restart_normalizes_generation_but_preserves_cap_argv(launch, supervised):
    code, stderr, entries, pid = launch(
        **supervised, SLURM_RESTART_COUNT="0002", AUTOML_CONTROLLER_MAX_ALLOCATIONS="003"
    )
    assert (code, stderr) == (0, "")
    assert [entry["argv"][1] for entry in entries[:2]] == ["probe", "fanout"]
    assert all(entry["attempt"] == "/shared/startup/123-g2" for entry in entries)
    assert entries[-1]["argv"] == expected_cli(cap="003")
    assert entries[-1]["pid"] == pid


def test_exhausted_cap_does_not_start_probe(launch, supervised):
    code, stderr, entries, _ = launch(**supervised, SLURM_RESTART_COUNT="3")
    assert code == 2 and "allocation cap exhausted" in stderr
    assert entries == []


def test_probe_failure_does_not_start_cli_or_leak_requeue_exit(launch, supervised):
    code, stderr, entries, _ = launch(**supervised, HELPER_EXIT="75")
    assert code == 2 and "probe failed (exit 75)" in stderr
    assert len(entries) == 1 and entries[0]["argv"][1] == "probe"


@pytest.mark.parametrize("workflow", ["sft", "prompts"])
def test_explicit_resume_preserves_exec(launch, workflow):
    code, stderr, entries, pid = launch(AUTOML_ACTION="resume", AUTOML_WORKFLOW=workflow)
    assert (code, stderr) == (0, "")
    assert entries == [{"argv": expected_cli(workflow, "resume"), "pid": pid, "attempt": None}]


@pytest.mark.parametrize("action", ["plan", "status", "cancel", "migrate", "run;true"])
def test_nonlaunch_action_cannot_reach_helper(launch, supervised, action):
    code, stderr, entries, _ = launch(**supervised, AUTOML_ACTION=action)
    assert code == 2 and "AUTOML_ACTION must be run or resume" in stderr
    assert entries == []
