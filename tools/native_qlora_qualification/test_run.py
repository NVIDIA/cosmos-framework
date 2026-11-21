# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Orchestration contracts; these do not qualify accelerator kernels or performance."""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from protocol import file_sha256
from run import Stages, validate_action_checkpoint, verify_kernel_gate


@pytest.mark.parametrize("action,checkpoint", [("train-eval", None), ("frozen", None), ("evaluate", "/own/checkpoint")])
def test_explicit_native_action_checkpoint_contract(action, checkpoint):
    validate_action_checkpoint(action, checkpoint)


@pytest.mark.parametrize(
    "action,checkpoint",
    [
        ("evaluate", None),
        ("frozen", "/trained/checkpoint"),
        ("train-eval", "/other/checkpoint"),
        ("evaluate", ""),
        ("evaluate", "   "),
        ("unknown", None),
    ],
)
def test_ambiguous_action_fails_before_config_read_or_allocation(tmp_path, monkeypatch, action, checkpoint):
    module = importlib.import_module("run")
    monkeypatch.setattr(module.Experiment, "read", lambda path: pytest.fail("read configuration before action guard"))
    monkeypatch.setattr(module, "verify_build", lambda path: pytest.fail("verified runtime before action guard"))
    output = tmp_path / "new"
    with pytest.raises(ValueError):
        module.execute(SimpleNamespace(action=action, checkpoint=checkpoint, output=str(output), spec="unread.json"))
    assert not output.exists()


@pytest.mark.parametrize("module_name", ["run", "lifecycle"])
def test_missing_process_capability_refuses_stages_before_launch(tmp_path, monkeypatch, module_name):
    module = importlib.import_module(module_name)

    def unavailable():
        raise RuntimeError("pidfd capability unavailable")

    monkeypatch.setattr(module.OwnedProcessTree, "check_support", staticmethod(unavailable))
    with pytest.raises(RuntimeError, match="pidfd capability unavailable"):
        if module_name == "run":
            module.Stages(tmp_path, Path("/usr/bin/time"))
        else:
            module.Stages(tmp_path, {}, 10)
    assert not (tmp_path / "stages").exists()


def test_failed_stage_retains_command_log_and_exit_code(tmp_path):
    stages = Stages(tmp_path, Path("/usr/bin/time"))
    with pytest.raises(RuntimeError, match="exited 7"):
        stages.run("intentional_failure", [sys.executable, "-c", "print('retained'); raise SystemExit(7)"])
    stage = tmp_path / "stages/intentional_failure"
    assert "retained" in (stage / "output.log").read_text()
    assert json.loads((stage / "outcome.json").read_text())["exit_code"] == 7
    assert json.loads((stage / "command.json").read_text())[0] == sys.executable
    with pytest.raises(FileExistsError):
        stages.run("intentional_failure", [sys.executable, "-c", "pass"])


@pytest.mark.parametrize("suite", ["gpu", "higher_precision"])
@pytest.mark.parametrize(
    "invalid",
    [
        "skipped",
        "failure",
        "error",
        "missing_case",
        "extra_case",
        "missing_path",
        "missing_sha",
        "missing_file",
        "changed_xml",
        "malformed_xml",
        None,
    ],
)
def test_gpu_gate_requires_both_exact_test_suites_without_skips(tmp_path, suite, invalid):
    receipt = {"code_revision": "a" * 40, "wheel_sha256": "b" * 64}
    value = {"completed": True, "build": receipt}
    for name, expected_count, prefix in (("gpu", 14, ""), ("higher_precision", 1, "higher_precision_")):
        selected = name == suite
        count = expected_count + (int(invalid == "extra_case") - int(invalid == "missing_case") if selected else 0)
        body = "".join(
            f'<testcase name="test_{index}">'
            + (f"<{invalid}/>" if selected and index == 0 and invalid in {"skipped", "failure", "error"} else "")
            + "</testcase>"
            for index in range(count)
        )
        xml = tmp_path / f"{name}.xml"
        xml.write_text("<testsuite>" + body + "</testsuite>" if not (selected and invalid == "malformed_xml") else "<")
        value[prefix + "junit_xml"] = str(xml)
        value[prefix + "junit_sha256"] = file_sha256(xml)
        if selected:
            if invalid == "missing_path":
                del value[prefix + "junit_xml"]
            elif invalid == "missing_sha":
                del value[prefix + "junit_sha256"]
            elif invalid == "missing_file":
                xml.unlink()
            elif invalid == "changed_xml":
                xml.write_text(xml.read_text() + "\n")
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps(value))
    if invalid is None:
        assert verify_kernel_gate(gate, receipt)["completed"]
    else:
        with pytest.raises(ValueError):
            verify_kernel_gate(gate, receipt)


@pytest.mark.parametrize("completed,build", [(False, {"version": 1}), (True, {"version": 2})])
def test_gpu_gate_requires_completed_matching_build_before_xml(tmp_path, completed, build):
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"completed": completed, "build": build}))
    with pytest.raises(ValueError, match="completed real-kernel gate for this exact build"):
        verify_kernel_gate(gate, {"version": 1})
