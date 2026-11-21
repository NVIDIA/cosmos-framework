# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Source/installation contracts; does not substitute for a fresh CUDA environment run."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.L0, pytest.mark.CPU]
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "docker/posttrain/native_action/install.py"
if not SCRIPT.is_file():
    pytest.skip("native action runtime profile tests require source-checkout assets", allow_module_level=True)
spec = importlib.util.spec_from_file_location("native_action_install", SCRIPT)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def test_profile_pins_actual_family_runtime_without_global_or_private_distribution_resolution():
    pins = runtime.requirements()
    assert {
        key: pins[key]
        for key in ("torch", "torchvision", "torchcodec", "transformers", "diffusers", "numpy", "bitsandbytes")
    } == {
        "torch": "2.10.0+cu128",
        "torchvision": "0.25.0+cu128",
        "torchcodec": "0.10.0+cu128",
        "transformers": "4.57.6",
        "diffusers": "0.35.1",
        "numpy": "1.26.4",
        "bitsandbytes": "0.49.2",
    }
    assert not any(name.startswith("cosmos-framework") for name in pins)
    assert not {"gr00t", "alpamayo-r1", "vllm", "libero", "robosuite", "mujoco"} & set(pins)
    assert runtime.CHECKOUT == ROOT


@pytest.mark.parametrize(
    "value",
    [
        "torch>=2.10",
        "torch==2.10\ntorch==2.11",
        "cosmos-framework-flow-runtime==1.2.2",
        "--extra-index-url https://example.invalid",
    ],
)
def test_nonexact_or_private_profile_dependencies_fail_before_install(tmp_path, value):
    path = tmp_path / "requirements.txt"
    path.write_text(value)
    with pytest.raises(ValueError):
        runtime.requirements(path)


def source_fixture(tmp_path):
    package = tmp_path / "native"
    package.mkdir()
    code = package / "forward.py"
    code.write_text("def forward(x):\n    return x\n")
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    files = {"forward.py": hashlib.sha256(code.read_bytes()).hexdigest()}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"package": "native", "directories": ["."], "revision": revision, "sha256": digest}, code


def test_exact_clean_source_is_verified_without_importing_optional_packages(tmp_path):
    expected, _ = source_fixture(tmp_path)
    assert runtime.verify_source(tmp_path, expected) == {"revision": expected["revision"], "sha256": expected["sha256"]}


@pytest.mark.parametrize("mutation", ["revision", "hash", "tracked_change", "untracked_module"])
def test_source_changes_fail_explicitly_instead_of_using_latest_or_falling_back(tmp_path, mutation):
    expected, code = source_fixture(tmp_path)
    if mutation == "revision":
        expected["revision"] = "0" * 40
    elif mutation == "hash":
        expected["sha256"] = "0" * 64
    elif mutation == "tracked_change":
        code.write_text("def forward(x):\n    return -x\n")
    else:
        code.with_name("extra.py").write_text("raise RuntimeError('unexpected source')\n")
    with pytest.raises(ValueError):
        runtime.verify_source(tmp_path, expected)


def test_activation_handles_spaces_without_altering_scheduler_or_shared_library_policy(tmp_path):
    path = tmp_path / "native env"
    script = runtime.activation_script(path)
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert str(ROOT) in script and "Isaac-GR00T" in script and "alpamayo/src" in script
    assert not any(name in script for name in ("SLURM_", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES"))


def test_installer_refuses_existing_environment_before_calling_uv(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime, "run", lambda *_args, **_kwargs: pytest.fail("must refuse before any command"))
    with pytest.raises(FileExistsError):
        runtime.main(["--venv", str(tmp_path)])


def test_installer_refuses_unqualified_platform_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "aarch64")
    with pytest.raises(ValueError, match="x86_64"):
        runtime.main(["--venv", str(tmp_path / "new")])
    assert not (tmp_path / "new").exists()
