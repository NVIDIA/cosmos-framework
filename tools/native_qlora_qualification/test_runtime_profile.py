# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Explicit diagnostic admission; these CPU tests are not native GPU evidence."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import runtime


def test_profile_environment_is_bounded_and_owned(tmp_path, monkeypatch):
    monkeypatch.setenv("COSMOS_SFT_PROFILE_DIR", "/untrusted/inherited")
    monkeypatch.setenv("COSMOS_SFT_PROFILE_ACTIVE", "999999")
    assert runtime.trace_environment(None, tmp_path) == {}
    for window, wait in [("early", "0"), ("steady", "5")]:
        assert runtime.trace_environment(window, tmp_path) == {
            "COSMOS_SFT_PROFILE_DIR": str(tmp_path / "trace"),
            "COSMOS_SFT_PROFILE_WAIT": wait,
            "COSMOS_SFT_PROFILE_WARMUP": "1",
            "COSMOS_SFT_PROFILE_ACTIVE": "2",
        }
    with pytest.raises(ValueError, match="trace window"):
        runtime.trace_environment("custom", tmp_path)


@pytest.mark.parametrize("window", ["early", "steady"])
def test_actual_runtime_cli_routes_explicit_profile_without_model_import(tmp_path, monkeypatch, window):
    called = []
    monkeypatch.setattr(runtime, "execute", lambda args: called.append(args))
    argv = ["runtime.py", "--trace-window", window]
    for key in (
        "spec",
        "workloads",
        "profile",
        "workload-manifest-sha256",
        "profile-manifest-sha256",
        "model",
        "output",
        "build",
        "kernel-gate",
        "time-binary",
    ):
        argv.extend(["--" + key, str(tmp_path / key)])
    monkeypatch.setattr(sys, "argv", argv)
    runtime.main()
    assert len(called) == 1 and called[0].trace_window == window
    assert called[0].stage_timeout == 3600
