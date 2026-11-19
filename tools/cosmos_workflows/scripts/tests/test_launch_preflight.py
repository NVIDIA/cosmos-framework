# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for launch-preflight GPU architecture handling."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_cosmos_launch_preflight as preflight  # noqa: E402


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("10.3", "sm_103"),
        ("10.3a", "sm_103a"),
        ("103a", "sm_103a"),
        ("sm_103a", "sm_103a"),
        ("compute_103a", "sm_103a"),
    ],
)
def test_normalize_gpu_arch_accepts_arch_specific_forms(value, expected):
    assert preflight.normalize_gpu_arch(value) == expected


def test_no_architecture_support_is_invented_for_unvalidated_images():
    assert preflight.KNOWN_IMAGE_SMS == {}


def test_architecture_specific_target_matches_family_target():
    assert preflight.gpu_arch_is_supported("sm_103", {"sm_103a"})
    assert preflight.gpu_arch_is_supported("sm_103a", {"sm_103"})


def test_different_architecture_families_do_not_match():
    assert not preflight.gpu_arch_is_supported("sm_121", {"sm_120"})


def test_gpu_resources_accept_cumulative_memory_across_devices(capsys):
    assert preflight.check_gpu_resources(None, None, 256, 4, [80], True)
    assert "required_total_memory_gb=256" in capsys.readouterr().out


def test_gpu_resources_accept_nominal_single_device_capacity(capsys):
    # Target memory is supplied in GiB, matching nvidia-smi's MiB-based report.
    assert preflight.check_gpu_resources(None, None, 256, 1, [250.7], True)
    assert "GPU resources OK" in capsys.readouterr().out


def test_gpu_resources_reject_insufficient_cumulative_memory(capsys):
    assert not preflight.check_gpu_resources(None, None, 256, 2, [100], True)
    assert "GPU resource check failed" in capsys.readouterr().out


def test_target_gpu_selection_excludes_unallocated_display_gpu(capsys):
    gpus = [
        {"index": "0", "name": "A100"},
        {"index": "1", "name": "A100"},
        {"index": "3", "name": "Display"},
        {"index": "4", "name": "A100"},
    ]
    ok, selected = preflight.filter_target_gpus(gpus, ["0", "1", "4"])
    assert ok
    assert [gpu["index"] for gpu in selected] == ["0", "1", "4"]
    assert "indices=0,1,4" in capsys.readouterr().out


def test_target_count_without_memory_uses_detected_hardware(monkeypatch, capsys):
    monkeypatch.setattr(preflight, "detect_local_gpu_memory_gb", lambda: [80.0, 80.0])
    assert preflight.check_gpu_resources(2, 40, None, 2, [], False)
    assert "detected=0.0GiB" not in capsys.readouterr().out


def test_target_gpu_selection_rejects_missing_index(capsys):
    ok, selected = preflight.filter_target_gpus([{"index": "0"}], ["0", "2"])
    assert not ok and selected == []
    assert "missing index(es)=2" in capsys.readouterr().out


def _host_gpus():
    return [
        {
            "index": str(index),
            "uuid": f"GPU-0000000{index}",
            "name": "A100",
            "driver_version": "600.0",
            "memory_mib": 81920.0,
            "sm": "sm_80",
        }
        for index in range(4)
    ]


def test_cuda_visible_devices_filters_and_renumbers_host_inventory(capsys):
    ok, visible = preflight.filter_cuda_visible_gpus(_host_gpus(), "2,0")
    assert ok
    assert [gpu["index"] for gpu in visible] == ["2", "0"]
    assert [gpu["visible_index"] for gpu in visible] == ["0", "1"]
    assert "physical=4, visible=2" in capsys.readouterr().out


def test_cuda_visible_devices_accepts_unique_gpu_uuid_prefix():
    ok, visible = preflight.filter_cuda_visible_gpus(_host_gpus(), "GPU-00000002")
    assert ok
    assert [gpu["index"] for gpu in visible] == ["2"]


@pytest.mark.parametrize("visibility", ["", "-1"])
def test_cuda_visible_devices_rejects_empty_allocation(visibility):
    ok, visible = preflight.filter_cuda_visible_gpus(_host_gpus(), visibility)
    assert not ok
    assert visible == []


@pytest.mark.parametrize("visibility", ["0,,1", "7", "GPU-unknown", "0,0"])
def test_cuda_visible_devices_fails_closed_on_invalid_selection(visibility):
    ok, visible = preflight.filter_cuda_visible_gpus(_host_gpus(), visibility)
    assert not ok
    assert visible == []


def test_gpu_resource_count_uses_cuda_visible_inventory(monkeypatch, capsys):
    monkeypatch.setattr(
        preflight,
        "query_host_gpus",
        lambda **_kwargs: (True, _host_gpus()[:2]),
    )
    assert not preflight.check_gpu_resources(4, None, None, None, [], False)
    output = capsys.readouterr().out
    assert "qualifying_gpus=2 < required=4" in output


def test_multi_device_docker_gpu_request_preserves_csv_quotes():
    assert preflight.docker_gpu_request(["0", "1", "2", "4"]) == '"device=0,1,2,4"'
    assert preflight.docker_gpu_request([]) == "all"


def test_free_disk_requirement_passes_for_existing_parent(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        preflight.shutil,
        "disk_usage",
        lambda _path: preflight.shutil._ntuple_diskusage(500 * 1024**3, 100, 400 * 1024**3),
    )
    output = tmp_path / "not-created-yet" / "results"
    assert preflight.check_free_disk_space([("results", str(output))], {"results": 256}, skip_access=False)
    assert "Free-disk OK" in capsys.readouterr().out


def test_free_disk_requirement_rejects_insufficient_capacity(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        preflight.shutil,
        "disk_usage",
        lambda _path: preflight.shutil._ntuple_diskusage(500 * 1024**3, 450, 50 * 1024**3),
    )
    assert not preflight.check_free_disk_space([("results", str(tmp_path))], {"results": 256}, skip_access=False)
    assert "free=50.0GiB < required=256GiB" in capsys.readouterr().out


def test_slurm_preflight_checks_remote_scheduler_pyxis_and_enroot(monkeypatch, tmp_path):
    key = tmp_path / "id_ed25519"
    key.write_text("fixture")
    for name, value in {
        "SLURM_USER": "user",
        "SLURM_HOSTNAME": "login.example",
        "SLURM_PARTITION": "compute",
        "SSH_KEY_PATH": str(key),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(preflight.socket, "getaddrinfo", lambda *_args: [(None,)])
    monkeypatch.setattr(preflight, "check_slurm_runtime", lambda _platform: True)
    commands = []

    def fake_run(command, timeout=30, env=None):
        commands.append(command)
        remote = command[-1]
        if remote == "echo COSMOS_SSH_OK":
            return subprocess.CompletedProcess(command, 0, stdout="COSMOS_SSH_OK\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(preflight, "run", fake_run)
    platform = {
        "required_credentials": [
            {"name": "SLURM_USER", "source": "env_var"},
            {"name": "SLURM_HOSTNAME", "source": "env_var"},
            {"name": "SLURM_PARTITION", "source": "env_var"},
        ],
        "credential_groups": [{"require_one_of": ["SSH_KEY_PATH", "SSH_AUTH_SOCK"]}],
    }
    assert preflight.check_slurm(platform, [("data", "/shared/data")], {}, 20, False)
    remote_commands = [command[-1] for command in commands]
    assert any("command -v sbatch" in command for command in remote_commands)
    assert any("command -v enroot" in command for command in remote_commands)
    assert any("--container-image" in command for command in remote_commands)


def test_slurm_preflight_rejects_missing_remote_pyxis(monkeypatch, tmp_path, capsys):
    key = tmp_path / "id_ed25519"
    key.write_text("fixture")
    for name, value in {
        "SLURM_USER": "user",
        "SLURM_HOSTNAME": "login.example",
        "SLURM_PARTITION": "compute",
        "SSH_KEY_PATH": str(key),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(preflight.socket, "getaddrinfo", lambda *_args: [(None,)])
    monkeypatch.setattr(preflight, "check_slurm_runtime", lambda _platform: True)

    def fake_run(command, timeout=30, env=None):
        remote = command[-1]
        if remote == "echo COSMOS_SSH_OK":
            return subprocess.CompletedProcess(command, 0, stdout="COSMOS_SSH_OK\n", stderr="")
        if "command -v sbatch" in remote:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="enroot missing")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(preflight, "run", fake_run)
    platform = {
        "required_credentials": [
            {"name": "SLURM_USER", "source": "env_var"},
            {"name": "SLURM_HOSTNAME", "source": "env_var"},
            {"name": "SLURM_PARTITION", "source": "env_var"},
        ],
        "credential_groups": [{"require_one_of": ["SSH_KEY_PATH", "SSH_AUTH_SOCK"]}],
    }
    assert not preflight.check_slurm(platform, [], {}, 20, False)
    assert "Remote SLURM/Pyxis/Enroot preflight failed" in capsys.readouterr().out
