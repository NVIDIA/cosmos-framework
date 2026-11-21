# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Actual CPU filesystem operations and injected failures; no quota/Slurm queries."""

import errno
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

DIRECTORY = Path(__file__).parents[1]
sys.path.insert(0, str(DIRECTORY))
import storage


def test_real_nested_probe_fsync_rename_readback_and_cleanup(tmp_path):
    sentinel = tmp_path / "existing"
    sentinel.write_bytes(b"preserve")
    result = storage.require_storage([tmp_path], context="login")
    assert result["passed"] and result["reservation_created"] is False
    assert result["uid"] == result["effective_uid"] == os.geteuid()
    row = result["paths"][0]
    assert row["cleanup_complete"] and len(row["operations"]) == 7
    assert sum(x.get("bytes", 0) for x in row["operations"]) == 262144
    assert sorted(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"preserve"


@pytest.mark.parametrize("operation", ["mkdir", "fsync", "rename", "readback"])
def test_operational_failure_refuses_and_cleans_only_owned_files(tmp_path, monkeypatch, operation):
    sentinel = tmp_path / "keep"
    sentinel.write_bytes(b"untouched")
    if operation == "mkdir":
        original = Path.mkdir

        def fail(path, *args, **kwargs):
            if path.name == "new-parent":
                raise OSError(errno.EDQUOT, "quota exhausted")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", fail)
    elif operation == "fsync":
        original = os.fsync
        calls = 0

        def fail(descriptor):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError(errno.ENOSPC, "no space")
            return original(descriptor)

        monkeypatch.setattr(os, "fsync", fail)
    elif operation == "rename":
        monkeypatch.setattr(Path, "rename", lambda *args: (_ for _ in ()).throw(OSError(errno.EIO, "rename failed")))
    else:
        original = Path.read_bytes
        monkeypatch.setattr(Path, "read_bytes", lambda p: b"corrupt" if p.name.endswith(".bin") else original(p))
    with pytest.raises(storage.StorageProbeError) as error:
        storage.require_storage([tmp_path], context="login")
    row = error.value.report["paths"][0]
    assert not row["passed"] and row["cleanup_complete"]
    assert "error" in row
    assert sorted(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"untouched"


def test_cleanup_failure_never_reports_success(tmp_path, monkeypatch):
    original = Path.unlink

    def fail(path, *args, **kwargs):
        if path.suffix == ".bin":
            raise PermissionError(errno.EPERM, "cleanup refused")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail)
    report = storage.probe_storage([tmp_path], context="login")
    assert not report["passed"]
    assert report["paths"][0]["cleanup_errors"] and not report["paths"][0]["cleanup_complete"]


def test_changed_file_identity_is_not_deleted(tmp_path, monkeypatch):
    original = storage._identity

    def changed(path):
        identity = original(path)
        return (identity[0], identity[1] + 1) if path.suffix == ".bin" else identity

    monkeypatch.setattr(storage, "_identity", changed)
    report = storage.probe_storage([tmp_path], context="login")
    assert not report["passed"]
    row = report["paths"][0]
    assert "changed inode" in row["cleanup_errors"][0]["message"]
    assert len(list(Path(row["probe"]).rglob("*.bin"))) == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"files_per_path": 0},
        {"files_per_path": 17},
        {"files_per_path": True},
        {"bytes_per_file": 0},
        {"bytes_per_file": 65537},
    ],
)
def test_bounded_sizes_refuse_before_writing(tmp_path, kwargs):
    with pytest.raises(ValueError):
        storage.probe_storage([tmp_path], context="login", **kwargs)
    assert not list(tmp_path.iterdir())


def test_duplicate_symlink_missing_and_relative_paths_refuse(tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    for paths in ([tmp_path, tmp_path], [alias], [tmp_path / "missing"], [Path("relative")]):
        with pytest.raises((ValueError, FileNotFoundError)):
            storage.probe_storage(paths, context="login")
    assert list(tmp_path.iterdir()) == [alias]


def test_compute_identity_requires_job_and_matching_uid(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(storage.StorageProbeError) as error:
        storage.require_storage([tmp_path], context="compute")
    assert "actual SLURM_JOB_ID" in error.value.report["error"]["message"]
    assert "uid_map" in error.value.report["identity"]
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_JOB_UID", str(os.geteuid() + 1))
    with pytest.raises(storage.StorageProbeError) as error:
        storage.require_storage([tmp_path], context="compute")
    assert "UID differs" in error.value.report["error"]["message"]
    monkeypatch.setenv("SLURM_JOB_UID", str(os.geteuid()))
    report = storage.require_storage([tmp_path], context="compute")
    assert report["passed"] and report["allocation_id"] == "12345"
    assert not list(tmp_path.iterdir())


def test_cli_preserves_receipt_and_nonzero_refusal(tmp_path):
    output = tmp_path / "receipt.json"
    environment = {key: value for key, value in os.environ.items() if key != "SLURM_JOB_ID"}
    result = subprocess.run(
        [
            sys.executable,
            str(DIRECTORY / "storage.py"),
            "--context",
            "compute",
            "--path",
            str(tmp_path),
            "--output",
            str(output),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report == json.loads(output.read_text()) and report["passed"] is False
    assert "actual SLURM_JOB_ID" in report["error"]["message"]


@pytest.mark.parametrize("mode", ["restart", "native"])
def test_runner_storage_failure_precedes_build_or_model_load(tmp_path, monkeypatch, capsys, mode):
    run = importlib.import_module("run")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_JOB_UID", str(os.geteuid()))
    monkeypatch.setattr(run.Experiment, "read", lambda _: object())
    monkeypatch.setattr(run, "read_validation_target", lambda *args: None)
    monkeypatch.setattr(run, "verify_build", lambda _: pytest.fail("build reached before failed storage guard"))
    original = Path.mkdir

    def fail(path, *args, **kwargs):
        if path.name == "work":
            raise OSError(errno.EDQUOT, "quota exhausted")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail)
    output = tmp_path / "experiment"
    args = SimpleNamespace(
        action="train-eval",
        checkpoint=None,
        spec="unused",
        output=str(output),
        build="unread",
        evaluation_mode=mode,
        validation_target="target.json",
        split="development",
    )
    with pytest.raises(storage.StorageProbeError):
        run.execute(args)
    assert not list(output.iterdir())
    assert '"errno": 122' in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["restart", "native"])
def test_runner_without_slurm_preserves_local_process_usage(tmp_path, monkeypatch, mode):
    run = importlib.import_module("run")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(run.Experiment, "read", lambda _: object())
    monkeypatch.setattr(run, "read_validation_target", lambda *args: None)
    monkeypatch.setattr(run, "verify_build", lambda _: (_ for _ in ()).throw(ValueError("reached build")))
    output = tmp_path / "experiment"
    args = SimpleNamespace(
        action="train-eval",
        checkpoint=None,
        spec="unused",
        output=str(output),
        build="unread",
        evaluation_mode=mode,
        validation_target="target.json",
        split="development",
    )
    with pytest.raises(ValueError, match="reached build"):
        run.execute(args)
    report = json.loads((output / "storage-probe.json").read_text())
    assert report["passed"] and report["context"] == "process" and report["allocation_id"] is None
    assert report["uid"] == os.getuid()


def test_identity_read_failure_after_creation_does_not_claim_cleanup(tmp_path, monkeypatch):
    original = storage._identity

    def fail(path):
        if path.name == "new-parent":
            raise OSError(errno.EIO, "metadata read failed")
        return original(path)

    monkeypatch.setattr(storage, "_identity", fail)
    report = storage.probe_storage([tmp_path], context="process")
    assert not report["passed"]
    row = report["paths"][0]
    assert not row["cleanup_complete"] and row["cleanup_errors"]
    assert (Path(row["probe"]) / "new-parent").exists()


def identity_fixture(raw="0 41852 1\n", uid=0, host="41852"):
    identity = storage.process_identity()
    identity.update(uid=uid, effective_uid=uid, slurm_job_id="12345", slurm_job_uid=host)
    identity["uid_map"] = {"path": "/proc/self/uid_map", "raw": raw, "sha256": hashlib.sha256(raw.encode()).hexdigest()}
    return identity


@pytest.mark.parametrize(
    "raw,uid,host,method",
    [
        ("0 0 4294967295\n", 41852, "41852", "direct"),
        ("0 41852 1\n", 0, "41852", "kernel_uid_map"),
        ("0 1000 10\n20 41850 5\n", 22, "41852", "kernel_uid_map"),
    ],
)
def test_direct_and_namespace_identity_proofs(raw, uid, host, method):
    proof = storage.compute_identity(identity_fixture(raw, uid, host), expected_host_uid=41852)
    assert proof["host_uid"] == proof["slurm_job_uid"] == 41852 and proof["method"] == method


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "\n",
        "0 41852",
        "0 41852 1 extra",
        "-1 41852 1",
        "0 -1 1",
        "0 41852 -1",
        "0 41852 0",
        "0 41852 4294967295",
        "4294967295 0 1",
        "0 4294967295 1",
        "0 41852 2\n1 50000 1",
        "0 41852 2\n5 41853 1",
        "0 41852 1\n0 41852 1",
        "1 41852 1",
        "0 40000 1",
        "0 41852 1\n\n",
        "0 41852 1\n" + " " * 8192,
    ],
)
def test_malformed_ambiguous_overflow_unmapped_and_wrong_uid_refuse(raw):
    with pytest.raises(ValueError):
        storage.compute_identity(identity_fixture(raw), expected_host_uid=41852)


@pytest.mark.parametrize("host", [None, "", "root", "-1", "4294967295", "41853"])
def test_missing_invalid_or_unexpected_host_uid_refuses(host):
    with pytest.raises(ValueError):
        storage.compute_identity(identity_fixture(host=host), expected_host_uid=41852)


def test_compute_mapping_refusal_retains_raw_ids_maps_before_writes(tmp_path, monkeypatch):
    identity = identity_fixture("0 40000 1\n")
    monkeypatch.setattr(storage, "process_identity", lambda: identity)
    report = storage.probe_storage([tmp_path], context="compute", expected_host_uid=41852)
    assert not report["passed"] and not list(tmp_path.iterdir())
    assert report["identity"] == identity and report["paths"] == []
    assert report["identity"]["effective_uid"] == 0 and report["identity"]["slurm_job_uid"] == "41852"
    assert report["identity"]["uid_map"]["raw"] == "0 40000 1\n"


def test_namespace_fixture_allows_real_disposable_storage_operations(tmp_path, monkeypatch):
    identity = identity_fixture()
    monkeypatch.setattr(storage, "process_identity", lambda: identity)
    report = storage.probe_storage([tmp_path], context="compute", expected_host_uid=41852)
    assert report["passed"] and report["identity_proof"]["method"] == "kernel_uid_map"
    assert report["paths"][0]["cleanup_complete"] and not list(tmp_path.iterdir())


def test_unavailable_map_refuses_and_preserves_read_error(tmp_path, monkeypatch):
    identity = identity_fixture()
    identity["uid_map"]["error"] = {"errno": errno.EACCES, "message": "unavailable"}
    monkeypatch.setattr(storage, "process_identity", lambda: identity)
    report = storage.probe_storage([tmp_path], context="compute", expected_host_uid=41852)
    assert not report["passed"] and report["identity"]["uid_map"]["error"]["errno"] == errno.EACCES
    assert not list(tmp_path.iterdir())
