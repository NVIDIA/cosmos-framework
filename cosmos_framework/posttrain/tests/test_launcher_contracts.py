# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Dependency-free launcher run-reservation and Slurm contracts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from cosmos_framework.posttrain.launcher.run import RunReservation
from cosmos_framework.posttrain.launcher.slurm import (
    job_activity,
    job_state,
    normalize_state,
    parse_accounting,
    parse_job_id,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_run_reservation_is_exclusive_and_release_is_identity_guarded(tmp_path: Path) -> None:
    reservation = RunReservation(str(tmp_path), "run", 300)
    marker = Path(reservation.reserve())
    assert marker.read_text() == "300\n"
    with pytest.raises(FileExistsError, match="already reserved or launched"):
        reservation.reserve()
    with pytest.raises(FileExistsError):
        RunReservation(str(tmp_path), "run", 200).reserve()
    with pytest.raises(RuntimeError, match="refusing to release"):
        RunReservation(str(tmp_path), "run", 200).release()
    reservation.release()
    assert not marker.exists()


def test_run_reservation_recovers_an_accepted_dependency_without_reopening_a_launched_run(tmp_path: Path) -> None:
    reservation = RunReservation(str(tmp_path), "run", 300)
    artifact = str(tmp_path / "hf")
    assert reservation.begin(artifact) is None
    reservation.record_dependency("123", artifact)
    with pytest.raises(RuntimeError, match="accepted scheduler state"):
        reservation.release()
    dependency = reservation.begin(artifact)
    assert dependency is not None
    assert (dependency.phase, dependency.job_id, dependency.artifact) == ("dependency", "123", artifact)
    with pytest.raises(RuntimeError, match="dependency artifact"):
        reservation.begin(str(tmp_path / "other"))
    reservation.prepare_submission(artifact)
    submitting = reservation.read_state()
    assert submitting is not None
    assert (submitting.phase, submitting.job_id, submitting.dependency_job_id) == ("submitting", None, "123")
    with pytest.raises(FileExistsError, match="already reserved or launched"):
        reservation.begin(artifact)
    reservation.abort_submission()
    assert reservation.read_state() is not None and reservation.read_state().phase == "dependency"
    reservation.prepare_submission(artifact)
    with pytest.raises(ValueError, match="job id must be numeric"):
        reservation.record_launched("not-a-job", artifact)
    assert reservation.read_state() is not None and reservation.read_state().phase == "submitting"
    with pytest.raises(FileExistsError, match="already reserved or launched"):
        reservation.begin(artifact)
    reservation.record_launched("456", artifact)
    with pytest.raises(FileExistsError, match="already reserved or launched"):
        reservation.begin(artifact)
    assert reservation.read_state() is not None and reservation.read_state().phase == "launched"

    fresh = RunReservation(str(tmp_path), "fresh", 400)
    fresh_artifact = str(tmp_path / "fresh-hf")
    assert fresh.begin(fresh_artifact) is None
    fresh.prepare_submission(fresh_artifact)
    with pytest.raises(FileExistsError, match="already reserved or launched"):
        fresh.begin(fresh_artifact)
    fresh.abort_submission()
    assert fresh.read_state() is None
    fresh.release()


def test_slurm_parsers_share_normalized_states_and_job_ids() -> None:
    assert normalize_state("RUNNING+\n") == "RUNNING"
    assert normalize_state("CANCELLED by 123") == "CANCELLED"
    assert parse_job_id("12345;cluster\n") == "12345"
    assert parse_job_id("Submitted batch job 98765\n") == "98765"
    assert parse_job_id("submission failed") is None
    assert parse_accounting("7.batch|FAILED|1:0\n7|CANCELLED by 5|0:15\n") == ("CANCELLED", 0)


def test_slurm_job_state_fails_closed_on_unknown_accounting() -> None:
    def completed(returncode: int, stdout: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["sacct"], returncode, stdout, "")

    assert job_state("1", run=lambda *args, **kwargs: completed(0, "PENDING+\n")) == "PENDING"
    assert job_activity("1", run=lambda *args, **kwargs: completed(0, "RUNNING+\n")) == ("ACTIVE", "RUNNING")
    assert job_activity("1", run=lambda *args, **kwargs: completed(0, "FAILED\n")) == ("TERMINAL", "FAILED")
    assert job_state("1", run=lambda *args, **kwargs: completed(0, "")) is None
    assert job_state("1", run=lambda *args, **kwargs: completed(1, "")) is None


def test_operational_contracts_import_without_torch() -> None:
    code = """
import sys
sys.modules['torch'] = None
from cosmos_framework.posttrain.checkpoint import ContinuationPlan
from cosmos_framework.posttrain.checkpoint.conversion.hf import hf_export_complete
from cosmos_framework.posttrain.evaluation.results import align_coverage
from cosmos_framework.posttrain.launcher.run import RunReservation
assert ContinuationPlan and hf_export_complete and align_coverage and RunReservation
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
