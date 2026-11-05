# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Slurm-native duration forms accepted by the shared submission preflight."""

import subprocess

import pytest

from cosmos_framework.posttrain.launcher.slurm import parse_duration
from cosmos_framework.posttrain.utils.launch_multinode import check_partition_time

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize(
    ("duration", "seconds"),
    [
        ("60", 3600),
        ("1:30", 90),
        ("1:30:00", 5400),
        ("1-2", 93600),
        ("1-2:30", 95400),
        ("1-2:30:00", 95400),
        ("1:90:00", 9000),
        ("0", None),
        ("00:00", None),
        ("UNLIMITED", None),
        ("infinite", None),
    ],
)
def test_duration_preserves_slurm_units(duration: str, seconds: int | None) -> None:
    assert parse_duration(duration) == seconds


@pytest.mark.parametrize("duration", ["", "-1", "1:2:3:4", "1--2", "1-", "1.5", "1:bad", " 30"])
def test_invalid_duration_is_rejected(duration: str) -> None:
    with pytest.raises(ValueError, match="invalid Slurm duration"):
        parse_duration(duration)


def test_preflight_compares_all_duration_forms_and_unlimited(monkeypatch: pytest.MonkeyPatch) -> None:
    limit = "04:00:00"

    def query(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        assert command == ["scontrol", "show", "partition", "batch", "--oneliner"]
        return subprocess.CompletedProcess(command, 0, f"PartitionName=batch MaxTime={limit}")

    monkeypatch.setattr(subprocess, "run", query)
    for duration in ("30", "04:00", "240", "240:00", "4:00:00", "0-4", "0-4:00", "0-4:00:00"):
        assert check_partition_time("batch", duration) == limit
    for duration in ("240:01", "04:00:01", "0", "UNLIMITED"):
        with pytest.raises(ValueError, match="exceeds"):
            check_partition_time("batch", duration)
    limit = "UNLIMITED"
    assert check_partition_time("batch", "0") == limit
    with pytest.raises(ValueError, match="invalid Slurm duration"):
        check_partition_time("batch", "1-foo")
