# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Evaluation result and accepted-attempt integrity contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cosmos_framework.posttrain.evaluation.results import (
    CoverageMismatch,
    DuplicateResultId,
    align_coverage,
    unique_jsonl_rows,
)
from cosmos_framework.posttrain.evaluation.state import AttemptJournal, read_accepted_job, write_accepted_job

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_unique_jsonl_rows_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join([json.dumps({"id": "x", "score": 1}), "not json", json.dumps({"id": "x", "score": 0})]))
    with pytest.raises(DuplicateResultId, match="duplicate result id"):
        unique_jsonl_rows(str(path))


def test_align_coverage_refuses_or_explicitly_intersects() -> None:
    rows = {"base": {"a": 1, "b": 1}, "candidate": {"b": 1, "c": 1}}
    with pytest.raises(CoverageMismatch, match="do not cover the same"):
        align_coverage(rows, dataset="d")
    assert align_coverage(rows, dataset="d", intersect=True)[0] == {"b"}


def test_attempt_and_accepted_job_journals_preserve_existing_formats(tmp_path: Path) -> None:
    journal = AttemptJournal(str(tmp_path / "attempts"))
    key = ("wave", "seed")
    assert journal.count(key) == 0
    assert journal.record_accepted(key) == 1
    assert journal.record_accepted(key) == 2
    marker = tmp_path / "job.json"
    write_accepted_job(str(marker), "12345", 2)
    assert read_accepted_job(str(marker)).job_id == "12345"
    marker.write_text("99999")
    assert read_accepted_job(str(marker)).attempts == 1
