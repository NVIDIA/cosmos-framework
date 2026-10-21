# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Evaluation result identity and comparable-coverage contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


class DuplicateResultId(ValueError):
    """One result file contains the same sample identity more than once."""


class CoverageMismatch(ValueError):
    """Result sets being compared do not cover the same sample identities."""


def unique_jsonl_rows(path: str, id_field: str = "id") -> list[tuple[str, dict[str, Any]]]:
    """Read valid identified JSON objects, rejecting duplicate identities.

    Invalid JSON and rows without an identity remain ignored for compatibility with existing evaluation readers.
    """
    rows: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                continue
            if not isinstance(row, dict) or row.get(id_field) is None:
                continue
            sample_id = str(row[id_field])
            if sample_id in seen:
                raise DuplicateResultId(f"{path}:{line_number}: duplicate result id {sample_id!r}")
            seen.add(sample_id)
            rows.append((sample_id, row))
    return rows


def align_coverage(
    rows_by_source: Mapping[str, Mapping[str, object]],
    *,
    dataset: str,
    intersect: bool = False,
    item_label: str = "sample ids",
) -> tuple[set[str], list[str]]:
    """Return comparable IDs and coverage notes, or reject unequal sets."""
    if not rows_by_source:
        return set(), []
    sets = {source: set(rows) for source, rows in rows_by_source.items()}
    common = set.intersection(*sets.values())
    if all(ids == common for ids in sets.values()):
        return common, []
    detail = "; ".join(
        f"{source}: {len(ids)} ids, {len(ids - common)} not shared (e.g. {', '.join(sorted(ids - common)[:3]) or '-'})"
        for source, ids in sets.items()
    )
    if not intersect:
        raise CoverageMismatch(
            f"{dataset}: the result files do not cover the same {item_label} ({detail}); "
            f"{len(common)} ids are common to all -- pass --intersect to compare on those, or complete the files"
        )
    return common, [f"{dataset}: compared on the {len(common)} ids common to every file ({detail})"]
