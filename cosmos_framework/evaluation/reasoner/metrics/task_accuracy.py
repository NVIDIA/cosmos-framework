# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Deterministic task-aware accuracy shared by RL and Framework exports."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Sequence

ACCURACY_TASKS = {"bcq", "mcq", "binary"}
EVALUATOR_VERSION = "cosmos-shared-v2"


def normalize_task(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def normalize_text(text: str, *, preserve_articles: bool = False) -> str:
    value = str(text).lower()
    value = "".join(
        character if not unicodedata.category(character).startswith(("P", "S")) else " " for character in value
    )
    if not preserve_articles:
        value = re.sub(r"\b(a|an|the)\b", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def extract_answer(prediction: str, task: str) -> str:
    value = str(prediction)
    if "</think>" in value:
        value = value.split("</think>", 1)[1]
    task = normalize_task(task)
    if task == "mcq":
        match = re.search(r"\b([A-D])\b", value.strip(), re.IGNORECASE)
        return match.group(1).upper() if match else value.strip().upper()
    if task in {"bcq", "binary"}:
        tokens = re.split(r"[,\s]+", value.strip().lower())
        return tokens[0] if tokens else ""
    return value


def score_accuracy(
    predictions: Sequence[str], references: Sequence[Sequence[str]], tasks: Sequence[str]
) -> dict[str, Any]:
    if not (len(predictions) == len(references) == len(tasks)):
        raise ValueError("prediction, reference, and task lengths differ")
    per_task: dict[str, dict[str, Any]] = {}
    correct = 0
    total = 0
    for prediction, refs, raw_task in zip(predictions, references, tasks):
        task = normalize_task(raw_task)
        bucket = per_task.setdefault(task or "unspecified", {"correct": 0, "total": 0, "accuracy": None})
        if task not in ACCURACY_TASKS:
            continue
        extracted = extract_answer(prediction, task)
        preserve = task == "mcq"
        matched = any(
            normalize_text(extracted, preserve_articles=preserve)
            == normalize_text(
                extract_answer(reference, task),
                preserve_articles=preserve,
            )
            for reference in refs
        )
        bucket["correct"] += int(matched)
        bucket["total"] += 1
        correct += int(matched)
        total += 1
    for values in per_task.values():
        if values["total"]:
            values["accuracy"] = values["correct"] / values["total"]
    included = sorted(task for task, values in per_task.items() if values["total"])
    excluded = sorted(task for task, values in per_task.items() if not values["total"])
    return {
        "overall": {"accuracy": correct / total if total else None, "correct": correct, "total": total},
        "per_task": per_task,
        "aggregation": "example_weighted_over_accuracy_defined_tasks",
        "coverage": {"included_tasks": included, "excluded_tasks": excluded},
        "excluded_tasks": excluded,
        "evaluator_version": EVALUATOR_VERSION,
    }


def score_saved_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Score persisted evaluator rows with the same task-aware extraction."""
    predictions = [str(result.get("full_response", result.get("response", ""))) for result in results]
    references = [
        [str(value) for value in result.get("gt", [])]
        if isinstance(result.get("gt"), list)
        else [str(result.get("gt", ""))]
        for result in results
    ]
    tasks = [str(result.get("task") or "unspecified") for result in results]
    return score_accuracy(predictions, references, tasks)
