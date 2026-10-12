# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Batch-safe prompt objectives; use native answer parsing, never drop invalid outputs."""

from cosmos_framework.evaluation.reasoner.metrics.binary_classification import extract_binary_response
from cosmos_framework.evaluation.reasoner.metrics.task_accuracy import score_accuracy


def aggregate(outputs, golds):
    if not outputs or len(outputs) != len(golds):
        raise ValueError("Prompt metrics need one output per labeled example")
    tasks = {gold["task"] for gold in golds}
    if tasks == {"mcq"}:
        return score_accuracy(outputs, [gold["references"] for gold in golds], ["mcq"] * len(golds))["overall"]
    if not tasks <= {"bcq", "binary"}:
        raise ValueError("Prompt metrics require MCQ or binary tasks")
    predicted = [extract_binary_response(output) for output in outputs]
    expected = [extract_binary_response(gold["references"][0]) for gold in golds]
    if any(label is None for label in expected):
        raise ValueError("Binary reference labels must parse as yes/no or A/B")
    f1 = []
    for label in ("yes", "no"):
        tp = sum(p == label and g == label for p, g in zip(predicted, expected))
        fp = sum(p == label and g != label for p, g in zip(predicted, expected))
        fn = sum(p != label and g == label for p, g in zip(predicted, expected))
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    correct = sum(p == g for p, g in zip(predicted, expected))
    return {
        "accuracy": correct / len(outputs),
        "correct": correct,
        "total": len(outputs),
        "macro_f1": sum(f1) / 2,
        "unparseable": sum(p is None for p in predicted),
    }
