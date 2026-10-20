# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Binary-response parsing and metrics shared by rank-local and merged evaluation."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence, Tuple

BINARY_TOKEN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
LEADING_BINARY_TOKEN = re.compile(r"^\s*[\W_]*(yes|no)\b", re.IGNORECASE)
ANSWER_BINARY_TOKEN = re.compile(
    r"\b(?:final\s+)?answer\s*(?:is|:)?\s*[\W_]*(yes|no)\b",
    re.IGNORECASE,
)


def extract_binary_response(value: str) -> Optional[str]:
    """Extract one unambiguous canonical yes/no label from free-form text."""
    text = str(value)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.strip()

    alias = text.lower().strip(" \t\r\n.,;:!?()[]{}\"'")
    if alias == "a":
        return "yes"
    if alias == "b":
        return "no"
    if re.fullmatch(
        r"\s*[\W_]*(?:yes\s+or\s+no|no\s+or\s+yes)[\W_]*",
        text,
        re.IGNORECASE,
    ):
        return None

    tokens = [match.group(1).lower() for match in BINARY_TOKEN.finditer(text)]
    if not tokens:
        return None
    if len(set(tokens)) == 1:
        return tokens[0]

    leading = LEADING_BINARY_TOKEN.search(text)
    if leading:
        return leading.group(1).lower()

    answer_matches = list(ANSWER_BINARY_TOKEN.finditer(text))
    if answer_matches:
        answer_labels = {match.group(1).lower() for match in answer_matches}
        if len(answer_labels) == 1:
            return answer_matches[-1].group(1).lower()
    return None


def compute_binary_classification_metrics(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> Dict[str, Any]:
    """Compute binary metrics without dropping unparseable predictions."""
    if len(predictions) != len(references):
        raise ValueError(
            "binary metrics require one ground truth for every prediction: "
            f"predictions={len(predictions)}, references={len(references)}"
        )
    if not predictions:
        raise ValueError("binary metrics require at least one evaluated example")

    tp = fp = tn = fn = 0
    positive_samples = negative_samples = 0
    unparseable_predictions = 0
    unparseable_positive_predictions = 0
    unparseable_negative_predictions = 0
    for prediction, refs in zip(predictions, references):
        parsed_prediction = extract_binary_response(prediction)
        ground_truth_raw = refs[0] if refs else ""
        ground_truth = extract_binary_response(str(ground_truth_raw))
        if ground_truth is None:
            raise ValueError(f"binary ground truth does not contain one clear yes/no label: {ground_truth_raw!r}")

        if ground_truth == "yes":
            positive_samples += 1
        else:
            negative_samples += 1

        if parsed_prediction is None:
            unparseable_predictions += 1
            if ground_truth == "yes":
                unparseable_positive_predictions += 1
            else:
                unparseable_negative_predictions += 1
            continue

        if parsed_prediction == "yes" and ground_truth == "yes":
            tp += 1
        elif parsed_prediction == "no" and ground_truth == "no":
            tn += 1
        elif parsed_prediction == "yes":
            fp += 1
        else:
            fn += 1

    if positive_samples == 0 or negative_samples == 0:
        raise ValueError(
            f"balanced accuracy requires both ground-truth classes; found yes={positive_samples}, no={negative_samples}"
        )

    def percentage(num: float, den: float) -> float:
        return num / den * 100 if den else 0.0

    precision = percentage(tp, tp + fp)
    positive_recall = percentage(tp, positive_samples)
    negative_recall = percentage(tn, negative_samples)
    f1 = 2 * precision * positive_recall / (precision + positive_recall) if precision + positive_recall else 0.0
    total_samples = len(predictions)
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "total_samples": total_samples,
        "parsed_predictions": total_samples - unparseable_predictions,
        "unparseable_predictions": unparseable_predictions,
        "unparseable_positive_predictions": unparseable_positive_predictions,
        "unparseable_negative_predictions": unparseable_negative_predictions,
        "positive_ground_truth": positive_samples,
        "negative_ground_truth": negative_samples,
        "precision": round(precision, 2),
        "recall": round(positive_recall, 2),
        "f1_score": round(f1, 2),
        "positive_accuracy": round(positive_recall, 2),
        "negative_accuracy": round(negative_recall, 2),
        "balanced_accuracy": round((positive_recall + negative_recall) / 2, 2),
        "accuracy": round(percentage(tp + tn, total_samples), 2),
    }


def required_saved_field(result: Dict[str, Any], fields: Tuple[str, ...], label: str) -> str:
    """Return the first non-empty supported result field."""
    for field in fields:
        value = result.get(field)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"saved binary result is missing a non-empty {label} (supported fields: {', '.join(fields)})")


def compute_saved_binary_metrics(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute binary metrics after deterministic multi-rank result merging."""
    predictions = [
        str(result.get("full_response"))
        if isinstance(result.get("full_response"), str) and result["full_response"].strip()
        else required_saved_field(result, ("response", "answer", "prediction"), "prediction")
        for result in results
    ]
    references = [[required_saved_field(result, ("gt", "ground_truth"), "ground truth")] for result in results]
    return compute_binary_classification_metrics(predictions, references)
