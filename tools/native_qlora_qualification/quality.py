# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Paired WTS quality diagnostics with recording-cluster uncertainty.

This analyzes existing scores; it does not train, tune thresholds, or infer a
release result from one seed. Slice and influence results are exploratory.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.native_qlora_qualification.common import (
    digest_file,
    digest_json,
    inventory_groups,
    read_jsonl,
    write_json_new,
)

INPUT_KEYS = (
    "id",
    "source_line",
    "raw_row_sha256",
    "row_identity",
    "gold",
    "choices",
    "media_sha256",
    "input_sha256",
    "inputs",
    "candidate_token_ids",
    "answer_token_position",
    "prediction_logit_position",
    "native_supervised_tokens",
    "sequence_length",
)


def validate_scores(rows: list[dict[str, Any]]) -> None:
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("scores require nonempty unique row IDs")
    for row in rows:
        choices = row["choices"]
        if choices not in (["A", "B"], ["A", "B", "C"], ["A", "B", "C", "D"]):
            raise ValueError("invalid actual question choices")
        if (
            row["gold"] not in choices
            or set(row["candidate_logits"]) != set(choices)
            or set(row["candidate_token_ids"]) != set(choices)
        ):
            raise ValueError("gold/logit/token inventory differs from actual choices")
        if len(set(row["candidate_token_ids"].values())) != len(choices):
            raise ValueError("candidate token IDs must be distinct")
        if (
            row["input_sha256"] != digest_json(row["inputs"])
            or row["prediction_logit_position"] != row["answer_token_position"] - 1
        ):
            raise ValueError("processed input identity or causal shift differs")
        if type(row["native_supervised_tokens"]) is not int or row["native_supervised_tokens"] < 1:
            raise ValueError("invalid supervised token count")
        for key in ("native_sft_ce", "answer_nll", "restricted_choice_nll"):
            if (
                isinstance(row[key], bool)
                or not isinstance(row[key], (int, float))
                or not math.isfinite(row[key])
                or row[key] < 0
            ):
                raise ValueError("nonfinite or negative quality metric")
        if any(
            not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
            for v in row["candidate_logits"].values()
        ):
            raise ValueError("nonfinite candidate logit")
        expected = max(choices, key=lambda key: row["candidate_logits"][key])
        if expected != row["restricted_choice_prediction"] or row["restricted_choice_correct"] is not (
            expected == row["gold"]
        ):
            raise ValueError("restricted prediction/correctness differs from logits")
        if row["strict_first_token_correct"] is not (
            row["strict_first_token_id"] == row["candidate_token_ids"][row["gold"]]
        ):
            raise ValueError("strict correctness differs")
        if row["valid_first_letter"] is not (row["strict_first_token_id"] in row["candidate_token_ids"].values()):
            raise ValueError("valid-letter flag differs")


def metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot summarize empty scores")
    tokens = sum(row["native_supervised_tokens"] for row in rows)
    return {
        "rows": len(rows),
        "supervised_tokens": tokens,
        "restricted_correct": sum(row["restricted_choice_correct"] for row in rows),
        "restricted_accuracy": sum(row["restricted_choice_correct"] for row in rows) / len(rows),
        "strict_accuracy": sum(row["strict_first_token_correct"] for row in rows) / len(rows),
        "valid_letter_rate": sum(row["valid_first_letter"] for row in rows) / len(rows),
        "token_weighted_ce": math.fsum(row["native_sft_ce"] * row["native_supervised_tokens"] for row in rows) / tokens,
        "answer_nll": math.fsum(row["answer_nll"] for row in rows) / len(rows),
        "restricted_choice_nll": math.fsum(row["restricted_choice_nll"] for row in rows) / len(rows),
    }


def paired_metrics(candidate: list[dict[str, Any]], reference: list[dict[str, Any]]) -> dict[str, Any]:
    a, b = metrics(candidate), metrics(reference)
    if b["token_weighted_ce"] <= 0:
        raise ValueError("relative NLL requires positive reference CE")
    return {
        "candidate": a,
        "reference": b,
        "relative_token_nll_increase": a["token_weighted_ce"] / b["token_weighted_ce"] - 1,
        "accuracy_difference": a["restricted_accuracy"] - b["restricted_accuracy"],
    }


def cluster_bootstrap(candidate, reference, groups, *, repetitions=10000, seed=2026):
    if type(repetitions) is not int or repetitions < 1 or type(seed) is not int:
        raise ValueError("positive bootstrap repetitions and integer seed required")
    by_group = defaultdict(list)
    for a, b in zip(candidate, reference, strict=True):
        by_group[groups[a["id"]]].append((a, b))
    if len(by_group) < 2:
        return {"clusters": len(by_group), "estimable": False, "reason": "fewer than two independent group labels"}
    totals = []
    for key in sorted(by_group):
        pairs = by_group[key]
        totals.append(
            (
                len(pairs),
                sum(a["native_supervised_tokens"] for a, _ in pairs),
                math.fsum(a["native_sft_ce"] * a["native_supervised_tokens"] for a, _ in pairs),
                math.fsum(b["native_sft_ce"] * b["native_supervised_tokens"] for _, b in pairs),
                sum(int(a["restricted_choice_correct"]) - int(b["restricted_choice_correct"]) for a, b in pairs),
            )
        )
    rng = random.Random(seed)
    ratios, accuracy = [], []
    for _ in range(repetitions):
        sample = rng.choices(totals, k=len(totals))
        rows, tokens, a, b, correct = (math.fsum(values) for values in zip(*sample, strict=True))
        if b <= 0:
            raise ValueError("bootstrap reference NLL is zero; relative uncertainty is undefined")
        ratios.append(a / b - 1)
        accuracy.append(correct / rows)
    ratios.sort()
    accuracy.sort()

    def percentile(values, probability):
        return values[math.ceil(len(values) * probability) - 1]

    return {
        "clusters": len(totals),
        "estimable": True,
        "repetitions": repetitions,
        "seed": seed,
        "method": "paired connected-recording-cluster percentile bootstrap; whole clusters retained; token-weighted ratio",
        "relative_nll_interval_95": [percentile(ratios, 0.025), percentile(ratios, 0.975)],
        "relative_nll_upper_one_sided_95": percentile(ratios, 0.95),
        "accuracy_difference_interval_95": [percentile(accuracy, 0.025), percentile(accuracy, 0.975)],
        "scope": "One-seed exploratory diagnostic on previously exposed rows; not the multi-seed release bound.",
    }


def analyze(arms: dict[str, list[dict[str, Any]]], inventory, *, repetitions=10000, seed=2026):
    expected = {"frozen_nf4", "trained_nf4", "frozen_bf16", "trained_bf16"}
    if set(arms) != expected:
        raise ValueError("all four explicit frozen/trained precision arms are required")
    for rows in arms.values():
        validate_scores(rows)
    anchor = arms["trained_nf4"]
    for rows in arms.values():
        if len(rows) != len(anchor) or any(
            any(a[key] != b[key] for key in INPUT_KEYS) for a, b in zip(anchor, rows, strict=True)
        ):
            raise ValueError("four arms differ in ordered inputs, masks, media, candidate tokens or supervised counts")
    groups = inventory_groups(inventory)
    inventory_by_id = {item["id"]: item for item in inventory}
    if any(row["id"] not in groups or inventory_by_id[row["id"]]["sha256"] != row["media_sha256"] for row in anchor):
        raise ValueError("scored media/rows do not match grouping inventory")
    a, b = anchor, arms["trained_bf16"]
    slices = {}
    for name, key in (("choice_count", lambda row: str(len(row["choices"]))), ("gold_letter", lambda row: row["gold"])):
        slices[name] = {}
        for value in sorted({key(row) for row in a}):
            indices = [i for i, row in enumerate(a) if key(row) == value]
            slices[name][value] = paired_metrics([a[i] for i in indices], [b[i] for i in indices])
    differences = sorted(
        (
            {
                "row_identity": row["row_identity"],
                "group_sha256": groups[row["id"]],
                "candidate_minus_reference_ce": row["native_sft_ce"] - ref["native_sft_ce"],
                "candidate_correct": row["restricted_choice_correct"],
                "reference_correct": ref["restricted_choice_correct"],
            }
            for row, ref in zip(a, b, strict=True)
        ),
        key=lambda row: (-row["candidate_minus_reference_ce"], row["row_identity"]),
    )
    discordance = Counter(
        (row["restricted_choice_correct"], ref["restricted_choice_correct"]) for row, ref in zip(a, b, strict=True)
    )
    return {
        "protocol": "cosmos.wts-quality-diagnostic.v1",
        "arms": {name: metrics(rows) for name, rows in arms.items()},
        "trained_paired": paired_metrics(a, b),
        "discordance": {
            f"candidate_{str(x).lower()}_reference_{str(y).lower()}": discordance[x, y]
            for x in (False, True)
            for y in (False, True)
        },
        "slices_exploratory": slices,
        "per_row_influence_exploratory": differences,
        "recording_cluster_uncertainty": cluster_bootstrap(a, b, groups, repetitions=repetitions, seed=seed),
        "exact_video_clusters": len({row["media_sha256"] for row in a}),
        "grouping": "connected_date_record_families_and_exact_video_sha256",
        "release_qualified": False,
        "limitations": [
            "Previously exposed regression subset; no outlier trimming or selection from these slices is a release test.",
            "Related cameras/trials are joined; cross-recording actor/event dependence may remain.",
            "Formal 2% NLL margin, training seeds and release scope remain unchanged.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores", action="append", required=True, help="frozen_nf4=PATH, etc.; all four arms required"
    )
    parser.add_argument("--media-inventory", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {}
    for item in args.scores:
        name, path = item.split("=", 1)
        if name in paths:
            raise ValueError("duplicate score arm")
        paths[name] = Path(path)
    result = analyze(
        {name: read_jsonl(path) for name, path in paths.items()},
        read_jsonl(args.media_inventory),
        repetitions=args.repetitions,
        seed=args.seed,
    )
    result["source_sha256"] = {name: digest_file(path) for name, path in paths.items()} | {
        "media_inventory": digest_file(args.media_inventory)
    }
    write_json_new(args.output, result)


if __name__ == "__main__":
    main()
