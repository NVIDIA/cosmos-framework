# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Partition WTS by related recordings, retaining historical exposure explicitly.

Outputs are local dataset artifacts and may contain private source rows. Nothing
is uploaded. A small untouched partition is a diagnostic, never a release claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.native_qlora_qualification.common import (
    digest_file,
    digest_json,
    inventory_groups,
    read_jsonl,
    write_json_new,
)

PROTOCOL = "cosmos.wts-group-split.v1"


def plan_partitions(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    exposed_ids: set[str],
    *,
    development_groups: int = 12,
    seed: int = 20261030,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, str]]:
    if type(development_groups) is not int or development_groups < 1 or type(seed) is not int:
        raise ValueError("positive development group count and integer seed required")
    all_rows = train + validation
    ids = [row["id"] for row in all_rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate source row ID across train/validation")
    groups = inventory_groups(inventory)
    if set(ids) != set(groups) or not exposed_ids <= set(ids):
        raise ValueError("inventory/exposure IDs do not match the complete source rows")
    inventory_by_id = {item["id"]: item for item in inventory}
    for label, rows in (("train", train), ("validation", validation)):
        if any(inventory_by_id[row["id"]].get("split") != label for row in rows):
            raise ValueError("inventory source split differs")
    train_groups = {groups[row["id"]] for row in train}
    validation_groups = {groups[row["id"]] for row in validation}
    exposed_groups = {groups[identifier] for identifier in exposed_ids}
    available = train_groups - validation_groups - exposed_groups
    if len(available) <= development_groups:
        raise ValueError("insufficient training-disjoint groups for development and remaining training")
    selected = set(
        sorted(available, key=lambda key: (digest_json({"protocol": PROTOCOL, "seed": seed, "group": key}), key))[
            :development_groups
        ]
    )
    partitions = {
        "train": [row for row in train if groups[row["id"]] not in validation_groups | selected | exposed_groups],
        "development": [row for row in train if groups[row["id"]] in selected],
        "regression": [row for row in validation if groups[row["id"]] in exposed_groups],
        "untouched_diagnostic": [row for row in validation if groups[row["id"]] not in exposed_groups | train_groups],
        "excluded": [],
    }
    kept = {row["id"] for name, rows in partitions.items() if name != "excluded" for row in rows}
    partitions["excluded"] = [row for row in all_rows if row["id"] not in kept]
    active = {name: {groups[row["id"]] for row in rows} for name, rows in partitions.items() if name != "excluded"}
    for name, keys in active.items():
        for other, values in active.items():
            if name != other and keys & values:
                raise RuntimeError("connected recording/media group crossed an active partition")
    summary = {
        "protocol": PROTOCOL,
        "seed": seed,
        "grouping": "connected_date_record_families_and_exact_video_sha256",
        "original_train_validation_shared_groups": len(train_groups & validation_groups),
        "original_shared_group_rows": {
            label: sum(groups[row["id"]] in train_groups & validation_groups for row in rows)
            for label, rows in (("train", train), ("validation", validation))
        },
        "partitions": {
            name: {"rows": len(rows), "groups": len({groups[row["id"]] for row in rows})}
            for name, rows in partitions.items()
        },
        "development_historically_training_exposed": True,
        "new_training_must_start_from_original_base": True,
        "exposed_ids": len(exposed_ids),
        "untouched_independent_clusters": len(active["untouched_diagnostic"]),
        "untouched_cluster_uncertainty_estimable": len(active["untouched_diagnostic"]) >= 2,
        "release_qualified": False,
        "limitations": [
            "Development was selected from historically trained groups; it is not an untouched test.",
            "Known exposures are an explicit input ledger; completeness requires reviewing all earlier probes.",
            "Recording-family separation does not prove actor/event independence across different recordings.",
            "An untouched diagnostic is not a new final release set; one cluster cannot estimate cluster uncertainty.",
            "The previously scored 256-row subset remains a separate frozen regression selection.",
        ],
    }
    return partitions, summary, groups


def verify_conversion(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads((root / "manifest.json").read_text())
    for label in ("train", "validation"):
        expected = manifest["splits"][label]
        path = root / (label + ".jsonl")
        if digest_file(path) != expected["output"]["sha256"]:
            raise ValueError("converted source bytes differ from their manifest")
        if len(read_jsonl(path)) != expected["selected_rows"]:
            raise ValueError("converted row count differs from manifest")
    if digest_file(root / "media_inventory.jsonl") != manifest["media_inventory"]["sha256"]:
        raise ValueError("media inventory differs from conversion manifest")
    return tuple(read_jsonl(root / name) for name in ("train.jsonl", "validation.jsonl", "media_inventory.jsonl"))


def write_partition(
    output: Path,
    name: str,
    rows: list[dict[str, Any]],
    groups: dict[str, str],
    media: dict[str, str],
    *,
    provenance: dict[str, Any],
    role: str | None = None,
) -> None:
    if not rows:
        return
    path = output / (name + ".jsonl")
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    manifest = {
        "protocol": PROTOCOL,
        "role": role or name,
        "data_sha256": digest_file(path),
        "rows": [
            {
                "id": row["id"],
                "raw_row_sha256": digest_json(row),
                "group_sha256": groups[row["id"]],
                "media_sha256": media[row["id"]],
            }
            for row in rows
        ],
        "group_count": len({groups[row["id"]] for row in rows}),
        "provenance": provenance,
        "release_qualified": False,
    }
    write_json_new(output / (name + ".manifest.json"), manifest)


def select_sealed_rows(
    source: Path, manifest_path: Path, manifest_sha256: str
) -> tuple[list[tuple[int, dict[str, Any]]], dict[str, Any]]:
    if digest_file(manifest_path) != manifest_sha256:
        raise ValueError("split manifest SHA256 differs from the precommitted identity")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol") != PROTOCOL or manifest.get("role") not in {
        "development",
        "regression",
        "untouched_diagnostic",
        "final",
    }:
        raise ValueError("unsupported sealed scoring split")
    if digest_file(source) != manifest["data_sha256"]:
        raise ValueError("sealed scoring data bytes differ")
    rows = read_jsonl(source)
    if len(rows) != len(manifest["rows"]) or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("sealed row count or unique IDs differ")
    for row, expected in zip(rows, manifest["rows"], strict=True):
        if row["id"] != expected["id"] or digest_json(row) != expected["raw_row_sha256"]:
            raise ValueError("sealed ordered row identity differs")
    groups = {row["group_sha256"] for row in manifest["rows"]}
    if len(groups) != manifest["group_count"]:
        raise ValueError("sealed group count differs")
    if manifest["role"] == "final" and (
        len(groups) < 2
        or manifest.get("provenance", {}).get("historical_exposure") != "none"
        or manifest.get("provenance", {}).get("training_development_regression_group_disjoint") is not True
    ):
        raise ValueError(
            "final split requires untouched/disjoint provenance and at least two clusters; this does not establish adequate release power"
        )
    return list(enumerate(rows, 1)), {
        "protocol": PROTOCOL,
        "source": str(source),
        "source_sha256": manifest["data_sha256"],
        "source_rows": len(rows),
        "selected_rows": len(rows),
        "selection_rule": "all_rows_from_precommitted_group_split_in_source_order",
        "split_manifest_sha256": manifest_sha256,
        "split_role": manifest["role"],
        "group_count": manifest["group_count"],
        "sealed_rows": manifest["rows"],
        "release_qualified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--exposed-scores", type=Path, action="append", default=[])
    parser.add_argument("--regression-scores", type=Path, help="also preserve this exact historical scored subset")
    parser.add_argument(
        "--exposed-ids", type=Path, action="append", default=[], help="JSON list of additional probe row IDs"
    )
    parser.add_argument("--exposed-validation-prefix", type=int, default=64)
    parser.add_argument("--development-groups", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train, validation, inventory = verify_conversion(args.data_root)
    if not 0 <= args.exposed_validation_prefix <= len(validation):
        raise ValueError("invalid exposed validation prefix")
    exposures = {row["id"] for row in validation[: args.exposed_validation_prefix]}
    for path in args.exposed_scores:
        exposures.update(row["id"] for row in read_jsonl(path))
    if args.regression_scores:
        exposures.update(row["id"] for row in read_jsonl(args.regression_scores))
    for path in args.exposed_ids:
        ids = json.loads(path.read_text())
        if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
            raise ValueError("extra exposures must be a JSON list of IDs")
        exposures.update(ids)
    partitions, summary, groups = plan_partitions(
        train, validation, inventory, exposures, development_groups=args.development_groups, seed=args.seed
    )
    provenance = {
        "conversion_manifest_sha256": digest_file(args.data_root / "manifest.json"),
        "exposure_file_sha256": [digest_file(path) for path in args.exposed_scores + args.exposed_ids],
        "regression_scores_sha256": digest_file(args.regression_scores) if args.regression_scores else None,
        "exposed_validation_prefix": args.exposed_validation_prefix,
        "partition_summary": summary,
    }
    selected = None
    if args.regression_scores:
        original = {row["id"]: row for row in validation}
        scores = read_jsonl(args.regression_scores)
        if len({row["id"] for row in scores}) != len(scores):
            raise ValueError("historical regression scores contain duplicate IDs")
        selected = []
        for score in scores:
            row = original.get(score["id"])
            if row is None or digest_json(row) != score["raw_row_sha256"]:
                raise ValueError("historical regression row differs from original validation")
            selected.append(row)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in partitions.items():
        write_partition(
            args.output, name, rows, groups, {row["id"]: row["sha256"] for row in inventory}, provenance=provenance
        )
    if selected is not None:
        write_partition(
            args.output,
            "regression_selection",
            selected,
            groups,
            {row["id"]: row["sha256"] for row in inventory},
            provenance=provenance,
            role="regression",
        )
    write_json_new(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
