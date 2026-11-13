# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Normalize independently reviewed, preserved text-reference executions.

This is an evidence adapter, not an update oracle. The admitted source review
owns the tensor comparison implementation. Raw receipts and the independent
review must agree; a summary PASS alone cannot produce a gate. Failed early
comparisons remain FAIL and do not claim that later comparisons executed.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

from .evidence import Store, digest, producer_reviews, read_json, require, same
from .primary import write


def load_review(store, reference):
    review = store.json(reference)
    require(review.get("format") == "selected-text-terminal-qualification-review-v1", "unsupported reference review")
    require(
        review.get("status") == "REVIEW_COMPLETE_NINE_PASS_FOUR_FAIL_NO_CAMPAIGN_PROMOTION",
        "reference review is incomplete",
    )
    table = store.json(review["result_table"])
    require(table["format"] == "selected-text-corrected-reference-interim-review-v1", "unsupported result table")
    require(same(review["counts"], table["counts"]), "review/table count disagreement")
    source_review = store.json(review["source_review"])
    require(source_review["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS", "oracle source review failed")
    sources = {}
    for ref in review["evidence"]:
        data = store.bytes(ref)
        if not ref["path"].endswith("/index.json"):
            continue
        capture = read_json(data)
        envelope = store.json(capture["source_envelope"])
        require(envelope["returncode"] == 0, "source capture failed")
        for item in capture["files"]:
            local = {key: item[key] for key in ("path", "sha256")}
            raw = store.bytes(local)
            require(len(raw) == item["bytes"], "capture length differs")
            old = sources.get(item["source_path"])
            require(old is None or old["sha256"] == local["sha256"], "conflicting captured source bytes")
            sources[item["source_path"]] = local
    rows = table["reference_corrections"] + table["failed_references"]
    require(len(rows) == len({r["cell"] for r in rows}) == review["counts"]["selected"], "duplicate/missing references")
    require(
        sum(r["status"] == "PASS" for r in rows) == review["counts"]["pass"]
        and sum(r["status"] == "FAIL" for r in rows) == review["counts"]["fail"]
        and review["counts"]["pending"] == 0,
        "unreviewed outcome counts",
    )
    require(
        set(review["numerical_failures_preserved"]) == {r["cell"] for r in rows if r["status"] == "FAIL"},
        "review dropped a failed reference",
    )
    return rows, sources


def normalize(store, cell, row, captured, gates_sha256):
    def raw(path, expected=None):
        require(path in captured, "missing raw capture: " + path)
        reference = captured[path]
        require(expected is None or reference["sha256"] == expected, "review/raw hash disagreement: " + path)
        return store.json(reference)

    require(
        row["cell"] == cell["id"] and row["full_config_sha256"] == cell["identity"]["config_sha256"], "wrong full cell"
    )
    require(row["oracle_version"] == "text_micro_reduce_v3", "unadmitted reference mathematics")
    plan = raw(row["plan"], row["plan_sha256"])
    receipt = raw(row["receipt"], row["receipt_sha256"])
    spec = next((r for r in plan["cells"] if r["id"] == row["cell"]), None)
    require(spec is not None and spec["sha256"] == row["spec_sha256"], "wrong preflight cell")
    require(spec["primary_receipt_sha256"] == row["primary_receipt_sha256"], "wrong distributed primary")
    # This gate compares the preserved primary trajectory. A continuation's
    # independent receipt is required by the separate recovery gate.
    require(
        receipt["cell"] == row["cell"]
        and receipt["job"] == row["job"]
        and receipt["plan_sha256"] == row["plan_sha256"]
        and receipt["status"] == row["status"],
        "raw reference outcome/identity disagrees",
    )
    result_path = str(Path(row["receipt"]).parent / "oracle/results.json")
    expected = row["result_sha256"] if row["status"] == "PASS" else row["partial_result"]["sha256"]
    result = raw(result_path, expected)
    provenance = result["source"]
    source = {
        "revision": row["source_revision"],
        "wheel_sha256": row["wheel_sha256"],
        "runtime_sha256": row["image_sha256"],
    }
    for key, target in [
        ("source_revision", "revision"),
        ("wheel_sha256", "wheel_sha256"),
        ("container_sha256", "runtime_sha256"),
    ]:
        require(provenance[key] == source[target], "reference installation changed")
    for key in ("source_revision", "wheel_sha256", "image_sha256"):
        require(plan[key] == row[key], "launch source differs from reviewed execution")
    require(
        provenance["spec_sha256"] == row["spec_sha256"]
        and provenance["oracle_version"] == row["oracle_version"]
        and provenance["slurm_job_id"] == row["job"]
        and provenance["verified_package_files"] > 1000
        and provenance["strict_determinism"] is True
        and provenance["initial_trainable_exact"] is True,
        "incomplete installed-reference provenance",
    )
    require(
        provenance["oracle_sha256"] == plan["files_sha256"]["independent_text_update_reference_v3.py"],
        "reference executed a different oracle",
    )
    steps = result["steps"]
    require([s["step"] for s in steps] == list(range(1, len(steps) + 1)), "reference reset, repeated or skipped update")
    for step in steps:
        require(step["status"] == "PASS", "malformed completed reference update")
        counts = step["independent_micro_tokens"]
        require(
            len(counts) == 2
            and all(type(n) is int and n >= 0 for n in counts)
            and sum(counts) == step["global_valid_tokens"] > 0,
            "wrong global native loss denominator",
        )
        require(
            all(
                math.isfinite(step[k])
                for k in ("global_loss", "distributed_loss", "max_gradient_error", "max_parameter_error")
            ),
            "nonfinite reference measurement",
        )
        require(
            abs(step["global_loss"] - step["distributed_loss"]) <= 1e-6 + 1e-5 * abs(step["distributed_loss"]),
            "global objective differs",
        )
    passed = row["status"] == "PASS"
    if passed:
        require(receipt["returncode"] == 0 and receipt["results_sha256"] == expected, "failed command claimed PASS")
        require(
            same(steps, row["steps"]) and same(steps, receipt["steps"]) and len(steps) == 3,
            "incomplete three-update comparison",
        )
        require(steps[-1]["independent_micro_tokens"][1] == 0, "missing incomplete-tail comparison")
        require(
            any(s["independent_micro_tokens"][0] != s["independent_micro_tokens"][1] for s in steps[:2]),
            "missing uneven-length comparison",
        )
        require(
            set(receipt["gradient_artifacts"]) == {f"gradient-{s:06d}.pt" for s in (1, 2, 3)},
            "missing saved reference gradients",
        )
        require(
            set(row["gradient_artifacts"]) == set(receipt["gradient_artifacts"]),
            "review omitted saved reference gradients",
        )
        require(
            all(receipt["gradient_artifacts"][n] == v["sha256"] for n, v in row["gradient_artifacts"].items()),
            "gradient evidence changed",
        )
    else:
        require(
            receipt["returncode"] != 0 and row["first_failed_check"] == "gradient comparison before optimizer step",
            "unsupported failed-reference kind",
        )
        require(
            row["first_failed_update"] == len(steps) + 1 and row["mismatched_elements"] > 0,
            "failed trajectory contradicts completed updates",
        )
        require(row["gradient_gates"] == {"atol": 2e-6, "rtol": 2e-4}, "failure used different gates")
        require(
            store.bytes(captured[row["log"]["path"]])
            and captured[row["log"]["path"]]["sha256"] == row["log"]["sha256"],
            "missing preserved numerical failure log",
        )
    measurement = {
        "same_global_inputs": True,
        "independent_objective_weighting": True,
        "uninterrupted_multi_update_reference": passed,
        "tail_and_uneven_lengths": passed,
        "raw_gradients_within_gates": passed,
        "optimizer_within_gates": passed,
        "native_rng_replayed": True,
        "gates_sha256": gates_sha256,
        "completed_steps": steps,
        "failure": None if passed else row,
        "basis": "Hash-bound independent source/result review plus original raw receipts. Elementwise gradient/parameter/Adam checks are enforced by the reviewed oracle before each completed step; unsaved independent Adam tensors were not reread by this normalizer.",
        "scope": "Independent-update prerequisite only. Failed early comparisons do not establish later updates, full training, lifecycle, deployment or task qualification.",
    }
    return measurement, source


def produce(root, contract, review_reference, output, gates_sha256):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root), "output escapes evidence root")
    store = Store(root)
    rows, captured = load_review(store, review_reference)
    cells = {c["id"]: c for c in contract["cells"]}
    producer = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    index = {"format": "adapter-campaign-evidence-index-v1", "contract_sha256": digest(contract), "cells": {}}
    for row in rows:
        cell = cells[row["cell"]]
        measurement, source = normalize(store, cell, row, captured, gates_sha256)
        gate = next(r for r in cell["requirements"] if r["id"] == "independent_global_update")
        require(
            producer in gate["producers"] and source in gate["sources"], "normalizer/source needs independent admission"
        )
        require(review_reference in producer_reviews(gate, producer), "reference review needs independent admission")
        require(
            any(c["pointer"] == "/gates_sha256" and c["expected"] == gates_sha256 for c in gate["assertions"]),
            "wrong numerical contract",
        )
        supporting = [{"path": path, "sha256": h} for path, h in sorted(store.verified.items())]
        directory = output / cell["id"]
        m = write(root, directory / "measurement.json", measurement)
        provenance = {
            "identity": cell["identity"],
            "source": source,
            "profile": None,
            "producer_sha256": producer,
            "lineage": None,
            "final_checkpoint": None,
            "documents_sha256": {"measurement": m["sha256"]},
            "supporting_evidence": supporting,
        }
        p = write(root, directory / "provenance.json", provenance)
        receipt = {
            "format": "adapter-campaign-gate-v1",
            "synthetic": False,
            "identity": cell["identity"],
            "gate": gate["id"],
            "profile": None,
            "source": source,
            "producer_sha256": producer,
            "lineage": None,
            "final_checkpoint": None,
            "outcome": row["status"],
            "executed": True,
            "documents": {"measurement": m, "provenance": p},
            "supporting_evidence": supporting,
        }
        index["cells"][cell["id"]] = {"gates": {gate["id"]: [write(root, directory / "receipt.json", receipt)]}}
    write(root, output / "index.json", index)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "contract", "review", "review-sha256", "gates-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    produce(
        args.root,
        read_json(Path(args.contract).read_bytes()),
        {"path": args.review, "sha256": args.review_sha256},
        args.output,
        args.gates_sha256,
    )
