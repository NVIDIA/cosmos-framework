# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Normalize reviewed final-portable warm starts, distinct from exact resume.

The admitted two-host worker owns the executed loading/negative controls. Its
independent actual-result review compared saved rank tensors, moments and child
serialization. This adapter revalidates their complete captured-byte closure,
the original final-checkpoint link, identities, topology and warm-state clocks.
It neither runs training nor transfers the two reviewed DoRA results to other
methods or model bindings.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

from .evidence import Store, check_receipt, digest, producer_reviews, read_json, require, same
from .final_checkpoint import Capture as CheckpointCapture
from .primary import file_ref, write

GATE = "final_portable_warm_reload"
DEPENDENCIES = (
    "portable_warm.py",
    "final_checkpoint.py",
    "inline_midpoint.py",
    "primary.py",
    "recovered.py",
    "evidence.py",
)
CELLS = tuple("llama_hf__dora__factorized_v1__default_v1__" + layout for layout in ("fsdp2", "hsdp"))
WORKER = "ca7561efda16c9ddc36878bda3e16b0c52594b19c6b7e2d733bf37034f885b21"


class Capture(Store):
    def __init__(self, root, review_ref, cell_id):
        super().__init__(root)
        review = self.json(review_ref)
        require(
            review["status"] == "PASS_ACTUAL_WARM_ONLY" and review["open_findings"] == [],
            "warm execution lacks independent review",
        )
        self.review = review
        self.folder = Path(review_ref["path"]).parent
        self.reviewed = review["reviewed_files"]
        self.source_review = self.review_json(review["source_review"])
        require(
            self.source_review["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS"
            and self.source_review["open_findings"] == [],
            "warm worker source is not admitted",
        )
        for name, sha in self.source_review["reviewed_files"].items():
            self.bytes({"path": name, "sha256": sha})
        actual = self.review_json(review["actual_audit"])
        require(
            actual["status"] == "PASS_BOTH_ACTUAL_WARM_LIFECYCLES" and same(actual["cases"], review["cases"]),
            "warm actual review disagrees with its audit",
        )
        rows = review["cases"]
        require(len(rows) == 2 and {r["cell"] for r in rows} == set(CELLS), "wrong reviewed warm population")
        require(cell_id in CELLS, "warm producer admits only the two actually reviewed DoRA bindings")
        self.case = next(r for r in rows if r["cell"] == cell_id)
        capture_ref = {key: review["actual_capture"][key] for key in ("path", "sha256")}
        capture = self.review_json(capture_ref)
        require(
            capture["format"] == "warm-final-two-host-readonly-capture-v1" and capture["read_only"] is True,
            "invalid warm capture",
        )
        matches = [r for r in capture["rows"] if r["job"] == self.case["job"]]
        require(len(matches) == 1, "ambiguous/missing warm allocation")
        self.row = matches[0]
        self.remote = Path(self.row["root"])
        self.refs = {}
        # Individual captured witnesses are each below Store's 64 MiB cap.
        # The redundant large transfer archive need not be loaded into memory.
        for name, item in capture["files"].items():
            relative = str(self.folder / "files" / name)
            ref = {"path": relative, "sha256": item["sha256"]}
            require(self.reviewed.get(relative) == ref["sha256"], "unreviewed raw warm witness")
            require(len(self.bytes(ref)) == item["bytes"], "warm capture size changed")
            self.refs[str(self.remote.parent / name)] = ref
        require(self.case["plan_sha256"] == self.row["plan_sha256"], "warm plan changed")
        self.capture = capture

    def review_json(self, ref):
        require(self.reviewed.get(ref["path"]) == ref["sha256"], "review omitted a required witness")
        return self.json(ref)

    def raw_ref(self, path):
        key = str(path)
        require(key in self.refs, "uncaptured warm source: " + key)
        return self.refs[key]

    def raw(self, path):
        return self.json(self.raw_ref(path))

    def document(self, name):
        return self.raw(self.remote / name)


def validate_identity(cell, case, plan, parent, checkpoint, checkpoint_measurement, artifact_receipt):
    identity, lineage = cell["identity"], checkpoint["lineage"]
    require(
        case["cell"] == plan["cell"] == cell["id"] and plan["files_sha256"]["cell.json"] == identity["config_sha256"],
        "wrong full-cell configuration",
    )
    expected_portable = str(Path(lineage["primary_root"]) / "output/primary/audit/final-portable")
    require(plan["portable"] == case["parent_portable"] == expected_portable, "warm start used another primary")
    require(plan["primary_plan_sha256"] == lineage["primary_plan_sha256"], "warm source primary plan changed")
    require(
        same(case["base"], identity["base"])
        and same(case["processor"], identity["processor"])
        and same(parent["identity"]["base"], identity["base"])
        and same(parent["identity"]["processor"], identity["processor"]),
        "warm base/processor differs from the frozen binding",
    )
    require(parent["kind"] == "portable_adapter", "incomplete or wrong artifact type")
    require(
        parent["weights_sha256"] == case["parent_tensor_sha256"] == plan["portable_inventory"]["adapter.safetensors"],
        "wrong final portable weights",
    )
    require(
        same(
            plan["portable_inventory"],
            {name: value["sha256"] for name, value in artifact_receipt["direct_inventory"].items()},
        )
        and checkpoint_measurement["checkpoint"]["inventory_sha256"] == checkpoint["final_checkpoint"]
        and same(checkpoint_measurement["checkpoint"]["identity"], identity),
        "portable is not linked to the admitted final checkpoint",
    )
    require(
        case["status"] == "PASS_ACTUAL_TWO_HOST_FRESH_WARM_START"
        and case["logical_updates"] == 3
        and case["per_rank_cursor"] == 6
        and case["optimizer_moments_reset_then_steps"] == [1, 2, 3]
        and case["scheduler_reset_then_steps"] == [0, 1, 2, 3]
        and case["parameters"] == 180
        and case["dora_magnitudes"] == 60
        and case["unique_examples"] == 24
        and all(case[key] is False for key in ("full_training_credit", "resume_credit", "task_or_deployment_credit")),
        "warm audit is incomplete or claims another lifecycle",
    )


def validate_rank(values, plan, case, parent_identity, rank):
    provenance, initial, warm, final, completed, public = values
    cell = plan["cell"]
    for value in (warm, final):
        require(value["cell"] == cell and value["plan_sha256"] == case["plan_sha256"], "rank identity changed")
        require(value["original_final_update"] == 935, "warm start is not from the scheduled final update")
    require(
        provenance["source_revision"] == case["source_revision"] == plan["source_revision"]
        and provenance["wheel_sha256"] == case["wheel_sha256"] == plan["wheel_sha256"]
        and provenance["container_sha256"] == case["image_sha256"] == plan["image_sha256"]
        and provenance["campaign_spec_sha256"] == plan["files_sha256"]["cell.json"]
        and provenance["warm_plan_sha256"] == case["plan_sha256"]
        and provenance["harness_sha256"] == plan["files_sha256"]["final_two_host_warm_v2.py"] == WORKER
        and provenance["slurm_job_id"] == case["job"]
        and provenance["verified_package_files"] == provenance["tracked_package_files_verified"] == 1932,
        "unattested installed warm execution",
    )
    require(
        warm["status"] == "PASS_EXACT_FINAL_PORTABLE_WITH_RESET_TRAINING_STATE"
        and same(warm["warm_update"], 0)
        and same(warm["warm_cursor"], 0)
        and warm["negative_snapshot_guards"] is True
        and same(warm["parent_inventory"], plan["portable_inventory"]),
        "initial load, negative guards or warm reset missing",
    )
    require(
        final["status"] == "PASS_THREE_TWO_HOST_PUBLIC_WARM_UPDATES"
        and final["warm_update"] == public["final_step"] == completed["step"] == 3
        and final["warm_cursor"] == 6
        and [s["step"] for s in final["schedulers"]] == [0, 1, 2, 3]
        and all(
            final[key] is True for key in ("ownership_and_aliases_unchanged", "parent_immutable", "child_strict_state")
        )
        and completed["frozen_preserved"] is True,
        "incomplete warm updates or mutated source/ownership",
    )
    require(
        same(warm["aliases"], final["aliases"])
        and same(warm["aliases"], case["aliases"])
        and same(initial["adapter_identity"], parent_identity)
        and same(initial["resolved"], parent_identity["resolved"]),
        "warm alias or adapter identity changed",
    )
    owners = [n for group in initial["optimizer"] for n in group["names"]]
    require(
        len(owners) == len(set(owners)) == case["parameters"]
        and set(owners) == set(initial["parameter_layouts"]) == set(completed["changed"])
        and all(value is True for value in completed["changed"].values())
        and set(completed["finite_gradients"]) == set(owners)
        and len(completed["finite_gradients"]) == len(owners),
        "missing/duplicated/unupdated warm owners",
    )
    require(
        len(initial["resolved"]["targets"]) == 60
        and initial["resolved"]["extra_parameters"] == []
        and initial["resolved"]["expected_trainable_parameters"] == case["trainable_elements"],
        "changed target, initial-factor or extra-module scope",
    )
    for target in initial["resolved"]["targets"]:
        settings = target["settings"]
        require(
            target["component"] == "language"
            and settings["method"] == "dora"
            and settings["rank"] == 8
            and settings["alpha"] == 16
            and settings["use_rslora"] is False
            and settings["dropout"] == 0
            and settings["dtype"] == "float32"
            and target["scaling"] == 2,
            "unreviewed warm method/component combination",
        )
    peers = initial["topology"]["peers"]
    require(
        same(peers, case["peers"])
        and len(peers) == 4
        and {p["rank"] for p in peers} == set(range(4))
        and len({p["host"] for p in peers}) == 2
        and len({p["gpu_uuid"] for p in peers}) == 4
        and all(p["job"] == case["job"] for p in peers),
        "warm allocation is not the reviewed two physical hosts",
    )
    require(bool(initial["topology"]["layouts"]), "missing actual warm mesh")
    rank_hosts = {p["rank"]: p["host"] for p in peers}
    require(
        rank_hosts[0] == rank_hosts[1]
        and rank_hosts[2] == rank_hosts[3]
        and sorted(p["local_rank"] for p in peers) == [0, 0, 1, 1],
        "warm rank-to-host mapping changed",
    )
    for layout in initial["topology"]["layouts"].values():
        full_shard = cell.endswith("__fsdp2")
        require(
            same(layout["shape"], [4] if full_shard else [2, 2])
            and same(layout["coordinate"], [rank] if full_shard else [rank // 2, rank % 2])
            and set(layout["groups"]) == ({"0"} if full_shard else {"0", "1"}),
            "warm mesh dimensions or coordinates changed",
        )
        for group in layout["groups"].values():
            require(
                rank in group["ranks"]
                and len(set(group["ranks"])) == len(group["ranks"])
                and set(group["ranks"]) <= set(rank_hosts)
                and set(group["hosts"]) == {rank_hosts[r] for r in group["ranks"]},
                "warm process group disagrees with actual hosts",
            )
            if full_shard:
                valid = len(group["ranks"]) == 4 and len(group["hosts"]) == 2 and group["placement"].startswith("S(")
            elif group["placement"] == "R":
                valid = len(group["ranks"]) == 2 and len(group["hosts"]) == 2
            else:
                valid = len(group["ranks"]) == 2 and len(group["hosts"]) == 1 and group["placement"].startswith("S(")
            require(valid, "wrong warm sharding/replication ownership")
        if not full_shard:
            require(
                layout["groups"]["0"]["placement"] == "R" and layout["groups"]["1"]["placement"].startswith("S("),
                "HSDP axes were interchanged",
            )
    require(
        len(case["rank_losses"][rank]) == 3 and all(math.isfinite(v) for v in case["rank_losses"][rank]),
        "nonfinite/incomplete warm loss",
    )


def normalize(root, contract, cell_id, *, review_ref, checkpoint_admission_ref, checkpoint_capture_ref):
    c = Capture(root, review_ref, cell_id)
    cell = next(row for row in contract["cells"] if row["id"] == cell_id)
    admission = c.json(checkpoint_admission_ref)
    admitted_contract, index = c.json(admission["contract"]), c.json(admission["index"])
    require(
        admission["status"] == "ACTIVE_REVIEWED_PARTIAL_FINAL_GATE_INDEX"
        and admitted_contract["synthetic"] is False
        and admission["canonical_contract_sha256"] == digest(admitted_contract) == index["contract_sha256"],
        "checkpoint prerequisite is not a coherent admitted real index",
    )
    previous_cell = next(row for row in admitted_contract["cells"] if row["id"] == cell_id)
    require(same(previous_cell["identity"], cell["identity"]), "checkpoint admission belongs to another binding")
    subject = index["cells"][cell_id]
    refs = subject["gates"]["final_checkpoint"]
    require(len(refs) == 1 and subject["final_checkpoint"] is not None, "missing/ambiguous final checkpoint")
    requirement = next(row for row in previous_cell["requirements"] if row["id"] == "final_checkpoint")
    outcome = check_receipt(
        c, refs[0], requirement, cell["identity"], admitted_contract, subject["final_checkpoint"], subject["lineage"]
    )
    require(outcome["outcome"] == "PASS" and outcome["executed"] is True, "final checkpoint did not pass")
    checkpoint = c.json(refs[0])
    require(checkpoint_capture_ref in checkpoint["supporting_evidence"], "unbound checkpoint extraction")
    checkpoint_measurement = c.json(checkpoint["documents"]["measurement"])
    artifacts = CheckpointCapture(root, checkpoint_capture_ref, cell_id)
    artifact_receipt = artifacts.document("output/receipt.json")
    plan, case = c.document("inputs/plan.json"), c.case
    require(c.raw_ref(c.remote / "inputs/plan.json")["sha256"] == case["plan_sha256"], "raw warm plan hash differs")
    require(c.source_review["plans"].get(cell_id) == case["plan_sha256"], "warm plan not source-reviewed")
    for name, sha in plan["files_sha256"].items():
        require(c.raw_ref(c.remote / "inputs" / name)["sha256"] == sha, "executed warm input differs from plan")
    parent = c.raw(Path(plan["portable"]) / "adapter_manifest.json")
    validate_identity(cell, case, plan, parent, checkpoint, checkpoint_measurement, artifact_receipt)
    for name, sha in plan["portable_inventory"].items():
        require(c.raw_ref(Path(plan["portable"]) / name)["sha256"] == sha, "parent portable bytes changed")
    receipt = c.document("output/receipt.json")
    require(
        receipt["passed"] is True
        and type(receipt["returncode"]) is int
        and receipt["returncode"] == 0
        and receipt["cell"] == cell_id
        and receipt["job"] == case["job"]
        and receipt["plan_sha256"] == case["plan_sha256"]
        and all(receipt[key] == case[key] for key in ("source_revision", "wheel_sha256", "image_sha256")),
        "warm worker did not complete successfully",
    )
    accounting = c.row["accounting"]
    require(type(accounting["returncode"]) is int and accounting["returncode"] == 0, "scheduler capture failed")
    rows = [line.split("|") for line in accounting["stdout"].splitlines() if line.strip()]
    require(
        len(rows) == 1 and rows[0][0] == case["job"] and rows[0][2:4] == ["COMPLETED", "0:0"],
        "scheduler outcome disagrees",
    )
    for rank in range(4):
        names = ("provenance", "initial", "warm-initial", "warm-final", "completed-updates", "public-result")
        values = [c.document(f"output/primary/audit/{name}-rank{rank}.json") for name in names]
        validate_rank(values, plan, case, parent["identity"], rank)
    c.verified.update(artifacts.verified)
    source = {
        "revision": case["source_revision"],
        "wheel_sha256": case["wheel_sha256"],
        "runtime_sha256": case["image_sha256"],
    }
    measurement = {
        "strict_reload": True,
        "base_processor_identity": True,
        "initial_factors_extras_ties": True,
        "source_immutable": True,
        "warm_start_reset": True,
        "warm_updates_and_ownership": True,
        "warm_two_node_topology": True,
        "incomplete_and_wrong_base_rejected": True,
        "actual_warm_result": case,
        "basis": "Reviewed installed public-worker negative/load/update checks plus independent saved-tensor audit; all captured witnesses rehashed. This producer rechecks identity/topology/reset receipts, not model forwards or optimizer tensor arithmetic.",
        "scope": "Only the exact two final935 Smol DoRA bindings. Warm updates are diagnostic work, not primary coverage, exact resume, deployment numerics or task quality.",
    }
    support = [{"path": name, "sha256": sha} for name, sha in c.verified.items()]
    return measurement, source, checkpoint["lineage"], checkpoint["final_checkpoint"], support


def produce(
    root, contract, output, *, cell_id, review_ref, checkpoint_admission_ref, checkpoint_capture_ref, producer_review
):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root) and not output.exists(), "fresh output directory required")
    require(contract.get("synthetic") is False, "synthetic warm proof is not campaign evidence")
    store = Store(root)
    review = store.json(producer_review)
    require(
        review["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS" and review["open_findings"] == [], "unreviewed producer"
    )
    for name in DEPENDENCIES:
        require(
            review["reviewed_files"].get("tools/adapter_campaign/" + name)
            == hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest(),
            "producer dependency changed",
        )
    for ref in (review_ref, checkpoint_admission_ref, checkpoint_capture_ref):
        require(review["reviewed_files"].get(ref["path"]) == ref["sha256"], "unreviewed warm/checkpoint input")
    producer = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cell = next(row for row in contract["cells"] if row["id"] == cell_id)
    requirement = next(row for row in cell["requirements"] if row["id"] == GATE)
    require(
        producer in requirement["producers"] and producer_review in producer_reviews(requirement, producer),
        "producer requires exact independent admission",
    )
    measurement, source, lineage, anchor, support = normalize(
        root,
        contract,
        cell_id,
        review_ref=review_ref,
        checkpoint_admission_ref=checkpoint_admission_ref,
        checkpoint_capture_ref=checkpoint_capture_ref,
    )
    require(source in requirement["sources"], "warm installed revision needs explicit admission")
    support.append(producer_review)
    for ref in support:
        store.bytes(ref)
    directory = output / cell_id / GATE
    m = write(root, directory / "measurement.json", measurement)
    common = {
        "identity": cell["identity"],
        "source": source,
        "profile": None,
        "producer_sha256": producer,
        "lineage": lineage,
        "final_checkpoint": anchor,
        "supporting_evidence": support,
    }
    p = write(root, directory / "provenance.json", {**common, "documents_sha256": {"measurement": m["sha256"]}})
    receipt = {
        **common,
        "format": "adapter-campaign-gate-v1",
        "synthetic": False,
        "gate": GATE,
        "outcome": "PASS",
        "executed": True,
        "documents": {"measurement": m, "provenance": p},
    }
    index = {
        "format": "adapter-campaign-evidence-index-v1",
        "contract_sha256": digest(contract),
        "cells": {
            cell_id: {
                "lineage": lineage,
                "final_checkpoint": anchor,
                "gates": {GATE: [write(root, directory / "receipt.json", receipt)]},
            }
        },
    }
    write(root, output / "index.json", index)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "root",
        "contract",
        "output",
        "cell-id",
        "warm-review",
        "checkpoint-admission",
        "checkpoint-capture",
        "producer-review",
    ):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    produce(
        root,
        read_json(Path(args.contract).read_bytes()),
        args.output,
        cell_id=args.cell_id,
        review_ref=file_ref(root, Path(args.warm_review).resolve()),
        checkpoint_admission_ref=file_ref(root, Path(args.checkpoint_admission).resolve()),
        checkpoint_capture_ref=file_ref(root, Path(args.checkpoint_capture).resolve()),
        producer_review=file_ref(root, Path(args.producer_review).resolve()),
    )
