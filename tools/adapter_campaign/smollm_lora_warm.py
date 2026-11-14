# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Admit the recorded own-final Smol LoRA warm start without repeating training.

This narrowly bound producer consumes immutable copied receipts. It preserves
the distinct source, original final checkpoint, and failed predecessor. It does
not reinterpret the older DoRA worker or confer resume/deployment/task credit.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

from .evidence import Store, check_receipt, digest, producer_reviews, read_json, require, same
from .final_checkpoint import Capture as CheckpointCapture
from .primary import file_ref, write

CELL = "llama_hf__lora__factorized_v1__default_v1__fsdp2"
GATE = "final_portable_warm_reload"
JOB = "38537918"
WORKER = "15960fa464cfd8f1e239280b898057504f7d73b30b76f55b5e01b2d8c1dcbea1"
SOURCE = {
    "revision": "5fbdf7d189a7f1be84af1d5175d8936b6e5a47e2",
    "wheel_sha256": "4907e00c809b31f74885bf185a674f9dc6dbfb29f2ff33f1a2625aa49fd6d9f5",
    "runtime_sha256": "2ec9f4d3d2a079705de15a8bc5c141fa681f84aabbf7f996f20b18e4b71d2a8e",
}
DEPENDENCIES = (
    "smollm_lora_warm.py",
    "final_checkpoint.py",
    "inline_midpoint.py",
    "primary.py",
    "recovered.py",
    "evidence.py",
)


class Capture(Store):
    """Resolve original absolute review names through a byte-exact copy map."""

    def __init__(self, root, reference):
        super().__init__(root)
        self.extraction = self.json(reference)
        x = self.extraction
        require(
            x["format"] == "reviewed-smol-lora-warm-copy-v1" and x["synthetic"] is False,
            "wrong or synthetic warm capture",
        )
        self.aliases = x["aliases"]
        require(isinstance(self.aliases, dict) and self.aliases, "missing original-byte copy map")
        actual = self.json(x["actual_review"])
        require(
            actual["format"] == "smol-own-final-warm-independent-actual-review-v1"
            and actual["status"] == "PASS_OWN_FINAL_WARM_ONLY"
            and actual["open_findings"] == []
            and actual["job"] == JOB
            and actual["full_campaign_pass"] is False
            and same(actual["new_full_schedule_credit"], 0),
            "missing independent actual warm review",
        )
        self.review = actual
        for name, expected in actual["reviewed_files"].items():
            require(name in self.aliases and self.aliases[name]["sha256"] == expected, "incomplete reviewed copy")
            self.bytes(self.aliases[name])
        require(set(self.aliases) == set(actual["reviewed_files"]), "unexpected or unreviewed warm source alias")
        self.capture = self.original(x["original_capture"])
        self.source_review = self.original(x["original_source_review"])
        require(
            self.source_review["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS"
            and self.source_review["open_findings"] == [],
            "warm worker source review did not pass",
        )
        for name, expected in self.source_review["reviewed_files"].items():
            require(actual["reviewed_files"].get(name) == expected, "source review lost actual-byte linkage")
        cap = self.capture
        require(
            cap["format"] == "smol-lora-final-warm-terminal-capture-v3"
            and cap["job"] == JOB
            and cap["namespace_unchanged"] is True
            and cap["all_files_freshly_hashed"] is True
            and same(cap["new_gpu_executions"], 0)
            and cap["qualification_pass"] is False,
            "incomplete or wrong terminal capture",
        )
        self.original_files = Path(x["original_files_directory"])
        require(self.original_files.is_absolute(), "raw capture source location is ambiguous")
        for name in cap["mirrored_small_files"]:
            path = Path(name)
            require(not path.is_absolute() and ".." not in path.parts, "escaping raw warm path")
            row = cap["files"][name]
            raw = self.raw_bytes(name)
            require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"], "stale raw file")

    def original(self, name):
        require(name in self.aliases, "required original review source was not copied")
        return self.json(self.aliases[name])

    def raw_bytes(self, name):
        key = str(self.original_files / name)
        require(key in self.aliases, "missing captured warm witness: " + name)
        return self.bytes(self.aliases[key])

    def raw(self, name):
        return read_json(self.raw_bytes(name))


def validate_rank(values, plan, rank, hosts, journal):
    """Recheck reset, real topology, intended ownership and all three updates."""
    initial, start, final, done, public, provenance = values
    owners = plan["owner_names"]
    require(len(owners) == len(set(owners)) == 120, "incorrect LoRA owner population")
    require(
        same(initial["adapter_identity"], plan["adapter_identity"])
        and same(initial["resolved"], plan["adapter_identity"]["resolved"])
        and sorted(initial["parameter_layouts"]) == owners
        and same(initial["initial_clock"], plan["initial_clock"])
        and initial["strict_determinism"] is True,
        "warm initialization differs from the original artifact",
    )
    actual_owners = [n for group in initial["optimizer"] for n in group["names"]]
    require(
        len(actual_owners) == len(set(actual_owners)) == 120 and sorted(actual_owners) == owners, "optimizer owners"
    )
    for key in ("data_identity", "finite_coverage", "validation_identity", "frozen_sha256"):
        require(same(initial[key], plan["parent_initial_by_rank"][str(rank)][key]), "changed frozen data/base state")
    for value in (start, final):
        require(
            value["cell"] == CELL
            and value["plan_sha256"] == plan["actual_plan_sha256"]
            and same(value["original_final_update"], 935)
            and same(value["owner_names"], owners)
            and same(value["target_names"], plan["target_names"])
            and same(value["journal_sha256"], journal),
            "wrong warm source or execution identity",
        )
    require(
        start["status"] == "PASS_EXACT_FINAL_PORTABLE_WITH_RESET_TRAINING_STATE"
        and same(start["warm_update"], 0)
        and same(start["warm_cursor"], 0)
        and same(start["initial_clock"], plan["initial_clock"])
        and start["optimizer_empty"] is True
        and start["negative_snapshot_guards"] is True
        and same(start["parent_inventory"], plan["portable_inventory"]),
        "not a strict fresh warm start with negative controls",
    )
    clock = dict(
        iter=3,
        optimizer_step=3,
        accum=0,
        n_mini=1,
        attempt_id=3,
        nonfinite_skips_total=0,
        nonfinite_skips_consecutive=0,
    )
    require(
        final["status"] == "PASS_THREE_TWO_HOST_PUBLIC_WARM_UPDATES"
        and all(same(value, 3) for value in (final["warm_update"], done["step"], public["final_step"]))
        and same(final["warm_cursor"], 6)
        and same(final["clock"], clock)
        and same(final["optimizer_steps"], [3])
        and same(final["aliases"], start["aliases"])
        and all(
            final[key] is True for key in ("ownership_and_aliases_unchanged", "parent_immutable", "child_strict_state")
        )
        and done["frozen_preserved"] is True
        and sorted(done["changed"]) == owners
        and all(v is True for v in done["changed"].values())
        and sorted(done["finite_gradients"]) == owners,
        "missing updates, wrong clocks, duplicate gradients or changed parameter ownership",
    )
    require(same([s["step"] for s in final["schedulers"]], [0, 1, 2, 3]), "wrong scheduler update sequence")
    require(same(start["initial_scheduler"], final["schedulers"][0]), "scheduler did not reset")
    for row in final["schedulers"]:
        names = [name for group in row["groups"] for name in group["names"]]
        require(len(names) == 120 and sorted(names) == owners, "changed scheduler ownership")
        require(same(row["identity"], final["schedulers"][0]["identity"]), "changed scheduler identity")
    peers, layouts = initial["topology"]["peers"], initial["topology"]["layouts"]
    require(len(hosts) == len(set(hosts)) == 2 and len(peers) == 4 and layouts, "missing two-host topology")
    require(len({p["gpu_uuid"] for p in peers}) == 4, "GPU ownership duplicated")
    require(len({p["gpu_name"] for p in peers}) == len({p["driver"] for p in peers}) == 1, "heterogeneous runtime")

    def host(value):
        return value.split(".")[0]

    require(len(set(map(host, hosts))) == 2, "host aliases do not constitute two nodes")
    for i, peer in enumerate(peers):
        require(
            same(peer["rank"], i)
            and same(peer["local_rank"], i % 2)
            and str(peer["node_id"]) == str(i // 2)
            and host(peer["host"]) == host(hosts[i // 2])
            and peer["job"] == JOB
            and "A100" in peer["gpu_name"]
            and bool(peer["driver"]),
            "wrong rank to physical host/device mapping",
        )
    for key, layout in layouts.items():
        require(
            same(read_json(key), [0, 1, 2, 3])
            and same(layout["shape"], [4])
            and same(layout["coordinate"], [rank])
            and set(layout["groups"]) == {"0"},
            "wrong FSDP mesh",
        )
        group = layout["groups"]["0"]
        require(
            same(group["ranks"], [0, 1, 2, 3])
            and set(map(host, group["hosts"])) == set(map(host, hosts))
            and group["placement"] == "S(0)",
            "sharding does not cross both hosts",
        )
    require(
        provenance["source_revision"] == SOURCE["revision"]
        and provenance["wheel_sha256"] == SOURCE["wheel_sha256"]
        and provenance["container_sha256"] == SOURCE["runtime_sha256"]
        and provenance["warm_plan_sha256"] == plan["actual_plan_sha256"]
        and provenance["campaign_spec_sha256"] == plan["files_sha256"]["cell.json"]
        and provenance["harness_sha256"] == plan["files_sha256"]["smol_final_warm_worker_v2.py"] == WORKER
        and provenance["slurm_job_id"] == JOB
        and same(provenance["verified_package_files"], 1932)
        and same(provenance["tracked_package_files_verified"], 1932),
        "wrong installed warm implementation",
    )


def validate_closure(c, folder, plan, hosts, journal):
    """Require clean parent and both node command domains at this exact job."""
    phase = c.raw(folder + "/primary-receipt.json")
    parent = c.raw(folder + "/primary-ownership.json")
    command = c.raw(folder + "/primary-command.json")
    remote = plan["root"] + "/" + folder

    def clean(value, argv):
        require(
            same(value["command"], argv)
            and same(value["returncode"], 0)
            and value["cleanup_error"] is None
            and value.get("exception") is None
            and value["cleanup"]["remaining"] == [],
            "unclean owned command",
        )

    require(
        phase["status"] == "PASS_OWN_FINAL_WARM_ONLY"
        and phase["passed"] is True
        and phase["plan_sha256"] == plan["actual_plan_sha256"]
        and phase["job"] == JOB,
        "incomplete warm phase",
    )
    clean(command, phase["command"])
    require(
        phase["command_receipt_sha256"] == hashlib.sha256(c.raw_bytes(folder + "/primary-command.json")).hexdigest()
        and same(parent["command"], command["command"])
        and same(parent["uid"], 41852)
        and parent["plan_sha256"] == plan["actual_plan_sha256"]
        and command["ownership_path"] == remote + "/primary-ownership.json",
        "wrong parent ownership",
    )
    tokens = {parent["token"]}
    parent_token = hashlib.sha256(parent["token"].encode()).hexdigest()
    for node in range(2):
        prefix = folder + f"/owned-primary-node{node}"
        records = {name: c.raw(prefix + "-" + name + ".json") for name in ("launch", "ownership", "command", "result")}
        launch, owned, measured, result = (records[key] for key in ("launch", "ownership", "command", "result"))
        clean(measured, launch["command"])
        require(
            result["status"] == "PASS_COMMAND"
            and result["execution_error"] is None
            and result["journals_unchanged"] is True
            and same(result["returncode"], 0),
            "missing node postcheck",
        )
        for value in (launch, result):
            require(
                value["job"] == JOB
                and same(value["node_id"], node)
                and value["plan_sha256"] == plan["actual_plan_sha256"]
                and same(value["journal_sha256"], journal)
                and value["source_revision"] == SOURCE["revision"]
                and value["image_sha256"] == SOURCE["runtime_sha256"]
                and value["spec_sha256"] == plan["files_sha256"]["cell.json"]
                and value["cell"] == CELL
                and value["phase"] == "primary"
                and same(value["uid"], 0)
                and value["host"].split(".")[0] == hosts[node].split(".")[0],
                "wrong node command identity",
            )
        require(
            launch["parent_owned_token_sha256"] == parent_token
            and owned["token"] not in tokens
            and same(owned["command"], launch["command"])
            and same(owned["uid"], 0)
            and owned["host"] == launch["host"]
            and owned["plan_sha256"] == plan["actual_plan_sha256"],
            "node ownership domain",
        )
        tokens.add(owned["token"])
        require(
            measured["ownership_path"] == remote + f"/owned-primary-node{node}-ownership.json"
            and measured["pidfd_backend"] == owned["pidfd_backend"]
            and "--nnodes=2" in measured["command"]
            and "--nproc-per-node=2" in measured["command"]
            and f"--node-rank={node}" in measured["command"],
            "wrong node process geometry",
        )
        pins = {key: hashlib.sha256(c.raw_bytes(prefix + "-" + key + ".json")).hexdigest() for key in records}
        require(
            all(result[key + "_sha256"] == pins[key] for key in ("launch", "ownership", "command"))
            and phase["node_owned_commands"][str(node)]["status"] == "PASS_COMMAND"
            and same(phase["node_owned_commands"][str(node)]["files_sha256"], pins),
            "node closure hash mismatch",
        )


def normalize(root, contract, *, extraction_ref, checkpoint_admission_ref, checkpoint_capture_ref):
    c = Capture(root, extraction_ref)
    cell = next(row for row in contract["cells"] if row["id"] == CELL)
    admission = c.json(checkpoint_admission_ref)
    previous, index = c.json(admission["contract"]), c.json(admission["index"])
    require(
        admission["status"] == "ACTIVE_REVIEWED_PARTIAL_FINAL_GATE_INDEX"
        and previous["synthetic"] is False
        and admission["canonical_contract_sha256"] == digest(previous) == index["contract_sha256"],
        "checkpoint admission",
    )
    old_cell = next(row for row in previous["cells"] if row["id"] == CELL)
    require(same(old_cell["identity"], cell["identity"]), "different checkpoint binding")
    subject = index["cells"][CELL]
    refs = subject["gates"]["final_checkpoint"]
    require(len(refs) == 1 and subject["final_checkpoint"] is not None, "missing final checkpoint prerequisite")
    requirement = next(row for row in old_cell["requirements"] if row["id"] == "final_checkpoint")
    outcome = check_receipt(
        c, refs[0], requirement, cell["identity"], previous, subject["final_checkpoint"], subject["lineage"]
    )
    require(outcome["outcome"] == "PASS" and outcome["executed"] is True, "final checkpoint did not pass")
    checkpoint = c.json(refs[0])
    require(checkpoint_capture_ref in checkpoint["supporting_evidence"], "checkpoint extraction is not bound")
    artifacts = CheckpointCapture(root, checkpoint_capture_ref, CELL)
    artifact = artifacts.document("output/receipt.json")
    proof = c.json(checkpoint["documents"]["measurement"])
    plan = c.raw("inputs/plan.json")
    raw_plan = hashlib.sha256(c.raw_bytes("inputs/plan.json")).hexdigest()
    cap, review = c.capture, c.review
    require(raw_plan == cap["plan_sha256"] == review["plan_sha256"], "warm plan changed")
    require(
        plan["cell"] == CELL and plan["files_sha256"]["cell.json"] == cell["identity"]["config_sha256"], "wrong config"
    )
    require(
        plan["portable"] == checkpoint["lineage"]["primary_root"] + "/output/primary/audit/final-portable"
        and plan["primary_plan_sha256"] == checkpoint["lineage"]["primary_plan_sha256"],
        "cross-primary warm artifact",
    )
    require(
        same(proof["checkpoint"]["identity"], cell["identity"])
        and proof["checkpoint"]["inventory_sha256"] == checkpoint["final_checkpoint"],
        "wrong final checkpoint anchor",
    )
    # The raw historical source capture contains the final portable manifest and
    # byte inventory; no saved state is created or numerically reinterpreted.
    historical = c.raw("inputs/source-capture.json")
    parent_name = plan["portable"] + "/adapter_manifest.json"
    parent_row = historical["files"][parent_name]
    parent_bytes = parent_row["text"].encode()
    parent = read_json(parent_bytes)
    require(
        hashlib.sha256(parent_bytes).hexdigest()
        == parent_row["sha256"]
        == plan["portable_inventory"]["adapter_manifest.json"],
        "parent manifest hash",
    )
    require(
        parent["kind"] == "portable_adapter"
        and same(parent["identity"], plan["adapter_identity"])
        and parent["weights_sha256"] == plan["portable_inventory"]["adapter.safetensors"],
        "parent artifact identity",
    )
    require(
        same(plan["portable_inventory"], {name: row["sha256"] for name, row in artifact["direct_inventory"].items()}),
        "warm artifact differs from the independently admitted final checkpoint extraction",
    )
    for key in ("base", "processor"):
        require(same(parent["identity"][key], cell["identity"][key]), "base/processor differs from frozen binding")
    for name, expected in plan["files_sha256"].items():
        require(hashlib.sha256(c.raw_bytes("inputs/" + name)).hexdigest() == expected, "changed executed worker input")
    require(
        plan["invocation_profile"] == "smol_final_warm_nonmutating_observation_v2", "wrong warm observation semantics"
    )
    resolved = plan["adapter_identity"]["resolved"]
    require(len(resolved["targets"]) == 60 and resolved["extra_parameters"] == [], "changed component scope")
    for target in resolved["targets"]:
        settings = target["settings"]
        require(
            target["component"] == "language"
            and settings["method"] == "lora"
            and same(settings["rank"], 8)
            and settings["alpha"] == 16
            and settings["use_rslora"] is False
            and settings["dropout"] == 0
            and settings["dtype"] == "float32"
            and target["scaling"] == 2,
            "wrong method or scaling",
        )
    allocation, result = c.raw("output/allocation.json"), c.raw("output/results.json")
    require(
        result["status"] == "PASS_OWN_FINAL_WARM_ONLY"
        and result["job"] == allocation["job"] == JOB
        and result["plan_sha256"] == allocation["plan_sha256"] == raw_plan
        and result["qualification_pass"] is False
        and same(result["campaign_credit"], 0),
        "wrong warm completion",
    )
    for value in (plan, result, allocation, review):
        require(
            value["source_revision"] == SOURCE["revision"] and value["image_sha256"] == SOURCE["runtime_sha256"],
            "wrong actual source",
        )
    require(plan["wheel_sha256"] == review["wheel_sha256"] == SOURCE["wheel_sha256"], "wrong wheel")
    journal = allocation["journal_sha256"]
    require(
        same(journal, {n: hashlib.sha256(c.raw_bytes(n)).hexdigest() for n in ("intent.json", "submission.json")}),
        "changed scheduler journal",
    )
    rows = [row.split("|") for row in cap["sacct"]["stdout"].splitlines() if row]
    require(
        cap["sacct"]["returncode"] == 0
        and len(rows) == 1
        and rows[0][:4] == [JOB, "COMPLETED", "0:0", "150"]
        and "gres/gpu=4" in rows[0][4],
        "scheduler completion/cost differs",
    )
    plan = {**plan, "actual_plan_sha256": raw_plan}
    folder = "output/" + CELL
    validate_closure(c, folder, plan, allocation["hosts"], journal)
    examples = []
    for rank in range(4):
        names = ("initial", "warm-initial", "warm-final", "completed-updates", "public-result", "provenance")
        values = [c.raw(f"{folder}/primary/audit/{name}-rank{rank}.json") for name in names]
        validate_rank(values, plan, rank, allocation["hosts"], journal)
        coverage = [
            read_json(line) for line in c.raw_bytes(f"{folder}/primary/audit/coverage-rank{rank}.jsonl").splitlines()
        ]
        require(
            same(
                [{"step": row["step"], "inputs": row["inputs"]} for row in coverage], plan["expected_inputs"][str(rank)]
            ),
            "warm examples changed",
        )
        require(
            len(coverage) == 3
            and all(
                row["metrics"]["optim/update_outcome"] == "applied"
                and type(row["metrics"]["train/loss_avg"]) in (int, float)
                and math.isfinite(row["metrics"]["train/loss_avg"])
                for row in coverage
            ),
            "incomplete/nonfinite updates",
        )
        for row in coverage:
            for item in row["inputs"]:
                require(
                    item["padding"] is False and item["weight"] == 1 and same(item["epoch"], 0), "wrong warm coverage"
                )
                examples.append(item["sample_id"])
    require(len(examples) == len(set(examples)) == 24, "duplicate or omitted warm examples")
    c.verified.update(artifacts.verified)
    measurement = {
        key: True
        for key in (
            "strict_reload",
            "base_processor_identity",
            "initial_factors_extras_ties",
            "source_immutable",
            "warm_start_reset",
            "warm_updates_and_ownership",
            "warm_two_node_topology",
            "incomplete_and_wrong_base_rejected",
        )
    }
    measurement.update(
        actual_warm_job=JOB,
        original_final_update=935,
        warm_updates=3,
        parameter_owners=120,
        actual_review=c.extraction["actual_review"],
        allocation_gpu_seconds=600,
        basis="Executed source-pinned worker and independent actual review; raw reset/ownership/topology/coverage receipts rechecked. No model execution in this producer.",
        scope="Only own-final935 ordinary-LoRA FSDP2 warm start; no resume, full schedule, deployment or task credit.",
    )
    return (
        measurement,
        SOURCE,
        checkpoint["lineage"],
        checkpoint["final_checkpoint"],
        [{"path": n, "sha256": h} for n, h in c.verified.items()],
    )


def produce(
    root, contract, output, *, extraction_ref, checkpoint_admission_ref, checkpoint_capture_ref, producer_review
):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(
        output.is_relative_to(root) and not output.exists() and contract.get("synthetic") is False,
        "fresh real output required",
    )
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
    for ref in (extraction_ref, checkpoint_admission_ref, checkpoint_capture_ref):
        require(review["reviewed_files"].get(ref["path"]) == ref["sha256"], "unreviewed source or prerequisite")
    producer = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cell = next(row for row in contract["cells"] if row["id"] == CELL)
    requirement = next(row for row in cell["requirements"] if row["id"] == GATE)
    require(
        producer in requirement["producers"] and producer_review in producer_reviews(requirement, producer),
        "unadmitted producer",
    )
    measurement, source, lineage, anchor, support = normalize(
        root,
        contract,
        extraction_ref=extraction_ref,
        checkpoint_admission_ref=checkpoint_admission_ref,
        checkpoint_capture_ref=checkpoint_capture_ref,
    )
    require(source in requirement["sources"], "unadmitted execution source")
    support.append(producer_review)
    directory = output / CELL / GATE
    m = write(root, directory / "measurement.json", measurement)
    common = dict(
        identity=cell["identity"],
        source=source,
        profile=None,
        producer_sha256=producer,
        lineage=lineage,
        final_checkpoint=anchor,
        supporting_evidence=support,
    )
    provenance = write(
        root, directory / "provenance.json", {**common, "documents_sha256": {"measurement": m["sha256"]}}
    )
    receipt = {
        **common,
        "format": "adapter-campaign-gate-v1",
        "synthetic": False,
        "gate": GATE,
        "outcome": "PASS",
        "executed": True,
        "documents": {"measurement": m, "provenance": provenance},
    }
    index = {
        "format": "adapter-campaign-evidence-index-v1",
        "contract_sha256": digest(contract),
        "cells": {
            CELL: {
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
        "extraction",
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
        extraction_ref=file_ref(root, Path(args.extraction).resolve()),
        checkpoint_admission_ref=file_ref(root, Path(args.checkpoint_admission).resolve()),
        checkpoint_capture_ref=file_ref(root, Path(args.checkpoint_capture).resolve()),
        producer_review=file_ref(root, Path(args.producer_review).resolve()),
    )
