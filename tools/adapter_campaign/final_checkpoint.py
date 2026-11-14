# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Normalize reviewed full-text final DCP/consolidation proof, without exporting.

Only final_checkpoint is emitted. The original DCP inventory anchors later
artifacts. Full model/optimizer equality is the existing installed producer's
proof; small controls and DCP metadata are independently checked here on CPU.
The artifact job's numerical failures remain failures. A complete adapter alone,
a directory, or a successful command is insufficient checkpoint evidence.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import pickle
import re
from pathlib import Path

from tools.adapter_campaign.evidence import (
    Store,
    check_receipt,
    digest,
    producer_reviews,
    read_json,
    require,
    same,
)
from tools.adapter_campaign.inline_midpoint import check_checkpoint, check_storage_metadata, exact_tree
from tools.adapter_campaign.primary import file_ref, write

GATE = "final_checkpoint"
DEPENDENCIES = ("final_checkpoint.py", "inline_midpoint.py", "primary.py", "recovered.py", "evidence.py")
RAW_FILES = (
    "output/receipt.json",
    "output/execution-intent.json",
    "output/consolidation-result.json",
    "output/consolidate-command.json",
    "output/selective-extract-command.json",
    "output/consolidated-extract-command.json",
    "output/control/provenance.json",
    "output/control/construction-provenance.json",
    "output/checkpoint-portable/adapter_manifest.json",
    "output/consolidated-portable/adapter_manifest.json",
)
WORKERS = {
    "run_final_text_artifacts_v3.py": "4e9808d6e3278a230a42ebede43cf95c47a4166108623be6aec5ec38036eb987",
    "final_text_artifact_state_v2.py": "992fdbbc5f1c2da310de397c0105a8b977750fd8dd7be1e3e4e22de2e0a8db95",
    "final_text_deployment_v3.py": "09e531b2535bfbbca13f22b227ad5ec4709c8e95cfc48e52304a62c6ae870ce4",
}


def positive_integer(value):
    return type(value) is int and value > 0


def inventory(value, names):
    require(isinstance(value, dict) and set(value) == set(names), "partial/unexpected checkpoint inventory")
    for item in value.values():
        require(
            isinstance(item, dict)
            and set(item) == {"bytes", "sha256"}
            and positive_integer(item["bytes"])
            and isinstance(item["sha256"], str)
            and re.fullmatch("[0-9a-f]{64}", item["sha256"]) is not None,
            "invalid checkpoint hash/length",
        )
    return value


class Capture(Store):
    def __init__(self, root, reference, cell_id):
        super().__init__(root)
        self.extraction = self.json(reference)
        x = self.extraction
        require(
            x["format"] == "reviewed-final-checkpoint-capture-extraction-v1"
            and x["status"] == "EXACT_CAPTURE_BYTES_EXTRACTED",
            "uncommitted extraction",
        )
        self.review = self.json(x["review"])
        review = self.review
        require(
            review["format"] == "smollm-final-artifact-terminal-result-review-v1"
            and review["status"] == "REVIEW_COMPLETE_LEGACY_NUMERICAL_FAILURES_PRESERVED"
            and review["open_confirmed_capture_consistency_findings"] == [],
            "missing independent terminal capture review",
        )
        require(
            same(x["original_capture"], review["capture"])
            and same(x["inventory"], review["captured_file_inventory"])
            and review["reviewed_files"].get(x["original_capture"]["path"]) == x["original_capture"]["sha256"]
            and review["reviewed_files"].get(x["inventory"]["path"]) == x["inventory"]["sha256"],
            "extraction lost original capture/inventory anchor",
        )
        envelope = x["envelope"]
        require(
            type(envelope["returncode"]) is int
            and envelope["returncode"] == 0
            and envelope["stderr"] == ""
            and envelope["started_utc"] == review["capture_started_utc"]
            and same(x["capture_metadata"]["scheduler"], review["scheduler"]),
            "extraction changed original execution metadata",
        )
        self.bytes(x["extractor"])
        entries = self.json(x["inventory"])
        require(
            same(x["original_text_files_verified"], review["counts"]["captured_text_files_rehashed"])
            and len(entries) == x["original_text_files_verified"],
            "incomplete reviewed text inventory",
        )
        self.row = x["rows"][cell_id]
        row = self.row
        require(row["cell"] == cell_id and set(row["files"]) == set(RAW_FILES), "wrong cell/raw capture set")
        reviewed = {}
        for item in entries:
            key = (item["row_index"], item["relative_name"])
            require(key not in reviewed and same(item["capture"], x["original_capture"]), "ambiguous capture inventory")
            reviewed[key] = item
        self.raw_documents = {}
        for name, ref in row["files"].items():
            item = reviewed[(row["row_index"], name)]
            require(
                str(Path(row["root"]) / name) == item["producer_path"] and ref["sha256"] == item["sha256"],
                "extracted bytes are not the reviewed original source",
            )
            self.raw_documents[name] = self.json(ref)
        self.controls, self.inventories = {}, {}

    def reviewed(self, ref):
        require(self.review["reviewed_files"].get(ref["path"]) == ref["sha256"], "unreviewed source")
        return self.json(ref)

    def local_reviewed(self, name):
        require(name in self.review["reviewed_files"], "missing source admission")
        return self.reviewed({"path": name, "sha256": self.review["reviewed_files"][name]})

    def document(self, name):
        return self.raw_documents[name]

    def load_controls(self, reference, source, destination):
        envelope = self.json(reference)
        require(
            type(envelope["returncode"]) is int and envelope["returncode"] == 0 and envelope["stderr"] == "",
            "failed small control capture",
        )
        values = read_json(envelope["stdout"])
        require(
            values["format"] == "final-checkpoint-small-control-capture-v1"
            and values["status"] == "EXACT_REVIEWED_FINAL_CONTROLS_CAPTURED",
            "wrong control capture",
        )
        expected = {}
        for path, names in (
            (source, ["trainer.pt", "data.pt", "model/.metadata", "optim/.metadata"]),
            (destination, ["trainer.pt", "data.pt", "optim/.metadata"]),
        ):
            for name in names:
                expected[str(path / name)] = self.inventories[str(path)][name]
        require(set(expected) <= set(values["files"]), "missing final control/metadata")
        for name, want in expected.items():
            item = values["files"][name]
            raw = base64.b64decode(item["base64"], validate=True)
            require(
                same(item["bytes"], want["bytes"])
                and len(raw) == want["bytes"] <= 1024**2
                and item["sha256"] == want["sha256"] == hashlib.sha256(raw).hexdigest(),
                "control differs from original final checkpoint hash",
            )
            self.controls[name] = raw

    def inventory(self, path):
        return self.inventories[str(path)]

    def raw(self, path):
        return self.controls[str(path)]

    def tensor_state(self, path):
        import torch

        # Pickle is admitted only after the independently reviewed original
        # inventory and exact small-control hashes have all been validated.
        return torch.load(io.BytesIO(self.raw(path)), map_location="cpu", weights_only=False)


def validate_command(command, expected):
    require(
        type(command["returncode"]) is int and command["returncode"] == 0 and same(command["command"], expected),
        "checkpoint command failed or used another source/entry point",
    )


def normalize(root, contract, cell_id, *, capture_ref, controls_ref, primary_admission_ref):
    c = Capture(root, capture_ref, cell_id)
    admission = c.json(primary_admission_ref)
    require(admission["status"] == "ACTIVE_REVIEWED_PARTIAL_FINAL_GATE_INDEX", "unadmitted primary lineage")
    old_contract, old_index = c.json(admission["contract"]), c.json(admission["index"])
    require(
        digest(old_contract) == admission["canonical_contract_sha256"] == old_index["contract_sha256"],
        "primary contract/index mismatch",
    )
    cell = next(row for row in contract["cells"] if row["id"] == cell_id)
    old_cell = next(row for row in old_contract["cells"] if row["id"] == cell_id)
    require(same(cell["identity"], old_cell["identity"]), "primary cell identity changed")
    lineage = old_index["cells"][cell_id]["lineage"]
    primary_refs = old_index["cells"][cell_id]["gates"]["full_schedule"]
    require(len(primary_refs) == 1, "ambiguous primary proof")
    original = c.json(primary_refs[0])
    checked = check_receipt(
        c,
        primary_refs[0],
        next(r for r in old_cell["requirements"] if r["id"] == "full_schedule"),
        cell["identity"],
        old_contract,
        None,
        lineage,
    )
    require(checked["outcome"] == "PASS", "full training not established")
    row = c.row
    stage = Path("build-inputs") / Path(row["root"]).name
    plan = c.reviewed({"path": str(stage / "plan.json"), "sha256": row["plan_sha256"]})
    require(
        plan["format"] == "final-text-artifacts-v1"
        and plan["cell"] == cell_id
        and plan["root"] == row["root"]
        and plan["primary_root"] == lineage["primary_root"]
        and plan["primary_plan_sha256"] == lineage["primary_plan_sha256"]
        and plan["training_revision"] == original["source"]["revision"],
        "artifact belongs to another original training lineage",
    )
    for name, value in plan["files_sha256"].items():
        require(Path(name).name == name and name not in (".", ".."), "unsafe staged source")
        c.bytes({"path": str(stage / name), "sha256": value})
    for name, value in WORKERS.items():
        require(plan["files_sha256"].get(name) == value, "unreviewed checkpoint producer schema")
    spec = c.json({"path": str(stage / "cell.json"), "sha256": cell["identity"]["config_sha256"]})
    require(
        spec["binding"]["id"] == "llama_hf"
        and spec["binding"]["dataset"] == "gsm8k"
        and digest(spec["cell"]) == cell["identity"]["inventory_cell_sha256"]
        and same(spec["cell"]["topology"], cell["identity"]["topology"]),
        "unsupported or changed full-text checkpoint binding",
    )
    producer_review = c.json(c.review["source_identity"]["producer_review"])
    require(
        producer_review["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS" and producer_review["open_findings"] == [],
        "unreviewed installed checkpoint worker",
    )
    for name, value in producer_review["reviewed_files"].items():
        c.bytes({"path": name, "sha256": value})
    for name, value in WORKERS.items():
        require(
            producer_review["reviewed_files"].get("inputs/" + name) == value, "producer review omitted executed helper"
        )
    ack = c.local_reviewed("final-smol-artifacts-v3-submission/" + cell_id + ".json")
    require(
        type(ack["returncode"]) is int and ack["returncode"] == 0 and ack["job_id"] == row["job"],
        "artifact allocation not acknowledged",
    )
    intent, receipt, consolidation = [
        c.document("output/" + name + ".json") for name in ("execution-intent", "receipt", "consolidation-result")
    ]
    require(same(intent, {"job": row["job"], "plan_sha256": row["plan_sha256"]}), "artifact execution changed job/plan")
    require(receipt["status"] == "FAIL_NUMERICAL_GATES", "unreviewed terminal outcome; preserve actual failures")
    for key, value in {
        "cell": cell_id,
        "job": row["job"],
        "plan_sha256": row["plan_sha256"],
        "training_revision": plan["training_revision"],
        "export_revision": plan["source_revision"],
    }.items():
        require(same(receipt[key], value), "artifact receipt changed " + key)
    source = {
        "revision": plan["source_revision"],
        "wheel_sha256": plan["wheel_sha256"],
        "runtime_sha256": plan["image_sha256"],
    }
    reviewed_source = c.review["source_identity"]
    require(
        source
        == {
            "revision": reviewed_source["export_and_probe_revision"],
            "wheel_sha256": reviewed_source["wheel_sha256"],
            "runtime_sha256": reviewed_source["container_sha256"],
        },
        "artifact source differs from actual review",
    )
    for name in ("provenance", "construction-provenance"):
        provenance = c.document("output/control/" + name + ".json")
        require(
            provenance["source_revision"] == source["revision"]
            and provenance["wheel_sha256"] == source["wheel_sha256"]
            and provenance["container_sha256"] == source["runtime_sha256"]
            and provenance["source_dirty"] is False
            and same(provenance["verified_package_files"], 1933)
            and same(provenance["tracked_package_files_verified"], 1933)
            and provenance["job"] == provenance["slurm_job_id"] == row["job"]
            and provenance["script_sha256"] == WORKERS["final_text_deployment_v3.py"]
            and provenance["spec_sha256"] == cell["identity"]["config_sha256"],
            "installed source/config was not attested",
        )
    requirement = next(r for r in cell["requirements"] if r["id"] == GATE)
    expected = requirement["assertions"][0]["expected"]
    require(
        requirement["profile"] is None
        and requirement["final_checkpoint"] is True
        and requirement["checkpoint_anchor"] == {"document": "measurement", "pointer": "/checkpoint"}
        and requirement["assertions"]
        == [
            {
                "document": "measurement",
                "pointer": "/checkpoint",
                "operator": "complete_checkpoint",
                "expected": expected,
            },
            {
                "document": "measurement",
                "pointer": "/consolidated_and_dcp_verified",
                "operator": "eq",
                "expected": True,
            },
        ]
        and same(
            expected,
            {"components": ["model", "optimizer", "training_state"], "update": spec["cell"]["optimizer_updates"]},
        ),
        "changed final checkpoint gate",
    )
    update = expected["update"]
    require(positive_integer(update), "invalid final update")
    checkpoint = (
        Path(lineage["primary_root"])
        / "output/primary/adapter_all_family_two_node/llama_hf"
        / cell_id
        / "campaign/checkpoints"
        / f"iter_{update:06d}"
    )
    destination = Path(plan["root"]) / "output/consolidated" / checkpoint.name
    require(
        receipt["checkpoint"] == consolidation["source"] == str(checkpoint)
        and consolidation["destination"] == str(destination),
        "wrong final checkpoint location",
    )
    require(
        consolidation["status"] == "PASS_LOSSLESS_CONSOLIDATION"
        and consolidation["all_model_optimizer_control_values_exact"] is True
        and consolidation["source_immutable"] is True
        and same(consolidation["iteration"], update)
        and receipt["immutable_sources"] is True,
        "incomplete or lossy final checkpoint lifecycle",
    )
    source_names = {"trainer.pt", "data.pt", "model/.metadata", "optim/.metadata"}
    source_names |= {f"{p}/__{r}_0.distcp" for p in ("model", "optim") for r in range(4)}
    source_inventory = inventory(consolidation["source_inventory"], source_names)
    destination_inventory = inventory(
        consolidation["destination_inventory"],
        {"trainer.pt", "data.pt", "model.pt", "optim/.metadata", "optim/__0_0.distcp"},
    )
    require(same(receipt["checkpoint_inventory"], source_inventory), "DCP inventory changed across lifecycle")
    require(source_inventory["data.pt"] == destination_inventory["data.pt"], "consolidation changed data state bytes")
    c.inventories = {str(checkpoint): source_inventory, str(destination): destination_inventory}
    commands = {
        "consolidate": [
            "/opt/campaign/runtime/bin/python",
            "/run/inputs/final_text_artifact_state_v2.py",
            "--source",
            str(checkpoint),
            "--destination",
            str(destination),
            "--out",
            str(Path(plan["root"]) / "output/consolidation-result.json"),
        ],
        "selective-extract": [
            "/opt/campaign/runtime/bin/python",
            "-m",
            "cosmos_framework.posttrain.checkpoint.tools.adapters",
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(Path(plan["root"]) / "output/checkpoint-portable"),
        ],
        "consolidated-extract": [
            "/opt/campaign/runtime/bin/python",
            "-m",
            "cosmos_framework.posttrain.checkpoint.tools.adapters",
            "--checkpoint",
            str(destination),
            "--out",
            str(Path(plan["root"]) / "output/consolidated-portable"),
        ],
    }
    for name, command in commands.items():
        measured = c.document("output/" + name + "-command.json")
        require(same(measured, receipt["commands"][name]), "command receipt is not the captured original")
        validate_command(measured, command)
    initials = []
    for rank in range(4):
        candidates = [
            ref
            for ref in original["supporting_evidence"]
            if ref["path"].endswith(
                f"/{Path(lineage['primary_root']).name}/output/primary/audit/initial-rank{rank}.json"
            )
        ]
        require(len(candidates) == 1, "missing original trainable ownership/identity")
        initials.append(c.json(candidates[0]))
    initial = initials[0]
    for value in initials:
        require(
            same(value["adapter_identity"], consolidation["adapter_identity"])
            and same(value["optimizer"], initial["optimizer"]),
            "checkpoint changed adapter/optimizer identity",
        )
    owners = [n for group in initial["optimizer"] for n in group["names"]]
    require(
        positive_integer(consolidation["optimizer_parameter_states"])
        and len(set(owners)) == len(owners) == consolidation["optimizer_parameter_states"],
        "partial or duplicate optimizer state",
    )
    c.load_controls(controls_ref, checkpoint, destination)
    _, trainer, data = check_checkpoint(c, checkpoint, initial, update, spec["cell"]["grad_accum"])
    exact_tree(c.tensor_state(destination / "trainer.pt"), trainer)
    exact_tree(c.tensor_state(destination / "data.pt"), data)
    require(bool(trainer["scheduler_identity"]), "missing scheduler identity")
    source_metadata = pickle.loads(c.raw(checkpoint / "model/.metadata"))
    require(same(len(source_metadata.state_dict_metadata), consolidation["model_tensors"]), "partial model metadata")
    original_optim = pickle.loads(c.raw(checkpoint / "optim/.metadata"))
    final_optim = pickle.loads(c.raw(destination / "optim/.metadata"))
    require(
        {name for name in original_optim.state_dict_metadata if name.startswith("state.")}
        == {f"state.{name}.{key}" for name in owners for key in ("step", "exp_avg", "exp_avg_sq")},
        "optimizer contains missing or unintended parameter state",
    )
    check_storage_metadata(final_optim, "optim", destination_inventory)
    require(
        set(original_optim.state_dict_metadata) == set(final_optim.state_dict_metadata),
        "consolidated optimizer omitted state",
    )
    for name, value in original_optim.state_dict_metadata.items():
        other = final_optim.state_dict_metadata[name]
        require(type(value) is type(other), "optimizer metadata type changed")
        if hasattr(value, "size"):
            require(
                value.size == other.size and value.properties == other.properties, "optimizer tensor metadata changed"
            )
    for name, comparison in (
        ("checkpoint-portable", "dcp_portable_exact"),
        ("consolidated-portable", "consolidated_portable_exact"),
    ):
        proof = receipt[comparison]
        require(
            proof["identity_exact"] is True
            and proof["all_tensors_exact"] is True
            and same(proof["tensor_count"], len(owners))
            and proof["direct_manifest_sha256"] == receipt["direct_inventory"]["adapter_manifest.json"]["sha256"]
            and proof["extracted_manifest_sha256"]
            == row["files"]["output/" + name + "/adapter_manifest.json"]["sha256"],
            "portable state comparison is incomplete or unbound",
        )
        manifest = c.document("output/" + name + "/adapter_manifest.json")
        require(same(manifest["identity"], consolidation["adapter_identity"]), "extraction changed adapter identity")
        require(manifest["kind"] == "portable_adapter", "incomplete extracted artifact")
    files = {name: value["sha256"] for name, value in source_inventory.items()}
    measurement = {
        "checkpoint": {
            "committed": True,
            "validated_by_framework": True,
            "update": update,
            "components": expected["components"],
            "files": files,
            "inventory_sha256": digest(files),
            "identity": cell["identity"],
            "lineage": lineage,
        },
        "consolidated_and_dcp_verified": True,
        "verification": {
            "original_training_source": original["source"],
            "verification_source": source,
            "checkpoint": str(checkpoint),
            "consolidated": str(destination),
            "source_inventory": source_inventory,
            "destination_inventory": destination_inventory,
            "model_tensors": consolidation["model_tensors"],
            "optimizer_parameter_states": len(owners),
            "local_control_and_metadata_verification": True,
            "framework_full_state_comparison": "existing installed producer, exact dtype/schema/value comparison",
            "legacy_job_outcome": receipt["status"],
            "legacy_probe_outcomes": receipt["probes"],
            "artifact_job": row["job"],
            "artifact_plan_sha256": row["plan_sha256"],
            "scope": "Final checkpoint only. Full model/optimizer values were compared by the original source-pinned installed worker; this CPU producer rereads small controls/metadata, not full shards. Numerical failures, warm reload, task and whole-cell qualification remain separate.",
        },
    }
    supporting = [{"path": name, "sha256": sha} for name, sha in c.verified.items()]
    return measurement, source, lineage, supporting


def produce(root, contract, output, *, cell_id, capture_ref, controls_ref, primary_admission_ref, producer_review):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root) and not output.exists(), "fresh output directory required")
    require(contract.get("synthetic") is False, "synthetic proof is not production evidence")
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
    for ref in (capture_ref, controls_ref, primary_admission_ref):
        require(
            review["reviewed_files"].get(ref["path"]) == ref["sha256"], "unreviewed actual capture/primary admission"
        )
    producer = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cell = next(row for row in contract["cells"] if row["id"] == cell_id)
    requirement = next(r for r in cell["requirements"] if r["id"] == GATE)
    require(
        producer in requirement["producers"] and producer_review in producer_reviews(requirement, producer),
        "producer requires exact independent admission",
    )
    measurement, source, lineage, support = normalize(
        root,
        contract,
        cell_id,
        capture_ref=capture_ref,
        controls_ref=controls_ref,
        primary_admission_ref=primary_admission_ref,
    )
    require(source in requirement["sources"], "installed verification revision needs explicit source admission")
    support = list({ref["path"]: ref for ref in [*support, producer_review]}.values())
    for ref in support:
        store.bytes(ref)
    directory = output / cell_id / GATE
    m = write(root, directory / "measurement.json", measurement)
    checkpoint = measurement["checkpoint"]["inventory_sha256"]
    common = {
        "identity": cell["identity"],
        "source": source,
        "profile": None,
        "producer_sha256": producer,
        "lineage": lineage,
        "final_checkpoint": checkpoint,
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
                "final_checkpoint": checkpoint,
                "gates": {GATE: [write(root, directory / "receipt.json", receipt)]},
            }
        },
    }
    write(root, output / "index.json", index)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "contract", "output", "cell-id", "capture", "controls", "primary-admission", "producer-review"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    produce(
        root,
        read_json(Path(args.contract).read_bytes()),
        args.output,
        cell_id=args.cell_id,
        capture_ref=file_ref(root, Path(args.capture).resolve()),
        controls_ref=file_ref(root, Path(args.controls).resolve()),
        primary_admission_ref=file_ref(root, Path(args.primary_admission).resolve()),
        producer_review=file_ref(root, Path(args.producer_review).resolve()),
    )
