# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Fail-closed aggregation of hash-bound campaign gate receipts.

The sealed contract owns required gates, profiles, numerical assertions and
admitted producers. This verifier never infers success from a job exit code or
a training-complete flag. It does not execute GPU tests: measurements must come
from the independently reviewed, source-pinned producers named in the contract.
Historical raw receipts remain immutable; adapters to their schemas must also be
pinned producers. Synthetic proof sets require an explicitly synthetic contract
and cannot be accepted by a production invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def same(actual, expected):
    """JSON equality must distinguish booleans from numbers at every depth."""
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(same(actual[k], expected[k]) for k in actual)
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(same(a, b) for a, b in zip(actual, expected, strict=True))
    return actual == expected


def producer_reviews(requirement, producer):
    """Optional exact review anchors owned by the independently sealed contract."""
    if "producer_reviews" not in requirement:
        return []
    mapping = requirement["producer_reviews"]
    require(isinstance(mapping, dict) and mapping, "empty producer review admissions")
    require(set(mapping) == set(requirement["producers"]), "reviews must cover exactly the admitted producers")
    for key, references in mapping.items():
        require(re.fullmatch(r"[0-9a-f]{64}", key) is not None, "invalid reviewed producer digest")
        require(isinstance(references, list) and references, "producer has no admitted review")
        require(len({digest(ref) for ref in references}) == len(references), "duplicate admitted review")
        for ref in references:
            require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "invalid admitted review reference")
            require(isinstance(ref["path"], str) and isinstance(ref["sha256"], str), "invalid review path/digest")
            path = Path(ref["path"])
            require(not path.is_absolute() and ".." not in path.parts, "review path escapes evidence root")
            require(re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]) is not None, "invalid review digest")
    require(producer in mapping, "unadmitted review producer")
    return mapping[producer]


def _pairs(pairs):
    result = {}
    for name, value in pairs:
        require(name not in result, f"duplicate JSON key: {name}")
        result[name] = value
    return result


def read_json(data):
    return json.loads(data, object_pairs_hook=_pairs, parse_constant=lambda x: (_ for _ in ()).throw(EvidenceError(x)))


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.verified = {}

    def bytes(self, reference):
        require(set(reference) == {"path", "sha256"}, "a file reference requires only path and sha256")
        require(isinstance(reference["path"], str), "file path must be a string")
        name = Path(reference["path"])
        require(not name.is_absolute() and ".." not in name.parts, "evidence must be rooted relative paths")
        require(re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]) is not None, "invalid evidence digest")
        path = (self.root / name).resolve()
        require(path.is_relative_to(self.root) and path.is_file(), f"missing or escaping evidence: {name}")
        require(
            path.stat().st_size <= 64 * 1024**2, "receipt exceeds 64 MiB; use a reviewed streaming inventory producer"
        )
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == reference["sha256"], f"stale evidence hash: {name}")
        self.verified[str(name)] = reference["sha256"]
        return data

    def json(self, reference):
        return read_json(self.bytes(reference))


def pointer(value, path):
    require(isinstance(path, str) and (path == "" or path.startswith("/")), "invalid JSON pointer")
    if not path:
        return value
    for part in path[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            require(part.isdigit() and int(part) < len(value), f"missing list item: {path}")
            value = value[int(part)]
        else:
            require(isinstance(value, dict) and part in value, f"missing evidence field: {path}")
            value = value[part]
    return value


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), "metric must be finite numeric data")
    return value


def predicate(clause, documents):
    """Assertions are fixed by the contract, never supplied by result writers."""
    require(set(clause) == {"document", "pointer", "operator", "expected"}, "invalid assertion fields")
    require(clause["document"] in documents, "assertion names an absent document")
    actual = pointer(documents[clause["document"]], clause["pointer"])
    expected, operator = clause["expected"], clause["operator"]
    if operator == "eq":
        return same(actual, expected)
    if operator == "le":
        return number(actual) <= number(expected)
    if operator == "ge":
        return number(actual) >= number(expected)
    if operator == "unique_ids":
        require(isinstance(actual, list) and all(isinstance(x, str) for x in actual), "IDs must be strings")
        require(isinstance(expected, dict) and set(expected) == {"count", "sorted_ids_sha256"}, "invalid ID census")
        require(type(expected["count"]) is int and expected["count"] >= 0, "ID count must be a nonnegative integer")
        return (
            len(actual) == len(set(actual)) == expected["count"]
            and digest(sorted(actual)) == expected["sorted_ids_sha256"]
        )
    if operator == "complete_checkpoint":
        require(isinstance(actual, dict), "checkpoint proof must be a mapping")
        require(isinstance(expected, dict) and set(expected) == {"components", "update"}, "invalid checkpoint contract")
        files = actual.get("files", {})
        return (
            actual.get("committed") is True
            and same(actual.get("update"), expected["update"])
            and sorted(actual.get("components", [])) == sorted(expected["components"])
            and isinstance(files, dict)
            and bool(files)
            and all(
                isinstance(k, str)
                and not k.endswith((".partial", ".incomplete"))
                and isinstance(v, str)
                and re.fullmatch(r"[0-9a-f]{64}", v)
                for k, v in files.items()
            )
            and actual.get("inventory_sha256") == digest(files)
        )
    raise EvidenceError(f"unknown assertion operator: {operator}")


def source_admitted(store, receipt, requirement, identity, compatibility):
    source = receipt["source"]
    require(
        isinstance(source, dict) and set(source) == {"revision", "wheel_sha256", "runtime_sha256"},
        "incomplete source identity",
    )
    require(
        isinstance(source["revision"], str) and re.fullmatch(r"[0-9a-f]{40}", source["revision"]),
        "invalid source revision",
    )
    require(
        all(
            isinstance(source[k], str) and re.fullmatch(r"[0-9a-f]{64}", source[k])
            for k in ("wheel_sha256", "runtime_sha256")
        ),
        "invalid source digest",
    )
    if source in requirement["sources"]:
        require(receipt.get("compatibility") is None, "unexpected compatibility override")
        return
    reference = receipt.get("compatibility")
    require(reference is not None, "later execution requires an explicit reviewed compatibility receipt")
    proof = store.json(reference)
    require(reference["sha256"] in compatibility, "compatibility proof has not been independently admitted")
    require(proof.get("format") == "adapter-campaign-source-compatibility-v1", "wrong compatibility format")
    require(proof.get("identity_sha256") == digest(identity), "wrong compatibility subject")
    require(
        proof.get("old_source") in requirement["sources"] and proof.get("new_source") == source,
        "wrong compatibility revisions",
    )
    require(requirement["id"] in proof.get("gates", []), "compatibility does not cover this gate")
    require(
        proof.get("requirements_sha256", {}).get(requirement["id"]) == digest(requirement),
        "compatibility does not cover this exact profile and gate contract",
    )
    require(
        proof.get("status") == "PASS" and proof.get("preserves_historical_attribution") is True,
        "unqualified compatibility",
    )
    require(proof.get("validation") and proof.get("review"), "compatibility needs validation and independent review")
    for ref in [*proof["validation"], proof["review"]]:
        store.bytes(ref)


def check_receipt(store, reference, requirement, identity, contract, final_checkpoint, lineage=None):
    receipt = store.json(reference)
    require(receipt.get("format") == "adapter-campaign-gate-v1", "unsupported gate receipt")
    require(receipt.get("synthetic") is contract["synthetic"], "synthetic evidence cannot enter a real campaign")
    require(same(receipt.get("identity"), identity), "wrong cell/model/data/config/processor/topology identity")
    require(receipt.get("gate") == requirement["id"], "wrong gate")
    require(same(receipt.get("profile"), requirement["profile"]), "wrong loader/profile/precision")
    require(receipt.get("producer_sha256") in requirement["producers"], "unreviewed receipt producer")
    unexecuted = receipt.get("outcome") in {"BLOCKED", "NOT_EXECUTED_UPSTREAM_FAILURE"}
    require(unexecuted or not requirement.get("pending_admission"), "required execution admission is pending")
    if requirement.get("training_lineage") and not (unexecuted and receipt.get("lineage") is None):
        require(
            lineage is not None and same(receipt.get("lineage"), lineage),
            "wrong or missing authoritative training lineage",
        )
    elif not requirement.get("training_lineage"):
        require(receipt.get("lineage") is None, "prerequisite evidence cannot claim authoritative primary lineage")
    if not unexecuted or receipt.get("source") is not None:
        source_admitted(store, receipt, requirement, identity, contract.get("compatibility_receipts", []))
    if requirement["final_checkpoint"] and not (unexecuted and receipt.get("final_checkpoint") is None):
        require(
            final_checkpoint is not None and receipt.get("final_checkpoint") == final_checkpoint,
            "wrong or missing final checkpoint",
        )
    elif not requirement["final_checkpoint"]:
        require(receipt.get("final_checkpoint") is None, "unexpected final-checkpoint claim")
    require(
        receipt.get("outcome") in {"PASS", "FAIL", "BLOCKED", "NOT_EXECUTED_UPSTREAM_FAILURE"}, "invalid gate outcome"
    )
    if receipt["outcome"] in {"BLOCKED", "NOT_EXECUTED_UPSTREAM_FAILURE"}:
        require(
            receipt.get("executed") is False and isinstance(receipt.get("reason"), str) and receipt["reason"],
            "blocked evidence needs a concrete reason",
        )
        require(
            receipt.get("blocker_kind") in {"external", "dependency", "implementation"},
            "blocker needs an explicit classification",
        )
        require(receipt.get("blocker_evidence"), "blocked claim requires evidence")
        require(not receipt.get("documents"), "unexecuted gate cannot claim measurement documents")
        for ref in receipt["blocker_evidence"]:
            store.bytes(ref)
        return {
            "outcome": receipt["outcome"],
            "executed": False,
            "receipt": reference,
            "source": receipt["source"],
            "reason": receipt["reason"],
            "blocker_kind": receipt["blocker_kind"],
        }
    require(receipt.get("executed") is True, "unexecuted checks cannot claim a numerical outcome")
    require(isinstance(receipt.get("documents"), dict), "missing raw evidence documents")
    require(set(receipt["documents"]) == set(requirement["documents"]), "missing or unexpected raw documents")
    documents = {name: store.json(ref) for name, ref in receipt["documents"].items()}
    # A producer is trusted only for the exact schema admitted in the reviewed
    # contract. Its normalized provenance MUST bind the raw measurement files,
    # cell, installed source and checkpoint; an exit code alone never suffices.
    provenance = documents[requirement["provenance_document"]]
    require(same(provenance["identity"], identity), "raw provenance belongs to another cell")
    require(same(provenance["source"], receipt["source"]), "raw provenance source mismatch")
    require(same(provenance["profile"], requirement["profile"]), "raw provenance profile mismatch")
    require(provenance["producer_sha256"] == receipt["producer_sha256"], "raw producer mismatch")
    require(provenance["final_checkpoint"] == receipt.get("final_checkpoint"), "raw checkpoint mismatch")
    require(
        same(provenance.get("lineage"), receipt.get("lineage")), "raw execution belongs to another training lineage"
    )
    bound = {
        name: ref["sha256"] for name, ref in receipt["documents"].items() if name != requirement["provenance_document"]
    }
    require(same(provenance["documents_sha256"], bound), "raw documents are not bound to the execution provenance")
    supporting = receipt.get("supporting_evidence", [])
    require(isinstance(supporting, list), "raw evidence references must be a list")
    if requirement.get("raw_evidence_required"):
        require(supporting, "required raw evidence is absent")
        require(
            same(provenance.get("supporting_evidence"), supporting),
            "raw evidence references are not bound to provenance",
        )
    for ref in supporting:
        store.bytes(ref)
    reviews = producer_reviews(requirement, receipt["producer_sha256"])
    if reviews:
        require(any(ref in supporting for ref in reviews), "receipt omits its independently admitted review")
        require(same(provenance.get("supporting_evidence"), supporting), "review is not bound to raw provenance")
    checks = [predicate(clause, documents) for clause in requirement["assertions"]]
    require(checks, "a required gate cannot pass without contract assertions")
    observed = "PASS" if all(checks) else "FAIL"
    require(receipt["outcome"] == observed, "reported outcome contradicts measured assertions")
    if requirement.get("checkpoint_anchor") and observed == "PASS":
        checkpoint = pointer(
            documents[requirement["checkpoint_anchor"]["document"]], requirement["checkpoint_anchor"]["pointer"]
        )
        require(
            final_checkpoint is not None and checkpoint["inventory_sha256"] == final_checkpoint,
            "final checkpoint is not anchored to its verified committed inventory",
        )
        require(same(checkpoint["identity"], identity), "checkpoint identity mismatch")
        if requirement.get("training_lineage"):
            require(same(checkpoint.get("lineage"), lineage), "checkpoint belongs to another primary run")
        require(
            checkpoint["committed"] is True and checkpoint["validated_by_framework"] is True,
            "checkpoint completeness was not verified through the framework",
        )
    return {
        "outcome": observed,
        "executed": True,
        "receipt": reference,
        "source": receipt["source"],
        "assertions": checks,
    }


def resolve_attempts(store, checked, requirement, identity, contract, supersession):
    """Keep all history; replacing a failed result needs a separately reviewed seal."""
    if len({r["outcome"] for r in checked}) == 1 and supersession is None:
        return {**checked[0], "attempts": checked}
    require(supersession is not None, "contradictory receipts require explicit reviewed supersession")
    require(supersession["sha256"] in contract.get("supersession_receipts", []), "unreviewed supersession")
    proof = store.json(supersession)
    require(proof.get("format") == "adapter-campaign-gate-supersession-v1", "wrong supersession format")
    require(
        same(proof.get("identity"), identity) and proof.get("gate") == requirement["id"], "wrong supersession subject"
    )
    require(same(proof.get("profile"), requirement["profile"]), "supersession cannot change execution semantics")
    refs = [c["receipt"] for c in checked]
    require(same(proof.get("attempts"), refs), "supersession must cover every preserved attempt")
    require(
        proof.get("selected") in refs and proof.get("reason") and proof.get("review") and proof.get("validation"),
        "incomplete supersession",
    )
    for ref in [proof["review"], *proof["validation"]]:
        store.bytes(ref)
    selected = next(c for c in checked if c["receipt"] == proof["selected"])
    return {**selected, "attempts": checked, "supersession": supersession}


def aggregate(contract, index, root, *, allow_synthetic=False, expected_contract_sha256=None):
    require(contract.get("format") == "adapter-campaign-completion-contract-v1", "wrong completion contract")
    require(type(contract.get("synthetic")) is bool, "contract must explicitly declare synthetic scope")
    require(not contract["synthetic"] or allow_synthetic, "synthetic contract requires explicit test-only opt-in")
    require(index.get("format") == "adapter-campaign-evidence-index-v1", "wrong evidence index")
    require(index.get("contract_sha256") == digest(contract), "evidence index belongs to another contract")
    require(
        contract["synthetic"] or expected_contract_sha256 == digest(contract),
        "production requires the externally reviewed contract digest",
    )
    store = Store(root)
    inventory = store.json(contract["inventory"])
    require(isinstance(inventory.get("cells"), list), "sealed inventory is missing cells")
    for reference in contract["sealed_inputs"]:
        store.bytes(reference)
    cells = contract["cells"]
    require(
        len({c["id"] for c in cells}) == len(cells) == contract["required_cells"],
        "incomplete or duplicate cell inventory",
    )
    require(
        inventory["total_required_cells"] == len(cells)
        and {c["id"] for c in inventory["cells"]} == {c["id"] for c in cells},
        "completion contract does not cover the sealed inventory",
    )
    original_cells = {c["id"]: c for c in inventory["cells"]}
    require(len(original_cells) == len(cells), "sealed inventory has duplicate cells")
    require(set(index["cells"]) <= {c["id"] for c in cells}, "evidence names an unknown cell")
    results = []
    for cell in cells:
        require(
            cell["identity"].get("inventory_cell_sha256") == digest(original_cells[cell["id"]]),
            "cell differs from frozen inventory",
        )
        requirements = cell["requirements"]
        gates = {g["id"]: g for g in requirements}
        require(len(gates) == len(requirements) and gates, "empty or duplicate required gates")
        require(all(g["admission"] == "REQUIRED" for g in requirements), "required gate admission cannot be inferred")
        require(
            all(
                x["admission"] in {"NOT_APPLICABLE", "EXPLICITLY_UNSUPPORTED_ADDITIONAL_PROFILE"}
                and x.get("reason")
                and x.get("contract_pointer")
                for x in cell["exclusions"]
            ),
            "unjustified scope exclusion",
        )
        evidence = index["cells"].get(cell["id"], {})
        require(
            set(evidence) <= {"gates", "final_checkpoint", "supersessions", "lineage"}, "unknown cell evidence fields"
        )
        provided = evidence.get("gates", {})
        require(set(provided) <= set(gates), "unknown gate evidence")
        checkpoint = evidence.get("final_checkpoint")
        lineage = evidence.get("lineage")
        if lineage is not None:
            require(
                isinstance(lineage, dict) and set(lineage) == {"primary_plan_sha256", "primary_job", "primary_root"},
                "invalid authoritative training lineage",
            )
            require(
                isinstance(lineage["primary_plan_sha256"], str)
                and re.fullmatch(r"[0-9a-f]{64}", lineage["primary_plan_sha256"]),
                "invalid primary plan identity",
            )
            require(
                all(isinstance(lineage[k], str) and lineage[k] for k in ("primary_job", "primary_root")),
                "incomplete primary execution identity",
            )
        if checkpoint is not None:
            require(
                isinstance(checkpoint, str) and re.fullmatch(r"[0-9a-f]{64}", checkpoint),
                "invalid final checkpoint identity",
            )
        supersessions = evidence.get("supersessions", {})
        require(set(supersessions) <= set(gates), "supersession names an unknown gate")
        require(
            sum(bool(g.get("checkpoint_anchor")) for g in requirements) == 1,
            "exactly one required final checkpoint anchor is needed",
        )
        outcomes = {}
        for gate_id, requirement in gates.items():
            refs = provided.get(gate_id, [])
            require(isinstance(refs, list), "gate receipts must be an explicit attempt list")
            if not refs:
                outcomes[gate_id] = {"outcome": "MISSING", "executed": False}
                continue
            checked = [
                check_receipt(store, ref, requirement, cell["identity"], contract, checkpoint, lineage) for ref in refs
            ]
            outcomes[gate_id] = resolve_attempts(
                store, checked, requirement, cell["identity"], contract, supersessions.get(gate_id)
            )
        values = list(outcomes.values())
        trained = all(outcomes.get(g, {}).get("outcome") == "PASS" for g in cell["training_gates"])
        require(
            cell["training_gates"] and set(cell["training_gates"]) <= set(gates), "invalid training gate definition"
        )
        passed = all(r["outcome"] == "PASS" for r in values)
        status = "PASS" if passed else "FAIL" if any(r["outcome"] == "FAIL" for r in values) else "BLOCKED"
        results.append(
            {
                "cell_id": cell["id"],
                "status": status,
                "fully_qualified": passed,
                "full_training_completed": trained,
                "all_required_checks_executed": all(r["executed"] for r in values),
                "checks_executed": sum(r["executed"] for r in values),
                "failures": [g for g, r in outcomes.items() if r["outcome"] == "FAIL"],
                "external_blockers": [g for g, r in outcomes.items() if r.get("blocker_kind") == "external"],
                "dependency_blockers": [g for g, r in outcomes.items() if r.get("blocker_kind") == "dependency"],
                "implementation_blockers": [
                    g for g, r in outcomes.items() if r.get("blocker_kind") == "implementation"
                ],
                "missing": [g for g, r in outcomes.items() if r["outcome"] == "MISSING"],
                "gates": outcomes,
            }
        )
    return {
        "format": "adapter-campaign-completion-v1",
        "synthetic": contract["synthetic"],
        "contract_sha256": digest(contract),
        "index_sha256": digest(index),
        "cells": results,
        "summary": {
            "required_cells": len(results),
            "checks_executed": sum(r["checks_executed"] for r in results),
            "full_training_completed": sum(r["full_training_completed"] for r in results),
            "all_required_checks_executed": sum(r["all_required_checks_executed"] for r in results),
            "fully_passing_qualification": sum(r["fully_qualified"] for r in results),
            "cells_with_failures": sum(bool(r["failures"]) for r in results),
            "cells_with_external_blockers": sum(bool(r["external_blockers"]) for r in results),
            "cells_with_dependency_blockers": sum(bool(r["dependency_blockers"]) for r in results),
            "cells_with_implementation_blockers": sum(bool(r["implementation_blockers"]) for r in results),
            "states": dict(Counter(r["status"] for r in results)),
        },
        "verified_files": store.verified,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True, help="Externally reviewed canonical contract digest")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(
        read_json(args.contract.read_bytes()),
        read_json(args.index.read_bytes()),
        args.root,
        expected_contract_sha256=args.contract_sha256,
    )
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
