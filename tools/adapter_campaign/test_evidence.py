# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Synthetic receipt sets only: none of these fixtures are campaign evidence."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.adapter_campaign.evidence import EvidenceError, aggregate, digest, read_json, same


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = {"revision": "1" * 40, "wheel_sha256": "2" * 64, "runtime_sha256": "3" * 64}
        inventory = {"total_required_cells": 1, "cells": [{"id": "synthetic", "seed": 0, "epochs": 1}]}
        self.identity = {
            "cell": "synthetic",
            "model": "4" * 64,
            "data": "5" * 64,
            "config": "6" * 64,
            "processor": "7" * 64,
            "topology": {"hosts": 2, "mesh": [2, 2]},
            "inventory_cell_sha256": digest(inventory["cells"][0]),
        }
        self.profile = {"loader": "synthetic-only", "execution": "factorized_v1", "dtype": "float32"}
        files = {"model/__0.distcp": "8" * 64, "trainer.pt": "9" * 64, "optimizer/__0.distcp": "a" * 64}
        self.checkpoint = {
            "committed": True,
            "validated_by_framework": True,
            "update": 3,
            "identity": self.identity,
            "components": ["model", "optimizer", "trainer", "rng", "data"],
            "files": files,
            "inventory_sha256": digest(files),
        }
        requirements = [self.requirement("training"), self.requirement("checkpoint"), self.requirement("task")]
        requirements[1]["checkpoint_anchor"] = {"document": "measurements", "pointer": "/checkpoint"}
        requirements[1]["assertions"] = [
            {
                "document": "measurements",
                "pointer": "/checkpoint",
                "operator": "complete_checkpoint",
                "expected": {"update": 3, "components": self.checkpoint["components"]},
            }
        ]
        requirements[2]["assertions"] = [
            {
                "document": "measurements",
                "pointer": "/ids",
                "operator": "unique_ids",
                "expected": {"count": 3, "sorted_ids_sha256": digest(["a", "b", "c"])},
            }
        ]
        self.contract = {
            "format": "adapter-campaign-completion-contract-v1",
            "synthetic": True,
            "inventory": self.write("inventory.json", inventory),
            "sealed_inputs": [],
            "required_cells": 1,
            "cells": [
                {
                    "id": "synthetic",
                    "identity": self.identity,
                    "requirements": requirements,
                    "exclusions": [],
                    "training_gates": ["training"],
                }
            ],
        }
        self.index = {
            "format": "adapter-campaign-evidence-index-v1",
            "contract_sha256": digest(self.contract),
            "cells": {"synthetic": {"final_checkpoint": digest(files), "gates": {}}},
        }
        self.add("training", {"loss": 0.1})
        self.add("checkpoint", {"checkpoint": self.checkpoint})
        self.add("task", {"ids": ["b", "c", "a"]})

    def requirement(self, name):
        return {
            "id": name,
            "admission": "REQUIRED",
            "profile": self.profile,
            "sources": [self.source],
            "producers": ["b" * 64],
            "final_checkpoint": name != "training",
            "documents": ["provenance", "measurements"],
            "provenance_document": "provenance",
            "assertions": [{"document": "measurements", "pointer": "/loss", "operator": "le", "expected": 0.2}],
        }

    def write(self, name, value):
        data = (json.dumps(value, sort_keys=True) + "\n").encode()
        (self.root / name).write_bytes(data)
        return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}

    def add(self, name, measurements, *, outcome="PASS", suffix="", source=None, lineage=None):
        source = source or self.source
        final = self.index["cells"]["synthetic"]["final_checkpoint"] if name != "training" else None
        measured = self.write(name + suffix + "-measurements.json", measurements)
        provenance = {
            "identity": self.identity,
            "source": source,
            "profile": self.profile,
            "producer_sha256": "b" * 64,
            "final_checkpoint": final,
            "lineage": lineage,
            "documents_sha256": {"measurements": measured["sha256"]},
        }
        receipt = {
            "format": "adapter-campaign-gate-v1",
            "synthetic": True,
            "gate": name,
            "identity": self.identity,
            "source": source,
            "producer_sha256": "b" * 64,
            "profile": self.profile,
            "outcome": outcome,
            "executed": True,
            "final_checkpoint": final,
            "lineage": lineage,
            "documents": {
                "measurements": measured,
                "provenance": self.write(name + suffix + "-provenance.json", provenance),
            },
        }
        reference = self.write(name + suffix + ".json", receipt)
        self.index["cells"]["synthetic"]["gates"].setdefault(name, []).append(reference)
        return reference

    def result(self):
        return aggregate(self.contract, self.index, self.root, allow_synthetic=True)

    def change_receipt(self, name, change):
        refs = self.index["cells"]["synthetic"]["gates"][name]
        value = read_json((self.root / refs[0]["path"]).read_bytes())
        change(value)
        refs[0] = self.write(refs[0]["path"], value)

    def test_complete_synthetic_set_traverses_success(self):
        result = self.result()
        self.assertEqual(result["summary"]["fully_passing_qualification"], 1)
        self.assertTrue(result["synthetic"])
        self.assertEqual(result["summary"]["all_required_checks_executed"], 1)

    def test_synthetic_is_rejected_by_production_entrypoint(self):
        with self.assertRaisesRegex(EvidenceError, "test-only"):
            aggregate(self.contract, self.index, self.root)

    def test_production_requires_reviewed_contract_digest(self):
        self.contract["synthetic"] = False
        self.index["contract_sha256"] = digest(self.contract)
        with self.assertRaisesRegex(EvidenceError, "externally reviewed"):
            self.result()

    def test_missing_gate_cannot_pass(self):
        del self.index["cells"]["synthetic"]["gates"]["task"]
        result = self.result()
        self.assertEqual(result["summary"]["fully_passing_qualification"], 0)
        self.assertEqual(result["summary"]["full_training_completed"], 1)
        self.assertFalse(result["cells"][0]["all_required_checks_executed"])

    def test_missing_file(self):
        (self.root / "task-measurements.json").unlink()
        with self.assertRaisesRegex(EvidenceError, "missing"):
            self.result()

    def test_stale_hash(self):
        (self.root / "task-measurements.json").write_text("{}")
        with self.assertRaisesRegex(EvidenceError, "stale"):
            self.result()

    def test_wrong_identity_and_nested_bool_are_rejected(self):
        self.change_receipt("task", lambda r: r["identity"]["topology"]["mesh"].__setitem__(0, True))
        with self.assertRaisesRegex(EvidenceError, "identity"):
            self.result()
        self.assertFalse(same({"epoch": 1, "ranks": [0, 1]}, {"epoch": True, "ranks": [False, True]}))

    def test_wrong_profile(self):
        self.change_receipt("task", lambda r: r["profile"].update(execution="merged_weight_bf16_v1"))
        with self.assertRaisesRegex(EvidenceError, "profile"):
            self.result()

    def test_producer_not_admitted(self):
        self.change_receipt("task", lambda r: r.update(producer_sha256="c" * 64))
        with self.assertRaisesRegex(EvidenceError, "producer"):
            self.result()

    def test_wrong_final_checkpoint(self):
        self.change_receipt("task", lambda r: r.update(final_checkpoint="c" * 64))
        with self.assertRaisesRegex(EvidenceError, "checkpoint"):
            self.result()

    def test_partial_checkpoint_is_a_failure_not_success(self):
        self.index["cells"]["synthetic"]["gates"]["checkpoint"] = []
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint["committed"] = False
        self.add("checkpoint", {"checkpoint": checkpoint}, outcome="FAIL")
        result = self.result()
        self.assertEqual(result["summary"]["fully_passing_qualification"], 0)
        self.assertEqual(result["summary"]["all_required_checks_executed"], 1)
        self.assertEqual(result["cells"][0]["failures"], ["checkpoint"])

    def test_checkpoint_anchor_cannot_be_arbitrary_digest(self):
        refs = self.index["cells"]["synthetic"]["gates"]
        refs["checkpoint"] = []
        bad = copy.deepcopy(self.checkpoint)
        bad["files"]["trainer.pt"] = "c" * 64
        bad["inventory_sha256"] = digest(bad["files"])
        self.add("checkpoint", {"checkpoint": bad})
        with self.assertRaisesRegex(EvidenceError, "anchored"):
            self.result()

    def test_incomplete_and_duplicate_questions_fail(self):
        for ids in (["a", "b"], ["a", "a", "c"], ["a", "b", "d"]):
            with self.subTest(ids=ids):
                self.index["cells"]["synthetic"]["gates"]["task"] = []
                self.add("task", {"ids": ids}, outcome="FAIL")
                self.assertEqual(self.result()["cells"][0]["status"], "FAIL")

    def test_contradictory_outcome_fails_closed(self):
        self.index["cells"]["synthetic"]["gates"]["task"] = []
        self.add("task", {"ids": ["a"]}, outcome="PASS")
        with self.assertRaisesRegex(EvidenceError, "contradicts"):
            self.result()

    def test_raw_source_mismatch(self):
        ref = self.index["cells"]["synthetic"]["gates"]["task"][0]
        receipt = read_json((self.root / ref["path"]).read_bytes())
        raw = read_json((self.root / receipt["documents"]["provenance"]["path"]).read_bytes())
        raw["source"]["revision"] = "c" * 40
        receipt["documents"]["provenance"] = self.write("task-provenance.json", raw)
        self.index["cells"]["synthetic"]["gates"]["task"] = [self.write("task.json", receipt)]
        with self.assertRaisesRegex(EvidenceError, "source mismatch"):
            self.result()

    def test_foreign_source_needs_reviewed_compatibility(self):
        self.change_receipt("task", lambda r: r["source"].update(revision="c" * 40))
        with self.assertRaisesRegex(EvidenceError, "compatibility receipt"):
            self.result()

    def test_inventory_cell_omission_rejected(self):
        self.contract["cells"] = []
        self.contract["required_cells"] = 0
        self.index["contract_sha256"] = digest(self.contract)
        with self.assertRaisesRegex(EvidenceError, "sealed inventory"):
            self.result()

    def test_changing_frozen_config_rejected(self):
        self.contract["cells"][0]["identity"]["inventory_cell_sha256"] = "c" * 64
        self.index["contract_sha256"] = digest(self.contract)
        with self.assertRaisesRegex(EvidenceError, "frozen inventory"):
            self.result()

    def test_conflicting_attempts_require_explicit_supersession(self):
        self.add("task", {"ids": ["a"]}, outcome="FAIL", suffix="-failed")
        with self.assertRaisesRegex(EvidenceError, "supersession"):
            self.result()

    def test_reviewed_supersession_keeps_old_failure(self):
        failed = self.add("task", {"ids": ["a"]}, outcome="FAIL", suffix="-failed")
        refs = self.index["cells"]["synthetic"]["gates"]["task"]
        proof = {
            "format": "adapter-campaign-gate-supersession-v1",
            "identity": self.identity,
            "gate": "task",
            "profile": self.profile,
            "attempts": refs,
            "selected": refs[0],
            "reason": "Synthetic retry after infrastructure loss, preserving original failure",
            "review": self.write("review.json", {"synthetic": True}),
            "validation": [self.write("validation.json", {"synthetic": True})],
        }
        ref = self.write("supersession.json", proof)
        self.contract["supersession_receipts"] = [ref["sha256"]]
        self.index["contract_sha256"] = digest(self.contract)
        self.index["cells"]["synthetic"]["supersessions"] = {"task": ref}
        result = self.result()
        self.assertEqual(result["summary"]["fully_passing_qualification"], 1)
        self.assertEqual(result["cells"][0]["gates"]["task"]["attempts"][1]["receipt"], failed)

    def test_duplicate_json_keys_and_nonfinite(self):
        for data in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.subTest(data=data), self.assertRaises(EvidenceError):
                read_json(data)

    def test_external_blocker_without_invented_training_environment(self):
        def blocked(receipt):
            receipt.update(
                outcome="BLOCKED",
                executed=False,
                source=None,
                final_checkpoint=None,
                documents={},
                reason="Synthetic absent dataset",
                blocker_kind="external",
                blocker_evidence=[self.write("missing-assets.json", {"synthetic": True, "missing": ["scene"]})],
            )

        self.change_receipt("task", blocked)
        result = self.result()
        self.assertEqual(result["cells"][0]["external_blockers"], ["task"])
        self.assertEqual(result["summary"]["fully_passing_qualification"], 0)
        self.assertEqual(result["summary"]["full_training_completed"], 1)

    def test_compatible_later_source_preserves_original_attribution(self):
        newer = {**self.source, "revision": "c" * 40, "wheel_sha256": "d" * 64}
        self.index["cells"]["synthetic"]["gates"]["task"] = []
        ref = self.add("task", {"ids": ["a", "b", "c"]}, source=newer)
        proof = {
            "format": "adapter-campaign-source-compatibility-v1",
            "identity_sha256": digest(self.identity),
            "old_source": self.source,
            "new_source": newer,
            "gates": ["task"],
            "requirements_sha256": {"task": digest(self.contract["cells"][0]["requirements"][2])},
            "status": "PASS",
            "preserves_historical_attribution": True,
            "review": self.write("compat-review.json", {"synthetic": True}),
            "validation": [self.write("compat-checks.json", {"synthetic": True})],
        }
        compatibility = self.write("compatibility.json", proof)
        self.contract["compatibility_receipts"] = [compatibility["sha256"]]
        self.index["contract_sha256"] = digest(self.contract)
        self.change_receipt("task", lambda r: r.update(compatibility=compatibility))
        result = self.result()
        self.assertEqual(result["summary"]["fully_passing_qualification"], 1)
        self.assertEqual(result["cells"][0]["gates"]["training"]["source"], self.source)
        self.assertEqual(result["cells"][0]["gates"]["task"]["source"], newer)

    def test_measurement_substitution_without_provenance_update(self):
        replacement = self.write("unrelated-task.json", {"ids": ["a", "b", "c"]})
        # Change bytes while retaining the same apparent scores/counts.
        replacement = self.write("unrelated-task.json", {"ids": ["a", "b", "c"], "unrelated": True})
        self.change_receipt("task", lambda r: r["documents"].update(measurements=replacement))
        with self.assertRaisesRegex(EvidenceError, "not bound"):
            self.result()

    def test_source_compatibility_cannot_be_reused_for_changed_profile_requirement(self):
        self.test_compatible_later_source_preserves_original_attribution()
        self.contract["cells"][0]["requirements"][2]["profile"] = {**self.profile, "execution": "another-profile"}
        self.index["contract_sha256"] = digest(self.contract)
        self.change_receipt("task", lambda r: r.update(profile={**self.profile, "execution": "another-profile"}))
        with self.assertRaisesRegex(EvidenceError, "exact profile and gate contract"):
            self.result()

    def test_generic_producer_cannot_admit_missing_asset_or_simulator_protocol(self):
        self.contract["cells"][0]["requirements"][2]["pending_admission"] = "Synthetic missing protocol"
        self.index["contract_sha256"] = digest(self.contract)
        with self.assertRaisesRegex(EvidenceError, "admission is pending"):
            self.result()

    def with_lineage(self):
        lineage = {"primary_job": "synthetic-job", "primary_root": "/synthetic/run", "primary_plan_sha256": "e" * 64}
        self.index["cells"]["synthetic"].update(lineage=lineage, gates={})
        for requirement in self.contract["cells"][0]["requirements"]:
            requirement["training_lineage"] = True
        self.checkpoint["lineage"] = lineage
        self.add("training", {"loss": 0.1}, lineage=lineage)
        self.add("checkpoint", {"checkpoint": self.checkpoint}, lineage=lineage)
        self.add("task", {"ids": ["a", "b", "c"]}, lineage=lineage)
        self.index["contract_sha256"] = digest(self.contract)
        return lineage

    def test_one_authoritative_primary_lineage_can_complete(self):
        self.with_lineage()
        self.assertEqual(self.result()["summary"]["fully_passing_qualification"], 1)

    def test_same_cell_other_primary_cannot_supply_training_or_recovery(self):
        lineage = self.with_lineage()
        self.change_receipt("training", lambda r: r.update(lineage={**lineage, "primary_plan_sha256": "f" * 64}))
        with self.assertRaisesRegex(EvidenceError, "authoritative training lineage"):
            self.result()

    def test_checkpoint_from_another_training_lineage_cannot_anchor_final_artifacts(self):
        lineage = self.with_lineage()
        self.index["cells"]["synthetic"]["gates"]["checkpoint"] = []
        self.add(
            "checkpoint",
            {"checkpoint": {**self.checkpoint, "lineage": {**lineage, "primary_job": "another"}}},
            lineage=lineage,
        )
        with self.assertRaisesRegex(EvidenceError, "another primary run"):
            self.result()

    def test_missing_authoritative_lineage_is_not_inferred_from_config(self):
        self.with_lineage()
        del self.index["cells"]["synthetic"]["lineage"]
        with self.assertRaisesRegex(EvidenceError, "authoritative training lineage"):
            self.result()

    def test_required_raw_evidence_cannot_be_replaced_by_normalized_flags(self):
        self.contract["cells"][0]["requirements"][0]["raw_evidence_required"] = True
        self.index["contract_sha256"] = digest(self.contract)
        with self.assertRaisesRegex(EvidenceError, "required raw evidence is absent"):
            self.result()


if __name__ == "__main__":
    unittest.main()
