# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU-only synthetic boundary tests; these fixtures are never campaign evidence."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.adapter_campaign import recovered
from tools.adapter_campaign.evidence import EvidenceError, Store, check_receipt
from tools.adapter_campaign.primary import file_ref


class Documents:
    def __init__(self, documents):
        self.documents = documents

    def json(self, name):
        return copy.deepcopy(self.documents[str(name)])

    def sha(self, name):
        return "4" * 64

    def raw(self, name):
        return self.documents[str(name)]


class RankTests(unittest.TestCase):
    def fixture(self, layout):
        audit, plan = Path("/remote/recovery/audit"), Path("/remote/recovery")
        peers = [
            {"rank": rank, "host": "host-a" if rank < 2 else "host-b", "gpu_uuid": f"gpu-{rank}", "job": "123"}
            for rank in range(4)
        ]
        if layout == "fsdp2":
            topology = {
                "shape": [4],
                "groups": {"0": {"ranks": [0, 1, 2, 3], "hosts": ["host-a", "host-b"], "placement": "S(0)"}},
            }
        else:
            topology = {
                "shape": [2, 2],
                "groups": {
                    "0": {"ranks": [0, 2], "hosts": ["host-a", "host-b"], "placement": "R"},
                    "1": {"ranks": [0, 1], "hosts": ["host-a"], "placement": "S(0)"},
                },
            }
        source = {"revision": "1" * 40, "wheel_sha256": "2" * 64, "runtime_sha256": "3" * 64}
        initial = {
            "data_identity": "train-fingerprint",
            "validation_identity": "validation-fingerprint",
            "adapter_identity": {"base": "frozen-base", "processor": "frozen-processor"},
            "optimizer": [{"names": ["A", "B"]}],
            "parameter_layouts": {"A": {}, "B": {}},
            "topology": {"peers": peers, "layouts": {"parameter": topology}},
        }
        finished = {
            "status": "UPDATE_CHECKS_PASS_TASK_AND_EXPORT_PENDING",
            "step": 935,
            "mode": "full",
            "frozen_preserved": True,
            "finite_gradients": ["A", "B"],
            "changed": {"A": True, "B": True},
        }
        provenance = {
            "source_revision": source["revision"],
            "wheel_sha256": source["wheel_sha256"],
            "container_sha256": source["runtime_sha256"],
            "campaign_spec_sha256": "4" * 64,
            "slurm_job_id": "123",
            "mode": "full",
            "harness_sha256": recovered.WRAPPERS["campaign_full_run_v1.py"],
            "verified_package_files": 200,
            "source_dirty": False,
        }
        documents = {
            str(audit / "initial-rank0.json"): initial,
            str(audit / "completed-updates-rank0.json"): finished,
            str(audit / "provenance-rank0.json"): provenance,
            str(audit / "public-result-rank0.json"): {"final_step": 935},
        }
        kwargs = {
            "spec": {"cell": {"topology": {"layout": layout}}},
            "plan": plan,
            "source": source,
            "job": "123",
            "hosts": ["host-a", "host-b"],
            "admission": {"expected_adapter_identity": copy.deepcopy(initial["adapter_identity"])},
            "fingerprints": {"train": "train-fingerprint", "validation": "validation-fingerprint"},
            "rank": 0,
            "mode": "full",
            "final_step": 935,
        }
        return audit, documents, kwargs, initial, finished, provenance

    def test_both_physical_layouts_are_accepted(self):
        for layout in ["fsdp2", "hsdp"]:
            with self.subTest(layout=layout):
                audit, documents, kwargs, initial, _, _ = self.fixture(layout)
                self.assertEqual(recovered.validate_rank(Documents(documents), audit, **kwargs), initial)

    def test_source_owners_physical_placement_and_completion_are_required(self):
        mutations = {
            "wrong_source": lambda i, f, p: p.update(source_revision="0" * 40),
            "wrong_wrapper": lambda i, f, p: p.update(harness_sha256="0" * 64),
            "dirty_package": lambda i, f, p: p.update(source_dirty=True),
            "unverified_package": lambda i, f, p: p.update(verified_package_files=0),
            "wrong_job": lambda i, f, p: p.update(slurm_job_id="456"),
            "wrong_split": lambda i, f, p: i.update(data_identity="other-data"),
            "wrong_processor": lambda i, f, p: i["adapter_identity"].update(processor="other-processor"),
            "duplicate_owner": lambda i, f, p: i["optimizer"][0]["names"].append("A"),
            "missing_owner": lambda i, f, p: i["parameter_layouts"].pop("B"),
            "empty_owner_gradient_and_update_sets": lambda i, f, p: (
                i.update(optimizer=[], parameter_layouts={}),
                f.update(finite_gradients=[], changed={}),
            ),
            "missing_gradient": lambda i, f, p: f["finite_gradients"].remove("B"),
            "unchanged_target": lambda i, f, p: f["changed"].update(B=False),
            "numeric_changed_flag": lambda i, f, p: f["changed"].update(B=1),
            "frozen_mutation": lambda i, f, p: f.update(frozen_preserved=False),
            "early_stop": lambda i, f, p: f.update(step=934),
            "wrong_mode": lambda i, f, p: f.update(mode="branch"),
            "duplicate_gpu": lambda i, f, p: i["topology"]["peers"][1].update(gpu_uuid="gpu-0"),
            "wrong_peer_rank": lambda i, f, p: i["topology"]["peers"][1].update(rank=0),
            "wrong_peer_job": lambda i, f, p: i["topology"]["peers"][1].update(job="456"),
            "one_physical_host": lambda i, f, p: [peer.update(host="host-a") for peer in i["topology"]["peers"]],
            "missing_mesh": lambda i, f, p: i["topology"].update(layouts={}),
            "wrong_mesh_shape": lambda i, f, p: i["topology"]["layouts"]["parameter"].update(shape=[1, 4]),
            "group_excludes_rank": lambda i, f, p: i["topology"]["layouts"]["parameter"]["groups"]["0"].update(
                ranks=[1, 2]
            ),
        }
        for layout in ["fsdp2", "hsdp"]:
            for name, change in mutations.items():
                with self.subTest(layout=layout, mutation=name):
                    audit, documents, kwargs, initial, finished, provenance = self.fixture(layout)
                    change(initial, finished, provenance)
                    with self.assertRaises(EvidenceError):
                        recovered.validate_rank(Documents(documents), audit, **kwargs)


class CoverageTests(unittest.TestCase):
    def fixture(self):
        ids = list("abcdefg")
        weights = {name: index + 1 for index, name in enumerate(ids)}
        native = {"train": {"loss_tokens": sum(weights.values())}}
        metrics = {
            "train/loss_avg": 1.25,
            "train/valid_token_count": sum(weights.values()),
            "train/valid_sequence_count": len(ids),
            "optim/update_outcome": "applied",
            "optim/step_skipped": 0,
            "optim/nonfinite_skips_total": 0,
            "optim/optimizer_step": 1,
            "optim/attempt_id": 1,
        }
        inputs = [
            {"epoch": 0, "sample_id": name, "source_index": i, "padding": False, "weight": 1.0}
            for i, name in enumerate(ids)
        ] + [{"epoch": 0, "sample_id": "a", "source_index": 0, "padding": True, "weight": 0.0}]
        logs = [
            [{"step": 1, "inputs": inputs[2 * rank : 2 * rank + 2], "metrics": copy.deepcopy(metrics)}]
            for rank in range(4)
        ]
        cell = {"optimizer_updates": 1, "training_epochs": 1, "grad_accum": 2}
        return logs, {"train": [{"id": name} for name in ids]}, {"train": weights}, native, cell

    def test_tail_has_zero_weight_and_native_per_id_denominator(self):
        counts, padding, tokens, curve = recovered.validate_coverage(*self.fixture())
        self.assertEqual(dict(counts[0]), dict.fromkeys("abcdefg", 1))
        self.assertEqual(padding, [1])
        self.assertEqual(tokens, 28)
        self.assertEqual(curve, [{"step": 1, "loss": 1.25, "valid_tokens": 28}])

    def test_missing_repeated_or_nonfinite_work_does_not_pass(self):
        mutations = {
            "missing_rank_row": lambda logs: logs[2].clear(),
            "repeated_update": lambda logs: logs[0].append(copy.deepcopy(logs[0][0])),
            "wrong_update_clock": lambda logs: logs[0][0]["metrics"].update({"optim/optimizer_step": 2}),
            "skipped_update": lambda logs: logs[0][0]["metrics"].update({"optim/update_outcome": "skipped"}),
            "prior_nonfinite_skip": lambda logs: logs[0][0]["metrics"].update({"optim/nonfinite_skips_total": 1}),
            "nonfinite_loss": lambda logs: logs[0][0]["metrics"].update({"train/loss_avg": float("nan")}),
            "rank_loss_disagreement": lambda logs: logs[2][0]["metrics"].update({"train/loss_avg": 2.0}),
            "rank_denominator_disagreement": lambda logs: logs[2][0]["metrics"].update({"train/valid_token_count": 27}),
            "wrong_native_denominator": lambda logs: [
                rank[0]["metrics"].update({"train/valid_token_count": 27}) for rank in logs
            ],
            "missing_microbatch": lambda logs: logs[0][0]["inputs"].pop(),
            "wrong_id": lambda logs: logs[0][0]["inputs"][0].update(sample_id="b"),
            "wrong_epoch": lambda logs: logs[0][0]["inputs"][0].update(epoch=1),
            "nonzero_padding": lambda logs: logs[3][0]["inputs"][1].update(weight=1.0),
            "numeric_padding_flag": lambda logs: logs[3][0]["inputs"][1].update(padding=1),
        }
        for name, change in mutations.items():
            with self.subTest(mutation=name):
                values = self.fixture()
                change(values[0])
                with self.assertRaises(EvidenceError):
                    recovered.validate_coverage(*values)

    def test_negative_source_index_cannot_alias_another_position(self):
        values = self.fixture()
        values[0][3][0]["inputs"][0]["source_index"] = -1  # Still selects "g" under Python indexing.
        with self.assertRaises(EvidenceError):
            recovered.validate_coverage(*values)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.remote = "/remote/campaign"
        self.reviewed = {}
        self.body = b'{"example":"synthetic"}\n'
        self.raw_ref = self.put("files/raw.json", self.body)
        self.remote_name = self.remote + "/input.json"
        item = {
            "remote_path": self.remote_name,
            "relative_path": "input.json",
            "canonical_path": self.remote_name,
            "sha256": self.raw_ref["sha256"],
            "bytes": len(self.body),
        }
        self.audit = {
            "cell": "synthetic-cell",
            "status": "PASS_PINNED_SEMANTIC_REVALIDATION_AND_RAW_MANIFEST",
            "helper_sha256": recovered.HELPERS,
            "remote_root": self.remote,
        }
        self.plan = {
            "format": "recovered-primary-small-evidence-capture-plan-v1",
            "remote_root": self.remote,
            "files": {self.remote_name: item},
            "large_objects_not_transferred": {},
            "continuation_links_not_materialized": {},
            "prior_attestations": self.put("prior.json", {"synthetic": True}),
            "capture_driver": self.put("driver.py", b"# synthetic test driver\n"),
            "remote_audit_program": self.put("remote.py", b"# synthetic test audit\n"),
        }
        self.resolver = {
            "remote_root": self.remote,
            "files": {
                self.remote_name: {
                    **self.raw_ref,
                    "bytes": len(self.body),
                    "canonical_remote_path": self.remote_name,
                }
            },
            "large_objects_not_transferred": {},
            "continuation_links_not_materialized": {},
        }
        self.envelope = self.put("cell-00-remote-envelope.json", {"synthetic": True, "returncode": 0})

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else (json.dumps(value, sort_keys=True) + "\n").encode())
        ref = file_ref(self.root, path)
        self.reviewed[name] = ref["sha256"]
        return ref

    def persist(self):
        self.plan["semantic_audits"] = [self.put("audit.json", self.audit)]
        plan = self.put("plan.json", self.plan)
        resolver = self.put("resolver.json", self.resolver)
        receipt = self.put(
            "result.json",
            {
                "status": "PASS_EXACT_RAW_EVIDENCE_CAPTURE",
                "plan": plan,
                "resolver": resolver,
                "files": 1,
                "files_sha256": {"input.json": self.raw_ref["sha256"]},
            },
        )
        review = {
            "format": "recovered-primary-raw-capture-independent-review-v1",
            "status": "PASS_NO_OPEN_CONFIRMED_FINDINGS",
            "open_confirmed_findings": [],
            "capture_result": receipt,
            "reviewed_files": dict(self.reviewed),
        }
        review_ref = self.put("review.json", review)
        return self.root / receipt["path"], review_ref

    def test_explicit_remote_resolver_verifies_exact_raw_bytes(self):
        receipt, review = self.persist()
        capture = recovered.Capture(self.root, receipt, review)
        self.assertEqual(capture.raw(self.remote_name), self.body)
        with self.assertRaises(EvidenceError):
            capture.raw(self.remote + "/uncaptured.json")

    def test_changed_raw_bytes_are_rejected(self):
        receipt, review = self.persist()
        (self.root / self.raw_ref["path"]).write_bytes(b"changed after review\n")
        with self.assertRaises(EvidenceError):
            recovered.Capture(self.root, receipt, review).raw(self.remote_name)

    def test_changed_review_bound_control_is_rejected(self):
        for name in ["result.json", "plan.json", "resolver.json", "audit.json", "driver.py", "remote.py"]:
            with self.subTest(control=name):
                receipt, review = self.persist()
                path = self.root / name
                saved = path.read_bytes()
                path.write_bytes(saved + b" ")
                with self.assertRaises(EvidenceError):
                    recovered.Capture(self.root, receipt, review)
                path.write_bytes(saved)

    def test_missing_review_bound_remote_envelope_is_rejected(self):
        receipt, review = self.persist()
        (self.root / self.envelope["path"]).unlink()
        with self.assertRaises(EvidenceError):
            recovered.Capture(self.root, receipt, review)

    def test_mismatched_resolver_binding_is_rejected(self):
        self.resolver["files"][self.remote_name]["sha256"] = "0" * 64
        receipt, review = self.persist()
        with self.assertRaises(EvidenceError):
            recovered.Capture(self.root, receipt, review)

    def test_remote_only_tensor_requires_exact_audited_identity(self):
        receipt, review = self.persist()
        capture = recovered.Capture(self.root, receipt, review)
        name = self.remote + "/original/update-000467-rank0.pt"
        item = {"remote_path": name, "bytes": 100, "stat": {"st_size": 100}, "sha256": "a" * 64}
        audit = {"large_objects": {name: copy.deepcopy(item)}}
        capture.plan["large_objects_not_transferred"][name] = copy.deepcopy(item)
        self.assertEqual(capture.remote_tensor_sha(name, audit), "a" * 64)
        for field, value in [("sha256", "b" * 64), ("bytes", 101), ("remote_path", name + ".other")]:
            with self.subTest(field=field):
                changed = copy.deepcopy(audit)
                changed["large_objects"][name][field] = value
                with self.assertRaises(EvidenceError):
                    capture.remote_tensor_sha(name, changed)
        with self.assertRaises(EvidenceError):
            capture.remote_tensor_sha(name, {"large_objects": {}})


class DatasetTests(unittest.TestCase):
    def fixture(self):
        base = Path("/remote/original")
        documents = {}
        identities, summaries, reports = {}, [], []
        paths = {}
        for split, ids in [("train", ["a", "b"]), ("validation", ["v"])]:
            paths[split] = "/remote/data/" + split + ".jsonl"
            data = [{"id": row_id, "messages": [{"role": "assistant", "content": row_id}]} for row_id in ids]
            raw = b"".join((json.dumps(row) + "\n").encode() for row in data)
            documents[paths[split]] = raw
            identities[split] = {"path": paths[split], "sha256": hashlib.sha256(raw).hexdigest()}
            summaries.append(
                {
                    "split": split,
                    "ok": True,
                    "failed": 0,
                    "unmeasured_rows": 0,
                    "over_long_rows": 0,
                    "rows": len(ids),
                    "matched_rows": len(ids),
                    "loss_tokens": 2 * len(ids),
                }
            )
            reports.extend(
                {"split": split, "row_id": row_id, "index": i, "mask_matches": True, "reasons": [], "loss_tokens": 2}
                for i, row_id in enumerate(ids)
            )
        census = {"status": "PASS", "summaries": summaries}
        spec = {"paths": paths, "dataset": {"rows_train": 2, "rows_validation": 1}}
        plan = {"expected_train_valid_tokens": 4, "expected_validation_valid_tokens": 2}

        def validate():
            report_bytes = b"".join((json.dumps(row) + "\n").encode() for row in reports)
            census["report_sha256"] = hashlib.sha256(report_bytes).hexdigest()
            documents[str(base / "inputs/proofs/preprocessing-rows.jsonl")] = report_bytes
            documents[str(base / "inputs/proofs/preprocessing-census.json")] = json.dumps(census).encode()
            documents[str(base / "inputs/proofs/data-identities.json")] = json.dumps(identities).encode()
            return recovered.validate_datasets(Documents(documents), base, plan, spec)

        return validate, reports, summaries, identities

    def test_full_native_census_derives_per_id_tokens(self):
        validate, _, _, _ = self.fixture()
        datasets, per_id, fingerprints, native = validate()
        self.assertEqual([r["id"] for r in datasets["train"]], ["a", "b"])
        self.assertEqual(per_id, {"train": {"a": 2, "b": 2}, "validation": {"v": 2}})
        self.assertEqual(set(fingerprints), {"train", "validation"})
        self.assertEqual(native["train"]["loss_tokens"], 4)

    def test_missing_reordered_bad_mask_or_unmeasured_rows_are_rejected(self):
        changes = {
            "missing_row": lambda r, s, i: r.pop(0),
            "duplicate_id": lambda r, s, i: r[1].update(row_id="a"),
            "wrong_index": lambda r, s, i: r[0].update(index=1),
            "mask_mismatch": lambda r, s, i: r[0].update(mask_matches=False),
            "preprocessing_failure": lambda r, s, i: r[0].update(reasons=["native error"]),
            "float_token_count": lambda r, s, i: r[0].update(loss_tokens=2.0),
            "zero_tokens": lambda r, s, i: r[0].update(loss_tokens=0),
            "wrong_total": lambda r, s, i: s[0].update(loss_tokens=5),
            "unmeasured": lambda r, s, i: s[0].update(unmeasured_rows=1),
            "dataset_changed": lambda r, s, i: i["train"].update(sha256="0" * 64),
        }
        for name, change in changes.items():
            with self.subTest(mutation=name):
                validate, rows, summaries, identities = self.fixture()
                change(rows, summaries, identities)
                with self.assertRaises(EvidenceError):
                    validate()


class ProducerTests(unittest.TestCase):
    """Isolate receipt production; synthetic normalizer stubs never admit real evidence."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.producer = hashlib.sha256(Path(recovered.__file__).read_bytes()).hexdigest()
        self.source = {"revision": "1" * 40, "wheel_sha256": "2" * 64, "runtime_sha256": "3" * 64}
        self.lineage = {"primary_job": "123", "primary_root": "/remote/recovery", "primary_plan_sha256": "4" * 64}
        self.capture_review = self.put("capture-review.json", {"synthetic": True})
        self.capture_ref = self.put("synthetic-capture.json", {"synthetic": True})
        self.review = self.put(
            "producer-review.json",
            {
                "status": "PASS_NO_OPEN_CONFIRMED_FINDINGS",
                "open_findings": [],
                "capture_review": self.capture_review,
                "reviewed_files": {
                    **{
                        "tools/adapter_campaign/" + name: hashlib.sha256(
                            Path(recovered.__file__).with_name(name).read_bytes()
                        ).hexdigest()
                        for name in ["recovered.py", "evidence.py", "primary.py"]
                    },
                    self.capture_review["path"]: self.capture_review["sha256"],
                    self.capture_ref["path"]: self.capture_ref["sha256"],
                },
            },
        )
        requirement = {
            "id": "mid_epoch_continuation",
            "profile": None,
            "final_checkpoint": False,
            "documents": ["measurement", "provenance"],
            "provenance_document": "provenance",
            "raw_evidence_required": True,
            "training_lineage": True,
            "sources": [self.source],
            "producers": [self.producer],
            "producer_reviews": {self.producer: [self.review]},
            "assertions": [
                {"document": "measurement", "pointer": "/updates_compared", "operator": "eq", "expected": 3}
            ],
        }
        self.contract = {
            "synthetic": False,
            "cells": [{"id": "synthetic-cell", "identity": {"cell": "synthetic-cell"}, "requirements": [requirement]}],
        }
        self.capture = Documents({"/remote/recovery/inputs/plan.json": {"cell": "synthetic-cell"}})
        self.capture.audits = {"synthetic-cell": {"recovery_root": "/remote/recovery"}}

    def put(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value, sort_keys=True) + "\n")
        return file_ref(self.root, path)

    def produce(self):
        normalized = (
            {"mid_epoch_continuation": {"updates_compared": 3}},
            self.source,
            self.lineage,
            [self.capture_review],
        )
        with (
            patch.object(recovered, "Capture", return_value=self.capture),
            patch.object(recovered, "normalize_recovered_primary", return_value=normalized),
        ):
            return recovered.produce(
                self.root,
                self.contract,
                self.root / "synthetic-capture.json",
                self.root / "output",
                capture_review=self.capture_review,
                producer_review=self.review,
            )

    def test_producer_review_is_bound_into_receipt_and_provenance(self):
        index = self.produce()
        entry = index["cells"]["synthetic-cell"]
        ref = entry["gates"]["mid_epoch_continuation"][0]
        cell = self.contract["cells"][0]
        observed = check_receipt(
            Store(self.root), ref, cell["requirements"][0], cell["identity"], self.contract, None, self.lineage
        )
        self.assertEqual(observed["outcome"], "PASS")
        receipt = Store(self.root).json(ref)
        self.assertIn(self.review, receipt["supporting_evidence"])
        self.assertIsNone(entry["final_checkpoint"])
        self.assertEqual(set(entry["gates"]), {"mid_epoch_continuation"})

    def test_wrong_admitted_review_is_rejected_before_outputs(self):
        other = self.put("other-review.json", {"synthetic": "another review"})
        self.contract["cells"][0]["requirements"][0]["producer_reviews"][self.producer] = [other]
        with self.assertRaises(EvidenceError):
            self.produce()
        self.assertFalse((self.root / "output").exists())

    def test_unadmitted_source_is_rejected_before_outputs(self):
        self.contract["cells"][0]["requirements"][0]["sources"] = []
        with self.assertRaises(EvidenceError):
            self.produce()
        self.assertFalse((self.root / "output").exists())

    def test_other_capture_review_cannot_borrow_the_normalizer_admission(self):
        self.capture_review = self.put("other-capture-review.json", {"synthetic": "another capture"})
        with self.assertRaises(EvidenceError):
            self.produce()
        self.assertFalse((self.root / "output").exists())

    def test_changed_dependency_is_not_covered_by_a_review_of_only_the_entrypoint(self):
        review = json.loads((self.root / self.review["path"]).read_text())
        review["reviewed_files"]["tools/adapter_campaign/evidence.py"] = "0" * 64
        self.review = self.put("producer-review.json", review)
        self.contract["cells"][0]["requirements"][0]["producer_reviews"][self.producer] = [self.review]
        with self.assertRaises(EvidenceError):
            self.produce()
        self.assertFalse((self.root / "output").exists())


if __name__ == "__main__":
    unittest.main()
