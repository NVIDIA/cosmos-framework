# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Synthetic inline evidence controls; never campaign or training executions."""

import copy
import hashlib
import json
import pickle
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tools.adapter_campaign import inline_midpoint
from tools.adapter_campaign.evidence import EvidenceError, Store
from tools.adapter_campaign.inline_midpoint import (
    Capture,
    check_checkpoint,
    check_runtime,
    checkpoint_inventory,
    compare_midpoint,
    compare_replay,
    exact_tree,
    metric_number,
    produce,
)

Index = namedtuple("Index", "fqn offset")


def state_fixture():
    clock = {
        "iter": 3,
        "optimizer_step": 3,
        "attempt_id": 3,
        "accum": 0,
        "nonfinite_skips_total": 0,
        "nonfinite_skips_consecutive": 0,
    }
    finite = {
        "version": 1,
        "kind": "finite_sft_zero_weight_update_tail",
        "epochs": 1,
        "samples_per_update_per_rank": 2,
        "dataset_rows": 8,
        "world_size": 4,
    }
    initial = {
        "optimizer": [{"names": ["adapter.A"], "lr": 0.1, "weight_decay": 0.0}],
        "parameter_layouts": {"adapter.A": {"shape": [2, 2], "local_shape": [1, 2], "dtype": "torch.float32"}},
        "initial_clock": clock,
        "finite_coverage": finite,
        "data_identity": "dataset-digest",
        "adapter_identity": {"method": "synthetic", "execution": "factorized_v1"},
    }
    data = {
        "version": 1,
        "world_size": 4,
        "states": {
            r: {
                "rank": r,
                "position": 6,
                "finite_sft": finite,
                "source": {"dataset_fingerprint": "dataset-digest", "dataset_len": 8},
            }
            for r in range(4)
        },
    }
    rng = {
        r: {
            "python": (3, (1, 2), None),
            "numpy": ("MT19937", np.arange(3, dtype=np.uint32), 0, 0, 0.0),
            "torch": torch.arange(3, dtype=torch.uint8),
            "cuda": torch.arange(4, dtype=torch.uint8),
        }
        for r in range(4)
    }
    original = {
        "step": 3,
        "clock": clock,
        "parameters": {"adapter.A": torch.ones(1, 2)},
        "optimizer": {
            "adapter.A": {"step": torch.tensor(3.0), "exp_avg": torch.zeros(1, 2), "exp_avg_sq": torch.ones(1, 2)}
        },
        "optimizer_groups": [{"parameter_names": ["adapter.A"], "lr": 0.1, "weight_decay": 0.0}],
        "scheduler": {"last_epoch": 3},
        "data": data,
        "rng": rng,
    }
    return original, copy.deepcopy(original), copy.deepcopy(initial)


class MidpointTests(unittest.TestCase):
    def test_exact_restoration_includes_optimizer_scheduler_cursor_and_rng(self):
        result = compare_midpoint(*state_fixture(), 3, 2)
        self.assertTrue(result["binary_equal"])
        self.assertEqual(result["trainable_tensors"], 1)

    def test_changed_values_reject_even_if_in_both_sources(self):
        mutations = [
            lambda v: v.pop("rng"),
            lambda v: v.update(step=4),
            lambda v: v["clock"].update(accum=1),
            lambda v: v["clock"].update(optimizer_step=True),
            lambda v: v["parameters"].clear(),
            lambda v: v["optimizer"].clear(),
            lambda v: v["parameters"].update(unintended_base=torch.ones(1, 2)),
            lambda v: v["optimizer"]["adapter.A"].pop("exp_avg"),
            lambda v: v["optimizer"]["adapter.A"].update(step=torch.tensor(2.0)),
            lambda v: v["optimizer_groups"][0]["parameter_names"].append("adapter.A"),
            lambda v: v["parameters"].update({"adapter.A": torch.ones(1, 2, dtype=torch.float64)}),
            lambda v: v["parameters"].update({"adapter.A": torch.ones(2, 2)}),
            lambda v: v["parameters"]["adapter.A"].fill_(float("nan")),
            lambda v: v["optimizer"]["adapter.A"]["exp_avg"].fill_(float("inf")),
            lambda v: v["scheduler"].update(last_epoch=4),
            lambda v: v["data"]["states"].pop(3),
            lambda v: v["data"]["states"][2].update(rank=1),
            lambda v: v["data"]["states"][2].update(position=7),
            lambda v: v["data"]["states"][2]["source"].update(dataset_fingerprint="wrong"),
            lambda v: v["rng"].pop(3),
            lambda v: v["rng"][0].pop("cuda"),
            lambda v: v["rng"][1].update(torch=None),
            lambda v: v["rng"][1].update(cuda=torch.tensor([], dtype=torch.uint8)),
            lambda v: v["rng"][1].update(numpy=("MT19937", np.array([], dtype=np.uint32), 0, 0, 0.0)),
            lambda v: v["optimizer_groups"][0].update(lr=0.2),
        ]
        for mutate in mutations:
            left, right, initial = state_fixture()
            mutate(left)
            mutate(right)
            with self.subTest(mutate=mutate), self.assertRaises(EvidenceError):
                compare_midpoint(left, right, initial, 3, 2)

    def test_same_schema_but_changed_resumed_state_rejected(self):
        mutations = [
            lambda v: v["parameters"]["adapter.A"].add_(1e-7),
            lambda v: v["optimizer"]["adapter.A"]["exp_avg"].add_(1e-7),
            lambda v: v["optimizer_groups"][0].update(lr=0.2),
            lambda v: v["scheduler"].update(other_clock=2),
            lambda v: v["rng"][0]["numpy"][1].__setitem__(0, 12),
            lambda v: v["rng"][1]["torch"].__setitem__(0, 12),
            lambda v: v["rng"][2].update(python=(3, (1, 4), None)),
        ]
        for mutate in mutations:
            left, right, initial = state_fixture()
            mutate(right)
            with self.subTest(mutate=mutate), self.assertRaises(EvidenceError):
                compare_midpoint(left, right, initial, 3, 2)

    def test_binary_comparison_distinguishes_signed_zero_types_and_arrays(self):
        for left, right in [
            (0.0, -0.0),
            (True, 1),
            ([1], (1,)),
            ({"x": 1}, {"y": 1}),
            (torch.tensor([0.0]), torch.tensor([-0.0])),
            (np.array([1], dtype=np.int32), np.array([1], dtype=np.int64)),
        ]:
            with self.subTest(left=left, right=right), self.assertRaises(EvidenceError):
                exact_tree(left, right)
        exact_tree(float("nan"), float("nan"))

    def test_only_metrics_accept_integral_float_counters(self):
        self.assertTrue(metric_number(0.0, 0))
        self.assertTrue(metric_number(3, 3))
        for value in (False, "0", None, float("nan"), float("inf"), 0.1):
            self.assertFalse(metric_number(value, 0))


def checkpoint_fixture():
    state, _, initial = state_fixture()
    names = ["trainer.pt", "data.pt", "model/.metadata", "optim/.metadata"]
    names += [f"{part}/__{r}_0.distcp" for part in ("model", "optim") for r in range(4)]
    inventory = {name: {"sha256": "1" * 64, "bytes": 100} for name in names}
    trainer = {
        **state["clock"],
        "rng": state["rng"],
        "lr_scheduler": state["scheduler"],
        "scheduler_identity": {"version": 1},
        "validation": {},
        "objective_identity": {
            "adapters": initial["adapter_identity"],
            "training_source": "dataset-digest",
            "training_coverage": initial["finite_coverage"],
        },
    }
    metadata = {}
    for part, keys in (
        ("model", ["adapter.A"]),
        ("optim", [f"state.adapter.A.{k}" for k in ("exp_avg", "exp_avg_sq", "step")]),
    ):
        metadata[part] = SimpleNamespace(
            state_dict_metadata={
                key: SimpleNamespace(size=(2, 2), chunks=[SimpleNamespace(offsets=(0, 0), sizes=(2, 2))])
                for key in keys
            },
            storage_data={
                Index(key, (0, 0)): SimpleNamespace(relative_path="__0_0.distcp", offset=i * 20, length=20)
                for i, key in enumerate(keys)
            },
        )
    c = SimpleNamespace(
        inventory=lambda p: inventory,
        tensor_state=lambda p: trainer if p.name == "trainer.pt" else state["data"],
        raw=lambda p: pickle.dumps(metadata[p.parent.name]),
    )
    return c, inventory, trainer, metadata, initial


class CheckpointTests(unittest.TestCase):
    def test_complete_twelve_file_checkpoint_and_controls(self):
        c, _, _, _, initial = checkpoint_fixture()
        inv, trainer, data = check_checkpoint(c, Path("/checkpoint"), initial, 3, 2)
        self.assertEqual(len(inv), 12)
        self.assertEqual(trainer["optimizer_step"], 3)
        self.assertEqual(data["states"][3]["position"], 6)

    def test_missing_empty_or_extra_checkpoint_files_rejected(self):
        for mutate in [
            lambda inv: inv.pop("optim/__3_0.distcp"),
            lambda inv: inv.update({"incomplete.tmp": {"bytes": 100, "sha256": "1" * 64}}),
            lambda inv: inv["trainer.pt"].update(bytes=0),
        ]:
            c, inv, _, _, _ = checkpoint_fixture()
            mutate(inv)
            with self.subTest(mutate=mutate), self.assertRaises(EvidenceError):
                checkpoint_inventory(c, Path("/checkpoint"))

    def test_partial_control_profile_metadata_or_storage_rejected(self):
        mutations = [
            lambda t, m: t.pop("scheduler_identity"),
            lambda t, m: t["objective_identity"]["adapters"].update(execution="other-profile"),
            lambda t, m: t["rng"].pop(1),
            lambda t, m: t["lr_scheduler"].update(last_epoch=4),
            lambda t, m: m["model"].state_dict_metadata.clear(),
            lambda t, m: m["optim"].state_dict_metadata.pop("state.adapter.A.step"),
            lambda t, m: setattr(m["model"].state_dict_metadata["adapter.A"], "size", (1, 2)),
            lambda t, m: setattr(next(iter(m["model"].storage_data.values())), "length", 101),
            lambda t, m: setattr(next(iter(m["model"].storage_data.values())), "offset", -1),
            lambda t, m: setattr(next(iter(m["model"].storage_data.values())), "relative_path", "../outside"),
            lambda t, m: m["model"].storage_data.clear(),
            lambda t, m: setattr(m["model"].state_dict_metadata["adapter.A"].chunks[0], "sizes", (1, 2)),
            lambda t, m: (
                m["model"].state_dict_metadata["adapter.A"].chunks.append(SimpleNamespace(offsets=(0, 0), sizes=(2, 2)))
            ),
            lambda t, m: setattr(next(iter(m["optim"].storage_data.values())), "length", 21),
        ]
        for mutate in mutations:
            c, _, trainer, metadata, initial = checkpoint_fixture()
            initial = copy.deepcopy(initial)
            mutate(trainer, metadata)
            with self.subTest(mutate=mutate), self.assertRaises(EvidenceError):
                check_checkpoint(c, Path("/checkpoint"), initial, 3, 2)


def replay_fixture():
    original = [
        {
            "step": s,
            "inputs": [{"sample_id": f"sample-{s}", "padding": False}],
            "metrics": {
                "train/loss_avg": 1.25,
                "optim/update_outcome": "applied",
                "optim/optimizer_step": s,
                "optim/attempt_id": s,
                "optim/step_skipped": 0.0,
                "optim/nonfinite_skips_total": 0,
                "train/valid_token_count": 17,
                "train/valid_sequence_count": 1,
            },
        }
        for s in range(1, 9)
    ]
    replay = copy.deepcopy(original[3:6])
    proof = {
        s: {"status": "PASS", "reference": "/primary", "atol": 1e-7, "rtol": 1e-5, "loss_error": 0.0} for s in (4, 5, 6)
    }
    large = {
        f"/{phase}/update-{s:06d}-rank0.pt": {"sha256": str(s) * 64, "bytes": 150}
        for s in (4, 5, 6)
        for phase in ("primary", "branch")
    }
    c = SimpleNamespace(
        rows=lambda p: original if p.parent == Path("/primary") else replay,
        json=lambda p: proof[int(p.name.split("-")[2])],
        large={"files": large},
    )
    gates = {
        "parameter_and_optimizer_atol": 1e-7,
        "parameter_and_optimizer_rtol": 1e-5,
        "loss_atol": 1e-6,
        "loss_rtol": 1e-5,
    }
    return c, original, replay, proof, large, gates


class ReplayTests(unittest.TestCase):
    def test_exact_three_updates_excludes_replay_from_primary_coverage(self):
        c, _, _, _, _, gates = replay_fixture()
        records = compare_replay(c, Path("/primary"), Path("/branch"), 0, 3, gates)
        self.assertEqual([r["step"] for r in records], [4, 5, 6])
        self.assertEqual(sum(r["real_rows"] for r in records), 3)

    def test_failed_partial_foreign_or_numerically_changed_replay_rejected(self):
        mutations = [
            lambda r, p, l: r.pop(),
            lambda r, p, l: r.append(r[0]),
            lambda r, p, l: r[0].update(step=1),
            lambda r, p, l: r[0]["inputs"][0].update(sample_id="wrong"),
            lambda r, p, l: p[4].update(status="FAIL"),
            lambda r, p, l: p[4].update(reference="/other"),
            lambda r, p, l: p[4].update(atol=1e-5),
            lambda r, p, l: r[0]["metrics"].update({"train/loss_avg": float("nan")}),
            lambda r, p, l: r[0]["metrics"].update({"train/loss_avg": 1.26}),
            lambda r, p, l: r[0]["metrics"].update({"optim/step_skipped": True}),
            lambda r, p, l: r[0]["metrics"].update({"optim/optimizer_step": 5}),
            lambda r, p, l: r[0]["metrics"].update({"train/valid_token_count": 16}),
            lambda r, p, l: l.pop("/primary/update-000005-rank0.pt"),
            lambda r, p, l: l["/branch/update-000006-rank0.pt"].update(sha256="9" * 64),
            lambda r, p, l: l["/branch/update-000004-rank0.pt"].update(bytes=149),
        ]
        for mutate in mutations:
            c, _, replay, proof, large, gates = replay_fixture()
            mutate(replay, proof, large)
            with self.subTest(mutate=mutate), self.assertRaises(EvidenceError):
                compare_replay(c, Path("/primary"), Path("/branch"), 0, 3, gates)


class AdmissionTests(unittest.TestCase):
    def test_producer_review_dependency_capture_and_contract_are_mandatory(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            audit = root / "synthetic-audit.json"
            audit.write_text(json.dumps({"completed": {"cell": "synthetic-cell"}}))
            audit_ref = {"path": audit.name, "sha256": hashlib.sha256(audit.read_bytes()).hexdigest()}
            deps = {
                "tools/adapter_campaign/" + n: hashlib.sha256(
                    Path(inline_midpoint.__file__).with_name(n).read_bytes()
                ).hexdigest()
                for n in inline_midpoint.DEPENDENCIES
            }
            source_sha = deps["tools/adapter_campaign/inline_midpoint.py"]
            for case in ("negative_review", "wrong_dependency", "wrong_capture", "missing_producer", "wrong_review"):
                review = {
                    "status": "PASS_NO_OPEN_CONFIRMED_FINDINGS",
                    "open_findings": [],
                    "reviewed_files": {**deps, audit.name: audit_ref["sha256"]},
                }
                if case == "negative_review":
                    review["open_findings"] = ["unresolved synthetic defect"]
                if case == "wrong_dependency":
                    review["reviewed_files"]["tools/adapter_campaign/primary.py"] = "0" * 64
                if case == "wrong_capture":
                    review["reviewed_files"][audit.name] = "0" * 64
                r = root / (case + ".json")
                r.write_text(json.dumps(review))
                review_ref = {"path": r.name, "sha256": hashlib.sha256(r.read_bytes()).hexdigest()}
                requirement = {
                    "id": "mid_epoch_continuation",
                    "producers": [source_sha],
                    "producer_reviews": {source_sha: [review_ref]},
                }
                if case == "missing_producer":
                    requirement["producers"] = []
                if case == "wrong_review":
                    requirement["producer_reviews"][source_sha] = [audit_ref]
                contract = {"cells": [{"id": "synthetic-cell", "requirements": [requirement]}]}
                output = root / (case + "-output")
                with self.subTest(case=case), self.assertRaises(EvidenceError):
                    produce(root, contract, output, audit_review=audit_ref, producer_review=review_ref)
                self.assertFalse(output.exists())

    def test_installed_runtime_requires_both_exact_image_and_wheel(self):
        plan = {"runtime": "/runtime", "image_sha256": "1" * 64, "wheel_sha256": "2" * 64}
        image, wheel = "/runtime/output/runtime.sqsh", "/runtime/inputs/cosmos_framework-1.2.2-py3-none-any.whl"
        original = {
            "same_runtime_stamps_rechecked": 2,
            "runtime_objects": {
                name: {"sha256": sha, "bytes": 100, "pre_post_stat": {"st_size": 100}}
                for name, sha in ((image, plan["image_sha256"]), (wheel, plan["wheel_sha256"]))
            },
        }
        check_runtime(original, plan)
        for mutate in [
            lambda p: p["runtime_objects"].pop(wheel),
            lambda p: p["runtime_objects"][image].update(sha256="3" * 64),
            lambda p: p["runtime_objects"][wheel]["pre_post_stat"].update(st_size=101),
            lambda p: p.update(same_runtime_stamps_rechecked=1),
        ]:
            value = copy.deepcopy(original)
            mutate(value)
            with self.subTest(mutate=mutate), self.assertRaises(EvidenceError):
                check_runtime(value, plan)

    def test_review_hash_size_binding_and_missing_sources(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "file.json").write_bytes(b"{}")
            sha = hashlib.sha256(b"{}").hexdigest()
            c = Capture.__new__(Capture)
            c.root, c.store, c.references, c.resolver = root, Store(root), {}, {}
            c.reviewed = {"file.json": sha}
            c.bind("/remote/file", Path("file.json"), {"sha256": sha, "bytes": 2})
            self.assertEqual(c.json("/remote/file"), {})
            for remote, item in [
                ("/remote/file", {"sha256": sha, "bytes": 2}),
                ("/new", {"sha256": sha, "bytes": 3}),
                ("/new", {"sha256": "0" * 64, "bytes": 2}),
            ]:
                with self.subTest(remote=remote, item=item), self.assertRaises(EvidenceError):
                    c.bind(remote, Path("file.json"), item)
            with self.assertRaises(EvidenceError):
                c.raw("/wrong/rank")
            (root / "file.json").write_bytes(b"[]")
            with self.assertRaises(EvidenceError):
                c.raw("/remote/file")

    def test_production_producer_rejects_synthetic_or_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(EvidenceError, "synthetic"):
                produce(d, {"synthetic": True}, Path(d) / "new", audit_review={}, producer_review={})
            with self.assertRaisesRegex(EvidenceError, "new evidence"):
                produce(d, {}, d, audit_review={}, producer_review={})


if __name__ == "__main__":
    unittest.main()
