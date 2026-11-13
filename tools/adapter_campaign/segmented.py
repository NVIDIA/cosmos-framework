# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Normalize the independently audited four-phase Mistral primary.

This is a schema adapter for existing immutable evidence, not another execution
or a checkpoint loader. Remote tensor comparisons remain attributed to their
installed producer and the explicitly admitted independent capture audit. Only
four training/continuation gates are emitted; deployment remains separate.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path

from tools.adapter_campaign.evidence import Store, digest, producer_reviews, read_json, require, same
from tools.adapter_campaign.primary import file_ref, write
from tools.adapter_campaign.recovered import validate_coverage, validate_datasets, validate_rank

CELL = "mistral_hf__lora__factorized_v1__default_v1__fsdp2"
GATES = {"full_schedule", "complete_sample_coverage", "initial_final_validation", "mid_epoch_continuation"}
GATES_SHA = "75c7db901369169327bc5b3ba5508da55d55efee6e8ebef530d3d477fcf62d3b"


class Capture:
    """Resolve exact remote names through a separately admitted raw capture."""

    def __init__(self, root, audit_ref):
        self.root, self.store, self.references = Path(root), Store(root), {}
        self.audit = self.document(audit_ref)
        a = self.audit
        require(
            a["format"] == "mistral-segmented-primary-independent-actual-audit-v1"
            and a["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS"
            and not a["open_findings"]
            and a["full_campaign_pass"] is False,
            "unadmitted segmented capture audit",
        )
        self.reviewed = a["reviewed_files"]
        for path, sha in self.reviewed.items():
            self.bytes({"path": path, "sha256": sha})
        folder = Path(audit_ref["path"]).parent
        self.small = self.local(folder / "capture01.json")
        supplement = self.local(folder / "capture02.json")
        self.remote_root = Path(a["root"]).parent
        require(self.small["root"] == a["root"] and not self.small["links"], "changed capture namespace")
        self.resolver = {}
        for name, item in self.small["files"].items():
            path = Path(name)
            require(not path.is_absolute() and ".." not in path.parts, "unsafe raw capture path")
            self.bind(self.remote_root / path, folder / "files" / path, item)
        train = self.small["assets"]["train"]
        self.bind(train["path"], folder / "files/external-assets/train/train.jsonl", train)
        for remote, item in supplement["files"].items():
            self.bind(remote, Path(item["local"]), item)
        self.large = self.local(folder / "large-object-hashes.json")
        require(self.large["status"] == "PASS_STABLE_REMOTE_OBJECT_SHA256", "unhashed remote checkpoint")
        self.closure = self.document(a["closure_audit"])
        require(self.closure["status"] == "PASS_FRESH_REMOTE_HASH_AND_RECEIPT_CLOSURE", "incomplete checkpoint closure")
        postcheck = self.document(a["capture"]["actual_small_source_postcheck"])
        require(
            postcheck["status"] == "PASS_ALL_CAPTURED_SOURCE_BYTES_AND_NAMESPACES_UNCHANGED"
            and a["capture"]["original_sources_immutable"] is True,
            "capture sources changed",
        )

    def bytes(self, ref):
        self.references[ref["path"]] = ref
        return self.store.bytes(ref)

    def document(self, ref):
        if hasattr(self, "reviewed"):
            require(self.reviewed.get(ref["path"]) == ref["sha256"], "unreviewed capture document")
        return read_json(self.bytes(ref))

    def local(self, name):
        return self.document({"path": str(name), "sha256": self.reviewed[str(name)]})

    def bind(self, remote, local, item):
        ref = {"path": str(local), "sha256": item["sha256"]}
        require(self.reviewed.get(str(local)) == ref["sha256"], "capture review omitted raw source")
        require(len(self.bytes(ref)) == item["bytes"], "raw capture size changed")
        require(str(remote) not in self.resolver, "ambiguous remote source")
        self.resolver[str(remote)] = ref

    def raw(self, remote):
        require(str(remote) in self.resolver, "uncaptured producer record: " + str(remote))
        return self.bytes(self.resolver[str(remote)])

    def json(self, remote):
        return read_json(self.raw(remote))

    def rows(self, remote):
        return [read_json(row) for row in self.raw(remote).split(b"\n") if row]

    def sha(self, remote):
        return hashlib.sha256(self.raw(remote)).hexdigest()


def join_phases(phases, logs, updates):
    """Exclude the replay; reject gaps, overlapping phases and invented updates."""
    require(len(phases) == len(logs) == 4, "requires four authoritative phases")
    previous = 0
    joined = [[] for _ in range(4)]
    for phase, rank_logs in zip(phases, logs, strict=True):
        start, end = phase["updates"]
        require(type(start) is int and type(end) is int and start == previous + 1 <= end, "phase overlap/gap")
        require(len(rank_logs) == 4, "missing phase rank")
        for rank, rows in enumerate(rank_logs):
            require([r["step"] for r in rows] == list(range(start, end + 1)), "incomplete phase updates")
            joined[rank].extend(rows)
        previous = end
    require(previous == updates, "unfinished schedule")
    return joined


def checkpoint_closure(capture, final_checkpoint):
    """Reconcile all five generations with freshly measured remote inventories."""
    c = capture
    generations = c.closure["checkpoint_generations"]
    require(len(generations) == 5, "missing boundary checkpoint")
    require(
        {Path(g["path"]).name for g in generations} == {f"iter_{s:06d}" for s in [348, 467, 600, 850, 935]},
        "wrong checkpoint generations",
    )
    expected = {"data.pt", "trainer.pt", "model/.metadata", "optim/.metadata"}
    expected |= {f"{part}/__{rank}_0.distcp" for part in ["model", "optim"] for rank in range(4)}
    require(sum(g["path"] == final_checkpoint for g in generations) == 1, "wrong final checkpoint")
    for g in generations:
        require(set(g["inventory"]) == expected and g["files"] == 12, "incomplete checkpoint")
        require(g["bytes"] == sum(v["bytes"] for v in g["inventory"].values()), "checkpoint byte census differs")
        for name, item in g["inventory"].items():
            remote = str(Path(g["path"]) / name)
            if remote in c.resolver:
                require(
                    c.sha(remote) == item["sha256"] and len(c.raw(remote)) == item["bytes"], "stale checkpoint metadata"
                )
            else:
                measured = c.large["files"][remote]
                require(
                    measured["sha256"] == item["sha256"]
                    and measured["bytes"] == measured["pre_post_stat"]["st_size"] == item["bytes"] > 0
                    and re.fullmatch("[0-9a-f]{64}", item["sha256"]) is not None,
                    "changed remote tensor inventory",
                )


def normalize(root, contract_cell, audit_ref):
    c = Capture(root, audit_ref)
    a, read = c.audit, c.json
    terminal = Path(a["root"])
    plan = read(terminal / "inputs/plan.json")
    original, prior = Path(plan["original_root"]), Path(plan["terminal_retry"]["previous_root"])
    spec, old = read(terminal / "inputs/cell.json"), read(original / "inputs/plan.json")
    cell, identity = spec["cell"], contract_cell["identity"]
    require(plan["format"] == "segmented-mistral-committed-resume-v1", "unsupported segmented format")
    require(a["cell"] == plan["cell"] == cell["id"] == contract_cell["id"] == CELL, "unadmitted segmented cell")
    require(
        spec["binding"]["id"] == "mistral_hf" and spec["binding"]["dataset"] == "gsm8k", "unadmitted dataset/binding"
    )
    require(c.sha(terminal / "inputs/plan.json") == a["plan_sha256"], "wrong audited plan")
    require(c.sha(original / "inputs/plan.json") == plan["original_plan_sha256"], "changed original ancestry")
    require(
        c.sha(prior / "inputs/plan.json") == plan["terminal_retry"]["previous_plan_sha256"],
        "changed segmented ancestry",
    )
    require(
        len({original, prior, terminal}) == 3 and all(p.parent == c.remote_root for p in [original, prior, terminal]),
        "foreign/aliased lineage",
    )
    require(digest(cell) == identity["inventory_cell_sha256"], "inventory changed")
    require(
        cell["optimizer_updates"] == 935
        and cell["training_epochs"] == 1
        and cell["checkpoint_contract"]["mid_epoch_update"] == 467,
        "unadmitted schedule",
    )
    require(
        cell["micro_batch_size"] == 1 and cell["grad_accum"] == 2 and cell["topology"]["world_size"] == 4,
        "changed global batch",
    )
    for base in [original, prior, terminal]:
        value = read(base / "inputs/plan.json")
        require(
            c.sha(base / "inputs/cell.json") == identity["config_sha256"] == plan["config_sha256"], "config changed"
        )
        for key in ["source_revision", "wheel_sha256", "image_sha256", "runtime", "cell"]:
            require(value[key] == plan[key], "segment changed source/runtime")
        for name, expected in value["files_sha256"].items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe payload name")
            require(c.sha(base / "inputs" / name) == expected, "executed payload changed")
    for key in ["failed_files_sha256", "previous_segments_sha256"]:
        for name, sha in plan["terminal_retry"][key].items():
            require(c.sha(prior / name) == sha, "preserved infrastructure history changed")
    source = {
        "revision": plan["source_revision"],
        "wheel_sha256": plan["wheel_sha256"],
        "runtime_sha256": plan["image_sha256"],
    }
    require(
        source
        == {
            "revision": a["actual_source"]["source_revision"],
            "wheel_sha256": a["actual_source"]["wheel_sha256"],
            "runtime_sha256": a["actual_source"]["image_sha256"],
        },
        "audit source mismatch",
    )
    training, replay, result = [
        read(terminal / "output" / n)
        for n in ["primary-training-receipt.json", "continuation-receipt.json", "result.json"]
    ]
    require(
        training["status"] == "FULL_TRAINING_COMPLETE_FROM_VERIFIED_SEGMENTS_DEPLOYMENT_PENDING", "incomplete primary"
    )
    require(
        result["status"] == "SEGMENTED_FULL_TRAINING_AND_RESTART_COMPLETE_DEPLOYMENT_PENDING"
        and result["full_campaign_pass"] is False,
        "wrong terminal result",
    )
    require(result["parent_plan_sha256"] == training["parent_plan_sha256"] == a["plan_sha256"], "receipt plan mismatch")
    for name, key in [
        ("primary-training-receipt.json", "primary_receipt_sha256"),
        ("continuation-receipt.json", "continuation_receipt_sha256"),
    ]:
        require(c.sha(terminal / "output" / name) == result[key] == a[key], "receipt chain changed")
    for receipt in [training, replay]:
        for name, sha in receipt["evidence"].items():
            require(c.sha(name) == sha, "stale execution receipt")
    checkpoint_closure(c, training["checkpoint"])
    datasets, tokens_by_id, fingerprints, native = validate_datasets(c, original, old, spec)
    initial_manifest = read(original / "output/primary/audit/initial-portable/adapter_manifest.json")
    expected_identity = initial_manifest["identity"]
    require(
        same(expected_identity["base"], identity["base"])
        and same(expected_identity["processor"], identity["processor"]),
        "original portable model/processor differs from contract",
    )
    admission = {"expected_adapter_identity": expected_identity}
    folders = [
        original / "output/primary",
        prior / "output/segment-0/training",
        prior / "output/segment-1/training",
        terminal / "output/segment-2/training",
        terminal / "output/continuation",
    ]
    require(
        [p["updates"] for p in a["phases"]] == [[1, 348], [349, 600], [601, 850], [851, 935], [468, 470]],
        "wrong logical/replay ranges",
    )
    logs, first = [], {}
    for index, (folder, phase) in enumerate(zip(folders, a["phases"], strict=True)):
        start, end = phase["updates"]
        rank_logs = []
        for rank in range(4):
            audit = folder / "audit"
            initial = validate_rank(
                c,
                audit,
                spec=spec,
                plan=terminal,
                source=source,
                job=phase["job"],
                hosts=phase["hosts"],
                admission=admission,
                fingerprints=fingerprints,
                rank=rank,
                mode="branch" if index == 4 else "full",
                final_step=None if index == 0 else end,
            )
            clock = initial["initial_clock"]
            require(
                clock["iter"] == clock["optimizer_step"] == clock["attempt_id"] == start - 1 and clock["accum"] == 0,
                "wrong restored update boundary",
            )
            require(initial["strict_determinism"] is True, "changed determinism")
            require(
                {x["name"] for x in initial["resolved"]["targets"]} == set(cell["resolved_target_names"]),
                "actual target placement differs",
            )
            if rank in first:
                for key in ["resolved", "adapter_identity", "optimizer", "parameter_layouts", "frozen_sha256"]:
                    # LR advances with the schedule; ownership/settings remain
                    # checked by the original exact-restoration producer.
                    if key != "optimizer":
                        require(same(first[rank][key], initial[key]), "segment changed parameter identity")
            else:
                first[rank] = initial
            if index in [1, 2, 3]:
                restored = read(audit / f"segment-restored-rank{rank}.json")
                require(
                    restored["status"] == "PASS_EXACT_SEGMENT_RESTORE_BEFORE_FORWARD" and restored["step"] == start - 1,
                    "segment warm-started or failed restore",
                )
            if index == 4:
                require(
                    read(audit / f"restored-state-rank{rank}.json")["status"] == "PASS", "replay failed exact restore"
                )
            rank_logs.append(c.rows(audit / f"coverage-rank{rank}.jsonl"))
        logs.append(rank_logs)
    joined = join_phases(a["phases"][:4], logs[:4], 935)
    for rank in range(4):
        require(
            same(joined[rank], c.rows(terminal / f"output/authoritative-coverage/coverage-rank{rank}.jsonl")),
            "coverage splice differs",
        )
    counts, pads, tokens, curve = validate_coverage(joined, datasets, tokens_by_id, native, cell)
    coverage = read(terminal / "output/primary-coverage.json")
    require(c.sha(terminal / "output/primary-coverage.json") == training["coverage_sha256"], "coverage receipt changed")
    require(
        coverage["per_id_counts"] == [dict(v) for v in counts] and coverage["valid_tokens"] == tokens,
        "coverage census differs",
    )
    validation = []
    suffix = Path("adapter_all_family_two_node/mistral_hf") / CELL / "campaign/metrics.jsonl"
    for folder, step in [(folders[0], 0), (folders[3], 935)]:
        rows = [v for v in c.rows(folder / suffix) if v["step"] == step and "val/loss_avg" in v.get("metrics", v)]
        require(len(rows) == 1, "missing/duplicate validation boundary")
        m = rows[0].get("metrics", rows[0])
        require(
            math.isfinite(m["val/loss_avg"])
            and m["val/valid_sequence_count"] == len(datasets["validation"])
            and m["val/valid_token_count"] == native["validation"]["loss_tokens"],
            "incomplete validation denominator",
        )
        validation.extend(rows)
    require(same(validation, training["initial_final_validation"]), "validation splice differs")
    require(
        replay["status"] == "PASS_EXACT_THREE_UPDATE_CONTINUATION"
        and replay["replay_is_not_primary_coverage"] is True
        and type(replay["process"]["returncode"]) is int
        and replay["process"]["returncode"] == 0,
        "incomplete continuation",
    )
    require(same(replay["process"], read(terminal / "output/continuation-process.json")), "replay process changed")
    for rank in range(4):
        require([x["step"] for x in logs[4][rank]] == [468, 469, 470], "partial replay")
        for offset, step in enumerate([468, 469, 470]):
            original_row, resumed_row = joined[rank][step - 1], logs[4][rank][offset]
            r = read(folders[4] / f"audit/resume-update-{step}-rank{rank}.json")
            require(
                r["status"] == "PASS"
                and r["atol"] == 1e-7
                and r["rtol"] == 1e-5
                and r["reference"] == str(folders[1] / "audit"),
                "wrong replay oracle/gates",
            )
            require(same(original_row["inputs"], resumed_row["inputs"]), "replay consumed different examples")
            loss = original_row["metrics"]["train/loss_avg"]
            error = abs(loss - resumed_row["metrics"]["train/loss_avg"])
            require(
                math.isfinite(error) and error == r["loss_error"] and error <= 1e-6 + 1e-5 * abs(loss),
                "replay loss failed",
            )
    lineage = {"primary_root": str(terminal), "primary_job": a["job"], "primary_plan_sha256": a["plan_sha256"]}
    history = {
        "kind": "four_phase_segmented_primary",
        "phases": a["phases"],
        "preserved_failures": a["preserved_failures"],
        "audit": audit_ref,
        "diagnostic_replayed_rows": 24,
        "allocated_gpu_hours": a["validated_measurements"]["allocated_gpu_hours_all_five_jobs"],
        "remote_tensor_scope": "Actual installed comparisons plus independently reviewed remote hash/namespace audit; no new tensor reload by normalizer.",
    }
    values = {
        "full_schedule": {
            "logical_updates": 935,
            "epochs": 1,
            "finite_applied_updates": True,
            "frozen_preserved": True,
            "two_physical_hosts": True,
            "loss_curve": curve,
        },
        "complete_sample_coverage": {
            "real_rows_per_epoch": [sum(v.values()) for v in counts],
            "zero_weight_padding_per_epoch": pads,
            "each_expected_id_once_per_epoch": True,
            "denominator_census_matches": True,
            "recovery_replays_excluded": True,
            "data_contract_sha256": identity["dataset_sha256"],
            "per_id_counts": [dict(v) for v in counts],
            "valid_tokens": tokens,
        },
        "initial_final_validation": {
            "steps": [0, 935],
            "counts": [1319, 1319],
            "complete_id_census": True,
            "native_denominators_verified": True,
            "finite_losses": True,
            "validation": validation,
            "census_basis": "Complete immutable split/native preprocessing census and exact global counts; no per-row runtime validation trace claimed.",
        },
        "mid_epoch_continuation": {
            "mid_update": 467,
            "updates_compared": 3,
            "same_topology": True,
            "input_ids_exact": True,
            "rng_exact": True,
            "cursor_exact": True,
            "clock_scheduler_exact": True,
            "loss_within_gates": True,
            "parameters_within_gates": True,
            "optimizer_within_gates": True,
            "initial_factors_unchanged": True,
            "source_immutable": True,
            "gates_sha256": GATES_SHA,
        },
    }
    for value in values.values():
        value["recovery_chain"] = history
    return values, source, lineage, list(c.references.values())


def produce(root, contract, output, *, audit_review, producer_review):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root), "output escapes evidence root")
    store = Store(root)
    reviewed = store.json(producer_review)
    require(
        reviewed["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS" and not reviewed["open_findings"], "unreviewed producer"
    )
    for name in ["segmented.py", "recovered.py", "evidence.py", "primary.py"]:
        require(
            reviewed["reviewed_files"].get("tools/adapter_campaign/" + name)
            == hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest(),
            "producer/dependency changed after review",
        )
    require(reviewed["reviewed_files"].get(audit_review["path"]) == audit_review["sha256"], "unadmitted capture audit")
    producer = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cell = next(c for c in contract["cells"] if c["id"] == CELL)
    values, source, lineage, supporting = normalize(root, cell, audit_review)
    result = {"lineage": lineage, "final_checkpoint": None, "gates": {}}
    for gate, measurement in values.items():
        requirement = next(r for r in cell["requirements"] if r["id"] == gate)
        require(
            source in requirement["sources"] and producer in requirement["producers"], "producer/source not admitted"
        )
        reviews = producer_reviews(requirement, producer)
        require(not reviews or producer_review in reviews, "wrong producer review admission")
        closure = list({r["path"]: r for r in [*supporting, producer_review, *reviews]}.values())
        for ref in closure:
            store.bytes(ref)
        folder = output / CELL / gate
        m = write(root, folder / "measurement.json", measurement)
        common = {
            "identity": cell["identity"],
            "source": source,
            "profile": None,
            "producer_sha256": producer,
            "lineage": lineage,
            "final_checkpoint": None,
            "supporting_evidence": closure,
        }
        p = write(root, folder / "provenance.json", {**common, "documents_sha256": {"measurement": m["sha256"]}})
        record = {
            **common,
            "format": "adapter-campaign-gate-v1",
            "synthetic": False,
            "gate": gate,
            "outcome": "PASS",
            "executed": True,
            "documents": {"measurement": m, "provenance": p},
        }
        result["gates"][gate] = [write(root, folder / "receipt.json", record)]
    index = {
        "format": "adapter-campaign-evidence-index-v1",
        "contract_sha256": digest(contract),
        "cells": {CELL: result},
    }
    write(root, output / "index.json", index)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "contract", "output", "audit-review", "producer-review"]:
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    produce(
        root,
        read_json(Path(args.contract).read_bytes()),
        args.output,
        audit_review=file_ref(root, Path(args.audit_review).resolve()),
        producer_review=file_ref(root, Path(args.producer_review).resolve()),
    )
