# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Normalize preserved complete text primaries without rerunning training.

This producer rederives coverage from each rank's raw log and the full immutable
dataset. It admits no deployment, recovery or task result. Historical execution
revision and primary lineage remain unchanged by the collector revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from tools.adapter_campaign.evidence import Store, digest, read_json, require, same


def file_ref(root, path):
    path = Path(path)
    return {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(root, path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.write("\n")
    return file_ref(root, path)


def normalize_primary(root, contract_cell, capture, directory):
    """Return three independently derived stage measurements and raw references."""
    store = Store(root)
    records = []
    capture_root = Path(capture).parent / "files"
    capture_ref = file_ref(root, capture)
    records.append(capture_ref)
    receipt = store.json(capture_ref)
    require(receipt["status"] == "PASS_EXACT_RAW_EVIDENCE_CAPTURE", "uncommitted raw capture")
    capture_plan_ref = {
        "path": str((Path(capture).parent / "plan.json").relative_to(root)),
        "sha256": receipt["plan_sha256"],
    }
    records.append(capture_plan_ref)
    capture_plan = store.json(capture_plan_ref)
    require(set(capture_plan["files"]) == set(receipt["files_sha256"]), "capture inventory differs from capture plan")
    require(
        all(receipt["files_sha256"].get(n) == h for n, h in capture_plan["expected_sha256"].items()),
        "capture differs from preserved historical evidence",
    )
    audit_ref = {
        "path": "evidence/completed-primary-readonly-audit01.json",
        "sha256": capture_plan["source_audit_sha256"],
    }
    records.append(audit_ref)
    historical = store.json(audit_ref)
    require(historical["returncode"] == 0, "historical read-only audit failed")
    historical_rows = {r["cell"]: r for r in read_json(historical["stdout"])["rows"]}

    def raw(name):
        name = str(name)
        require(name in receipt["files_sha256"], "uncaptured raw input: " + name)
        ref = {"path": str((capture_root / name).relative_to(root)), "sha256": receipt["files_sha256"][name]}
        records.append(ref)
        return store.bytes(ref)

    base = Path(directory)
    plan_bytes = raw(base / "inputs/plan.json")
    plan = read_json(plan_bytes)
    spec_bytes = raw(base / "inputs/cell.json")
    spec = read_json(spec_bytes)
    identity = contract_cell["identity"]
    require(plan["cell"] == contract_cell["id"] == spec["cell"]["id"], "primary cell mismatch")
    require(hashlib.sha256(spec_bytes).hexdigest() == identity["config_sha256"], "primary changed frozen config")
    require(digest(spec["cell"]) == identity["inventory_cell_sha256"], "primary changed frozen cell")
    for name, expected in plan["files_sha256"].items():
        require(hashlib.sha256(raw(base / "inputs" / name)).hexdigest() == expected, "changed primary launch inputs")
    process = read_json(raw(base / "output/primary-process.json"))
    historical_row = historical_rows[contract_cell["id"]]
    require(
        historical_row["spec_sha256"] == identity["config_sha256"]
        and historical_row["primary_process_sha256"]
        == receipt["files_sha256"][str(base / "output/primary-process.json")],
        "primary differs from historical audit anchor",
    )
    require(process["returncode"] == 0 and process["status"] == "COMMAND_COMPLETE", "primary did not finish")
    provenance = process["provenance"]
    require(provenance["plan_sha256"] == hashlib.sha256(plan_bytes).hexdigest(), "wrong execution plan")
    require(
        provenance["config_sha256"] == identity["config_sha256"] and provenance["cell_id"] == contract_cell["id"],
        "wrong execution config",
    )
    source = {
        "revision": plan["source_revision"],
        "wheel_sha256": plan["wheel_sha256"],
        "runtime_sha256": plan["image_sha256"],
    }
    for key in ("source_revision", "wheel_sha256", "image_sha256"):
        require(provenance[key] == plan[key], "primary runtime/source mismatch")
    lineage = {
        "primary_job": provenance["job"],
        "primary_root": plan["root"],
        "primary_plan_sha256": provenance["plan_sha256"],
    }
    require(
        historical_row["status"] == "PASS_COMPLETE_PRIMARY_ONLY"
        and historical_row["root"] == lineage["primary_root"]
        and historical_row["job"] == lineage["primary_job"]
        and historical_row["source_revision"] == source["revision"],
        "historical audit execution differs from primary",
    )
    for name, expected in historical_row["update_evidence"].items():
        captured_name = Path(name).relative_to(capture_plan["remote_root"])
        require(
            hashlib.sha256(raw(captured_name)).hexdigest() == expected,
            "raw update evidence differs from historical audit",
        )
    for name, expected in historical_row["coverage"]["input_files"].items():
        require(
            hashlib.sha256(raw(base / "output/primary/audit" / name)).hexdigest() == expected,
            "coverage log differs from historical audit",
        )
    cell = spec["cell"]
    require(spec["binding"]["dataset"] == "gsm8k", "this producer only admits the preserved text objective")
    require(cell["micro_batch_size"] == 1 and cell["topology"]["world_size"] == 4, "unadmitted primary shape")
    updates, epochs = cell["optimizer_updates"], cell["training_epochs"]
    identities = read_json(raw(base / "inputs/proofs/data-identities.json"))
    remote_root = str(Path(plan["root"]).parent)
    datasets = {}
    for split in ("train", "validation"):
        item = identities[split]
        require(item["path"] == spec["paths"][split], "census does not name the actual configured split")
        data_bytes = raw(Path(item["path"]).relative_to(remote_root))
        require(hashlib.sha256(data_bytes).hexdigest() == item["sha256"], "dataset content changed")
        datasets[split] = [read_json(line) for line in data_bytes.split(b"\n") if line]
        ids = [row["id"] for row in datasets[split]]
        require(len(ids) == len(set(ids)) == spec["dataset"]["rows_" + split], "wrong complete split census")
        require(
            all(
                set(row) == {"id", "messages"} and all(isinstance(m["content"], str) for m in row["messages"])
                for row in datasets[split]
            ),
            "producer only admits the frozen plain-text dataset fingerprint",
        )
    fingerprints = {split: digest({"rows": rows, "media": {}}) for split, rows in datasets.items()}
    census = read_json(raw(base / "inputs/proofs/preprocessing-census.json"))
    require(census["status"] == "PASS", "preprocessing census failed")
    native = {r["split"]: r for r in census["summaries"]}
    for split in datasets:
        r = native[split]
        require(
            r["ok"] is True and r["failed"] == r["unmeasured_rows"] == r["over_long_rows"] == 0,
            "incomplete preprocessing",
        )
        require(r["rows"] == r["matched_rows"] == len(datasets[split]), "partial preprocessing census")
        require(
            plan["expected_" + split.replace("validation", "validation").replace("train", "train") + "_valid_tokens"]
            == r["loss_tokens"],
            "wrong native denominator",
        )
    report_bytes = raw(base / "inputs/proofs/preprocessing-rows.jsonl")
    require(hashlib.sha256(report_bytes).hexdigest() == census["report_sha256"], "preprocessing rows changed")
    preprocessing = [read_json(line) for line in report_bytes.split(b"\n") if line]
    loss_tokens = {}
    for split, dataset in datasets.items():
        rows = [r for r in preprocessing if r.get("split") == split]
        require(
            [r["row_id"] for r in rows] == [r["id"] for r in dataset]
            and [r["index"] for r in rows] == list(range(len(dataset))),
            "preprocessing omitted/reordered/repeated an ID",
        )
        require(
            all(
                r["mask_matches"] is True
                and not r["reasons"]
                and type(r["loss_tokens"]) is int
                and r["loss_tokens"] > 0
                for r in rows
            ),
            "invalid native mask/token census",
        )
        require(
            sum(r["loss_tokens"] for r in rows) == native[split]["loss_tokens"],
            "preprocessing token total differs from rows",
        )
        loss_tokens[split] = {r["row_id"]: r["loss_tokens"] for r in rows}
    admission = read_json(raw(base / "inputs/readiness-seal.json"))
    require(
        admission["config_sha256"] == identity["config_sha256"] and admission["source_revision"] == source["revision"],
        "wrong primary admission",
    )
    rank_logs, expected_peers = [], None
    for rank in range(4):
        audit = base / "output/primary/audit"
        p = read_json(raw(audit / f"provenance-rank{rank}.json"))
        require(
            p["source_revision"] == source["revision"]
            and p["wheel_sha256"] == source["wheel_sha256"]
            and p["container_sha256"] == source["runtime_sha256"],
            "rank source mismatch",
        )
        require(
            p["campaign_spec_sha256"] == identity["config_sha256"]
            and p["mode"] == "full"
            and p["slurm_job_id"] == lineage["primary_job"],
            "rank execution mismatch",
        )
        require(
            p["harness_sha256"] == plan["files_sha256"]["campaign_full_run_v1.py"],
            "executed harness differs from sealed plan",
        )
        require(p["verified_package_files"] > 0 and p["source_dirty"] is False, "installed package was not attested")
        initial = read_json(raw(audit / f"initial-rank{rank}.json"))
        require(
            initial["data_identity"] == fingerprints["train"]
            and initial["validation_identity"] == fingerprints["validation"],
            "executed dataset fingerprints differ from complete split",
        )
        require(
            same(initial["adapter_identity"], admission["expected_adapter_identity"]),
            "actual adapter/processor execution differs from reviewed admission",
        )
        require(
            same(initial["adapter_identity"]["base"], identity["base"])
            and same(initial["adapter_identity"]["processor"], identity["processor"]),
            "actual base/processor differs from immutable binding",
        )
        peers = initial["topology"]["peers"]
        require(
            len(peers) == 4 and (expected_peers is None or same(peers, expected_peers)),
            "ranks disagree on physical topology",
        )
        expected_peers = peers
        require(bool(initial["topology"]["layouts"]), "no distributed parameter layout")
        owners = [n for group in initial["optimizer"] for n in group["names"]]
        require(
            len(owners) == len(set(owners)) and set(owners) == set(initial["parameter_layouts"]),
            "optimizer ownership mismatch",
        )
        require(
            {x["rank"] for x in peers} == set(range(4)) and len({x["gpu_uuid"] for x in peers}) == 4,
            "invalid physical ranks",
        )
        require(sorted(Counter(x["host"] for x in peers).values()) == [2, 2], "requires two actual equal-GPU hosts")
        require(
            {x["host"] for x in peers} == set(provenance["hosts"])
            and all(x["job"] == lineage["primary_job"] for x in peers),
            "wrong scheduler allocation",
        )
        for layout in initial["topology"]["layouts"].values():
            groups = layout["groups"]
            for group in groups.values():
                require(
                    rank in group["ranks"] and set(group["hosts"]) == {peers[r]["host"] for r in group["ranks"]},
                    "collective groups do not match actual rank placement",
                )
            if cell["topology"]["layout"] == "fsdp2":
                require(
                    layout["shape"] == [4]
                    and groups["0"]["ranks"] == list(range(4))
                    and len(groups["0"]["hosts"]) == 2
                    and groups["0"]["placement"].startswith("S"),
                    "FSDP must shard across hosts",
                )
            else:
                require(
                    layout["shape"] == [2, 2]
                    and len(groups["0"]["hosts"]) == 2
                    and groups["0"]["placement"] == "R"
                    and len(groups["1"]["hosts"]) == 1
                    and groups["1"]["placement"].startswith("S")
                    and all(len(g["ranks"]) == 2 for g in groups.values()),
                    "HSDP mesh does not match physical hosts",
                )
        finished = read_json(raw(audit / f"completed-updates-rank{rank}.json"))
        require(
            finished["status"] == "UPDATE_CHECKS_PASS_TASK_AND_EXPORT_PENDING", "completed update explicitly failed"
        )
        require(
            set(finished["finite_gradients"]) == set(finished["changed"]) == set(owners)
            and all(v is True for v in finished["changed"].values()),
            "missing intended parameter gradient or multi-update change",
        )
        require(
            finished["step"] == updates
            and finished["mode"] == "full"
            and finished["frozen_preserved"] is True
            and finished["finite_gradients"],
            "incomplete final rank update checks",
        )
        require(
            read_json(raw(audit / f"public-result-rank{rank}.json"))["final_step"] == updates,
            "public trainer stopped early",
        )
        rank_logs.append([read_json(line) for line in raw(audit / f"coverage-rank{rank}.jsonl").split(b"\n") if line])
    train_ids = [r["id"] for r in datasets["train"]]
    global_batch = 4 * cell["grad_accum"]
    per_epoch = math.ceil(len(train_ids) / global_batch)
    require(updates == per_epoch * epochs, "wrong full epoch schedule")
    require(
        all([r["step"] for r in log] == list(range(1, updates + 1)) for log in rank_logs), "missing/repeated updates"
    )
    counts, pads, tokens = [Counter() for _ in range(epochs)], [0] * epochs, 0
    curve = []
    for i in range(updates):
        e = i // per_epoch
        reference = rank_logs[0][i]["metrics"]
        real, expected_tokens = 0, 0
        for rank in range(4):
            record = rank_logs[rank][i]
            m = record["metrics"]
            require(
                math.isfinite(m["train/loss_avg"])
                and abs(m["train/loss_avg"] - reference["train/loss_avg"])
                <= 1e-6 + 1e-5 * abs(reference["train/loss_avg"]),
                "rank loss mismatch/nonfinite",
            )
            require(
                m["optim/update_outcome"] == "applied"
                and m["optim/step_skipped"] == m["optim/nonfinite_skips_total"] == 0
                and m["optim/optimizer_step"] == m["optim/attempt_id"] == i + 1,
                "skipped/missing update",
            )
            require(
                m["train/valid_token_count"] == reference["train/valid_token_count"]
                and m["train/valid_sequence_count"] == reference["train/valid_sequence_count"],
                "rank objective denominator mismatch",
            )
            require(len(record["inputs"]) == cell["grad_accum"], "wrong microbatch census")
            for item in record["inputs"]:
                require(
                    item["epoch"] == e and item["sample_id"] == train_ids[item["source_index"]], "wrong actual example"
                )
                require(
                    type(item["padding"]) is bool and item["weight"] == float(not item["padding"]),
                    "padding contributed to loss",
                )
                if item["padding"]:
                    pads[e] += 1
                else:
                    counts[e][item["sample_id"]] += 1
                    real += 1
                    expected_tokens += loss_tokens["train"][item["sample_id"]]
        require(
            real == reference["train/valid_sequence_count"] and reference["train/valid_token_count"] > 0,
            "incorrect objective census",
        )
        require(
            reference["train/valid_token_count"] == expected_tokens,
            "per-update native objective denominator differs from actual examples",
        )
        tokens += reference["train/valid_token_count"]
        curve.append(
            {"step": i + 1, "loss": reference["train/loss_avg"], "valid_tokens": reference["train/valid_token_count"]}
        )
    require(all(count == Counter(train_ids) for count in counts), "missing/duplicated logical examples")
    require(
        pads == [per_epoch * global_batch - len(train_ids)] * epochs
        and tokens == epochs * native["train"]["loss_tokens"],
        "tail or token denominator mismatch",
    )
    metrics = (
        base
        / "output/primary/adapter_all_family_two_node"
        / spec["binding"]["id"]
        / cell["id"]
        / "campaign/metrics.jsonl"
    )
    require(
        hashlib.sha256(raw(metrics)).hexdigest() == historical_row["metrics_sha256"],
        "validation metrics differ from historical audit",
    )
    validation = [
        r
        for r in (read_json(line) for line in raw(metrics).split(b"\n") if line)
        if "val/loss_avg" in r.get("metrics", r)
    ]
    require([r["step"] for r in validation] == [0, updates], "missing initial/final validation")
    for row in validation:
        m = row.get("metrics", row)
        require(
            math.isfinite(m["val/loss_avg"])
            and m["val/valid_sequence_count"] == len(datasets["validation"])
            and m["val/valid_token_count"] == native["validation"]["loss_tokens"],
            "invalid/incomplete native validation",
        )
    measurements = {
        "full_schedule": {
            "logical_updates": updates,
            "epochs": epochs,
            "finite_applied_updates": True,
            "frozen_preserved": True,
            "two_physical_hosts": True,
            "loss_curve": curve,
        },
        "complete_sample_coverage": {
            "real_rows_per_epoch": [sum(x.values()) for x in counts],
            "zero_weight_padding_per_epoch": pads,
            "each_expected_id_once_per_epoch": True,
            "denominator_census_matches": True,
            "recovery_replays_excluded": True,
            "data_contract_sha256": identity["dataset_sha256"],
            "per_id_counts": [dict(x) for x in counts],
            "valid_tokens": tokens,
        },
        "initial_final_validation": {
            "steps": [r["step"] for r in validation],
            "counts": [len(datasets["validation"])] * 2,
            "complete_id_census": True,
            "native_denominators_verified": True,
            "finite_losses": True,
            "validation": validation,
            "census_basis": "Complete immutable unique-ID split and complete native preprocessing census, exhaustive production validation iterator, exact global row/token counts; no per-row validation runtime trace is claimed.",
        },
    }
    return measurements, source, lineage, list({r["path"]: r for r in records}.values())


def produce(root, contract, capture, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root), "output must be in evidence root")
    receipt = read_json(Path(capture).read_bytes())
    directories = sorted(
        {Path(n).parts[0] for n in receipt["files_sha256"] if n.endswith("/output/primary-process.json")}
    )
    producer = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    index = {"format": "adapter-campaign-evidence-index-v1", "contract_sha256": digest(contract), "cells": {}}
    cells = {c["id"]: c for c in contract["cells"]}
    for directory in directories:
        plan = read_json((Path(capture).parent / "files" / directory / "inputs/plan.json").read_bytes())
        cell = cells[plan["cell"]]
        values, source, lineage, supporting = normalize_primary(root, cell, capture, directory)
        result = {"lineage": lineage, "final_checkpoint": None, "gates": {}}
        require(cell["id"] not in index["cells"], "ambiguous authoritative primary")
        for gate, measurement in values.items():
            requirement = next(r for r in cell["requirements"] if r["id"] == gate)
            require(
                source in requirement["sources"] and producer in requirement["producers"],
                "producer/source needs independent admission",
            )
            destination = output / cell["id"] / gate
            ref = write(root, destination / "measurement.json", measurement)
            provenance = {
                "identity": cell["identity"],
                "source": source,
                "profile": None,
                "producer_sha256": producer,
                "lineage": lineage,
                "final_checkpoint": None,
                "documents_sha256": {"measurement": ref["sha256"]},
                "supporting_evidence": supporting,
            }
            p = write(root, destination / "provenance.json", provenance)
            record = {
                "format": "adapter-campaign-gate-v1",
                "synthetic": False,
                "identity": cell["identity"],
                "gate": gate,
                "profile": None,
                "source": source,
                "producer_sha256": producer,
                "lineage": lineage,
                "final_checkpoint": None,
                "outcome": "PASS",
                "executed": True,
                "documents": {"measurement": ref, "provenance": p},
                "supporting_evidence": supporting,
            }
            result["gates"][gate] = [write(root, destination / "receipt.json", record)]
        index["cells"][cell["id"]] = result
    write(root, output / "index.json", index)
    return index


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "contract", "capture", "output"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    produce(args.root, read_json(Path(args.contract).read_bytes()), args.capture, args.output)
