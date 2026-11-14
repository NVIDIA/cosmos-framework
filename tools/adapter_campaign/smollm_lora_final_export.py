# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Normalize own-final Smol LoRA merged export, retaining cache failures.

The installed b583 worker already exported, loaded and compared this final935
artifact. CPU job38540430 subsequently interrupted, recovered and reused the
same export. This producer reads receipts only. It never executes a model or
converts a cache failure into deployment, task or whole-cell qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from .evidence import Store, check_receipt, digest, producer_reviews, read_json, require, same
from .final_checkpoint import RAW_FILES, WORKERS, inventory
from .final_checkpoint import Capture as CheckpointCapture
from .primary import file_ref, write

CELL = "llama_hf__lora__factorized_v1__default_v1__fsdp2"
GATE = "final_merged_export"
JOB = "38540430"
ORIGINAL_JOB = "38460748"
ACTUAL_REVIEW = "a46ccadd763d30cd26beeb681c4bda200f4479fe282a8adf3b16cff2030ee9f5"
CAPTURE = "503b6f93eb2e77ce622f98b0c2c7d25c8d117de40ff7e3919b3d9875ceb932ba"
PLAN = "ab95a7d94e56d5d4c1fa3e1925acae476d284e41d0964d223ea1f8b8cdc56705"
SOURCE = {
    "revision": "b58329ccc85949ffc4c9432fbb75d53e13d395b2",
    "wheel_sha256": "f3b2693272d66fc688fd22faeb321479accc18cc628d472785fc86f9aadcfd84",
    "runtime_sha256": "52b7e0001b124b9a7b97aa255b2ee5f500425315c673ca1b8142912f7e9a8f96",
}
POINTS = ("first_shard", "serialized", "marked")
MODES = ("interrupt", "recover", "reuse")
ASSERTIONS = (
    "stock_or_native_supported_loader",
    "complete_tensors_and_assets",
    "precision_identity",
    "live_and_source_immutable",
    "idempotent_reuse",
    "interruption_recovery",
    "peaks_measured",
)
EXTRA_RAW_FILES = (
    "output/merged-fp32/.overlay_source.json",
    "output/merged-fp32/model.safetensors.index.json",
    "output/merged-fp32/config.json",
    "output/stock-reload/result.json",
    "output/stock-reload/provenance.json",
    "output/stock-reload/construction-provenance.json",
    "output/stock-reload/cases.json",
    "output/stock-reload/input-population.json",
)
DEPENDENCIES = (
    "smollm_lora_final_export.py",
    "final_checkpoint.py",
    "inline_midpoint.py",
    "primary.py",
    "recovered.py",
    "evidence.py",
)


class Capture(Store):
    """Small exact copies, separately bound to the original absolute review names.

    The 79MB legacy envelope is deliberately not passed to Store. Its previously
    reviewed full digest and570-text inventory anchor a bounded lossless extraction.
    The exact extraction and all selected raw bytes need independent producer review.
    """

    def __init__(self, root, reference):
        super().__init__(root)
        self.extraction = self.json(reference)
        x = self.extraction
        require(
            x["format"] == "reviewed-smol-lora-final-export-copy-v1" and x["synthetic"] is False,
            "wrong or synthetic export capture",
        )
        require(x["actual_review"]["sha256"] == ACTUAL_REVIEW, "unreviewed actual export")
        self.review = self.json(x["actual_review"])
        review = self.review
        require(
            review["format"] == "smol-own-final-export-recovery-independent-actual-review-v1"
            and review["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS"
            and review["open_findings"] == []
            and review["actual_job"] == JOB
            and review["qualification_pass"] is False,
            "actual export review did not pass",
        )
        self.aliases = x["aliases"]
        require(isinstance(self.aliases, dict) and self.aliases, "missing exact copy map")
        for original, ref in self.aliases.items():
            require(review["reviewed_files"].get(original) == ref["sha256"], "unreviewed copy alias")
            self.bytes(ref)
        self.capture = self.original(x["original_capture"])
        require(self.aliases[x["original_capture"]]["sha256"] == CAPTURE, "wrong actual capture")
        self.source_review = self.original(x["original_source_review"])
        require(
            self.source_review["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS"
            and self.source_review["open_findings"] == [],
            "unreviewed executed worker",
        )
        self.original_files = Path(x["original_files_directory"])
        require(self.original_files.is_absolute(), "ambiguous original source location")
        cap = self.capture
        require(
            cap["job"] == JOB
            and cap["plan_sha256"] == PLAN
            and cap["all_files_freshly_hashed"] is True
            and cap["namespace_unchanged"] is True
            and same(cap["new_gpu_executions"], 0)
            and cap["qualification_pass"] is False,
            "incomplete terminal capture",
        )
        require(
            len(cap["files"]) == 159
            and len(cap["mirrored_small_files"]) == 150
            and len(set(cap["mirrored_small_files"])) == 150,
            "partial or duplicate capture",
        )
        remote = set(cap["files"]) - set(cap["mirrored_small_files"])
        expected = {f"output/recovered-{p}/model-{i:05d}-of-00003.safetensors" for p in POINTS for i in range(1, 4)}
        require(remote == expected, "unmirrored required receipt")
        require(
            set(self.aliases)
            == {
                x["original_capture"],
                x["original_source_review"],
                *(str(self.original_files / n) for n in cap["mirrored_small_files"]),
            },
            "unexpected copy map",
        )
        for name, row in cap["files"].items():
            require(
                not Path(name).is_absolute()
                and ".." not in Path(name).parts
                and type(row["bytes"]) is int
                and row["bytes"] >= 0
                and isinstance(row["sha256"], str)
                and re.fullmatch("[0-9a-f]{64}", row["sha256"]) is not None,
                "invalid captured file hash/length",
            )
        require(same(sum(v["bytes"] for v in cap["files"].values()), cap["total_bytes"]), "capture byte census")
        for name in cap["mirrored_small_files"]:
            raw = self.raw_bytes(name)
            require(len(raw) == cap["files"][name]["bytes"], "captured length changed")
        self.plan = self.raw("inputs/plan.json")
        p = self.plan
        require(self.hash("inputs/plan.json") == PLAN and p["root"] == cap["root"], "wrong plan/root")
        require(
            set(n for n in cap["files"] if n.startswith("inputs/"))
            == {"inputs/plan.json", *("inputs/" + n for n in p["files_sha256"])},
            "open staged namespace",
        )
        stage = str(Path(x["original_source_review"]).parents[1] / "stage02")
        for name, value in {"plan.json": PLAN, **p["files_sha256"]}.items():
            require(
                Path(name).name == name
                and self.hash("inputs/" + name) == value
                and self.source_review["reviewed_files"].get(stage + "/" + name) == value,
                "executed source differs from reviewed stage",
            )

    def original(self, name):
        require(name in self.aliases, "missing reviewed original")
        return self.json(self.aliases[name])

    def hash(self, name):
        return self.capture["files"][name]["sha256"]

    def raw_bytes(self, name):
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe capture name")
        key = str(self.original_files / name)
        require(key in self.aliases and self.aliases[key]["sha256"] == self.hash(name), "missing/stale raw receipt")
        return self.bytes(self.aliases[key])

    def raw(self, name):
        return read_json(self.raw_bytes(name))

    def legacy(self, checkpoint):
        x = self.json(self.extraction["legacy_extraction"])
        require(
            x["format"] == "reviewed-smol-lora-final-export-extraction-v1"
            and x["status"] == "EXACT_CAPTURE_BYTES_EXTRACTED",
            "incomplete legacy extraction",
        )
        for key in (
            "review",
            "original_capture",
            "inventory",
            "envelope",
            "capture_metadata",
            "original_capture_bytes",
            "original_text_files_verified",
        ):
            require(same(x[key], checkpoint.extraction[key]), "different historical capture: " + key)
        self.bytes(x["extractor"])
        entries = self.json(x["inventory"])
        require(same(len(entries), x["original_text_files_verified"]), "partial legacy text census")
        row = x["rows"][CELL]
        require(
            set(x["rows"]) == {CELL} and set(row["files"]) == set(RAW_FILES + EXTRA_RAW_FILES),
            "missing/foreign historical file",
        )
        require(
            all(same(row[k], checkpoint.row[k]) for k in ("cell", "job", "root", "plan_sha256", "row_index")),
            "wrong original artifact execution",
        )
        indexed = {(r["row_index"], r["relative_name"]): r for r in entries}
        require(len(indexed) == len(entries), "duplicate legacy inventory")
        values = {}
        for name, ref in row["files"].items():
            entry = indexed[(row["row_index"], name)]
            require(
                entry["sha256"] == ref["sha256"]
                and entry["capture"] == x["original_capture"]
                and entry["producer_path"] == str(Path(row["root"]) / name),
                "unbound legacy raw bytes",
            )
            values[name] = self.json(ref)
            if name in RAW_FILES:
                require(
                    same(values[name], checkpoint.document(name))
                    and ref["sha256"] == checkpoint.row["files"][name]["sha256"],
                    "changed checkpoint witness",
                )
        self.legacy_row = row
        return values


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def clean_command(record, expected):
    require(
        same(record["returncode"], expected)
        and record["exception"] is None
        and record["cleanup_error"] is None
        and record["cleanup"]["remaining"] == [],
        "unclean command",
    )
    require(
        type(record["peak_owned_process_tree_rss_bytes"]) is int
        and 0 < record["peak_owned_process_tree_rss_bytes"] < 32 * 1024**3
        and type(record["samples"]) is int
        and record["samples"] > 0
        and positive(record["seconds"])
        and "controller RSS excluded" in record["memory_scope"],
        "missing/invalid measured process peaks",
    )


def execution_identity(row, plan):
    require(
        row["job"] == JOB
        and row["plan_sha256"] == PLAN
        and (row["source_revision"], row["wheel_sha256"], row["image_sha256"])
        == (SOURCE["revision"], SOURCE["wheel_sha256"], SOURCE["runtime_sha256"])
        and row["portable"] == plan["portable"]
        and row["portable_manifest_sha256"] == plan["portable_inventory"]["adapter_manifest.json"]["sha256"],
        "different export execution/source/artifact",
    )
    att = row["installed_attestation"]
    require(
        (att["source_revision"], att["wheel_sha256"], att["container_sha256"])
        == (SOURCE["revision"], SOURCE["wheel_sha256"], SOURCE["runtime_sha256"])
        and att["build_input"] == "git archive " + SOURCE["revision"]
        and att["source_dirty"] is False
        and att["slurm_job_id"] == JOB
        and att["environment"] is None
        and same(att["verified_package_files"], 1933)
        and same(att["tracked_package_files_verified"], 1933)
        and same(att["versions"], plan["versions"])
        and att["wheels"]["cosmos_framework-1.2.2-py3-none-any.whl"] == SOURCE["wheel_sha256"],
        "missing installed source attestation",
    )


def tensor_census(comparison, weight_map, original_schema=None):
    require(
        comparison["all_tensors_exact"] is True
        and comparison["complete_assets_exact"] is True
        and comparison["identity_exact"] is True,
        "inexact or incomplete export",
    )
    rows = comparison["tensors"]
    require(
        isinstance(rows, list)
        and same(comparison["tensor_count"], 272)
        and len(rows) == 272
        and len(weight_map) == 272
        and len(set(weight_map.values())) == 3,
        "partial tensor/shard census",
    )
    schema = {}
    for row in rows:
        require(
            set(row) == {"name", "shape", "dtype", "exact"}
            and isinstance(row["name"], str)
            and row["name"] not in schema
            and row["dtype"] == "torch.float32"
            and row["exact"] is True
            and isinstance(row["shape"], list)
            and row["shape"]
            and all(type(n) is int and n > 0 for n in row["shape"]),
            "malformed/duplicate tensor proof",
        )
        schema[row["name"]] = row["shape"]
    require(set(schema) == set(weight_map), "missing or foreign tensor")
    if original_schema is not None:
        require(same(schema, original_schema), "tensor dimensions changed across recovery")
    return schema


def validate_accounting(c):
    cap = c.capture
    for key in ("sacct", "all_step_accounting", "queue", "steps"):
        row = cap[key]
        require(same(row["returncode"], 0) and row["stderr"] == "", "failed scheduler observation")
    rows = [line.split("|") for line in cap["all_step_accounting"]["stdout"].splitlines()]
    require(
        len(rows) == 4
        and all(len(r) == 7 for r in rows)
        and {r[0] for r in rows} == {JOB, JOB + ".batch", JOB + ".extern", JOB + ".0"},
        "incomplete accounting",
    )
    for row in rows:
        require(
            row[1:3] == ["COMPLETED", "0:0"]
            and row[5] == "cpu-00008"
            and row[4] == "cpu=4,mem=32G,node=1"
            and row[6] == ("rarunachalam" if row[0] == JOB else "")
            and row[3] == ("472" if row[0] == JOB + ".0" else "553"),
            "different allocation or unfinished step",
        )
    allocation = next(r for r in rows if r[0] == JOB)
    require(cap["sacct"]["stdout"].splitlines() == ["|".join(allocation)], "inconsistent allocation record")
    for key in ("queue", "steps"):
        require(
            all(not (s == JOB or s.startswith((JOB + ".", JOB + "_"))) for s in cap[key]["stdout"].splitlines()),
            "export allocation still active",
        )
    return {"cpu_seconds": 2212, "gpu_seconds": 0, "allocation_seconds": 553, "cpus": 4, "nodes": 1}


def validate_transactions(c, original):
    p, cap = c.plan, c.capture
    require(
        p["cell"] == CELL
        and same(p["checkpoint_step"], 935)
        and p["execution_profile"] is None
        and p["source_revision"] == SOURCE["revision"]
        and p["wheel_sha256"] == SOURCE["wheel_sha256"]
        and p["image_sha256"] == SOURCE["runtime_sha256"]
        and p["points"] == list(POINTS)
        and p["output_dtype"] == "float32"
        and same(p["merged_tensor_count"], 272)
        and same(p["merged_shard_count"], 3),
        "wrong export plan",
    )
    require(
        same(p["resources"], {"cpus": 4, "gpus": 0, "memory_gib": 32, "nodes": 1, "wall_seconds": 3600}),
        "resource scope changed",
    )
    result, final, complete = [c.raw("output/" + k + ".json") for k in ("result", "host-final", "complete")]
    require(
        same(result, cap["result"])
        and result["status"] == complete["status"] == "PASS_OWN_FINAL_EXPORT_INTERRUPTION_ONLY"
        and final["status"] == "PASS_CPU_EXPORT_RECOVERY_CLOSURE"
        and result["error"] is None
        and final["source_error"] is None
        and result["source_immutable"] is True,
        "incomplete export transaction",
    )
    for row in (result, final, complete):
        require(
            row["job"] == JOB and row["plan_sha256"] == PLAN and row["qualification_pass"] is False,
            "wrong completion identity",
        )
    require(
        result["cell"] == complete["cell"] == CELL
        and result["fixed_mma_v2_credit"] is complete["fixed_mma_v2_credit"] is False
        and all(same(result[k], 0) for k in ("new_training_updates", "full_schedule_credit", "allocated_gpus"))
        and same(complete["new_training_updates"], 0)
        and result["original_numerical_status"]
        == complete["legacy_numerical_status"]
        == p["original_numerical_status"]
        == "FAIL_NUMERICAL_GATES",
        "historical failures or scope changed",
    )
    for key, name in (
        ("result_sha256", "result"),
        ("host_final_sha256", "host-final"),
        ("process_receipt_sha256", "export-recovery-command"),
    ):
        require(complete[key] == c.hash("output/" + name + ".json"), "unbound completion receipt")
    require(result["journal_sha256"] == {n: c.hash(n) for n in ("intent.json", "submission.json")}, "journal binding")
    ack, intent = c.raw("submission.json"), c.raw("intent.json")
    require(
        ack["job_id"] == JOB
        and same(ack["returncode"], 0)
        and ack["stdout"] == JOB + "\n"
        and ack["stderr"] == ""
        and same(ack["command"], intent["command"])
        and ack["plan_sha256"] == intent["plan_sha256"] == PLAN
        and intent["review"]["path"] == c.extraction["original_source_review"]
        and intent["review"]["sha256"] == c.aliases[c.extraction["original_source_review"]]["sha256"],
        "unacknowledged execution",
    )
    before, sc = c.raw("output/host-source-before.json"), c.raw("inputs/source-capture.json")
    require(
        c.hash("inputs/source-capture.json") == p["source_capture_sha256"]
        and before["job"] == JOB
        and before["plan_sha256"] == PLAN
        and same(before["immutable_inputs"], p["immutable_inputs"]),
        "missing source check",
    )
    expected = {
        root: {n: {k: v[k] for k in ("bytes", "sha256")} for n, v in row["files"].items()}
        for root, row in sc["folders"].items()
    }
    require(
        same(before["sources"], expected)
        and len(p["immutable_inputs"]) == 47
        and all(
            p["immutable_inputs"][root + "/" + n] == row["sha256"]
            for root, inv in expected.items()
            for n, row in inv.items()
        ),
        "source folder inventory changed",
    )
    for root, inv in ((p["portable"], p["portable_inventory"]), (p["reference_merged"], p["merged_inventory"])):
        require(same(expected[root], inv), "different immutable artifact")
    process = c.raw("output/export-recovery-command.json")
    clean_command(process, 0)
    require(
        same(process, final["whole_process_resources"])
        and "--container-image=" + p["runtime"] + "/output/runtime.sqsh" in process["command"]
        and same(c.raw("output/export-recovery-ownership.json")["uid"], 41852)
        and type(final["peak_host_controller_rss_bytes"]) is int
        and final["peak_host_controller_rss_bytes"] > 0,
        "missing host ownership or peak",
    )
    require(set(result["cases"]) == set(POINTS), "missing interruption case")
    schema, cases = None, {}
    for point in POINTS:
        case = result["cases"][point]
        event, post = [c.raw("output/" + point + "-" + k + ".json") for k in ("interrupt-event", "post-interruption")]
        execution_identity(event, p)
        owner, child = [c.raw("output/" + point + "-interrupt-" + k + ".json") for k in ("ownership", "child")]
        require(
            same(event["process"], child["identity"])
            and same(event["signal"], 9)
            and event["owned_token_sha256"]
            == hashlib.sha256(child["token"].encode()).hexdigest()
            == hashlib.sha256(owner["token"].encode()).hexdigest(),
            "not the owned interruption",
        )
        dest = p["root"] + "/output/recovered-" + point
        require(
            event["point"] == point
            and event["destination"] == dest
            and event["lock"]["holder"]["pid"] == child["identity"]["pid"]
            and event["lock"]["holder"]["host"] == "cpu-00008"
            and event["lock"]["holder"]["source"] == p["portable"]
            and event["lock"]["holder"]["out"] == dest
            and event["lock"]["released"] is False
            and post["lock"]["released"] is True
            and event["lock"]["inode"] == post["lock"]["inode"] == case["lock_inode"]
            and event["lock"]["device"] == post["lock"]["device"],
            "transaction lock or target changed",
        )
        require(
            post["job"] == JOB
            and post["plan_sha256"] == PLAN
            and post["status"] == "PASS_OWNED_DEATH_UNPUBLISHED_LOCK_RELEASED"
            and same(post["partial"], event["partial"])
            and post["event_sha256"] == c.hash("output/" + point + "-interrupt-event.json")
            and post["command_sha256"] == c.hash("output/" + point + "-interrupt-command.json")
            and case["interruption_sha256"] == c.hash("output/" + point + "-post-interruption.json"),
            "partial death proof",
        )
        shards = sorted(set(p["merged_weight_map"].values()))
        partial = post["partial"]
        require(
            len(shards) == 3
            and partial["output_absent"] is True
            and partial["marker"] is (point == "marked")
            and partial["index"] is (point != "first_shard")
            and partial["shards"] == (shards[:1] if point == "first_shard" else shards)
            and all(same(partial["inventory"][n], p["merged_inventory"][n]) for n in partial["shards"]),
            "wrong interruption boundary",
        )
        prefix = "output/recovered-" + point + "/"
        inv = {
            n[len(prefix) :]: {k: v[k] for k in ("bytes", "sha256")}
            for n, v in cap["files"].items()
            if n.startswith(prefix)
        }
        require(
            set(inv) == set(p["merged_inventory"]) | ({"config.json.trainer.bak"} if point == "marked" else set())
            and same(inv, case["recovered_inventory"]),
            "incomplete served assets",
        )
        require(
            not any(
                n.startswith(("output/recovered-" + point + ".tmp", "output/recovered-" + point + ".previous"))
                for n in cap["directory_names"]
            ),
            "uncommitted temporary transaction",
        )
        marker, stamp = c.raw(prefix + ".materialized"), c.raw(prefix + ".overlay_source.json")
        require(
            same(
                {k: v for k, v in stamp.items() if k != "written_at"},
                {k: v for k, v in original["output/merged-fp32/.overlay_source.json"].items() if k != "written_at"},
            )
            and same(marker["identity"], stamp["identity"])
            and same(marker["source"], stamp["source"])
            and same(marker["base"], stamp["base"])
            and marker["identity_digest"]
            == hashlib.sha256(json.dumps(marker["identity"], sort_keys=True).encode()).hexdigest(),
            "export identity/precision changed",
        )
        require(
            same(
                marker["options"],
                {"expect_tensors": 272, "min_tensors": 1, "no_verify": False, "unified_chat_template": False},
            )
            and same(marker["verify"]["n_tensors"], 272)
            and same(marker["verify"]["n_shards"], 3)
            and marker["verify"]["output_dtype"] == "float32"
            and marker["reused_staging"] is (point != "first_shard"),
            "partial/unverified export marker",
        )
        require(
            same(
                marker["verify"]["tensor_sha256"],
                original["output/merged-fp32/.materialized"]["verify"]["tensor_sha256"],
            )
            and same(marker["verify"]["bitwise_equal_tensors"], 272)
            and marker["verify"]["bitwise_probe"] == "all_tensors",
            "recovery omitted the all-tensor byte proof",
        )
        for name in p["merged_inventory"]:
            if name not in (".materialized", ".overlay_source.json"):
                require(same(inv[name], p["merged_inventory"][name]), "changed shard or served asset")
        if point == "marked":
            require(
                same(inv["config.json.trainer.bak"], p["merged_inventory"]["config.json"]), "unexpected backup bytes"
            )
        for mode in MODES:
            stem = "output/" + point + "-" + mode
            command, own, child, entry = [
                c.raw(stem + "-" + k + ".json") for k in ("command", "ownership", "child", "entry")
            ]
            clean_command(command, -9 if mode == "interrupt" else 0)
            require(
                same(own["command"], command["command"])
                and own["token"] == child["token"]
                and own["plan_sha256"] == PLAN
                and same(own["uid"], 0)
                and same(child["identity"]["uid"], 0)
                and own["host"] == child["identity"]["host"] == "cpu-00008"
                and command["ownership_path"] == p["root"] + "/" + stem + "-ownership.json",
                "foreign process domain",
            )
            execution_identity(entry, p)
            require(
                entry["mode"] == mode and entry["point"] == point and same(entry["gpu_count"], 0), "wrong child entry"
            )
            if mode == "interrupt":
                continue
            row = c.raw(stem + "-result.json")
            execution_identity(row, p)
            require(
                row["mode"] == mode
                and row["point"] == point
                and same(row["gpu_count"], 0)
                and row["status"] == "PASS_EXACT_RECOVERED_EXPORT"
                and row["source_immutable"] is True
                and row["qualification_pass"] is False
                and case[mode + "_sha256"] == c.hash(stem + "-result.json"),
                "missing recovery/reuse",
            )
            require(
                same(
                    row["export"],
                    {
                        "public_result": "built" if mode == "recover" else "skipped",
                        "build_calls": [dest + ".tmp"] if mode == "recover" and point == "first_shard" else [],
                    },
                ),
                "not exact public reuse",
            )
            cmp = row["comparison"]
            require(
                same(cmp["output_inventory"], inv) and same(cmp["reference_inventory"], p["merged_inventory"]),
                "wrong comparison source",
            )
            schema = tensor_census(cmp, p["merged_weight_map"], schema)
        cases[point] = {
            "owned_interruption": True,
            "recovery_and_reuse": True,
            "tensors": 272,
            "shards": 3,
            "recovery_rebuild_calls": int(point == "first_shard"),
        }
    require(
        [r["phase"] for r in result["commands"]] == [p + "-" + m for p in POINTS for m in MODES], "missing/extra phase"
    )
    for row in result["commands"]:
        raw = c.raw("output/" + row["phase"] + "-command.json")
        require(
            same({k: row[k] for k in raw}, raw)
            and row["receipt_sha256"] == c.hash("output/" + row["phase"] + "-command.json"),
            "command summary differs from actual raw receipt",
        )
    return (
        cases,
        schema,
        {
            "sampled_owned_descendant_rss_bytes": process["peak_owned_process_tree_rss_bytes"],
            "separate_host_controller_rss_bytes": final["peak_host_controller_rss_bytes"],
            "scope": "Sampled descendants together, controller separately; neither allocation limit nor sum of separate maxima is a measured whole-job peak.",
        },
    )


def comparison_outcome(row, *, expected, merge=False):
    """Keep the original frozen merge and stricter cache predicates distinct."""
    tolerances = (0.0005, 0.0002, 0.00005) if merge else (0.00002, 0.00002, 0.000002)
    require(same([row[k] for k in ("atol", "rtol", "mean_gate")], list(tolerances)), "changed original numerical gates")
    for key in ("finite", "all_elements_close", "all_greedy_equal"):
        require(type(row[key]) is bool, "untyped numerical flag")
    for key in ("max_absolute_error", "mean_absolute_error"):
        require(
            type(row[key]) in (int, float) and math.isfinite(row[key]) and row[key] >= 0, "invalid error measurement"
        )
    require(
        isinstance(row["shape"], list) and row["shape"] and all(type(n) is int and n > 0 for n in row["shape"]),
        "missing comparison shape",
    )
    passed = (
        row["finite"]
        and row["all_elements_close"]
        and row["all_greedy_equal"]
        and row["mean_absolute_error"] <= row["mean_gate"]
    )
    require(
        row["status"] == expected == ("PASS" if passed else "FAIL"), "numerical status contradicts frozen predicate"
    )


def validate_original(c, artifacts, original, cell, checkpoint, proof):
    p = c.plan
    inventory(p["portable_inventory"], {"adapter_manifest.json", "adapter.safetensors"})
    inventory(
        p["merged_inventory"],
        {
            ".materialized",
            ".overlay_source.json",
            "config.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors.index.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            *(f"model-{i:05d}-of-00003.safetensors" for i in range(1, 4)),
        },
    )
    receipt = original["output/receipt.json"]
    manifest = original["output/checkpoint-portable/adapter_manifest.json"]
    stamp = original["output/merged-fp32/.overlay_source.json"]
    index = original["output/merged-fp32/model.safetensors.index.json"]
    stock = original["output/stock-reload/result.json"]
    require(
        receipt["cell"] == CELL
        and receipt["job"] == ORIGINAL_JOB
        and receipt["status"] == "FAIL_NUMERICAL_GATES"
        and receipt["export_revision"] == SOURCE["revision"]
        and receipt["training_revision"] == "5fbdf7d189a7f1be84af1d5175d8936b6e5a47e2"
        and receipt["immutable_sources"] is True,
        "different original artifact outcome/source",
    )
    require(
        same(receipt["checkpoint_inventory"], proof["verification"]["source_inventory"])
        and receipt["checkpoint"] == proof["verification"]["checkpoint"]
        and proof["verification"]["artifact_job"] == ORIGINAL_JOB
        and proof["verification"]["legacy_job_outcome"] == "FAIL_NUMERICAL_GATES"
        and same(proof["verification"]["verification_source"], SOURCE)
        and same(proof["checkpoint"]["update"], 935)
        and proof["checkpoint"]["committed"] is True
        and proof["checkpoint"]["validated_by_framework"] is True
        and proof["checkpoint"]["inventory_sha256"] == checkpoint["final_checkpoint"]
        and same(proof["checkpoint"]["identity"], cell["identity"])
        and same(proof["checkpoint"]["lineage"], checkpoint["lineage"]),
        "export is not the admitted final checkpoint",
    )
    require(
        receipt["artifacts"]["portable"] == p["portable"]
        and receipt["artifacts"]["merged"] == p["reference_merged"]
        and same(receipt["portable_inventory"], p["portable_inventory"])
        and same(receipt["merged_inventory"], p["merged_inventory"]),
        "cross-artifact export",
    )
    require(
        manifest["kind"] == "portable_adapter"
        and manifest["provenance"]["source"] == receipt["checkpoint"]
        and manifest["weights_sha256"] == p["portable_inventory"]["adapter.safetensors"]["sha256"]
        and artifacts.row["files"]["output/checkpoint-portable/adapter_manifest.json"]["sha256"]
        == p["portable_inventory"]["adapter_manifest.json"]["sha256"],
        "wrong checkpoint extraction",
    )
    require(
        same(
            receipt["dcp_portable_exact"],
            {
                "identity_exact": True,
                "tensor_count": 120,
                "all_tensors_exact": True,
                "direct_manifest_sha256": receipt["direct_inventory"]["adapter_manifest.json"]["sha256"],
                "extracted_manifest_sha256": p["portable_inventory"]["adapter_manifest.json"]["sha256"],
            },
        ),
        "unverified portable tensor equality",
    )
    for key in ("base", "processor"):
        require(same(manifest["identity"][key], cell["identity"][key]), "wrong base/processor identity")
    require(
        same(index["weight_map"], p["merged_weight_map"])
        and len(index["weight_map"]) == 272
        and len(set(index["weight_map"].values())) == 3,
        "partial merged weight map",
    )
    sc = c.raw("inputs/source-capture.json")
    marker_item = sc["folders"][p["reference_merged"]]["files"][".materialized"]
    marker_bytes = marker_item["text"].encode()
    require(
        hashlib.sha256(marker_bytes).hexdigest()
        == marker_item["sha256"]
        == p["merged_inventory"][".materialized"]["sha256"]
        and len(marker_bytes) == marker_item["bytes"] == p["merged_inventory"][".materialized"]["bytes"],
        "unbound original marker",
    )
    marker = read_json(marker_bytes)
    original["output/merged-fp32/.materialized"] = marker
    require(
        same(marker["identity"], stamp["identity"])
        and same(marker["source"], stamp["source"])
        and same(marker["base"], stamp["base"])
        and marker["identity_digest"]
        == hashlib.sha256(json.dumps(marker["identity"], sort_keys=True).encode()).hexdigest()
        and same(stamp["identity"]["export_metadata"]["adapter_identity"], manifest["identity"])
        and stamp["identity"]["export_metadata"]["adapter_weights_sha256"] == manifest["weights_sha256"]
        and stamp["identity"]["output_dtype"] == "float32"
        and stamp["source"]["path"] == p["portable"]
        and stamp["base"]["path"] == p["base"],
        "wrong merge precision or provenance",
    )
    hashes = marker["verify"]["tensor_sha256"]
    require(
        set(hashes) == set(index["weight_map"])
        and all(isinstance(h, str) and re.fullmatch("[0-9a-f]{64}", h) is not None for h in hashes.values())
        and same(marker["verify"]["bitwise_equal_tensors"], 272)
        and marker["verify"]["bitwise_probe"] == "all_tensors"
        and same(marker["verify"]["n_tensors"], 272)
        and same(marker["verify"]["n_shards"], 3)
        and marker["verify"]["output_dtype"] == "float32"
        and marker["verify"]["processor_identity"] == "equal"
        and same(marker["verify"]["sidecars_identical"], 6),
        "missing original all-tensor/asset proof",
    )
    legacy_root = artifacts.row["root"]
    plan = c.raw("inputs/original-plan.json")
    require(
        c.hash("inputs/original-plan.json") == receipt["plan_sha256"] == artifacts.row["plan_sha256"]
        and same(c.raw("inputs/original-receipt.json"), receipt)
        and plan["primary_root"] == checkpoint["lineage"]["primary_root"]
        and plan["primary_plan_sha256"] == checkpoint["lineage"]["primary_plan_sha256"]
        and plan["runtime"] == p["runtime"]
        and plan["root"] == legacy_root,
        "wrong original job/primary plan",
    )
    for name, sha in WORKERS.items():
        require(plan["files_sha256"][name] == sha, "changed original export/stock loader")
    require(
        receipt["probes"]["stock-reload"]
        == {
            "path": legacy_root + "/output/stock-reload/result.json",
            "sha256": c.legacy_row["files"]["output/stock-reload/result.json"]["sha256"],
            "status": "FAIL",
        },
        "unbound stock result",
    )
    require(
        stock["status"] == "FAIL"
        and stock["source_immutable"] is True
        and same(stock["identity"], manifest["identity"])
        and same(
            stock["ownership"],
            {
                "strict_stock_loader": {
                    "missing_keys": [],
                    "unexpected_keys": [],
                    "mismatched_keys": [],
                    "error_msgs": [],
                }
            },
        )
        and same(stock["aliases"], [["lm_head.weight", "model.embed_tokens.weight"]]),
        "stock loader/identity incomplete",
    )
    for name in ("provenance", "construction-provenance"):
        prov = original["output/stock-reload/" + name + ".json"]
        require(
            prov["source_revision"] == SOURCE["revision"]
            and prov["wheel_sha256"] == SOURCE["wheel_sha256"]
            and prov["container_sha256"] == SOURCE["runtime_sha256"]
            and prov["source_dirty"] is False
            and prov["slurm_job_id"] == prov["job"] == ORIGINAL_JOB
            and same(prov["verified_package_files"], 1933)
            and same(prov["tracked_package_files_verified"], 1933)
            and prov["spec_sha256"] == cell["identity"]["config_sha256"]
            and prov["script_sha256"] == WORKERS["final_text_deployment_v3.py"]
            and prov["loader"] == "stock"
            and prov["mode"] == "compare"
            and same(prov["portable_inventory"], p["portable_inventory"])
            and prov["reference_result_sha256"] == receipt["probes"]["control"]["sha256"]
            and prov["task_protocol_sha256"] == plan["files_sha256"]["text-task-protocol-v5.json"]
            and prov["deployment_protocol_sha256"] == plan["files_sha256"]["full-text-deployment-protocol-v1.json"],
            "stock installed attribution",
        )
    for key, name in (("input_population_sha256", "input-population"), ("provenance_sha256", "provenance")):
        require(
            stock[key] == c.legacy_row["files"]["output/stock-reload/" + name + ".json"]["sha256"],
            "stock raw document join",
        )
    population = original["output/stock-reload/input-population.json"]
    require(
        same(stock["cases"], original["output/stock-reload/cases.json"])
        and len(stock["cases"]) == len(population) == 34
        and len({r["id"] for r in population}) == len({r["id"] for r in stock["cases"]}) == 34,
        "incomplete/duplicate stock population",
    )
    require([row["id"] for row in population[-2:]] == ["context-2048", "context-8192"], "changed original long inputs")
    failures, merge_failures = [], []
    merge_passes = 0
    for ordinal, (case, inp) in enumerate(zip(stock["cases"], population, strict=True)):
        require(
            case["id"] == inp["id"]
            and case["input_ids_sha256"] == inp["input_ids_sha256"]
            and case["status"] == case["native_cache"]["status"] == "FAIL"
            and case["generation_equal"] is True
            and case["generation_matches_common_history"] is True,
            "wrong or unexecuted stock comparison",
        )
        expected_paths = {"prefill_logits", "cached_logits", "full_logits", "split_logits"}
        if ordinal >= 32:
            expected_paths.add("exact_context")
        require(set(case["comparisons"]) == expected_paths, "missing merge comparison")
        for path, value in case["comparisons"].items():
            # These are original failed numerical observations, not predicates
            # of the structural merged-export gate. Do not erase or reinterpret.
            expected = "FAIL" if ordinal == 33 and path in ("full_logits", "exact_context") else "PASS"
            comparison_outcome(value, expected=expected, merge=True)
            if expected == "FAIL":
                merge_failures.append({"id": case["id"], "path": path, "comparison": value})
            else:
                merge_passes += 1
        for key in ("common_history_full_cached", "split_prefix_full"):
            comparison_outcome(case["native_cache"][key], expected="FAIL")
        comparison_outcome(case["native_cache"]["repeated_prefix"], expected="PASS")
        require(case["native_cache"]["prefix_immutable"] is True, "mutated live cache")
        failures.append(
            {
                "id": case["id"],
                "full_cached": case["native_cache"]["common_history_full_cached"],
                "split_prefix": case["native_cache"]["split_prefix_full"],
            }
        )
    merge = [
        "/opt/campaign/runtime/bin/python",
        "-m",
        "cosmos_framework.posttrain.checkpoint.tools.adapters",
        "--adapter",
        p["portable"],
        "--base",
        p["base"],
        "--merged",
        "--output-dtype",
        "float32",
        "--out",
        p["reference_merged"],
    ]
    for name in ("merge", "merge-reuse"):
        command = receipt["commands"][name]
        require(
            same(command["command"], merge)
            and same(command["returncode"], 0)
            and positive(command["seconds"])
            and type(command["peak_process_tree_rss_bytes"]) is int
            and command["peak_process_tree_rss_bytes"] > 0,
            "original export/reuse/peaks missing",
        )
    stock_command = receipt["commands"]["stock-reload"]
    require(same(stock_command["returncode"], 1), "historical stock failure changed")
    argv = stock_command["command"]
    require(
        argv[:2] == ["/opt/campaign/runtime/bin/python", "/run/inputs/final_text_deployment_v3.py"]
        and len(argv[2:]) % 2 == 0
        and len(set(argv[2::2])) == len(argv[2::2]),
        "invalid stock command",
    )
    args = dict(zip(argv[2::2], argv[3::2], strict=True))
    require(
        args
        == {
            "--spec": "/run/inputs/cell.json",
            "--wheel": "/run/runtime-inputs/cosmos_framework-1.2.2-py3-none-any.whl",
            "--build-record": "/run/runtime-inputs/build-input.json",
            "--task-protocol": "/run/inputs/text-task-protocol-v5.json",
            "--deployment-protocol": "/run/inputs/full-text-deployment-protocol-v1.json",
            "--evaluation": str(Path(legacy_root).parent / "data/text-wts-v1/gsm8k/evaluation.jsonl"),
            "--loader": "stock",
            "--mode": "compare",
            "--portable": p["portable"],
            "--merged": p["reference_merged"],
            "--reference": legacy_root + "/output/control",
            "--reference-result-sha256": receipt["probes"]["control"]["sha256"],
            "--out": legacy_root + "/output/stock-reload",
        },
        "wrong stock loader artifact or comparison",
    )
    return {
        "original_job": ORIGINAL_JOB,
        "original_status": "FAIL_NUMERICAL_GATES",
        "stock_returncode": 1,
        "stock_cases": 34,
        "merge_comparison_paths_passed": merge_passes,
        "merge_comparison_paths_failed": len(merge_failures),
        "merge_failures": merge_failures,
        "full_cached_failures": 34,
        "split_prefix_failures": 34,
        "failures": failures,
        "fixed_mma_v2_credit": False,
        "storage_precision_credit": False,
    }


def checkpoint_prerequisite(c, contract, admission_ref, capture_ref):
    admission = c.json(admission_ref)
    previous, index = c.json(admission["contract"]), c.json(admission["index"])
    require(
        admission["status"] == "ACTIVE_REVIEWED_PARTIAL_FINAL_GATE_INDEX"
        and previous["synthetic"] is False
        and admission["canonical_contract_sha256"] == digest(previous) == index["contract_sha256"],
        "checkpoint admission",
    )
    cell = next(r for r in contract["cells"] if r["id"] == CELL)
    old_cell = next(r for r in previous["cells"] if r["id"] == CELL)
    require(same(cell["identity"], old_cell["identity"]), "different checkpoint cell identity")
    subject = index["cells"][CELL]
    refs = subject["gates"]["final_checkpoint"]
    require(len(refs) == 1 and subject["final_checkpoint"] is not None, "missing/ambiguous final checkpoint")
    requirement = next(r for r in old_cell["requirements"] if r["id"] == "final_checkpoint")
    checked = check_receipt(
        c, refs[0], requirement, cell["identity"], previous, subject["final_checkpoint"], subject["lineage"]
    )
    require(checked["outcome"] == "PASS" and checked["executed"] is True, "checkpoint not admitted")
    checkpoint = c.json(refs[0])
    require(capture_ref in checkpoint["supporting_evidence"], "unbound raw checkpoint capture")
    proof = c.json(checkpoint["documents"]["measurement"])
    return cell, checkpoint, proof


def normalize(root, contract, *, extraction_ref, checkpoint_admission_ref, checkpoint_capture_ref):
    c = Capture(root, extraction_ref)
    cell, checkpoint, proof = checkpoint_prerequisite(c, contract, checkpoint_admission_ref, checkpoint_capture_ref)
    requirement = next(r for r in cell["requirements"] if r["id"] == GATE)
    require(
        requirement["profile"] is None
        and requirement["final_checkpoint"] is True
        and same(
            requirement["assertions"],
            [{"document": "measurement", "pointer": "/" + k, "operator": "eq", "expected": True} for k in ASSERTIONS],
        ),
        "changed merged-export gate",
    )
    artifacts = CheckpointCapture(root, checkpoint_capture_ref, CELL)
    original = c.legacy(artifacts)
    failures = validate_original(c, artifacts, original, cell, checkpoint, proof)
    cost = validate_accounting(c)
    cases, schema, peaks = validate_transactions(c, original)
    measurement = {key: True for key in ASSERTIONS}
    measurement.update(
        actual_review=c.extraction["actual_review"],
        checkpoint_update=935,
        original_export_job=ORIGINAL_JOB,
        recovery_job=JOB,
        original_training_source=proof["verification"]["original_training_source"],
        export_and_recovery_source=SOURCE,
        transaction_cases=cases,
        tensor_schema_sha256=digest(schema),
        tensor_count=272,
        shard_count=3,
        output_dtype="float32",
        preserved_numerical_failures=failures,
        resources={**cost, **peaks},
        new_gpu_executions=0,
        full_campaign_pass=False,
        scope="Only own-final935 ordinary-LoRA FSDP2 merged export. No full/cached, split-prefix, storage-BF16, task or whole-cell qualification.",
        evidence_limits=[
            "Nine recovered weight shards were freshly remote-hashed, not recopied or reloaded here. Installed all272-tensor comparisons and equal original shard hashes are preserved.",
            "Original live/source immutability follows the exact source-pinned installed probes; CPU recovery compares immutable artifact sources and exported tensors, without constructing a live training model.",
            "The reviewed host/controller rechecked source hashes after execution; no separate complete second source inventory was serialized.",
            "Resource peaks are sampled owned descendants and a separate controller measurement, not a measured whole-job maximum.",
        ],
    )
    c.verified.update(artifacts.verified)
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
        require(review["reviewed_files"].get(ref["path"]) == ref["sha256"], "unreviewed actual capture/prerequisite")
    producer = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cell = next(r for r in contract["cells"] if r["id"] == CELL)
    requirement = next(r for r in cell["requirements"] if r["id"] == GATE)
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
    require(source in requirement["sources"], "unadmitted installed export source")
    support.append(producer_review)
    for ref in support:
        store.bytes(ref)
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
