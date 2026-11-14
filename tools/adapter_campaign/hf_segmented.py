# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Four typed gates from an independently reviewed segmented HF/WTS primary.

This is an evidence schema adapter, not another trainer or checkpoint loader.
It rederives coverage, denominators, physical placement and receipt lineage.
Saved tensor equality and large remote checkpoint inventories remain attributed
to the explicitly admitted actual audit. No deployment, task or campaign pass
is inferred. Each new capture still requires its own producer-review admission.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import os
import re
import stat
from pathlib import Path

from tools.adapter_campaign.evidence import Store, digest, producer_reviews, read_json, require, same
from tools.adapter_campaign.primary import file_ref, write
from tools.adapter_campaign.recovered import validate_coverage, validate_rank
from tools.adapter_campaign.segmented import GATES_SHA

GATES = {"full_schedule", "complete_sample_coverage", "initial_final_validation", "mid_epoch_continuation"}
DEPENDENCIES = ("hf_segmented.py", "segmented.py", "recovered.py", "evidence.py", "primary.py")
STAT_FIELDS = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
AUDIT_FORMAT = "hf-dora-hsdp-full-primary-independent-review-v1"
PERSISTENT = {"step", "parameters", "optimizer", "optimizer_groups", "scheduler", "data", "clock", "rng"}
UPDATE_FIELDS = PERSISTENT | {"batches", "flow_randomness", "gradients", "inputs", "loss", "rng_before"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def valid_sha(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def relative(value):
    path = Path(value)
    require(not path.is_absolute() and path.parts and ".." not in path.parts, "unsafe relative evidence path")
    return path


def identity(value):
    return tuple(getattr(value, field) for field in STAT_FIELDS)


def owned_bytes(path):
    """Bounded single-link read, with descriptor/path checks and no alias opens."""
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, "unsafe local evidence path")
    require(all(not p.is_symlink() for p in (path, *path.parents)), "aliased local evidence")
    before = path.lstat()
    require(
        stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid == os.getuid(),
        "foreign, linked or nonregular local evidence",
    )
    require(0 <= before.st_size <= 64 << 20, "bounded evidence reader exceeded")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        require(identity(os.fstat(stream.fileno())) == identity(before), "evidence replaced before read")
        raw = stream.read((64 << 20) + 1)
        require(identity(os.fstat(stream.fileno())) == identity(before), "evidence changed during read")
    require(identity(path.lstat()) == identity(before) and len(raw) == before.st_size, "evidence replaced after read")
    return raw, identity(before)


def validate_remote_item(item):
    require(type(item["bytes"]) is int and item["bytes"] >= 0 and valid_sha(item["sha256"]), "invalid remote digest")
    s = item["stat"]
    require(
        stat.S_ISREG(s["st_mode"])
        and s["st_nlink"] == 1
        and s["st_uid"] == 41852
        and s["st_gid"] == 30
        and s["st_size"] == item["bytes"],
        "unowned remote capture object",
    )


class Capture:
    """Join immutable local bytes to their original remote names without edits."""

    def __init__(self, root, audit_ref, *, capture_origin=None):
        self.root = Path(root).absolute()
        # Raw reviews retain their historical absolute names. Relocation copies
        # the exact bytes to the same relative names, never edits those reviews.
        self.capture_origin = Path(capture_origin) if capture_origin is not None else self.root
        require(
            self.capture_origin.is_absolute() and ".." not in self.capture_origin.parts,
            "capture origin must be an explicit absolute root without traversal",
        )
        self.references, self.identities = {}, {}
        self.audit = self.document(audit_ref)
        a = self.audit
        require(
            a["format"] == AUDIT_FORMAT
            and a["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS"
            and a["open_findings"] == []
            and a["full_campaign_pass"] is False
            and a["qualification_pass"] is False
            and a["ledger_mutation"] is False
            and a["new_gpu_executions"] == 0,
            "unadmitted actual HF primary review",
        )
        self.reviewed = {}
        for name, expected in a["reviewed_files"].items():
            ref = self.ref(name, expected)
            require(ref["path"] not in self.reviewed and valid_sha(expected), "duplicate/invalid reviewed source")
            self.reviewed[ref["path"]] = expected
            self.bytes(ref)
        self.folder = Path(a["capture"]["path"]).parent
        self.small = self.document(self.ref(a["capture"]["path"], a["capture"]["sha256"]))
        require(
            self.small["format"] == "hf-dora-hsdp-full-primary-terminal-capture-v1"
            and self.small["all_files_freshly_hashed"] is True
            and self.small["namespace_unchanged"] is True
            and self.small["qualification_pass"] is False
            and self.small["new_gpu_executions"] == 0,
            "partial or changed terminal capture",
        )
        self.remote_root = Path(self.small["root"])
        require(self.remote_root.is_absolute() and ".." not in self.remote_root.parts, "unsafe remote root")
        require(
            len(self.small["files"]) == a["capture"]["freshly_remote_hashed_files"]
            and sum(v["bytes"] for v in self.small["files"].values())
            == self.small["total_bytes"]
            == a["capture"]["freshly_remote_hashed_bytes"],
            "incomplete terminal inventory",
        )
        self.remote, self.resolver = {}, {}
        for name, item in self.small["files"].items():
            validate_remote_item(item)
            self.remote[str(self.remote_root / relative(name))] = item
        names = self.small["mirrored_small_files"]
        require(len(names) == len(set(names)) == a["capture"]["local_small_files"], "duplicate/partial raw mirror")
        for name in names:
            self.bind(self.remote_root / relative(name), self.folder / "files" / name, self.small["files"][name])
        supplement = self.local(self.folder / "supplement01.json")
        require(
            supplement["format"] == "hf-dora-readonly-raw-supplement-v1" and supplement["source_immutable"] is True,
            "uncommitted dataset/census supplement",
        )
        for name, item in supplement["files"].items():
            validate_remote_item(item)
            self.bind(name, self.folder / relative(item["destination"]), item)
        transfer_plan = self.local(self.folder / "tensor-transfer-plan01.json")
        transfer = self.local(self.folder / "tensor-transfer01.json")
        require(
            transfer["status"] == "PASS_EXACT_SAVED_CONTROLS_TRANSFER_ONLY"
            and transfer["plan_sha256"] == self.local_sha(self.folder / "tensor-transfer-plan01.json")
            and transfer_plan["capture_sha256"] == a["capture"]["sha256"]
            and transfer_plan["remote_root"] == str(self.remote_root)
            and transfer["files"] == len(transfer_plan["files"]) == a["capture"]["local_saved_tensors"]
            and transfer["bytes"] == sum(v["bytes"] for v in transfer_plan["files"].values()),
            "incomplete or foreign saved-state transfer",
        )
        for name, item in transfer_plan["files"].items():
            require(same(item, self.small["files"][name]), "saved tensor does not match remote inventory")
            self.bind(self.remote_root / relative(name), self.folder / "tensors" / name, item)
        self.saved = self.local(self.folder / "saved-state-audit01.json")
        self.measured = self.local(self.folder / "primary-receipts-audit01.json")
        independent = a["validation"]["independent_audit"]
        self.independent = self.document(self.ref(independent["path"], independent["sha256"]))
        i = self.independent
        require(
            i["status"] == "PASS_INDEPENDENT_FULL_PRIMARY_AND_SAVED_CONTROLS"
            and i["capture_sha256"] == a["capture"]["sha256"]
            and i["parent_saved_state_audit_sha256"] == self.local_sha(self.folder / "saved-state-audit01.json")
            and i["parent_receipt_audit_sha256"] == self.local_sha(self.folder / "primary-receipts-audit01.json")
            and i["cell"] == self.saved["cell"] == self.measured["cell"] == a["cell"]
            and i["saved_controls_rehashed"] == transfer["files"]
            and i["final_checkpoint_control_metadata_checked"] is True
            and i["full_campaign_pass"] is False
            and i["ledger_mutation"] is False,
            "incomplete independent saved-state audit",
        )

    def ref(self, path, expected):
        path = Path(path)
        if path.is_absolute():
            require(path.is_relative_to(self.capture_origin), "captured source escapes declared origin")
            path = path.relative_to(self.capture_origin)
        return {"path": str(relative(path)), "sha256": expected}

    def bytes(self, ref):
        path = self.root / relative(ref["path"])
        raw, current = owned_bytes(path)
        require(sha(raw) == ref["sha256"], "stale evidence: " + str(path))
        if str(path) in self.identities:
            require(self.identities[str(path)] == current, "evidence identity changed during normalization")
        self.identities[str(path)] = current
        self.references[ref["path"]] = ref
        return raw

    def document(self, ref):
        if hasattr(self, "reviewed"):
            require(self.reviewed.get(ref["path"]) == ref["sha256"], "actual review omitted evidence")
        return read_json(self.bytes(ref))

    def local_sha(self, path):
        return self.reviewed[self.ref(path, "")["path"]]

    def local(self, path):
        return self.document(self.ref(path, self.local_sha(path)))

    def bind(self, remote, local, item):
        name = str(remote)
        require(Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe remote object")
        ref = self.ref(local, item["sha256"])
        require(self.reviewed.get(ref["path"]) == ref["sha256"], "review omitted raw source")
        require(len(self.bytes(ref)) == item["bytes"], "raw source length changed")
        if name in self.remote:
            require(
                self.remote[name]["sha256"] == item["sha256"] and self.remote[name]["bytes"] == item["bytes"],
                "conflicting raw capture",
            )
        else:
            self.remote[name] = item
        if name in self.resolver:
            require(self.resolver[name]["sha256"] == ref["sha256"], "ambiguous raw source")
        self.resolver[name] = ref

    def raw(self, remote):
        require(str(remote) in self.resolver, "uncaptured producer document: " + str(remote))
        return self.bytes(self.resolver[str(remote)])

    def json(self, remote):
        return read_json(self.raw(remote))

    def rows(self, remote):
        return [read_json(row) for row in self.raw(remote).splitlines() if row]

    def sha(self, remote):
        return sha(self.raw(remote))

    def proof(self, remote, expected, size=None):
        item = self.remote.get(str(remote))
        require(item is not None and item["sha256"] == expected and valid_sha(expected), "unbound remote evidence")
        require(item["bytes"] > 0 and (size is None or item["bytes"] == size), "wrong remote evidence length")
        if str(remote) in self.resolver:
            require(sha(self.raw(remote)) == expected, "local/remote evidence mismatch")

    def finish(self):
        # Earlier bytes cannot be silently replaced while later receipts parse.
        for name, before in self.identities.items():
            path = Path(name)
            raw, current = owned_bytes(path)
            ref = self.references[str(path.relative_to(self.root))]
            require(current == before and sha(raw) == ref["sha256"], "evidence changed after verification")


def validate_journal(intent, ack, *, job, payload, plan, operation, index, script, review):
    require(type(ack["returncode"]) is int and ack["returncode"] == 0, "failed/untyped scheduler ACK")
    require(re.fullmatch("[0-9]+", job) is not None and ack["job_id"] == job, "wrong scheduler job")
    require(ack["stdout"] == job + "\n" and ack["stderr"] == "", "uncertain scheduler ACK")
    require(
        intent["operation"] == operation
        and same(intent["index"], index)
        and same(intent["review"], review)
        and same(intent["execution"], None),
        "wrong scheduler operation/review",
    )
    require(
        intent["payload_sha256"] == ack["payload_sha256"] == payload
        and same(intent["plan_sha256"], plan)
        and same(ack["plan_sha256"], plan)
        and same(intent["command"], ack["command"]),
        "unbound scheduler journal",
    )
    exports = f"ALL,NATIVE_FULL_PAYLOAD_SHA256={payload},NATIVE_FULL_REVIEW_SHA256={review['sha256']}"
    if operation == "allocation":
        exports += f",NATIVE_SEGMENT_PLAN_SHA256={plan},NATIVE_FULL_ATTEMPT=segment-{index}"
    require(ack["command"] == ["sbatch", "--parsable", "--export=" + exports, str(script)], "changed launch command")


def validate_accounting(capture, expected):
    command = capture["sacct"]
    require(
        type(command["returncode"]) is int and command["returncode"] == 0 and command["stderr"] == "",
        "failed accounting",
    )
    rows = [line.split("|") for line in command["stdout"].splitlines() if line]
    require([r[0] for r in rows] == [r["job"] for r in expected], "wrong allocation census")
    actual = []
    for row in rows:
        require(
            len(row) == 7 and row[1:3] == ["COMPLETED", "0:0"] and row[6] == "rarunachalam", "failed/foreign allocation"
        )
        tres = dict(v.split("=", 1) for v in row[4].split(","))
        require(tres.get("gres/gpu") == "4" and tres.get("node") == "2" and int(row[3]) > 0, "wrong resource topology")
        actual.append(
            {
                "job": row[0],
                "allocation_seconds": int(row[3]),
                "allocated_gpus": 4,
                "gpu_seconds": int(row[3]) * 4,
                "scheduler_nodes": row[5],
            }
        )
    require(same(actual, expected), "accounting disagrees with independent audit")
    for key in ("queue", "steps"):
        value = capture[key]
        require(
            type(value["returncode"]) is int and value["returncode"] == 0 and value["stderr"] == "",
            "failed live-job census",
        )
        require(
            not any(
                any(n == r["job"] or n.startswith((r["job"] + ".", r["job"] + "_")) for r in actual)
                for n in value["stdout"].splitlines()
            ),
            "allocation remained active",
        )
    return actual


def validate_predecessor(plan, receipt, previous, previous_refs, *, start, end, updates):
    require(
        plan["format"] == "public-hf-wts-full-schedule-segment-v1"
        and plan["phase"] == receipt["phase"] == "training"
        and plan["purpose"] == "authoritative_full_schedule"
        and type(start) is int
        and type(end) is int
        and 0 <= start < end <= updates
        and same(plan["start"], start)
        and same(receipt["start"], start)
        and same(plan["end"], end)
        and same(receipt["end"], end)
        and plan["global_updates"] == updates,
        "wrong authoritative interval",
    )
    require(
        receipt["status"] == "PASS_PLANNED_SEGMENT_ONLY"
        and receipt["full_schedule_pass"] is False
        and same(receipt["lineage_receipts"], previous_refs),
        "incomplete or reordered committed chain",
    )
    expected = None if previous is None else {**previous["next_predecessor"], "receipt": previous_refs[-1]}
    require(same(plan["previous"], expected), "wrong committed predecessor")


def join_phases(phases, logs, updates):
    require(bool(phases) and len(phases) == len(logs), "missing authoritative allocations")
    previous, joined = 0, [[] for _ in range(4)]
    for phase, ranks in zip(phases, logs, strict=True):
        start, end = phase["updates"]
        require(type(start) is int and type(end) is int and start == previous + 1 <= end, "phase gap/overlap")
        require(len(ranks) == 4, "missing physical rank coverage")
        for rank, rows in enumerate(ranks):
            require([r["step"] for r in rows] == list(range(start, end + 1)), "partial phase coverage")
            joined[rank].extend(rows)
        previous = end
    require(previous == updates, "unfinished logical schedule")
    return joined


def validate_owners(initial, cell):
    """Resolve the recorded settings against this cell, never another method."""
    configured = {}
    for component in cell["adapter_config"]["components"]:
        require(component["execution"] == "factorized_v1", "unadmitted component execution")
        for name in component["targets"]:
            require(name not in configured, "duplicate configured target")
            configured[name] = component
    require(set(configured) == set(cell["resolved_target_names"]), "resolved targets differ from config")
    targets = initial["resolved"]["targets"]
    require(len(targets) == len(configured) and {t["name"] for t in targets} == set(configured), "wrong actual targets")
    require(
        initial["resolved"]["model_execution"] == cell["model_execution"]
        and initial["resolved"]["extra_parameters"] == cell["adapter_config"]["extra_modules"] == [],
        "changed execution profile or extras",
    )
    owners, magnitudes = set(), 0
    fields = {
        "method",
        "initialization",
        "rank",
        "alpha",
        "use_rslora",
        "dropout",
        "dtype",
        "pissa_niter",
        "lr_multiplier",
        "weight_decay",
    }
    for target in targets:
        name, settings = target["name"], target["settings"]
        component = configured[name]
        require(target["component"] == component["component"] and target["owner"] == name, "wrong component owner")
        require(same(settings, {k: component[k] for k in fields}), "actual adapter settings differ")
        require(
            settings["method"] in {"lora", "dora"} and settings["initialization"] == "default",
            "unreviewed factor-initialization schema",
        )
        rank = settings["rank"]
        require(type(rank) is int and rank > 0 and settings["dtype"] == "float32", "wrong rank/dtype")
        scale = settings["alpha"] / (math.sqrt(rank) if settings["use_rslora"] else rank)
        require(target["scaling"] == scale, "changed rank scaling")
        owners.update({name + ".lora_A.weight", name + ".lora_B.weight"})
        if settings["method"] == "dora":
            owners.add(name + ".magnitude")
            magnitudes += 1
    actual = [name for group in initial["optimizer"] for name in group["names"]]
    require(
        len(actual) == len(owners) and set(actual) == owners == set(initial["parameter_layouts"]),
        "wrong trainable ownership",
    )
    for layout in initial["parameter_layouts"].values():
        expected_mesh = [[0, 1], [2, 3]] if cell["topology"]["layout"] == "hsdp" else [0, 1, 2, 3]
        dims = [None, 0] if cell["topology"]["layout"] == "hsdp" else [0]
        require(
            same(layout["mesh"], expected_mesh)
            and same(layout["shard_dims"], dims)
            and layout["dtype"] == "torch.float32",
            "wrong parameter sharding",
        )
    return len(owners), magnitudes


def validate_data(c, root, spec, plan, hosts, job, folder, expected_identity):
    """Join the actual two-host installed fingerprint and the complete own census.

    Remote media bytes are attributed to the sealed installed host capture;
    this local normalizer neither decodes the videos nor reopens their aliases.
    """
    request = c.json(root / "inputs/full-data-identity-plan.json")
    request_sha = c.sha(root / "inputs/full-data-identity-plan.json")
    require(
        request["format"] == "hf-wts-canonical-full-data-capture-v1" and request["root"] == str(root),
        "wrong data identity request",
    )
    for key in ("source_revision", "wheel_sha256", "image_sha256", "runtime"):
        require(request[key] == plan[key], "data capture used another runtime")
    admitted = c.json(folder / "canonical-data-admission.json")
    require(
        admitted["status"] == "PASS_TWO_HOST_CANONICAL_FULL_DATA_IDENTITIES"
        and admitted["plan_sha256"] == request_sha
        and admitted["job"] == job
        and same(admitted["hosts"], hosts)
        and len(set(hosts)) == len(hosts) == 2,
        "wrong actual data hosts",
    )
    records = []
    expected_names = {str(folder / "canonical-data" / f"full-data-{host}.json") for host in hosts}
    require(set(admitted["records"]) == expected_names, "missing actual-host data receipt")
    for name, expected in admitted["records"].items():
        c.proof(name, expected)
        value = c.json(name)
        require(
            value["format"] == "hf-wts-canonical-full-data-host-v1"
            and value["status"] == "PASS_IDENTITIES_ONLY"
            and value["plan_sha256"] == request_sha
            and value["job"] == job
            and value["host"] == hosts[int(value["node_id"])],
            "wrong rank-local data identity",
        )
        runtime = value["runtime"]
        require(
            runtime["source_revision"] == plan["source_revision"]
            and runtime["wheel_sha256"] == plan["wheel_sha256"]
            and runtime["container_sha256"] == plan["image_sha256"]
            and runtime["source_dirty"] is False,
            "data identity source not attested",
        )
        selected = [v for v in value["cases"] if v["binding"] == spec["binding"]["id"]]
        require(len(selected) == 1, "ambiguous checkpoint data binding")
        selected = copy.deepcopy(selected[0])
        for split in ("train", "validation"):
            cursor = selected["splits"][split]["initial_cursor"]
            require(
                cursor["position"] == 0 and cursor["rank"] == int(value["node_id"]) * 2,
                "wrong rank-local canonical cursor",
            )
            if split == "train":
                require(
                    cursor["source"]["rank"] == 0
                    and cursor["source"]["world_size"] == 1
                    and cursor["source"]["position"] == 0,
                    "wrong canonical underlying data source",
                )
            else:
                require(cursor["world_size"] == 4, "wrong held-out cursor world size")
            # This one rank-local operand is validated above, not compared as
            # if both hosts were rank zero. All other cursor fields still join.
            cursor["rank"] = 0
        records.append(selected)
    require(same(records[0], records[1]), "physical hosts consumed different data/preprocessing identities")
    case = records[0]
    require(
        case["binding"] == spec["binding"]["id"] and case["config"]["sha256"] == plan["config_sha256"],
        "wrong own processor case",
    )
    c.proof(case["config"]["path"], case["config"]["sha256"])
    require(
        case["assets"]["base"] == expected_identity["base"]["sha256"]
        and case["assets"]["processor"] == expected_identity["processor"]["sha256"],
        "wrong model/processor census",
    )
    require(
        same(case["processor_kwargs"], expected_identity["media"]["processor_kwargs"]),
        "canonical host preprocessing differs from initial model contract",
    )
    historical = case["historical_census"]
    require(
        historical["schema"] == "own-hf-complete-processor-census-v1"
        and historical["binding"] == spec["binding"]["id"]
        and same(historical["processor_identity"], expected_identity["processor"])
        and historical["source_revision"] == plan["source_revision"]
        and historical["wheel_sha256"] == plan["wheel_sha256"],
        "borrowed native census",
    )
    census_path = root / "inputs" / f"own-census-{spec['binding']['id']}-rows.jsonl"
    require(case["historical_census"]["rows_sha256"] == c.sha(census_path), "wrong native census rows")
    census = c.rows(census_path)
    summaries = [v["summary"] for v in census if set(v) == {"summary"}]
    require(
        len(summaries) == 2 and {v["split"] for v in summaries} == {"train", "validation"},
        "partial native census summaries",
    )
    native = {v["split"]: v for v in summaries}
    datasets, loss_tokens, fingerprints = {}, {}, {}
    for split in ("train", "validation"):
        source = case["splits"][split]
        raw = c.raw(spec["paths"][split])
        dataset = [read_json(line) for line in raw.splitlines() if line]
        ids = [v["id"] for v in dataset]
        require(len(ids) == len(set(ids)) == source["rows"] == spec["dataset"]["rows_" + split], "wrong full split IDs")
        require(
            source["jsonl_sha256"] == historical["source_jsonl_sha256"][split] == sha(raw)
            and source["ids_sha256"] == sha("\n".join(ids).encode()),
            "changed split content/order",
        )
        require(
            source["media"] and all(Path(p).is_absolute() and valid_sha(h) for p, h in source["media"].items()),
            "missing canonical media identities",
        )
        require(valid_sha(source["source_fingerprint"]), "invalid installed data fingerprint")
        measured = [v for v in census if v.get("split") == split]
        require(
            [v["row_id"] for v in measured] == ids and [v["index"] for v in measured] == list(range(len(ids))),
            "census repeated/omitted/reordered IDs",
        )
        require(
            all(
                v["mask_matches"] is True
                and v["reasons"] == []
                and type(v["loss_tokens"]) is int
                and v["loss_tokens"] > 0
                for v in measured
            ),
            "native supervised mask census failed",
        )
        summary = native[split]
        require(
            summary["ok"] is True
            and summary["failed"] == summary["unmeasured_rows"] == summary["over_long_rows"] == 0
            and summary["rows"] == summary["matched_rows"] == len(ids),
            "incomplete full preprocessing",
        )
        require(
            sum(v["loss_tokens"] for v in measured) == summary["loss_tokens"] == source["native_loss_tokens"],
            "wrong native denominator",
        )
        datasets[split], loss_tokens[split] = dataset, {v["row_id"]: v["loss_tokens"] for v in measured}
        fingerprints[split] = source["source_fingerprint"]
    expected_finite = {
        "version": 1,
        "kind": "finite_sft_zero_weight_update_tail",
        "epochs": spec["cell"]["training_epochs"],
        "samples_per_update_per_rank": spec["cell"]["grad_accum"],
        "dataset_rows": len(datasets["train"]),
        "world_size": 4,
    }
    require(same(case["finite_coverage"], expected_finite), "changed finite data contract")
    require(len(census) == 2 + sum(len(v) for v in datasets.values()), "unclassified census rows")
    return datasets, loss_tokens, fingerprints, native, case


def validate_saved_audit(saved, independent, *, cell, boundaries, midpoint, updates, owner_count, magnitudes):
    require(
        saved["status"] == "PASS_EXACT_PERSISTED_CONTINUATION_AND_OWNER_UPDATES"
        and saved["cell"] == cell
        and saved["qualification_pass"] is False
        and saved["new_gpu_executions"] == 0,
        "saved-state audit did not pass",
    )
    rows = saved["comparisons"]
    expected = {("midpoint_restore", r, midpoint) for r in range(4)}
    expected |= {("uninterrupted_next_update", r, s) for r in range(4) for s in range(midpoint + 1, midpoint + 4)}
    expected |= {("allocation_restore", r, s) for r in range(4) for s in boundaries[1:-1]}
    expected |= {("final_owner_updates", r, updates) for r in range(4)}
    require(
        len(rows) == len(expected) and {(v["kind"], v["rank"], v["step"]) for v in rows} == expected,
        "missing/repeated saved-state comparisons",
    )
    for row in rows:
        if row["kind"] == "final_owner_updates":
            require(
                row["changed_owners"] == owner_count and row["changed_magnitudes"] == magnitudes,
                "untrained final owner",
            )
        else:
            fields = UPDATE_FIELDS if row["kind"] == "uninterrupted_next_update" else PERSISTENT
            require(
                row["bitwise_equal"] is True and set(row["compared_fields"]) == fields,
                "failed/partial state comparison",
            )
            if row["kind"] == "midpoint_restore":
                require(row["transient_empty_fields_checked"] is True, "replay restored stale transient state")
    require(
        independent["saved_comparisons_repeated"] == len(expected)
        and independent["saved_controls_rehashed"] > 0
        and independent["final_checkpoint_control_metadata_checked"] is True
        and independent["final_validation_status"] == "PASS_LOCAL_FINAL_VALIDATION_COMMITTED",
        "incomplete independent state closure",
    )
    replicas = independent["replica_consistency"]
    points = {"initial", f"midpoint{midpoint}", f"final{updates}"} | {f"boundary{s}" for s in boundaries[1:-1]}
    expected_pairs = {(point, pair) for point in points for pair in ((0, 2), (1, 3))}
    require(
        len(replicas) == len(expected_pairs)
        and {(r["point"], tuple(r["ranks"])) for r in replicas} == expected_pairs
        and all(r["all_owned_parameters_moments_groups_clock_scheduler_exact"] is True for r in replicas),
        "missing/failed HSDP replica equality",
    )


def validate_validation_state(value, *, step, fingerprint, accumulation):
    require(
        value["status"] == "PASS_LOCAL_FINAL_VALIDATION_COMMITTED" and value["step"] == step,
        "uncommitted final validation",
    )
    require(
        same(
            value["clock"],
            {"iter": step, "optimizer_step": step, "attempt_id": step, "accum": 0, "nonfinite_skips_total": 0},
        )
        and value["data_rank_positions"] == [step * accumulation] * 4,
        "wrong final clock/cursors",
    )
    state = value["validation"]
    require(
        state["version"] == 2
        and state["identity"]["enabled"] is True
        and state["identity"]["source_fingerprint"] == fingerprint
        and same(state["progress"], {"pending_step": None, "pending_reasons": [], "pending_metrics": {}})
        and state["publication"] is None,
        "partial/pending/foreign validation",
    )


def resolve_identity(root, contract, evidence_index, *, cell_id, audit_review, capture_origin=None):
    """Prepare an explicit inactive resolution of an unpopulated cell identity.

    The only permitted change is replacing its null base/processor fields with
    the pre-update manifests. A prior gate under the null identity is a blocker.
    No contract/index is written or activated here, and no gate is awarded.
    """
    require(
        len(contract["cells"]) == 368
        and len({v["id"] for v in contract["cells"]}) == 368
        and len({v["id"].split("__")[0] for v in contract["cells"]}) == 16,
        "identity resolution changed the frozen inventory",
    )
    require(
        evidence_index["format"] == "adapter-campaign-evidence-index-v1"
        and evidence_index["contract_sha256"] == digest(contract),
        "wrong predecessor contract/index",
    )
    prior = evidence_index["cells"].get(cell_id)
    require(prior is None or prior.get("gates") == {}, "null identity already owns gate evidence")
    candidate = copy.deepcopy(contract)
    selected = [v for v in candidate["cells"] if v["id"] == cell_id]
    require(len(selected) == 1, "unknown identity cell")
    cell = selected[0]
    require(
        cell["identity"]["base"] is None and cell["identity"]["processor"] is None,
        "identity is already resolved or partially populated",
    )
    c = Capture(root, audit_review, capture_origin=capture_origin)
    terminal, read = c.remote_root, c.json
    plan, spec = read(terminal / "inputs/plan.json"), read(terminal / "inputs/cell.json")
    require(
        c.audit["cell"] == plan["cell"] == spec["cell"]["id"] == cell_id
        and digest(spec["cell"]) == cell["identity"]["inventory_cell_sha256"]
        and c.sha(terminal / "inputs/cell.json") == plan["config_sha256"] == cell["identity"]["config_sha256"],
        "identity resolution used another config",
    )
    binding = spec["binding"]
    require(
        binding["revision"] == cell["identity"]["model_revision"]
        and binding["id"] == "qwen2_vl_hf"
        and binding["implementation"] == "hf"
        and binding["repository"] == "Qwen/Qwen2-VL-2B-Instruct",
        "wrong frozen checkpoint revision",
    )
    audit = terminal / "output/segment-0/training/audit"
    initial_manifest = read(audit / "initial-portable/adapter_manifest.json")
    initial_identity = initial_manifest["identity"]
    c.proof(audit / "initial-portable/adapter.safetensors", initial_manifest["weights_sha256"])
    seal = read(terminal / "inputs/readiness-seal.json")
    require(
        same(seal["expected_adapter_identity"], initial_identity)
        and seal["cell"] == cell_id
        and seal["config_sha256"] == plan["config_sha256"],
        "initial manifest differs from the before-training admission",
    )
    request = read(terminal / "inputs/full-data-identity-plan.json")
    cases = [v for v in request["cases"] if v["binding"] == binding["id"]]
    require(len(cases) == 1, "ambiguous initial asset request")
    assets = cases[0]["model_assets"]
    require(
        all(assets[k] == binding[k] for k in ("id", "repository", "revision", "implementation")),
        "initial assets differ from the frozen binding",
    )
    for key in ("base", "processor"):
        require(same(initial_identity[key], assets[key + "_identity"]), "initial asset inventory differs")
        require(
            valid_sha(initial_identity[key]["sha256"])
            and initial_identity[key]["files"]
            and all(valid_sha(v) for v in initial_identity[key]["files"].values()),
            "incomplete immutable asset identity",
        )
        cell["identity"][key] = copy.deepcopy(initial_identity[key])
    manifest_provenance = initial_manifest["provenance"]
    for rank in range(4):
        first = read(audit / f"initial-rank{rank}.json")
        clock = first["initial_clock"]
        require(
            all(type(clock[k]) is int and clock[k] == 0 for k in ("iter", "optimizer_step", "attempt_id", "accum"))
            and same(clock["optimizer_pristine"], 1)
            and same(first["adapter_identity"], initial_identity),
            "identity selected after training began",
        )
        provenance = read(audit / f"provenance-rank{rank}.json")
        require(
            same(provenance, manifest_provenance)
            and provenance["source_dirty"] is False
            and provenance["source_revision"] == plan["source_revision"]
            and provenance["wheel_sha256"] == plan["wheel_sha256"]
            and provenance["container_sha256"] == plan["image_sha256"]
            and provenance["campaign_spec_sha256"] == plan["config_sha256"],
            "initial source/provenance mismatch",
        )
    # Full normalization also checks each actual training host's canonical
    # model/processor/data identities against this same initial manifest.
    c.finish()
    return candidate, {
        "format": "hf-segmented-pretraining-identity-resolution-v2",
        "cell": cell_id,
        "prior_identity": next(v["identity"] for v in contract["cells"] if v["id"] == cell_id),
        "prior_contract_sha256": digest(contract),
        "candidate_contract_sha256": digest(candidate),
        "prior_index_sha256": digest(evidence_index),
        "actual_audit": audit_review,
        "capture_origin": str(Path(capture_origin) if capture_origin is not None else Path(root).absolute()),
        "producer_files_sha256": {
            "tools/adapter_campaign/" + name: sha(Path(__file__).with_name(name).read_bytes()) for name in DEPENDENCIES
        },
        "before": {"base": None, "processor": None},
        "after": {key: cell["identity"][key] for key in ("base", "processor")},
        "basis": "Before-training readiness, initial portable and all four step-zero pristine-optimizer receipts; never final weights.",
        "supporting_evidence": list(c.references.values()),
        "qualification_pass": False,
        "active_index_written": False,
        "gate_evidence_emitted": False,
    }


def normalize(root, contract_cell, audit_ref, *, capture_origin=None):
    c = Capture(root, audit_ref, capture_origin=capture_origin)
    a, read, terminal = c.audit, c.json, c.remote_root
    plan = read(terminal / "inputs/plan.json")
    spec = read(terminal / "inputs/cell.json")
    cell, binding, immutable = spec["cell"], spec["binding"], contract_cell["identity"]
    require(
        plan["format"] == "segmented-hf-wts-full-v1"
        and plan["root"] == str(terminal)
        and plan["purpose"] == "authoritative_full_schedule",
        "unsupported segmented primary",
    )
    require(a["cell"] == plan["cell"] == cell["id"] == contract_cell["id"], "wrong actual cell")
    require(
        binding["id"] == cell["binding"] == plan["binding"] == "qwen2_vl_hf"
        and binding["implementation"] == "hf"
        and binding["dataset"] == "wts"
        and binding["revision"] == immutable["model_revision"],
        "unadmitted model/backend/data binding",
    )
    require(
        c.sha(terminal / "inputs/plan.json") == a["plan_sha256"] == c.small["plan_sha256"]
        and c.sha(terminal / "inputs/cell.json")
        == a["config_sha256"]
        == plan["config_sha256"]
        == immutable["config_sha256"]
        and digest(cell) == immutable["inventory_cell_sha256"],
        "changed immutable configuration",
    )
    source = {
        "revision": plan["source_revision"],
        "wheel_sha256": plan["wheel_sha256"],
        "runtime_sha256": plan["image_sha256"],
    }
    for key in ("source_revision", "wheel_sha256", "image_sha256"):
        require(plan[key] == a[key], "execution relabeled as another source")
    updates, midpoint = cell["optimizer_updates"], cell["checkpoint_contract"]["mid_epoch_update"]
    boundaries = plan["boundaries"]
    require(
        type(updates) is int
        and updates > 0
        and all(type(v) is int for v in boundaries)
        and boundaries[0] == 0
        and boundaries[-1] == updates
        and sorted(set(boundaries)) == boundaries
        and plan["global_updates"] == updates
        and 0 < midpoint < midpoint + 3 <= updates,
        "changed logical schedule",
    )
    require(
        cell["micro_batch_size"] == 1
        and cell["grad_accum"] == 2
        and cell["training_epochs"] == 1
        and cell["topology"]["world_size"] == 4
        and plan["physical_nodes"] == plan["gpus_per_node"] == 2
        and plan["topology"] == cell["topology"]["layout"] == "hsdp"
        and cell["component_execution"] == "factorized_v1"
        and cell["model_execution"] == "qwen_vl_hf_ordered_v1",
        "unadmitted actual execution contract",
    )
    require(
        cell["checkpoint_contract"]["continuation_updates"] == 3
        and cell["validation"]["initial"] is True
        and cell["validation"]["final"] is True
        and cell["validation"]["complete_split"] is True
        and cell["validation"]["max_samples"] is None,
        "shortened recovery/validation",
    )
    payload = read(terminal / "inputs/payload.json")
    metadata = read(terminal / "inputs/readiness-metadata.json")
    require(
        same(metadata, {**read(terminal / "inputs/readiness-metadata-template.json"), "files_sha256": payload}),
        "changed preparation metadata",
    )
    expected_plan = {
        **read(terminal / "inputs/plan-template.json"),
        "files_sha256": {
            **payload,
            "readiness-metadata.json": c.sha(terminal / "inputs/readiness-metadata.json"),
            "readiness-seal.json": c.sha(terminal / "inputs/readiness-seal.json"),
        },
    }
    require(same(plan, expected_plan), "runtime plan not derived from reviewed payload")
    for name, expected in plan["files_sha256"].items():
        c.proof(terminal / "inputs" / relative(name), expected)
    prep = read(terminal / "output/preparation-result.json")
    require(
        prep["status"] == "PASS_READY_TO_TRAIN_NOT_EXECUTED"
        and prep["full_campaign_pass"] is False
        and prep["final_numerical_gate_credit"] is False
        and prep["full_task_gate_credit"] is False
        and type(prep["preserved_cache_failures"]) is int
        and prep["preserved_cache_failures"] > 0,
        "readiness incorrectly erased outstanding failures",
    )
    source_review = c.local(Path(plan["dispatch_review_path"]))
    review_ref = {"path": plan["dispatch_review_path"], "sha256": c.local_sha(Path(plan["dispatch_review_path"]))}
    require(
        source_review["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS"
        and source_review["open_findings"] == []
        and prep["review_sha256"] == c.independent["source_review_sha256"] == review_ref["sha256"],
        "unreviewed executed stage",
    )
    stage_matches = c.independent["source_stage_matches"]
    require(len(stage_matches) == 1, "ambiguous reviewed source stage")
    for name, expected in payload.items():
        require(
            source_review["reviewed_files"].get(str(Path(stage_matches[0]) / relative(name))) == expected,
            "executed payload absent from source review",
        )
    payload_sha = c.sha(terminal / "inputs/payload.json")
    validate_journal(
        read(terminal / "preparation-intent.json"),
        read(terminal / "preparation-submission.json"),
        job=prep["job"],
        payload=payload_sha,
        plan=None,
        operation="prepare",
        index=None,
        script=terminal / "inputs/prepare.sbatch",
        review=review_ref,
    )
    accounts = validate_accounting(c.small, a["actual_jobs"])
    require(
        same(accounts, c.independent["accounting"]) and len(accounts) == len(boundaries) - 1,
        "wrong audited allocations",
    )
    first_manifest = read(terminal / "output/segment-0/training/audit/initial-portable/adapter_manifest.json")
    expected_identity = first_manifest["identity"]
    require(
        same(expected_identity["base"], immutable["base"])
        and same(expected_identity["processor"], immutable["processor"]),
        "wrong frozen base/processor",
    )
    require(
        same(expected_identity, read(terminal / "inputs/readiness-seal.json")["expected_adapter_identity"]),
        "initial artifact differs from its before-training admission",
    )
    logs, phases, previous_refs, first, checkpoints, data_cases = [], [], [], {}, {}, []
    previous, midpoint_index, owner_count, magnitudes = None, None, None, None
    for index, account in enumerate(accounts):
        job = account["job"]
        folder = terminal / "output" / f"segment-{index}"
        p, result, process = [read(folder / name) for name in ("plan.json", "result.json", "primary-process.json")]
        start, end = boundaries[index : index + 2]
        validate_predecessor(p, result, previous, previous_refs, start=start, end=end, updates=updates)
        require(
            result["segment_plan_sha256"] == c.sha(folder / "plan.json")
            and same(result["process"], process)
            and type(process["returncode"]) is int
            and process["returncode"] == 0
            and process["stop_notice"] is None,
            "failed or unbound authoritative process",
        )
        provenance = process["provenance"]
        require(
            provenance["job"] == job and provenance["plan_sha256"] == a["plan_sha256"], "foreign process allocation"
        )
        selection = {
            "format": "native-full-execution-selection-v1",
            "parent_plan_sha256": a["plan_sha256"],
            "index": index,
            "operation": "training",
            "attempt_name": f"segment-{index}",
            "start": start,
            "end": end,
            "lineage_receipts": previous_refs,
        }
        require(same(provenance["execution_selection"], selection), "changed authoritative allocation selection")
        for key in ("source_revision", "wheel_sha256", "image_sha256", "config_sha256", "cell"):
            require(result[key] == p[key] == plan[key], "segment changed source/config")
        require(p["parent_plan_sha256"] == result["parent_plan_sha256"] == a["plan_sha256"], "wrong parent plan")
        for name, expected in result["evidence_sha256"].items():
            c.proof(name, expected)
        validate_journal(
            read(terminal / f"segment-{index}-intent.json"),
            read(terminal / f"segment-{index}-submission.json"),
            job=job,
            payload=payload_sha,
            plan=a["plan_sha256"],
            operation="allocation",
            index=index,
            script=terminal / "inputs" / f"segment-{index:02d}.sbatch",
            review=review_ref,
        )
        hosts = provenance["hosts"]
        datasets, loss_tokens, fingerprints, native, data_case = validate_data(
            c, terminal, spec, plan, hosts, job, folder, expected_identity
        )
        if data_cases:
            require(same(data_cases[0], data_case), "allocation changed data/processor contract")
        data_cases.append(data_case)
        require(
            p["expected_data_identity"] == fingerprints["train"]
            and p["expected_validation_identity"] == fingerprints["validation"]
            and same(p["expected_finite_coverage"], data_case["finite_coverage"]),
            "segment used different source/cursor",
        )
        next_source = result["next_predecessor"]
        require(next_source["kind"] == "captured_boundary", "uncommitted segment boundary")
        checkpoints[next_source["checkpoint"]] = next_source["checkpoint_inventory"]
        if "midpoint" in result:
            require(
                midpoint_index is None and result["midpoint"]["step"] == midpoint and start < midpoint < end,
                "ambiguous midpoint generation",
            )
            midpoint_index = index
            checkpoints[result["midpoint"]["checkpoint"]] = result["midpoint"]["checkpoint_inventory"]
        rank_logs = []
        for rank in range(4):
            audit = folder / "training/audit"
            initial = validate_rank(
                c,
                audit,
                spec=spec,
                plan=terminal,
                source=source,
                job=job,
                hosts=hosts,
                admission={"expected_adapter_identity": expected_identity},
                fingerprints=fingerprints,
                rank=rank,
                mode="full",
                final_step=end,
            )
            require(
                same(initial["finite_coverage"], data_case["finite_coverage"]), "rank changed finite source contract"
            )
            clock = initial["initial_clock"]
            require(
                clock["iter"] == clock["optimizer_step"] == clock["attempt_id"] == start and clock["accum"] == 0,
                "incorrect initial allocation clock",
            )
            current_owners = validate_owners(initial, cell)
            if owner_count is None:
                owner_count, magnitudes = current_owners
            require(current_owners == (owner_count, magnitudes), "ownership changed between ranks")
            for peer in initial["topology"]["peers"]:
                require(
                    peer["node_id"] in ("0", "1")
                    and peer["host"] == hosts[int(peer["node_id"])]
                    and peer["local_rank"] == peer["rank"] % 2,
                    "rank/node identity differs",
                )
            if rank in first:
                for key in ("resolved", "adapter_identity", "parameter_layouts", "frozen_sha256", "finite_coverage"):
                    require(same(first[rank][key], initial[key]), "allocation changed model or persistent ownership")
            else:
                require(clock.get("optimizer_pristine") == 1, "first allocation was not a fresh optimizer")
                first[rank] = initial
            if index:
                restored = read(audit / f"segment-restored-rank{rank}.json")
                require(
                    restored["status"] == "PASS_EXACT_SEGMENT_RESTORE_BEFORE_FORWARD"
                    and restored["step"] == start
                    and restored["segment_plan_sha256"] == c.sha(folder / "plan.json")
                    and same(restored["boundary"], p["previous"]["boundaries"][str(rank)]),
                    "failed/wrong allocation restore",
                )
            boundary = next_source["boundaries"][str(rank)]
            require(boundary["step"] == end, "wrong boundary update")
            c.proof(boundary["path"], boundary["sha256"])
            identity_ref = next_source["identities"][str(rank)]
            c.proof(identity_ref["path"], identity_ref["sha256"])
            rank_logs.append(c.rows(audit / f"coverage-rank{rank}.jsonl"))
        logs.append(rank_logs)
        phases.append({"updates": [start + 1, end], "job": job, "root": str(folder), "hosts": hosts})
        previous_refs.append({"path": str(folder / "result.json"), "sha256": c.sha(folder / "result.json")})
        previous = result
    joined = join_phases(phases, logs, updates)
    for rank in range(4):
        require(
            same(joined[rank], c.rows(terminal / f"output/authoritative-coverage/coverage-rank{rank}.jsonl")),
            "concatenated coverage differs from actual allocations",
        )
    counts, pads, tokens, curve = validate_coverage(joined, datasets, loss_tokens, native, cell)
    training, replay, final = [
        read(terminal / "output" / name)
        for name in ("primary-training-receipt.json", "continuation-receipt.json", "result.json")
    ]
    require(
        training["status"] == "FULL_TRAINING_COMPLETE_FROM_VERIFIED_SEGMENTS_DEPLOYMENT_PENDING"
        and same(training["lineage_receipts"], previous_refs)
        and training["checkpoint"] == previous["next_predecessor"]["checkpoint"],
        "incomplete full training receipt",
    )
    require(
        final["status"] == "SEGMENTED_FULL_TRAINING_AND_RESTART_COMPLETE_DEPLOYMENT_PENDING"
        and final["full_campaign_pass"] is False
        and final["parent_plan_sha256"] == a["plan_sha256"],
        "wrong terminal outcome",
    )
    for name, key in (
        ("primary-training-receipt.json", "primary_receipt_sha256"),
        ("continuation-receipt.json", "continuation_receipt_sha256"),
    ):
        require(c.sha(terminal / "output" / name) == final[key], "stale terminal receipt")
    for receipt in (training, replay):
        require(receipt["parent_plan_sha256"] == a["plan_sha256"], "foreign terminal receipt")
        for name, expected in receipt["evidence"].items():
            c.proof(name, expected)
    expected_names = {"data.pt", "trainer.pt", "model/.metadata", "optim/.metadata"} | {
        f"{part}/__{rank}_0.distcp" for part in ("model", "optim") for rank in range(4)
    }
    require(
        midpoint_index is not None
        and len(checkpoints) == len(boundaries)
        and {Path(p).name for p in checkpoints} == {f"iter_{v:06d}" for v in [*boundaries[1:], midpoint]},
        "missing checkpoint generation",
    )
    for checkpoint, inventory in checkpoints.items():
        require(set(inventory) == expected_names, "partial committed checkpoint")
        for name, item in inventory.items():
            c.proof(Path(checkpoint) / name, item["sha256"], item["bytes"])
    coverage = read(terminal / "output/primary-coverage.json")
    require(
        c.sha(terminal / "output/primary-coverage.json") == training["coverage_sha256"]
        and coverage["status"] == "PASS_COMPLETE_PRIMARY_COVERAGE"
        and coverage["per_id_counts"] == [dict(v) for v in counts]
        and coverage["valid_tokens"] == tokens
        and coverage["zero_weight_padding_per_epoch"] == pads
        and coverage["logical_updates"] == updates,
        "coverage summary disagrees with authoritative examples",
    )
    expected_training_contract = {
        "binding": binding["id"],
        "kind": "wts",
        "implementation": "hf",
        "purpose": "authoritative_full_schedule",
        "full_schedule_eligible": True,
        "updates": updates,
        "epochs": cell["training_epochs"],
        "accumulation": cell["grad_accum"],
        "midpoint": midpoint,
        "replay_end": midpoint + 3,
        "train_rows": len(datasets["train"]),
        "validation_rows": len(datasets["validation"]),
        "padding": sum(pads),
        "train_denominator": native["train"]["loss_tokens"],
        "validation_denominator": native["validation"]["loss_tokens"],
        "validation_metric": "val/valid_token_count",
        "config_sha256": plan["config_sha256"],
        "relative_run": str(Path("adapter_all_family_two_node") / binding["id"] / cell["id"] / "campaign"),
    }
    require(same(training["contract"], expected_training_contract), "terminal receipt changed full objective contract")
    for name, expected in training["source_coverage_sha256"].items():
        c.proof(name, expected)
    validation = []
    relative_run = Path("adapter_all_family_two_node") / binding["id"] / cell["id"] / "campaign"
    for index in range(len(phases)):
        metrics = c.rows(terminal / f"output/segment-{index}/training" / relative_run / "metrics.jsonl")
        validation.extend(v for v in metrics if "val/loss_avg" in v.get("metrics", v))
    # This schema stores flattened validation rows, unlike ordinary primaries.
    validation = [{**v.get("metrics", v), "step": v["step"]} for v in validation]
    require(
        [v["step"] for v in validation] == [0, updates] and same(validation, training["initial_final_validation"]),
        "wrong initial/final validation splice",
    )
    for row in validation:
        require(
            math.isfinite(row["val/loss_avg"])
            and row["val/valid_sequence_count"] == len(datasets["validation"])
            and row["val/valid_token_count"] == native["validation"]["loss_tokens"],
            "partial or nonfinite validation",
        )
    for rank in range(4):
        value = read(
            terminal / f"output/segment-{len(phases) - 1}/training/audit/final-validation-state-rank{rank}.json"
        )
        validate_validation_state(
            value, step=updates, fingerprint=fingerprints["validation"], accumulation=cell["grad_accum"]
        )
        for name, expected in value["control_files_sha256"].items():
            c.proof(name, expected)
    require(
        replay["status"] == "PASS_EXACT_THREE_UPDATE_CONTINUATION"
        and replay["replay_is_not_primary_coverage"] is True
        and same(replay["source_receipts"], previous_refs[: midpoint_index + 1]),
        "wrong continuation lineage",
    )
    process = read(terminal / f"output/continuation-segment-{midpoint_index}-process.json")
    require(
        same(replay["process"], process)
        and type(process["returncode"]) is int
        and process["returncode"] == 0
        and process["stop_notice"] is None
        and process["provenance"]["job"] == accounts[midpoint_index]["job"],
        "failed replay process",
    )
    assembly = replay["reference_assembly"]
    c.proof(assembly["path"], assembly["sha256"])
    assembled = read(assembly["path"])
    require(assembled["midpoint"]["step"] == midpoint, "wrong midpoint assembly")
    for name, ref in assembled["sources"].items():
        c.proof(ref["path"], ref["sha256"])
        c.proof(Path(assembly["path"]).parent / name, ref["sha256"])
    for rank in range(4):
        folder = terminal / f"output/continuation-segment-{midpoint_index}/audit"
        initial = validate_rank(
            c,
            folder,
            spec=spec,
            plan=terminal,
            source=source,
            job=accounts[midpoint_index]["job"],
            hosts=phases[midpoint_index]["hosts"],
            admission={"expected_adapter_identity": expected_identity},
            fingerprints=fingerprints,
            rank=rank,
            mode="branch",
            final_step=midpoint + 3,
        )
        require(
            initial["initial_clock"]["iter"]
            == initial["initial_clock"]["optimizer_step"]
            == initial["initial_clock"]["attempt_id"]
            == midpoint
            and initial["initial_clock"]["accum"] == 0,
            "wrong midpoint restore clock",
        )
        require(
            read(folder / f"restored-state-rank{rank}.json")["status"] == "PASS", "failed persistent midpoint restore"
        )
        rows = c.rows(folder / f"coverage-rank{rank}.jsonl")
        require([v["step"] for v in rows] == list(range(midpoint + 1, midpoint + 4)), "partial midpoint continuation")
        for resumed in rows:
            step = resumed["step"]
            before = joined[rank][step - 1]
            comparison = read(folder / f"resume-update-{step}-rank{rank}.json")
            require(
                comparison["status"] == "PASS"
                and comparison["atol"] == 1e-7
                and comparison["rtol"] == 1e-5
                and comparison["reference"] == str(Path(assembly["path"]).parent),
                "wrong continuation comparison/gates",
            )
            require(same(resumed["inputs"], before["inputs"]), "replay changed input IDs/padding")
            loss = before["metrics"]["train/loss_avg"]
            error = abs(resumed["metrics"]["train/loss_avg"] - loss)
            require(
                math.isfinite(error) and error == comparison["loss_error"] and error <= 1e-6 + 1e-5 * abs(loss),
                "midpoint loss mismatch",
            )
    validate_saved_audit(
        c.saved,
        c.independent,
        cell=cell["id"],
        boundaries=boundaries,
        midpoint=midpoint,
        updates=updates,
        owner_count=owner_count,
        magnitudes=magnitudes,
    )
    require(c.independent["final_checkpoint"] == training["checkpoint"], "final checkpoint changed after audit")
    gate = next(r for r in contract_cell["requirements"] if r["id"] == "mid_epoch_continuation")
    require(
        next(r["expected"] for r in gate["assertions"] if r["pointer"] == "/gates_sha256") == GATES_SHA,
        "contract recovery gates changed",
    )
    history = {
        "kind": "segmented_hf_wts_primary",
        "phases": phases,
        "audit": audit_ref,
        "actual_source": source,
        "accounting": accounts,
        "allocated_gpu_hours": sum(r["gpu_seconds"] for r in accounts) / 3600,
        "cost_scope": "These primary allocations and their inline replay only; earlier failures/prerequisites remain separately accounted.",
        "diagnostic_replayed_rows": 4 * cell["grad_accum"] * 3,
        "preserved_cache_failures": prep["preserved_cache_failures"],
        "original_failed_preparation": plan.get("prior_failed_preparation"),
        "tensor_scope": "Actual saved-state comparisons and checkpoint inventories are attributed to the admitted independent audit; this producer does not load DCPs or rerun training.",
        "data_scope": "Local JSONL/native census rederived; canonical media fingerprints are the sealed installed captures on both actual hosts.",
    }
    lineage = {
        "primary_root": str(terminal),
        "primary_job": accounts[-1]["job"],
        "primary_plan_sha256": a["plan_sha256"],
    }
    values = {
        "full_schedule": {
            "logical_updates": updates,
            "epochs": cell["training_epochs"],
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
            "data_contract_sha256": immutable["dataset_sha256"],
            "per_id_counts": [dict(v) for v in counts],
            "valid_tokens": tokens,
        },
        "initial_final_validation": {
            "steps": [0, updates],
            "counts": [len(datasets["validation"])] * 2,
            "complete_id_census": True,
            "native_denominators_verified": True,
            "finite_losses": True,
            "validation": validation,
            "census_basis": "Complete immutable split and own native preprocessing census with exact global counts; no per-row runtime validation trace claimed.",
        },
        "mid_epoch_continuation": {
            "mid_update": midpoint,
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
    c.finish()
    return values, source, lineage, list(c.references.values())


def produce(root, contract, output, *, cell_id, audit_review, producer_review, capture_origin=None):
    root, output = Path(root).resolve(), Path(output).absolute()
    require(output.is_relative_to(root), "output escapes evidence root")
    require(output.resolve() == output and not output.exists(), "output is aliased or already populated")
    store = Store(root)
    reviewed = store.json(producer_review)
    require(
        reviewed["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS" and reviewed["open_findings"] == [],
        "unreviewed producer",
    )
    for name in DEPENDENCIES:
        require(
            reviewed["reviewed_files"].get("tools/adapter_campaign/" + name)
            == sha(Path(__file__).with_name(name).read_bytes()),
            "producer/dependency changed after review",
        )
    require(
        reviewed["reviewed_files"].get(audit_review["path"]) == audit_review["sha256"],
        "unadmitted actual capture review",
    )
    if capture_origin is not None:
        require(
            reviewed.get("capture_origins", {}).get(audit_review["sha256"]) == str(Path(capture_origin)),
            "capture origin has no exact independent review admission",
        )
    producer = sha(Path(__file__).read_bytes())
    selected = [v for v in contract["cells"] if v["id"] == cell_id]
    require(len(selected) == 1, "unknown/duplicate contract cell")
    cell = selected[0]
    values, source, lineage, supporting = normalize(root, cell, audit_review, capture_origin=capture_origin)
    require(set(values) == GATES, "producer attempted another qualification gate")
    requirements = {v["id"]: v for v in cell["requirements"]}
    # Admit every gate before publishing the first receipt.
    for gate in GATES:
        r = requirements[gate]
        require(source in r["sources"] and producer in r["producers"], "source/producer not admitted")
        reviews = producer_reviews(r, producer)
        if "producer_reviews" in r:
            require(producer_review in reviews, "missing exact independent producer review admission")
        else:
            require(
                gate in {"full_schedule", "complete_sample_coverage", "initial_final_validation"}
                and producer_review in contract.get("sealed_inputs", []),
                "legacy primary gate requires the exact reviewed producer seal in the contract",
            )
    result = {"lineage": lineage, "final_checkpoint": None, "gates": {}}
    for gate, measurement in values.items():
        closure = list(
            {
                r["path"]: r for r in [*supporting, producer_review, *producer_reviews(requirements[gate], producer)]
            }.values()
        )
        for ref in closure:
            store.bytes(ref)
        folder = output / cell_id / gate
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
        "cells": {cell_id: result},
    }
    write(root, output / "index.json", index)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "contract", "cell", "output", "audit-review", "producer-review"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--capture-origin", help="Original absolute root of byte-identical relocated captures")
    args = parser.parse_args()
    evidence_root = Path(args.root).resolve()
    produce(
        evidence_root,
        read_json(Path(args.contract).read_bytes()),
        args.output,
        cell_id=args.cell,
        audit_review=file_ref(evidence_root, Path(args.audit_review).resolve()),
        producer_review=file_ref(evidence_root, Path(args.producer_review).resolve()),
        capture_origin=args.capture_origin,
    )
