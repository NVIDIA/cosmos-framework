# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Normalize an audited inline midpoint replay without rerunning a full schedule.

This is the full-text-cell-v1 same-job replay schema, not a recovered or
segmented primary. Only mid_epoch_continuation is emitted. The existing primary
normalizer revalidates the three prerequisite training gates, without emitting
them again. Trusted captured midpoint states are compared on CPU; subsequent
snapshot byte equality remains attributed to the independently admitted remote
hash audit. No model/optimizer DCP shards are loaded by this producer.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import pickle
import re
import struct
from pathlib import Path

from tools.adapter_campaign.evidence import Store, digest, producer_reviews, read_json, require, same
from tools.adapter_campaign.primary import file_ref, normalize_primary, write
from tools.adapter_campaign.recovered import validate_rank

GATE = "mid_epoch_continuation"
GATES_SHA = "75c7db901369169327bc5b3ba5508da55d55efee6e8ebef530d3d477fcf62d3b"
DEPENDENCIES = ("inline_midpoint.py", "primary.py", "recovered.py", "evidence.py")
STATE_KEYS = ("parameters", "optimizer", "optimizer_groups", "scheduler", "data", "clock", "rng")


def valid_captured_size(base, name, item):
    """An empty orchestration lock is not an empty checkpoint/state object."""
    size = item.get("bytes")
    return type(size) is int and (
        size > 0
        or size == 0
        and name == str(base / "admission.lock")
        and item.get("sha256") == hashlib.sha256(b"").hexdigest()
    )


class Capture:
    """Bind the existing independent-completed-primary-review-v1 closure."""

    def __init__(self, root, audit_ref):
        self.root, self.store, self.references = Path(root), Store(root), {}
        self.audit = self.document(audit_ref)
        a = self.audit
        require(
            a["format"] == "independent-completed-primary-review-v1"
            and a["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS"
            and a["open_findings"] == []
            and a["full_campaign_pass"] is False,
            "unreviewed inline capture",
        )
        self.reviewed = a["reviewed_files"]
        for path, sha in self.reviewed.items():
            self.bytes({"path": path, "sha256": sha})
        self.folder = Path(audit_ref["path"]).parent
        self.small = self.document(a["capture"]["capture"])
        self.remote_root = Path(self.small["remote_root"])
        self.base = Path(self.small["root"])
        require(self.base.parent == self.remote_root, "foreign inline root")
        self.resolver = {}
        for name, item in self.small["files"].items():
            path = Path(name)
            require(not path.is_absolute() and ".." not in path.parts, "unsafe raw source path")
            self.bind(self.remote_root / path, self.folder / "capture/files" / path, item)
        self.large = self.document(a["capture"]["large_hashes"])
        require(self.large["status"] == "PASS_STABLE_REMOTE_OBJECT_SHA256", "unhashed remote state")
        require(
            set(self.large["files"]) == {str(self.remote_root / n) for n in self.small["large_unhashed"]},
            "changed large-state inventory",
        )
        for name, item in self.large["files"].items():
            relative = str(Path(name).relative_to(self.remote_root))
            require(
                same(item["pre_post_stat"], self.small["large_unhashed"][relative]["stat"])
                and valid_captured_size(self.base, name, item)
                and item["bytes"] == item["pre_post_stat"]["st_size"]
                and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None,
                "stale remote state hash",
            )
        self.postcheck = self.document(a["capture"]["postcheck"])
        require(
            self.postcheck["status"].startswith("PASS_COMPLETED_")
            and self.postcheck["small_rehashed"] == len(self.small["files"])
            and self.postcheck["large_stamps_rechecked"] == len(self.large["files"])
            and same(self.postcheck["accounting"], self.small["accounting"]),
            "source capture was not closed",
        )
        controls = self.local("controls-capture.json")
        for remote, item in controls["files"].items():
            self.bind(remote, Path(item["local"]), item)
            require(self.sha(remote) == self.large["files"][remote]["sha256"], "stale checkpoint control")
        midpoint = self.local("midpoint-transfer.json")
        require(midpoint["status"] == "PASS_EXACT_MIDPOINT_TRANSFER", "incomplete midpoint transfer")
        for item in midpoint["files"]:
            self.bind(item["path"], Path(item["local_path"]), item)
            measured = self.large["files"][item["path"]]
            require(
                item["sha256"] == measured["sha256"] and same(item["pre_post_stat"], measured["pre_post_stat"]),
                "midpoint transfer differs from remote state",
            )

    def bytes(self, ref):
        self.references[ref["path"]] = ref
        return self.store.bytes(ref)

    def document(self, ref):
        if hasattr(self, "reviewed"):
            require(self.reviewed.get(ref["path"]) == ref["sha256"], "capture review omitted a source")
        return read_json(self.bytes(ref))

    def local(self, name):
        path = str(self.folder / name)
        require(path in self.reviewed, "unreviewed inline proof: " + path)
        return self.document({"path": path, "sha256": self.reviewed[path]})

    def bind(self, remote, local, item):
        ref = {"path": str(local), "sha256": item["sha256"]}
        require(self.reviewed.get(str(local)) == ref["sha256"], "unreviewed raw source")
        require(type(item["bytes"]) is int and len(self.bytes(ref)) == item["bytes"] >= 0, "raw source size changed")
        require(str(remote) not in self.resolver, "ambiguous raw source")
        self.resolver[str(remote)] = ref

    def raw(self, remote):
        require(str(remote) in self.resolver, "uncaptured source: " + str(remote))
        return self.bytes(self.resolver[str(remote)])

    def json(self, remote):
        return read_json(self.raw(remote))

    def rows(self, remote):
        return [read_json(line) for line in self.raw(remote).split(b"\n") if line]

    def sha(self, remote):
        return hashlib.sha256(self.raw(remote)).hexdigest()

    def inventory(self, directory):
        directory = Path(directory)
        values = {str(self.remote_root / k): v for k, v in self.small["files"].items()}
        values.update(self.large["files"])
        return {
            str(Path(k).relative_to(directory)): {"sha256": v["sha256"], "bytes": v["bytes"]}
            for k, v in values.items()
            if Path(k).is_relative_to(directory)
        }

    def tensor_state(self, remote):
        # Only read after the independent audit and every hash in its closure
        # have been checked. Pickled local controls are trusted campaign inputs.
        import torch

        return torch.load(io.BytesIO(self.raw(remote)), map_location="cpu", weights_only=False)


def exact_tree(left, right, path="state"):
    """Binary equality, including RNG arrays and signed zero; no numeric waiver."""
    import numpy as np
    import torch

    require(type(left) is type(right), path + ": state type differs")
    if isinstance(left, torch.Tensor):
        require(left.dtype == right.dtype and left.shape == right.shape, path + ": tensor layout differs")
        require(
            torch.equal(
                left.contiguous().reshape(-1).view(torch.uint8), right.contiguous().reshape(-1).view(torch.uint8)
            ),
            path + ": tensor bytes differ",
        )
    elif isinstance(left, np.ndarray):
        require(
            left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes(),
            path + ": RNG array differs",
        )
    elif isinstance(left, dict):
        require(left.keys() == right.keys(), path + ": missing state key")
        for key in left:
            exact_tree(left[key], right[key], path + "/" + str(key))
    elif isinstance(left, (list, tuple)):
        require(len(left) == len(right), path + ": state sequence length differs")
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            exact_tree(a, b, path + "/" + str(i))
    elif isinstance(left, float):
        require(struct.pack("!d", left) == struct.pack("!d", right), path + ": scalar bytes differ")
    else:
        require(left == right, path + ": scalar differs")


def check_clock(clock, step):
    for key, value in {
        "iter": step,
        "optimizer_step": step,
        "attempt_id": step,
        "accum": 0,
        "nonfinite_skips_total": 0,
        "nonfinite_skips_consecutive": 0,
    }.items():
        require(type(clock.get(key)) is int and clock[key] == value, "wrong committed update clock: " + key)


def metric_number(value, expected):
    """Telemetry serializes integer counters as floats; booleans are invalid."""
    return type(value) in (int, float) and math.isfinite(value) and value == expected


def check_rng(states):
    import numpy as np
    import torch

    require(set(states) == set(range(4)), "missing RNG rank")
    for rank, rng in states.items():
        require(type(rank) is int and {"python", "numpy", "torch", "cuda"} <= set(rng), "partial RNG state")
        python, numpy = rng["python"], rng["numpy"]
        require(
            isinstance(python, tuple)
            and len(python) == 3
            and type(python[0]) is int
            and isinstance(python[1], tuple)
            and bool(python[1]),
            "partial Python RNG state",
        )
        require(
            isinstance(numpy, tuple)
            and len(numpy) == 5
            and isinstance(numpy[0], str)
            and isinstance(numpy[1], np.ndarray)
            and numpy[1].size > 0,
            "partial NumPy RNG state",
        )
        for key in ("torch", "cuda"):
            tensor = rng[key]
            require(
                isinstance(tensor, torch.Tensor)
                and tensor.dtype == torch.uint8
                and tensor.ndim == 1
                and tensor.numel() > 0,
                "partial tensor RNG state",
            )


def check_data(data, initial, step, grad_accum):
    require(same(data.get("version"), 1) and same(data.get("world_size"), 4), "wrong sampler schema/world")
    require(set(data["states"]) == set(range(4)), "missing sampler rank")
    for rank, state in data["states"].items():
        require(type(rank) is int and same(state["rank"], rank), "wrong sampler owner")
        require(same(state["position"], step * grad_accum), "wrong sampler cursor")
        require(same(state["finite_sft"], initial["finite_coverage"]), "changed tail coverage contract")
        require(state["source"]["dataset_fingerprint"] == initial["data_identity"], "changed sampler source")
        require(state["source"]["dataset_len"] == initial["finite_coverage"]["dataset_rows"], "wrong data census")


def compare_midpoint(original, restored, initial, step, grad_accum):
    import torch

    owners = [n for group in initial["optimizer"] for n in group["names"]]
    require(bool(owners) and len(set(owners)) == len(owners), "duplicate/missing optimizer owner")
    for value in (original, restored):
        require(set(STATE_KEYS) <= set(value) and same(value["step"], step), "partial/wrong midpoint state")
        check_clock(value["clock"], step)
        require(set(value["parameters"]) == set(value["optimizer"]) == set(owners), "partial trainable state")
        groups = [n for g in value["optimizer_groups"] for n in g["parameter_names"]]
        require(groups == owners and len(set(groups)) == len(groups), "optimizer state ownership differs")
        require(len(value["optimizer_groups"]) == len(initial["optimizer"]), "optimizer groups differ")
        for actual, expected in zip(value["optimizer_groups"], initial["optimizer"], strict=True):
            require(
                actual["parameter_names"] == expected["names"]
                and same(actual["lr"], expected["lr"])
                and same(actual["weight_decay"], expected["weight_decay"]),
                "restored optimizer policy differs",
            )
        for name in owners:
            tensor, state = value["parameters"][name], value["optimizer"][name]
            layout = initial["parameter_layouts"][name]
            require(
                isinstance(tensor, torch.Tensor)
                and list(tensor.shape) == layout["local_shape"]
                and str(tensor.dtype) == layout["dtype"]
                and torch.isfinite(tensor).all().item(),
                "invalid restored trainable tensor",
            )
            require({"step", "exp_avg", "exp_avg_sq"} <= set(state), "partial Adam state")
            require(float(state["step"]) == step, "wrong Adam update count")
            for key in ("exp_avg", "exp_avg_sq"):
                require(
                    state[key].shape == tensor.shape and torch.isfinite(state[key]).all().item(), "invalid Adam moment"
                )
        require(value["scheduler"] and same(value["scheduler"]["last_epoch"], step), "wrong scheduler cursor")
        check_data(value["data"], initial, step, grad_accum)
        check_rng(value["rng"])
    for key in STATE_KEYS:
        exact_tree(original[key], restored[key], key)
    require(same(restored["clock"], initial["initial_clock"]), "restore witness clock differs from rank startup")
    return {"step": step, "keys_compared": list(STATE_KEYS), "trainable_tensors": len(owners), "binary_equal": True}


def checkpoint_inventory(capture, path):
    inventory = capture.inventory(path)
    expected = {"trainer.pt", "data.pt", "model/.metadata", "optim/.metadata"}
    expected |= {f"{part}/__{rank}_0.distcp" for part in ("model", "optim") for rank in range(4)}
    require(set(inventory) == expected, "partial/unexpected DCP checkpoint inventory")
    require(all(type(v["bytes"]) is int and v["bytes"] > 0 for v in inventory.values()), "empty DCP checkpoint file")
    return inventory


def check_storage_metadata(metadata, part, inventory):
    """Require every declared global tensor chunk and serialized object range."""
    expected = set()
    for name, value in metadata.state_dict_metadata.items():
        if not hasattr(value, "chunks"):
            expected.add((name, None))
            continue
        chunks = value.chunks
        size = tuple(value.size)
        require(bool(chunks), "DCP tensor has no chunks")
        for i, chunk in enumerate(chunks):
            offset, lengths = tuple(chunk.offsets), tuple(chunk.sizes)
            require(
                len(offset) == len(lengths) == len(size)
                and all(
                    0 <= start and 0 < length and start + length <= limit
                    for start, length, limit in zip(offset, lengths, size, strict=True)
                ),
                "invalid DCP tensor chunk bounds",
            )
            key = (name, offset)
            require(key not in expected, "duplicate DCP tensor chunk")
            expected.add(key)
            for other in chunks[:i]:
                overlap = all(
                    max(a, b) < min(a + x, b + y)
                    for a, x, b, y in zip(offset, lengths, other.offsets, other.sizes, strict=True)
                )
                require(not overlap, "overlapping DCP tensor chunks")
        require(sum(math.prod(c.sizes) for c in chunks) == math.prod(size), "incomplete global DCP tensor coverage")
    actual, ranges = set(), {}
    for index, storage in metadata.storage_data.items():
        key = (index.fqn, None if index.offset is None else tuple(index.offset))
        require(key not in actual, "duplicate DCP storage mapping")
        actual.add(key)
        name = storage.relative_path
        require(Path(name).name == name and name not in (".", ".."), "escaping DCP storage name")
        require(
            f"{part}/{name}" in inventory
            and storage.offset >= 0
            and storage.length > 0
            and storage.offset + storage.length <= inventory[f"{part}/{name}"]["bytes"],
            "DCP range exceeds complete stored shard",
        )
        require(getattr(storage, "transform_descriptors", None) is None, "unsupported DCP storage transform")
        ranges.setdefault(name, []).append((storage.offset, storage.offset + storage.length))
    require(actual == expected, "DCP storage omitted/added a declared state chunk")
    for values in ranges.values():
        values.sort()
        require(all(a[1] <= b[0] for a, b in zip(values, values[1:])), "overlapping serialized DCP ranges")


def check_checkpoint(capture, path, initial, step, grad_accum):
    inventory = checkpoint_inventory(capture, path)
    trainer, data = [capture.tensor_state(path / (key + ".pt")) for key in ("trainer", "data")]
    check_clock(trainer, step)
    check_data(data, initial, step, grad_accum)
    require(
        {"rng", "objective_identity", "validation", "lr_scheduler", "scheduler_identity"} <= set(trainer),
        "partial trainer control",
    )
    objective = trainer["objective_identity"]
    require(same(objective["adapters"], initial["adapter_identity"]), "checkpoint changed adapter/profile identity")
    require(
        objective["training_source"] == initial["data_identity"]
        and same(objective["training_coverage"], initial["finite_coverage"]),
        "checkpoint changed training objective/data",
    )
    require(same(trainer["lr_scheduler"]["last_epoch"], step), "wrong checkpoint scheduler clock")
    check_rng(trainer["rng"])
    owners = set(initial["parameter_layouts"])
    for part in ("model", "optim"):
        metadata = pickle.loads(capture.raw(path / part / ".metadata"))
        check_storage_metadata(metadata, part, inventory)
        for name in owners:
            keys = [name] if part == "model" else [f"state.{name}.{k}" for k in ("exp_avg", "exp_avg_sq", "step")]
            require(set(keys) <= set(metadata.state_dict_metadata), "DCP omitted trainable/optimizer state")
            if part == "model":
                require(
                    list(metadata.state_dict_metadata[name].size) == initial["parameter_layouts"][name]["shape"],
                    "DCP trainable shape changed",
                )
    return inventory, trainer, data


def compare_replay(capture, original_audit, branch_audit, rank, mid, gates):
    original = capture.rows(original_audit / f"coverage-rank{rank}.jsonl")
    replay = capture.rows(branch_audit / f"coverage-rank{rank}.jsonl")
    steps = list(range(mid + 1, mid + 4))
    require([r["step"] for r in replay] == steps, "incomplete/duplicated inline replay")
    result = []
    for index, step in enumerate(steps):
        before, after = original[step - 1], replay[index]
        require(before["step"] == step and same(before["inputs"], after["inputs"]), "replay changed sample IDs/padding")
        proof = capture.json(branch_audit / f"resume-update-{step}-rank{rank}.json")
        require(proof["status"] == "PASS" and proof["reference"] == str(original_audit), "failed/foreign replay")
        require(
            proof["atol"] == gates["parameter_and_optimizer_atol"]
            and proof["rtol"] == gates["parameter_and_optimizer_rtol"],
            "changed replay thresholds",
        )
        loss, reference = after["metrics"]["train/loss_avg"], before["metrics"]["train/loss_avg"]
        error = abs(loss - reference)
        require(
            math.isfinite(loss)
            and math.isfinite(reference)
            and error == proof["loss_error"]
            and error <= gates["loss_atol"] + gates["loss_rtol"] * abs(reference),
            "replay loss failed",
        )
        for row in (before, after):
            m = row["metrics"]
            require(
                m["optim/update_outcome"] == "applied"
                and metric_number(m["optim/optimizer_step"], step)
                and metric_number(m["optim/attempt_id"], step)
                and metric_number(m["optim/step_skipped"], 0)
                and metric_number(m["optim/nonfinite_skips_total"], 0),
                "replay skipped/nonfinite update",
            )
        for key in ("train/valid_token_count", "train/valid_sequence_count"):
            require(same(before["metrics"][key], after["metrics"][key]), "replay objective denominator differs")
        paths = [str(p / f"update-{step:06d}-rank{rank}.pt") for p in (original_audit, branch_audit)]
        values = [capture.large["files"].get(p) for p in paths]
        require(all(v is not None for v in values), "missing saved update snapshot")
        require(
            values[0]["sha256"] == values[1]["sha256"] and values[0]["bytes"] == values[1]["bytes"],
            "inline producer requires audited byte-identical update snapshots",
        )
        result.append(
            {
                "rank": rank,
                "step": step,
                "paths": paths,
                "sha256": values[0]["sha256"],
                "bytes": values[0]["bytes"],
                "loss_error": error,
                "real_rows": sum(not x["padding"] for x in after["inputs"]),
            }
        )
    return result


def check_runtime(postcheck, plan):
    expected = {
        str(Path(plan["runtime"]) / "output/runtime.sqsh"): plan["image_sha256"],
        str(Path(plan["runtime"]) / "inputs/cosmos_framework-1.2.2-py3-none-any.whl"): plan["wheel_sha256"],
    }
    objects = postcheck["runtime_objects"]
    require(
        set(objects) == set(expected) and same(postcheck["same_runtime_stamps_rechecked"], 2),
        "runtime closure omitted/changed an installed artifact",
    )
    for name, sha in expected.items():
        item = objects[name]
        require(
            item["sha256"] == sha and item["bytes"] == item["pre_post_stat"]["st_size"] > 0,
            "runtime image/wheel differs from original execution",
        )


def portable_factor_spec(initial):
    """Derive complete tensor/alias coverage from the recorded artifact identity."""
    identity = initial["adapter_identity"]
    require(
        identity.get("pissa_representation") == "original_base_plus_trained_minus_initial_factors",
        "unsupported PiSSA residual representation",
    )
    resolved = identity["resolved"]
    require(same(resolved, initial["resolved"]), "portable target identity differs from installed resolution")
    targets, extras = resolved["targets"], resolved["extra_parameters"]
    require(isinstance(targets, list) and targets and isinstance(extras, list), "incomplete portable target manifest")
    named = {row["name"]: row for row in targets}
    extra_named = {row["name"]: row for row in extras}
    require(len(named) == len(targets) and len(extra_named) == len(extras), "duplicate portable target/extra name")
    spec, owners, factors, trainables = {}, {}, {}, set()

    def add(name, shape, owner, trainable):
        require(isinstance(name, str) and name and name not in spec, "duplicate/invalid portable tensor name")
        require(all(type(n) is int and n > 0 for n in shape), "invalid portable tensor dimensions")
        spec[name], owners[name] = tuple(shape), owner
        if trainable:
            trainables.add(owner)

    for row in targets:
        name, owner, settings = row["name"], row["owner"], row["settings"]
        require(isinstance(name, str) and name and isinstance(owner, str) and owner in named, "missing target owner")
        owner_row = named[owner]
        require(
            owner_row["owner"] == owner and all(same(row[k], owner_row[k]) for k in ("shape", "settings", "component")),
            "inconsistent/cyclic portable target aliases",
        )
        shape, rank, method = row["shape"], settings["rank"], settings["method"]
        require(
            isinstance(shape, list)
            and len(shape) == 2
            and all(type(n) is int and n > 0 for n in shape)
            and type(rank) is int
            and 0 < rank <= min(shape)
            and method in ("lora", "dora", "pissa")
            and settings["dtype"] == "float32",
            "invalid portable method/rank/shape/precision",
        )
        out_features, in_features = shape
        for suffix, size in (("lora_A.weight", (rank, in_features)), ("lora_B.weight", (out_features, rank))):
            add(name + "." + suffix, size, owner + "." + suffix, True)
        if method == "dora":
            add(name + ".magnitude", (out_features,), owner + ".magnitude", True)
        if method == "pissa":
            require(settings["initialization"] == "svd_lowrank", "PiSSA lacks its recorded decomposition")
            for suffix, size, trained in (
                ("initial_A", (rank, in_features), "lora_A.weight"),
                ("initial_B", (out_features, rank), "lora_B.weight"),
            ):
                key = name + "." + suffix
                add(key, size, owner + "." + suffix, False)
                factors[key] = name + "." + trained
    for row in extras:
        name, owner = row["name"], row["owner"]
        require(isinstance(owner, str) and owner in extra_named, "missing extra-parameter owner")
        owner_row = extra_named[owner]
        require(
            owner_row["owner"] == owner
            and all(
                same(row[k], owner_row[k]) for k in ("shape", "dtype", "component", "lr_multiplier", "weight_decay")
            )
            and row["dtype"] == "float32",
            "inconsistent/cyclic portable extra aliases",
        )
        add(name, row["shape"], owner, True)
    require(
        all(owner in spec and spec[name] == spec[owner] for name, owner in owners.items()),
        "incomplete portable aliases",
    )
    require(trainables == set(initial["parameter_layouts"]), "portable trainables differ from optimizer-owned state")
    require(
        type(resolved["expected_trainable_parameters"]) is int
        and sum(math.prod(spec[name]) for name in trainables) == resolved["expected_trainable_parameters"],
        "portable trainable element census differs",
    )
    for name in trainables:
        require(
            list(spec[name]) == initial["parameter_layouts"][name]["shape"]
            and initial["parameter_layouts"][name]["dtype"] == "torch.float32",
            "portable trainable shape/precision differs from installed owner",
        )
    return spec, owners, factors


def check_initial_factors(capture, primary, branch, initial):
    """Verify saved PiSSA buffers; non-PiSSA is N/A only from bound manifests.

    This uses the original initial/final/replay portable saver outputs. It is
    not a fresh model load, export qualification, or new GPU execution.
    """
    import torch
    from safetensors import SafetensorError
    from safetensors.torch import load

    spec, owners, factors = portable_factor_spec(initial)
    states, artifacts = {}, {}
    for label, audit, kind in (
        ("initial", primary, "initial"),
        ("final", primary, "final"),
        ("replay", branch, "final"),
    ):
        folder = audit / (kind + "-portable")
        manifest_path, tensor_path = folder / "adapter_manifest.json", folder / "adapter.safetensors"
        manifest = capture.json(manifest_path)
        require(
            manifest.get("kind") == "portable_adapter"
            and same(manifest.get("identity"), initial["adapter_identity"])
            and same(manifest.get("provenance"), capture.json(audit / "provenance-rank0.json")),
            "portable artifact identity/source differs from installed checkpoint lineage",
        )
        item = capture.large["files"].get(str(tensor_path))
        require(
            isinstance(item, dict)
            and type(item.get("bytes")) is int
            and item["bytes"] > 0
            and isinstance(item.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None
            and manifest.get("weights_sha256") == item["sha256"],
            "portable tensor inventory/manifest binding differs",
        )
        artifacts[label] = {"manifest": str(manifest_path), "weights": str(tensor_path), **item}
        if not factors:
            continue
        raw = capture.raw(tensor_path)
        require(
            len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"],
            "changed portable tensor bytes",
        )
        try:
            state = load(raw)
        except SafetensorError:
            require(False, "invalid portable safetensors payload")
        require(set(state) == set(spec), "missing/extra portable adapter or initial-factor tensors")
        for name, shape in spec.items():
            value = state[name]
            require(
                tuple(value.shape) == shape and value.dtype == torch.float32 and torch.isfinite(value).all().item(),
                "invalid portable tensor shape/precision/value: " + name,
            )
            exact_tree(value, state[owners[name]], "portable-alias/" + name)
        states[label] = state
    for name, trained in factors.items():
        exact_tree(states["initial"][name], states["initial"][trained], "initial-decomposition/" + name)
        for label in ("final", "replay"):
            exact_tree(states["initial"][name], states[label][name], label + "-initial-factor/" + name)
    return {
        "status": "PASS_EXACT_INITIAL_FACTORS" if factors else "NOT_APPLICABLE_NO_PISSA_TARGETS",
        "unchanged": True,
        "initial_factor_keys": sorted(factors),
        "unique_factor_owners": sorted({owners[name] for name in factors}),
        "expected_portable_keys": sorted(spec),
        "artifacts": artifacts,
        "portable_tensor_payloads_compared": bool(factors),
        "scope": "Bound initial/final/replay saver outputs; no fresh DCP model restore or deployment credit.",
    }


def normalize(root, contract_cell, audit_ref):
    c = Capture(root, audit_ref)
    base, read = c.base, c.json
    plan, spec = [read(base / "inputs" / name) for name in ("plan.json", "cell.json")]
    cell = spec["cell"]
    require(plan["format"] == "full-text-cell-v1" and plan["root"] == str(base), "not an inline full-text primary")
    require(c.audit["completed"]["cell"] == cell["id"] == contract_cell["id"], "wrong reviewed cell")
    require(
        same(contract_cell["identity"]["training_execution"], [cell["component_execution"], cell["model_execution"]]),
        "wrong training execution",
    )
    primary_capture = c.audit["normalizer"]["capture"]
    c.document(primary_capture)
    values, source, lineage, supporting = normalize_primary(
        root, contract_cell, Path(root) / primary_capture["path"], base.name
    )
    capture_plan = c.local("capture/plan.json")
    for ref in supporting:
        require(
            c.reviewed.get(ref["path"]) == ref["sha256"] or same(ref, capture_plan["source_audit"]),
            "primary proof outside admitted audit",
        )
        c.bytes(ref)
    require(
        same(source, c.audit["completed"]["source"]) and same(lineage, c.audit["completed"]["lineage"]),
        "review changed installed execution identity",
    )
    check_runtime(c.postcheck, plan)
    gates_ref = {"path": "inputs/campaign-gates-v1.json", "sha256": GATES_SHA}
    gates = read_json(c.bytes(gates_ref))["shared"]["resume"]
    require(
        gates
        == {
            "updates": 3,
            "parameter_and_optimizer_atol": 1e-7,
            "parameter_and_optimizer_rtol": 1e-5,
            "loss_atol": 1e-6,
            "loss_rtol": 1e-5,
            "exact": ["IDs", "RNG", "data cursor", "clock", "scheduler", "noise/timestep inputs"],
        },
        "unadmitted continuation thresholds",
    )
    mid = cell["checkpoint_contract"]["mid_epoch_update"]
    require(type(mid) is int and 0 < mid < mid + 3 < cell["optimizer_updates"], "invalid midpoint boundary")
    require(
        cell["checkpoint_contract"]["format"] == "dcp"
        and cell["checkpoint_contract"]["save_async"] is False
        and cell["checkpoint_contract"]["continuation_updates"] == 3
        and cell["checkpoint_contract"]["gradient_accumulation_boundary_only"] is True
        and cell["checkpoint_contract"]["diagnostic_branch_separate"] is True,
        "unadmitted checkpoint lifecycle",
    )
    requirement = next(r for r in contract_cell["requirements"] if r["id"] == GATE)
    require(requirement["profile"] is None and requirement["final_checkpoint"] is False, "wrong gate profile")
    expected = {r["pointer"]: r["expected"] for r in requirement["assertions"]}
    require(
        expected["/mid_update"] == mid
        and expected["/updates_compared"] == 3
        and expected["/gates_sha256"] == GATES_SHA,
        "contract changed replay boundary/gates",
    )
    result = read(base / "output/result.json")
    require(
        result["status"] == "FULL_TRAINING_AND_RECOVERY_COMPLETE_DEPLOYMENT_PENDING"
        and result["full_campaign_pass"] is False,
        "unfinished inline lifecycle",
    )
    receipts = [read(base / "output" / name) for name in ("primary-training-receipt.json", "continuation-receipt.json")]
    branch = receipts[1]
    require(
        branch["status"] == "PASS_EXACT_THREE_UPDATE_CONTINUATION" and branch["replay_is_not_primary_coverage"] is True,
        "failed/non-diagnostic branch",
    )
    for phase, receipt in zip(("primary", "continuation"), receipts, strict=True):
        receipt_name = "primary-training-receipt.json" if phase == "primary" else "continuation-receipt.json"
        require(result[phase + "_receipt_sha256"] == c.sha(base / "output" / receipt_name), "broken receipt chain")
        process = read(base / "output" / (phase + "-process.json"))
        require(
            same(receipt["process"], process)
            and type(process["returncode"]) is int
            and process["returncode"] == 0
            and process["status"] == "COMMAND_COMPLETE"
            and process["failure"] is None,
            "incomplete inline process",
        )
        require(
            same(receipt["provenance"], result["provenance"]) and same(process["provenance"], result["provenance"]),
            "branch changed source/job/config",
        )
        for path, sha in receipt["evidence"].items():
            require(c.sha(path) == sha, "changed original comparison receipt")
    require(same(result["provenance"], read(base / "output/primary-process.json")["provenance"]), "wrong lineage")
    accounting = c.small["accounting"]
    fields = accounting["stdout"].strip().split("|")
    require(
        type(accounting["returncode"]) is int
        and accounting["returncode"] == 0
        and fields[0] == lineage["primary_job"]
        and fields[3:5] == ["COMPLETED", "0:0"]
        and "gres/gpu=4" in fields[9].split(",")
        and "node=2" in fields[9].split(","),
        "not a completed two-host allocation",
    )
    a, b = base / "output/primary/audit", base / "output/continuation/audit"
    suffix = Path("adapter_all_family_two_node") / spec["binding"]["id"] / cell["id"] / "campaign/checkpoints"
    midpoint = base / "output/primary" / suffix / f"iter_{mid:06d}"
    command = branch["process"]["command"]
    for flag, value in {
        "--mode": "branch",
        "--branch-mid": str(mid),
        "--stop-after": str(mid + 3),
        "--resume": str(midpoint),
        "--reference": str(a),
        "--out": str(b.parent),
        "--spec": "/run/inputs/cell.json",
        "--recipe": "campaign_recipe",
    }.items():
        require(
            command.count(flag) == 1 and command[command.index(flag) + 1] == value, "wrong branch invocation: " + flag
        )
    require(
        "--diagnostic-strict-determinism" not in command and "--controlled-failure" not in command,
        "unadmitted branch execution override",
    )
    admission = read(base / "inputs/readiness-seal.json")
    snapshots, checkpoints, restores = [], {}, []
    original_initials, replay_initials = [], []
    for rank in range(4):
        original, restored = read(a / f"initial-rank{rank}.json"), read(b / f"initial-rank{rank}.json")
        validate_rank(
            c,
            b,
            spec=spec,
            plan=base,
            source=source,
            job=lineage["primary_job"],
            hosts=result["provenance"]["hosts"],
            admission=admission,
            fingerprints={"train": original["data_identity"], "validation": original["validation_identity"]},
            rank=rank,
            mode="branch",
            final_step=mid + 3,
        )
        for key in (
            "topology",
            "resolved",
            "adapter_identity",
            "parameter_layouts",
            "frozen_sha256",
            "data_identity",
            "finite_coverage",
            "validation_identity",
            "strict_determinism",
        ):
            require(same(original[key], restored[key]), "replay changed rank identity: " + key)
        check_clock(original["initial_clock"], 0)
        check_clock(restored["initial_clock"], mid)
        require(
            read(b / f"restored-state-rank{rank}.json")
            == {"status": "PASS", "comparison": "exact before first resumed forward"},
            "failed original restore witness",
        )
        marker = read(a / f"mid-checkpoint-rank{rank}.json")
        require(
            same(marker, {"path": str(midpoint), "step": mid, "frozen_preserved": True}), "wrong committed midpoint"
        )
        before, after = [
            c.tensor_state(p) for p in (a / f"update-{mid:06d}-rank{rank}.pt", b / f"initial-state-rank{rank}.pt")
        ]
        restores.append({"rank": rank, **compare_midpoint(before, after, restored, mid, cell["grad_accum"])})
        snapshots.extend(compare_replay(c, a, b, rank, mid, gates))
        original_initials.append(original)
        replay_initials.append(restored)
    for phase, step in (("primary", mid), ("primary", cell["optimizer_updates"]), ("continuation", mid + 3)):
        path = base / "output" / phase / suffix / f"iter_{step:06d}"
        inventory, trainer, data = check_checkpoint(c, path, original_initials[0], step, cell["grad_accum"])
        checkpoints[str(path)] = inventory
        if phase == "primary" and step == mid:
            for rank in range(4):
                restored = c.tensor_state(b / f"initial-state-rank{rank}.pt")
                for key, value in (
                    ("clock", {k: trainer[k] for k in restored["clock"]}),
                    ("rng", trainer["rng"]),
                    ("scheduler", trainer["lr_scheduler"]),
                    ("data", data),
                ):
                    exact_tree(restored[key], value, "checkpoint/" + key)
    require(receipts[0]["checkpoint"] in checkpoints, "primary final checkpoint differs")
    replayed = sum(v["real_rows"] for v in snapshots)
    require(same(branch["primary_rows_replayed"], replayed), "replayed rows counted incorrectly")
    factors = check_initial_factors(c, a, b, original_initials[0])
    measurement = {
        "mid_update": mid,
        "updates_compared": len(snapshots) // 4,
        "same_topology": True,
        "input_ids_exact": True,
        "rng_exact": True,
        "cursor_exact": True,
        "clock_scheduler_exact": True,
        "loss_within_gates": True,
        "parameters_within_gates": True,
        "optimizer_within_gates": True,
        "initial_factors_unchanged": factors["unchanged"],
        "source_immutable": True,
        "gates_sha256": GATES_SHA,
        "inline_replay": {
            "kind": "same_job_inline_midpoint_v1",
            "audit": audit_ref,
            "source": source,
            "lineage": lineage,
            "checkpoint_inventories": checkpoints,
            "restore_comparisons": restores,
            "initial_factor_proof": factors,
            "update_snapshots": snapshots,
            "primary_prerequisites_revalidated": sorted(values),
            "diagnostic_replayed_rows_excluded": replayed,
            "profile": contract_cell["identity"]["training_execution"],
            "scope": "Actual local midpoint/control comparisons; complete DCP shards and subsequent snapshot equality are bound to the independently reviewed remote hashes. No new DCP model/optimizer reload or deployment credit.",
        },
    }
    for pointer, expected_value in expected.items():
        require(same(measurement.get(pointer[1:]), expected_value), "measurement differs from frozen gate")
    return {GATE: measurement}, source, lineage, list(c.references.values())


def produce(root, contract, output, *, audit_review, producer_review):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root) and not output.exists(), "output must be a new evidence directory")
    require(contract.get("synthetic") is not True, "synthetic evidence cannot enter production normalization")
    store = Store(root)
    review = store.json(producer_review)
    require(
        review["status"] == "PASS_NO_OPEN_CONFIRMED_FINDINGS" and review["open_findings"] == [], "unreviewed producer"
    )
    for name in DEPENDENCIES:
        require(
            review["reviewed_files"].get("tools/adapter_campaign/" + name)
            == hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest(),
            "producer/dependency changed",
        )
    require(review["reviewed_files"].get(audit_review["path"]) == audit_review["sha256"], "unadmitted capture review")
    audit = store.json(audit_review)
    cell = next(c for c in contract["cells"] if c["id"] == audit["completed"]["cell"])
    producer = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    requirement = next(r for r in cell["requirements"] if r["id"] == GATE)
    require(producer in requirement["producers"], "inline producer needs independent contract admission")
    require(producer_review in producer_reviews(requirement, producer), "missing exact producer review admission")
    values, source, lineage, supporting = normalize(root, cell, audit_review)
    require(source in requirement["sources"], "unadmitted historical installed source")
    closure = list({r["path"]: r for r in [*supporting, producer_review]}.values())
    for ref in closure:
        store.bytes(ref)
    directory = output / cell["id"] / GATE
    m = write(root, directory / "measurement.json", values[GATE])
    common = {
        "identity": cell["identity"],
        "source": source,
        "profile": None,
        "producer_sha256": producer,
        "lineage": lineage,
        "final_checkpoint": None,
        "supporting_evidence": closure,
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
            cell["id"]: {
                "lineage": lineage,
                "final_checkpoint": None,
                "gates": {GATE: [write(root, directory / "receipt.json", receipt)]},
            }
        },
    }
    write(root, output / "index.json", index)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "contract", "output", "audit-review", "producer-review"):
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
