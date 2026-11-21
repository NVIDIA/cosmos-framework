# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU-only, fail-closed execution selection from sealed v5 runtime evidence.

No imports from the model package, checkpoint deserialization, jobs or training.
Input JSON binds every consumed file by SHA256 relative to --evidence-root, plus
recorded_root for translating the producer's absolute paths in a local mirror.
`plans` maps confirmation_plan/development_plan/development_selection to relative
paths. `profiles` contains two {activation_checkpointing,matrix_plan,attempt}
records. `files` maps relative filenames to their externally pinned SHA256s.
Only decision.json is a public-safe diagnostic; the executable receipt contains
local spec paths and belongs with private orchestration inputs.
All outputs are exclusive. Incomplete calibration produces only decision.json;
six specs and execution-selection.json appear only for a complete decision.
This receipt does NOT authorize untouched scoring or establish release quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any

from tools.native_qlora_qualification.common import digest_json
from tools.native_qlora_qualification.lifecycle import FIXED, NUMERICAL_METRICS, RESERVED, controlled_overrides
from tools.native_qlora_qualification.performance import (
    GIB,
    distribution,
    load_best_fit_selection,
    memory_summary,
    metric_summary,
    number,
)
from tools.native_qlora_qualification.protocol import Experiment, write_json

PROTOCOL = "cosmos.qlora-execution-selection.v1"
INPUT_PROTOCOL = "cosmos.qlora-execution-selection-inputs.v1"
PROFILE_INPUT_PROTOCOL = "cosmos.qlora-execution-selection-inputs.v2"
PROFILE_INPUT_PROTOCOL_V3 = "cosmos.qlora-execution-selection-inputs.v3"
PROFILE_INPUT_PROTOCOL_V4 = "cosmos.qlora-execution-selection-inputs.v4"
PROFILE_INPUT_PROTOCOL_V5 = "cosmos.qlora-execution-selection-inputs.v5"
PROFILE_INPUT_PROTOCOL_V6 = "cosmos.qlora-execution-selection-inputs.v6"
PROFILE_INPUT_PROTOCOL_V7 = "cosmos.qlora-execution-selection-inputs.v7"
PROFILE_PRODUCERS = {
    PROFILE_INPUT_PROTOCOL: ("gpu-runtime-v6", "proc-identity-state-read-attempts-v1"),
    PROFILE_INPUT_PROTOCOL_V3: ("gpu-runtime-v7", "proc-identity-state-read-attempts-v1"),
    PROFILE_INPUT_PROTOCOL_V4: ("gpu-runtime-v8", "proc-identity-state-read-attempts-v2"),
    PROFILE_INPUT_PROTOCOL_V5: ("gpu-runtime-v9", "proc-identity-state-read-attempts-v3"),
    PROFILE_INPUT_PROTOCOL_V6: ("gpu-runtime-v10", "proc-identity-state-read-attempts-v3"),
    PROFILE_INPUT_PROTOCOL_V7: ("gpu-runtime-v11", "proc-identity-state-read-attempts-v3"),
}
PRODUCER_PROFILE_PROTOCOL = "cosmos.qlora-runtime-producer-profile.v1"
TRAINING_REVISION = "55e7f14dfeca84002d0324108808a0e682579842"
WHEEL_SHA = "4939262dbf2de43be0b3f9d579b0bc2fbf72ab84250852397b0dc14087b5792b"
WORKFLOW_SHA = "8cbdb6fcc866cdb5990a8d6b26492f4fbd3e74302361e6aa09c4c9c9976990b7"
DRIVER_SHA = "4e9b0d236f0a3f42fdf7e34b592608ba14dce04511ff730ba1aca3d81c474a51"
BUDGET = 24 * GIB
SEEDS = (17, 42, 123)
ARMS = ("nf4", "bf16")
CELL_STAGES = (
    "workload_verification",
    "preflight",
    "training",
    "portable_publication",
    "probe_dcp",
    "probe_portable",
    "reload_exact",
)
PLACEMENT = (
    "uuid",
    "name",
    "device_bytes",
    "versions",
    "runtime_precision",
    "torch_git_version",
    "torch_cuda",
    "python",
    "cpu_affinity",
    "cpu_topology",
    "physical_cpu_cores",
    "allocation_id",
    "host_id_sha256",
    "thread_environment",
)


class InvalidExecutionSelection(ValueError):
    """Fixed explanation safe for public diagnostic projection."""


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise InvalidExecutionSelection(reason)


def is_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


class Evidence:
    """Parse exactly the pinned bytes; rehash every consumed source at publication."""

    def __init__(self, root: Path, inputs: dict):
        self.root, self.inputs, self.used = Path(root).resolve(), inputs, {}
        self.python_inventories: dict[str, set[str]] = {}
        require(
            inputs.get("protocol") in (INPUT_PROTOCOL, *PROFILE_PRODUCERS),
            "unsupported execution input schema",
        )
        require(
            ("producer_profile" in inputs) == (inputs.get("protocol") in PROFILE_PRODUCERS),
            "explicit producer profile requires the versioned input schema",
        )
        require(isinstance(inputs.get("files"), dict), "missing pinned evidence inventory")
        self.recorded_root = Path(inputs["recorded_root"])
        require(self.recorded_root.is_absolute(), "recorded evidence root must be absolute")
        self.aliases = [self.recorded_root, *[Path(value) for value in inputs.get("recorded_root_aliases", [])]]
        require(
            all(path.is_absolute() and ".." not in path.parts for path in self.aliases)
            and len(set(self.aliases)) == len(self.aliases),
            "recorded root aliases are invalid",
        )

    def relative(self, value: str) -> str:
        path = Path(value)
        if path.is_absolute():
            matches = []
            for root in self.aliases:
                try:
                    matches.append(path.relative_to(root))
                except ValueError:
                    continue
            require(len(matches) == 1, "recorded evidence path escapes or ambiguously matches its root")
            path = matches[0]
        require(bool(path.parts) and ".." not in path.parts, "evidence path escapes its root")
        return str(path)

    def path(self, value: str) -> Path:
        path = self.root / self.relative(value)
        require(path.resolve().is_relative_to(self.root), "evidence symlink escapes its root")
        return path

    def read(self, value: str, *, lines: bool = False) -> Any:
        raw = self.bytes(value)
        if lines:
            require(bool(raw.splitlines()) and all(line.strip() for line in raw.splitlines()), "empty evidence JSONL")
            return [json.loads(line) for line in raw.splitlines()]
        return json.loads(raw)

    def bytes(self, value: str) -> bytes:
        """Read and bind non-JSON inventories without executing producer code."""
        relative = self.relative(value)
        sha = self.inputs["files"].get(relative)
        require(is_sha(sha), "consumed evidence file lacks an external digest")
        raw = self.path(relative).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == sha, "pinned evidence bytes changed")
        self.used[relative] = sha
        return raw

    def sha(self, value: str) -> str:
        relative = self.relative(value)
        require(relative in self.used, "evidence digest requested before verified read")
        return self.used[relative]

    def stable(self) -> None:
        for path, sha in self.used.items():
            require(
                hashlib.sha256(self.path(path).read_bytes()).hexdigest() == sha, "evidence changed before publication"
            )
        for root, expected in self.python_inventories.items():
            directory = self.path(root)
            require(
                {str(path.relative_to(directory)) for path in directory.rglob("*.py")} == expected,
                "producer Python inventory changed before publication",
            )


def child(root: str, name: str) -> str:
    return str(Path(root) / name)


def producer_profile(evidence: Evidence, identity: dict) -> tuple[dict, dict]:
    """Bind one sealed producer, retaining the immutable v1/v5 interpretation.

    A v2 profile is data sealed before calibration, so no workflow needs to hash
    itself or its future private runner. It is not an arbitrary allowed-version
    list: every source byte, submitted plan and observed identity must agree.
    """
    plan = identity["plan"]
    if evidence.inputs["protocol"] == INPUT_PROTOCOL:
        require(
            identity.get("driver_sha256") == DRIVER_SHA and identity.get("workflow_receipt_sha256") == WORKFLOW_SHA,
            "runtime producer is not the pinned v5/workflow941 runner",
        )
        workflow = evidence.read("workflows/941b645/receipt.json")
        require(evidence.sha("workflows/941b645/receipt.json") == WORKFLOW_SHA, "workflow inventory differs")
        return workflow, {"input_protocol": INPUT_PROTOCOL, "workflow_receipt_sha256": WORKFLOW_SHA}

    binding = evidence.inputs["producer_profile"]
    require(set(binding) == {"path", "sha256"} and is_sha(binding["sha256"]), "invalid producer profile binding")
    profile = evidence.read(binding["path"])
    require(evidence.sha(binding["path"]) == binding["sha256"], "producer profile digest differs")
    require(
        set(profile) == {"protocol", "training", "workflow", "runner", "monitor", "correction_plan"}
        and profile["protocol"] == PRODUCER_PROFILE_PROTOCOL
        and profile["training"] == {"code_revision": TRAINING_REVISION, "wheel_sha256": WHEEL_SHA},
        "producer profile schema or unchanged native package differs",
    )
    bound = {
        row["path"]: row["sha256"] for row in plan["inputs"] if row.get("scope", "qualification") == "qualification"
    }
    require(
        plan.get("producer_profile") == binding and bound.get(evidence.relative(binding["path"])) == binding["sha256"],
        "calibration plan did not preseal the exact producer profile",
    )
    loaded = {}
    for kind, required in (
        ("workflow", {"root", "revision", "receipt_sha256", "source_archive_sha256"}),
        ("runner", {"root", "receipt_sha256", "driver_sha256"}),
    ):
        item = profile[kind]
        require(set(item) == required, "producer source binding schema differs")
        path = child(item["root"], "receipt.json")
        receipt = evidence.read(path)
        require(
            is_sha(item["receipt_sha256"])
            and evidence.sha(path) == item["receipt_sha256"] == bound.get(evidence.relative(path)),
            "producer receipt was not pinned by calibration",
        )
        files = receipt["files"]
        require(isinstance(files, dict) and files, "producer source inventory missing")
        for name, sha in files.items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts, "producer inventory path escaped")
            evidence.bytes(child(item["root"], name))
            require(is_sha(sha) and evidence.sha(child(item["root"], name)) == sha, "producer source changed")
        actual = {str(p.relative_to(evidence.path(item["root"]))) for p in evidence.path(item["root"]).rglob("*.py")}
        require(actual == {name for name in files if name.endswith(".py")}, "producer Python inventory differs")
        evidence.python_inventories[item["root"]] = actual
        loaded[kind] = receipt
    workflow, runner = loaded["workflow"], loaded["runner"]
    w, r = profile["workflow"], profile["runner"]
    expected_runner, expected_monitor = PROFILE_PRODUCERS[evidence.inputs["protocol"]]
    evidence.bytes(child(w["root"], "source.tar"))
    require(
        re.fullmatch("[0-9a-f]{40}", w["revision"] or "") is not None
        and w["root"] == "workflows/" + w["revision"][:7]
        and r["root"] == expected_runner
        and workflow.get("code_revision") == w["revision"]
        and workflow.get("training_package_revision") == TRAINING_REVISION
        and workflow.get("training_wheel_sha256") == WHEEL_SHA
        and evidence.sha(child(w["root"], "source.tar"))
        == workflow.get("source_archive_sha256")
        == w["source_archive_sha256"]
        and identity.get("workflow_receipt_sha256") == plan.get("workflow_receipt_sha256") == w["receipt_sha256"]
        and identity.get("driver_sha256") == runner["files"].get("driver.py") == r["driver_sha256"],
        "producer source or observed execution identity differs",
    )
    monitor = profile["monitor"]
    require(
        set(monitor) == {"method", "files"}
        and monitor["method"] == expected_monitor
        and isinstance(monitor["files"], dict)
        and set(monitor["files"]) == {"monitor.py", "monitor_host.py"},
        "strict host-monitor method and helper inventory required",
    )
    for name, sha in monitor["files"].items():
        require(
            Path(name).name == name
            and name.endswith(".py")
            and workflow["files"].get("tools/native_qlora_qualification/" + name) == sha,
            "monitor source is not in the exact workflow inventory",
        )
    correction = profile["correction_plan"]
    require(set(correction) == {"path", "sha256"}, "telemetry correction binding differs")
    value = evidence.read(correction["path"])
    require(
        evidence.sha(correction["path"]) == correction["sha256"] == bound.get(evidence.relative(correction["path"]))
        and value.get("protocol") == "cosmos.qlora-telemetry-correction-plan.v1"
        and value.get("preserve", {}).get("native_training_package") == TRAINING_REVISION
        and value.get("preserve", {}).get("all_quality_and_memory_thresholds") == "unchanged",
        "presealed telemetry correction policy missing or changed",
    )
    return workflow, {
        "input_protocol": evidence.inputs["protocol"],
        "profile_sha256": binding["sha256"],
        **profile,
    }


def healthy_outcomes(evidence: Evidence, root: str, records: list, expected: tuple[str, ...]) -> None:
    require([row.get("name") for row in records] == list(expected), "stage coverage or order differs")
    for row in records:
        actual = evidence.read(child(root, f"stages/{row['name']}/outcome.json"))
        require(actual == row, "stage outcome differs from completion")
        # run.Stages writes these three fields only after child wait and owned
        # cleanup succeed; unlike lifecycle.Stages it emits no timed_out field.
        require(
            set(row) == {"name", "exit_code", "elapsed_seconds"}
            and type(row.get("exit_code")) is int
            and row["exit_code"] == 0
            and number(row.get("elapsed_seconds"), positive=True),
            "runtime stage failed or has an invalid outcome",
        )


def budget(training: dict) -> dict:
    """Recompute the unchanged941 observed-budget predicate from raw measurements."""
    sampled = training["sampled"]
    peak = sampled["owned_device_process_peak_bytes"]
    reasons = []
    if not sampled["gpu_memory_evidence_valid"] or type(peak) is not int or peak <= 0:
        reasons.append("missing_or_invalid_owned_training_process_memory")
    if not training["allocator_evidence_complete"]:
        reasons.append("missing_or_invalid_cuda_allocator_evidence")
    if not sampled["host_pss_complete"]:
        reasons.append("incomplete_host_pss")
    for phase in ("loading", "first_update", "updates", "checkpoint_write"):
        item = training["phases"].get(phase)
        if item is None or item["allocator_peak_allocated_bytes"] is None:
            reasons.append("missing_phase_" + phase)
    gap = sampled["sampling_gap_seconds"]["maximum"]
    if gap is None or gap > 5:
        reasons.append("insufficient_sampling_cadence")
    peak = max(
        (
            v
            for v in (
                peak,
                training.get("allocator_peak_allocated_bytes"),
                training.get("allocator_peak_reserved_bytes"),
            )
            if type(v) is int
        ),
        default=None,
    )
    valid = not reasons
    within = peak <= BUDGET if valid else None
    return {
        "evidence_complete": valid,
        "observed_peak_within_budget": within,
        "observed_process_peak_lower_bound_bytes": peak,
        "eligible_for_higher_microbatch": within is True,
        "budget_bytes": BUDGET,
        "reasons": reasons + (["observed_training_process_peak_exceeds_budget"] if within is False else []),
    }


def native_metrics(records: list, expected: dict) -> dict:
    result = metric_summary(records, steps=20, effective_batch=8)
    require(len(expected["expected_updates"]) == 20, "measured cursor is not twenty updates")
    for actual, desired in zip(result["updates"], expected["expected_updates"], strict=True):
        require(
            all(actual[key] == desired[key] for key in ("step", "examples", "supervised_tokens")),
            "native tokens/examples differ from measured cursor",
        )
    require([row["step"] for row in result["validation"]] == [0, 10, 20], "runtime validation schedule differs")
    require(
        all(
            row["examples"] == 8 and row["supervised_tokens"] == expected["expected_validation_tokens"]
            for row in result["validation"]
        ),
        "runtime validation masks differ",
    )
    require(
        all(row.get("optim/finite_gradient_check_enabled") == 1 for row in records if "train/loss_avg" in row),
        "finite-gradient checks were disabled",
    )
    result["windows"] = {}
    for name, rows in (("all20", result["updates"]), ("after_warmup", result["updates"][2:])):
        elapsed = math.fsum(row["seconds"] for row in rows)
        result["windows"][name] = {
            "updates": len(rows),
            "seconds": distribution([row["seconds"] for row in rows]),
            "examples_per_second": sum(row["examples"] for row in rows) / elapsed,
            "supervised_tokens_per_second": sum(row["supervised_tokens"] for row in rows) / elapsed,
        }
    return result


def placement(runtime: dict, allocation: str) -> dict:
    # These producers are pinned to v5: its reviewed affinity launcher narrows
    # to one logical thread per eight physical cores before native preflight.
    value = {key: runtime[key] for key in PLACEMENT}
    require(
        value["allocation_id"] == allocation and value["physical_cpu_cores"] == 8,
        "runtime is not this eight-core allocation",
    )
    require(
        len(value["cpu_affinity"]) == 8 and len({(row["package"], row["core"]) for row in value["cpu_topology"]}) == 8,
        "runtime physical/logical placement differs",
    )
    require(
        bool(value["uuid"]) and "A100" in value["name"] and value["device_bytes"] >= 79_000_000_000,
        "runtime accelerator differs from qualified hardware",
    )
    return value


def cell(
    evidence: Evidence,
    output: dict,
    spec: Experiment,
    spec_sha: str,
    plan: dict,
    build: dict,
    allocation: str,
    profile: dict,
    tools: dict,
) -> tuple[dict, dict]:
    root = output["path"]
    completion = evidence.read(child(root, "completion.json"))
    sha = evidence.sha(child(root, "completion.json"))
    require(sha == output["completion_sha256"], "runtime completion digest differs")
    require(
        completion.get("protocol") == "cosmos.native-qlora-runtime-cell.v1"
        and completion.get("completed") is True
        and completion.get("failure") is None,
        "runtime cell failed or is incomplete",
    )
    require(not completion.get("diagnostic_profile"), "diagnostic traces cannot select execution settings")
    require(completion["experiment"] == spec.as_dict(), "runtime completion configuration differs")
    contract = evidence.read(child(root, "contract.json"))
    require(not contract.get("diagnostic_profile"), "diagnostic traces cannot select execution settings")
    require(
        contract.get("experiment") == spec.as_dict()
        and contract.get("spec_sha256") == spec_sha
        and contract.get("overrides") == spec.overrides()
        and contract.get("build") == build
        and contract.get("profile") == "short_video8"
        and contract.get("runtime_stress_only") is True
        and contract.get("quality_evidence") is False
        and contract.get("tools") == tools,
        "runtime contract differs from sealed experiment",
    )
    for key in ("workload_manifest_sha256", "profile_manifest_sha256"):
        require(contract.get(key) == plan[key], "runtime workload identity differs")
    healthy_outcomes(evidence, root, completion["outcomes"], CELL_STAGES)
    exact_comparison(
        evidence,
        root,
        "reload_exact",
        "compare-probes",
        reference=child(root, "probes/dcp"),
        candidate=child(root, "probes/portable"),
    )
    runtime = evidence.read(child(root, "runtime.json"))
    where = placement(runtime, allocation)
    expected = evidence.read(child(root, "workload-verification.json"))
    require(
        expected.get("completed") is True
        and expected.get("manifest_measured_shapes") == profile["measurements"]
        and expected.get("processor_identity_sha256") == profile["processor_identity_sha256"],
        "native workload measurement identity differs",
    )
    candidates = [
        name
        for name in evidence.inputs["files"]
        if name.startswith(evidence.relative(root) + "/native/") and name.endswith("/metrics.jsonl")
    ]
    require(len(candidates) == 1, "native metrics evidence is missing or ambiguous")
    native = native_metrics(evidence.read(candidates[0], lines=True), expected)
    require(
        native == completion.get("native") == evidence.read(child(root, "native-metrics-audit.json")),
        "runtime timing summary differs from raw native updates",
    )
    samples = evidence.read(child(root, "nvml-host.jsonl"), lines=True)
    phases = evidence.read(child(root, "stages/training/memory.jsonl"), lines=True)
    memory = memory_summary(phases, [row for row in samples if row.get("stage") == "training"], runtime)
    require(memory == completion.get("training_memory"), "training memory summary differs from raw telemetry")
    measured = budget(memory)
    require(
        all(completion.get("training_budget", {}).get(key) == value for key, value in measured.items()),
        "runtime eligibility differs from measured memory",
    )
    require(measured["evidence_complete"], "invalid memory evidence makes calibration incomplete")
    eligible = measured["eligible_for_higher_microbatch"]
    require(
        completion.get("eligible_for_higher_microbatch") is eligible
        and output.get("eligible_for_higher_microbatch") is eligible,
        "runtime escalation flags disagree",
    )
    final = evidence.read(child(root, "final-checkpoint.json"))
    final_path = Path(evidence.relative(final["checkpoint"]))
    require(
        final.get("optimizer_step") == 20
        and final_path.name == "iter_000020"
        and final_path.is_relative_to(Path(evidence.relative(child(root, "native")))),
        "runtime final checkpoint is not its own completed step20",
    )
    identity = final["metadata"]["identity"]
    require(
        all(identity["training_contract"].get(key) == build[key] for key in ("code_revision", "wheel_sha256"))
        and (identity.get("quantization") is not None) == (spec.arm == "nf4"),
        "runtime final checkpoint has incompatible build or base format",
    )
    probes = check_probes(
        evidence,
        root,
        spec.arm,
        checkpoints=(final["checkpoint"], child(root, "portable/checkpoint_artifacts/iter_000020")),
    )
    require(
        probes[0]["identity_sha256"]
        == digest_json({key: value for key, value in identity.items() if key != "training_contract"}),
        "fresh reload probes differ from committed model identity",
    )
    for probe in probes:
        require(
            probe.get("trainable_tensors") == probe.get("optimizer_parameters") == len(probe.get("adapters", {})) == 504
            and set(probe.get("adapter_updates", {})) == {"A", "B"}
            and all(type(v) is int and 0 < v <= 252 for v in probe["adapter_updates"].values()),
            "reload probe lacks intended trainables and updates",
        )
        video = probe["probes"]["video"]
        require(
            all(
                video[key] == profile["measurements"]["validation"]["rows"][0][key]
                for key in ("sequence_length", "supervised_tokens")
            ),
            "reload probe shapes differ",
        )
    require(
        probes[0]["adapters"] == probes[1]["adapters"] and probes[0]["identity_sha256"] == probes[1]["identity_sha256"],
        "fresh portable adapter identity differs",
    )
    require(number(completion.get("complete_elapsed_seconds"), positive=True), "runtime elapsed time is invalid")
    return {
        "arm": spec.arm,
        "microbatch": spec.microbatch,
        "activation_checkpointing": spec.activation_checkpointing,
        "state": "completed",
        "eligible": eligible,
        "training_budget": measured,
        "after_warmup_mean_seconds": native["windows"]["after_warmup"]["seconds"]["mean"],
        "complete_elapsed_seconds": completion["complete_elapsed_seconds"],
        "runtime_completion_sha256": sha,
        "runtime_identity_sha256": digest_json(where),
        "execution_environment_sha256": digest_json(
            {
                key: where[key]
                for key in (
                    "name",
                    "device_bytes",
                    "versions",
                    "runtime_precision",
                    "torch_git_version",
                    "torch_cuda",
                    "python",
                    "thread_environment",
                )
            }
        ),
        "workload_identity_sha256": digest_json([plan["workload_manifest_sha256"], plan["profile_manifest_sha256"]]),
        "spec_sha256": spec_sha,
        "runtime": {
            "accelerator": "NVIDIA A100 80GB",
            "device_bytes": where["device_bytes"],
            "software_identity_sha256": digest_json(where["versions"]),
        },
    }, where


def check_probes(evidence: Evidence, root: str, arm: str, *, checkpoints: tuple[str, str]) -> list[dict]:
    probes = [evidence.read(child(root, f"probes/{name}/probe.json")) for name in ("dcp", "portable")]
    for probe, checkpoint in zip(probes, checkpoints, strict=True):
        require(
            evidence.relative(probe["checkpoint"]) == evidence.relative(checkpoint),
            "probe loaded a different checkpoint or artifact",
        )
        require(
            probe.get("protocol") == "cosmos.qlora-integration-probe.v1"
            and probe.get("quantized_matrices") == (252 if arm == "nf4" else 0)
            and probe.get("trainable_tensors")
            == probe.get("optimizer_parameters")
            == len(probe.get("adapters", {}))
            == 504
            and set(probe.get("adapter_updates", {})) == {"A", "B"}
            and all(type(value) is int and 0 < value <= 252 for value in probe["adapter_updates"].values())
            and set(probe.get("probes", {})) == {"text", "image", "video"},
            "fresh probe coverage/trainability differs",
        )
        for name, row in probe["probes"].items():
            require(
                number(row.get("loss"))
                and type(row.get("supervised_tokens")) is int
                and row["supervised_tokens"] > 0
                and type(row.get("sequence_length")) is int
                and row["sequence_length"] > 1
                and (name == "text" or number(row.get("zero_media_max_logit_change"), positive=True)),
                "fresh text/media probe lacks finite loss or conditioning",
            )
    for key in ("identity_sha256", "processor_contract", "trainable_elements", "adapters", "probes"):
        require(probes[0][key] == probes[1][key], "fresh DCP/portable probe identity or controlled result differs")
    return probes


def continuation_metrics(records: list[dict], step: int) -> dict:
    rows = [row for row in records if row.get("step") == step and "train/loss_avg" in row]
    require(
        len(rows) == 1 and rows[0].get("optim/update_outcome") == "applied",
        "continuation lacks exactly one applied update",
    )
    result = {key: rows[0][key] for key in NUMERICAL_METRICS}
    require(
        all(type(value) in (int, float) and math.isfinite(value) for value in result.values()),
        "continuation numerical values are missing/nonfinite",
    )
    require(
        result["optim/optimizer_step"] == result["optim/attempt_id"] == step
        and result["optim/finite_gradient_check_enabled"] == 1
        and all(
            value == 0
            for key, value in result.items()
            if key.startswith("optim/")
            and key not in {"optim/optimizer_step", "optim/attempt_id", "optim/finite_gradient_check_enabled"}
        ),
        "continuation counters indicate skipped or unsafe updates",
    )
    return result


def exact_comparison(
    evidence: Evidence, root: str, stage: str, operation: str, *, reference: str, candidate: str
) -> None:
    command = evidence.read(child(root, f"stages/{stage}/command.json"))
    require(isinstance(command, list) and operation in command, "exact comparison command is missing")
    for key, expected in (("--reference", reference), ("--candidate", candidate)):
        require(
            command.count(key) == 1
            and command.index(key) + 1 < len(command)
            and evidence.relative(command[command.index(key) + 1]) == evidence.relative(expected),
            "comparison did not use the declared independent source/candidate",
        )
    for key in ("--atol", "--rtol"):
        require(
            command.count(key) == 1
            and command.index(key) + 1 < len(command)
            and command[command.index(key) + 1] == "0",
            "comparison tolerances were widened",
        )


def lifecycle(evidence: Evidence, output: dict, spec: Experiment, build: dict, profile: dict) -> str:
    root = output["path"]
    result = evidence.read(child(root, "completion.json"))
    sha = evidence.sha(child(root, "completion.json"))
    require(
        sha == output["completion_sha256"]
        and result.get("complete") is True
        and result.get("reload_exact") is True
        and result.get("protocol") == "cosmos.native-qlora-recovery-result.v1",
        "accepted-boundary lifecycle is incomplete",
    )
    experiment = evidence.read(child(root, "experiment.json"))
    requested = [item for item in spec.overrides() if item.split("=", 1)[0] not in set(FIXED) | RESERVED]
    expected_build = {key: build[key] for key in ("code_revision", "wheel_sha256")}
    require(
        result.get("experiment_sha256") == digest_json(experiment)
        and result.get("build_receipt") == expected_build == experiment.get("build_receipt")
        and experiment.get("config") == spec.recipe
        and experiment.get("requested_overrides") == requested
        and experiment.get("qualification_overrides") == controlled_overrides(requested)
        and experiment.get("data_sha256") == profile["files"]["validation"]
        and experiment.get("exact_atol") == experiment.get("exact_rtol") == 0,
        "accepted-boundary lifecycle configuration differs",
    )
    names = ["reference"]
    for mode in ("boundary", "accumulation"):
        names += [mode, mode + "_source_compare", mode + "_resumed", mode + "_compare_3", mode + "_compare_4"]
        event = evidence.read(child(root, mode + "-interruption.json"))
        live = event["live_state"]
        require(
            event.get("protocol") == "cosmos.qlora-injected-interruption.v1"
            and event.get("mode") == mode
            and event.get("exit_code") == 86
            and live.get("optimizer_step") == 2
            and live.get("attempt_id") == (3 if mode == "accumulation" else 2)
            and live.get("accumulation_step") == (1 if mode == "accumulation" else 0)
            and live.get("partial_checkpoint_rejected") is (mode == "accumulation")
            and live.get("frozen_base_validated") is True
            and live.get("exact_optimizer_membership_validated") is True
            and live.get("adapter_tensors") == 504,
            "real interruption boundary or frozen/trainability proof differs",
        )
        require(
            event["checkpoint"].get("optimizer_step") == 2
            and event["checkpoint"].get("healthy_completed_boundary") is True
            and evidence.relative(event["checkpoint"]["path"])
            == evidence.relative(child(root, f"native/qualification/recovery/{mode}/canary/checkpoints/iter_000002")),
            "interruption source checkpoint is not a healthy step2 boundary",
        )
        exact_comparison(
            evidence,
            root,
            mode + "_source_compare",
            "compare-checkpoints",
            reference=child(root, "native/qualification/recovery/reference/canary/checkpoints/iter_000002"),
            candidate=child(root, f"native/qualification/recovery/{mode}/canary/checkpoints/iter_000002"),
        )
        proof = result["recovery"][mode]
        require(set(proof) == {"3", "4"}, "lifecycle lacks immediate-next/final update proof")
        for step in ("3", "4"):
            exact_comparison(
                evidence,
                root,
                mode + "_compare_" + step,
                "compare-checkpoints",
                reference=child(
                    root, f"native/qualification/recovery/reference/canary/checkpoints/iter_{int(step):06d}"
                ),
                candidate=child(
                    root, f"native/qualification/recovery/{mode}_resumed/canary/checkpoints/iter_{int(step):06d}"
                ),
            )
            require(
                proof[step]["checkpoint"].get("optimizer_step") == int(step)
                and proof[step]["checkpoint"].get("healthy_completed_boundary") is True
                and evidence.relative(proof[step]["checkpoint"]["path"])
                == evidence.relative(
                    child(root, f"native/qualification/recovery/{mode}_resumed/canary/checkpoints/iter_{int(step):06d}")
                ),
                "resumed checkpoint is not the declared completed boundary",
            )
            metrics = evidence.read(child(root, f"stages/{mode}_compare_{step}/numerical-update.json"))
            left = continuation_metrics(
                evidence.read(child(root, "native/qualification/recovery/reference/canary/metrics.jsonl"), lines=True),
                int(step),
            )
            right = continuation_metrics(
                evidence.read(
                    child(root, f"native/qualification/recovery/{mode}_resumed/canary/metrics.jsonl"), lines=True
                ),
                int(step),
            )
            require(
                metrics == proof[step]["metrics"]
                and metrics.get("passed") is True
                and metrics.get("optimizer_step") == int(step)
                and metrics.get("numerical_metrics") == left == right,
                "lifecycle numerical comparison differs",
            )
    names += ["publish_portable", "probe_dcp", "probe_portable", "compare_reload"]
    require([row.get("name") for row in result["outcomes"]] == names, "lifecycle stage coverage differs")
    for row in result["outcomes"]:
        require(
            evidence.read(child(root, f"stages/{row['name']}/outcome.json")) == row,
            "lifecycle outcome differs from completion",
        )
        interrupted = row["name"] in ("boundary", "accumulation")
        require(
            row.get("injected_interruption_expected") is interrupted
            and row.get("timed_out") is False
            and (
                (type(row.get("exit_code")) is int and row["exit_code"] != 0)
                if interrupted
                else row.get("exit_code") == 0
            )
            and not row.get("owned_process_cleanup", {}).get("failed", False),
            "lifecycle execution failed",
        )
    exact_comparison(
        evidence,
        root,
        "compare_reload",
        "compare-probes",
        reference=child(root, "probes/dcp"),
        candidate=child(root, "probes/portable"),
    )
    check_probes(
        evidence,
        root,
        spec.arm,
        checkpoints=(
            child(root, "native/qualification/recovery/accumulation_resumed/canary/checkpoints/iter_000004"),
            child(root, "portable/checkpoint_artifacts/iter_000004"),
        ),
    )
    return sha


def choose(cells: list[dict]) -> dict:
    eligible = [row for row in cells if row.get("eligible") is True]
    require(bool(eligible), "no eligible runtime configuration for this precision")

    def key(row):
        return (
            row["after_warmup_mean_seconds"],
            row["training_budget"]["observed_process_peak_lower_bound_bytes"],
            row["microbatch"],
            0 if row["activation_checkpointing"] == "full" else 1,
        )

    require(
        all(
            number(row["after_warmup_mean_seconds"], positive=True)
            and type(row["microbatch"]) is int
            and row["microbatch"] in (1, 2, 4)
            and row["activation_checkpointing"] in ("full", "none")
            and type(row["training_budget"]["observed_process_peak_lower_bound_bytes"]) is int
            and 0 < row["training_budget"]["observed_process_peak_lower_bound_bytes"] <= BUDGET
            for row in eligible
        ),
        "eligible runtime rank inputs are invalid",
    )
    eligible.sort(key=key)
    require(len(eligible) == 1 or key(eligible[0]) != key(eligible[1]), "runtime selection has an unresolved exact tie")
    return eligible[0]


def runtime_spec(value: dict, arm: str, microbatch: int, ac: str) -> Experiment:
    spec = Experiment(**value)
    expected = Experiment(
        spec.name,
        arm,
        microbatch=microbatch,
        activation_checkpointing=ac,
        steps=20,
        warmup_steps=2,
        save_every=10,
        validation_every=10,
        validation_samples=8,
    )
    require(spec == expected, "runtime cell changed an unselected mathematical or workload setting")
    return spec


def audit_profile(evidence: Evidence, record: dict, confirmation: dict, *, collected: list | None = None) -> list[dict]:
    ac = record["activation_checkpointing"]
    require(ac in ("full", "none"), "unsupported activation-checkpointing calibration")
    matrix = evidence.read(record["matrix_plan"])
    binding = next(row for row in confirmation["runtime_matrix_plans"] if row["activation_checkpointing"] == ac)
    require(evidence.sha(record["matrix_plan"]) == binding["sha256"], "matrix differs from sealed confirmation plan")
    require(
        matrix.get("protocol") == "cosmos.qlora-runtime-matrix-plan.v1"
        and matrix.get("training_runtime_revision") == TRAINING_REVISION
        and matrix.get("stage_iterations") == 20
        and matrix.get("effective_batch") == 8
        and matrix.get("device_process_budget_bytes") == BUDGET,
        "calibration matrix scope differs",
    )
    entries = [row for row in matrix["order"] if row["profile"] == "short_video8"]
    require(
        len(entries) == 6 and all([r["microbatch"] for r in entries if r["arm"] == arm] == [1, 2, 4] for arm in ARMS),
        "matrix must contain each arm's ordered B1/B2/B4",
    )
    attempt = record["attempt"]
    identity = evidence.read(child(attempt, "identity.json"))
    top = evidence.read(child(attempt, "completion.json"))
    plan = identity["plan"]
    require(identity.get("plan_sha256") == digest_plan(evidence, identity), "runtime submitted plan digest differs")
    require(
        plan.get("protocol") == "cosmos.qlora-gpu-plan.v2"
        and plan.get("mode") == "runtime_profile"
        and plan.get("profile") == "short_video8"
        and evidence.relative(plan["matrix_plan"]) == evidence.relative(record["matrix_plan"]),
        "runtime attempt used another plan or workload",
    )
    require(top.get("completed") is True, "runtime profile failed or is incomplete")
    workflow, _ = producer_profile(evidence, identity)
    tools = {
        Path(name).name: sha
        for name, sha in workflow["files"].items()
        if Path(name).parent == Path("tools/native_qlora_qualification") and name.endswith(".py")
    }
    build = identity["build"]
    require(
        build.get("code_revision") == TRAINING_REVISION and build.get("wheel_sha256") == WHEEL_SHA,
        "runtime training package differs",
    )
    bound = {
        row["path"]: row["sha256"] for row in plan["inputs"] if row.get("scope", "qualification") == "qualification"
    }
    require(bound.get(evidence.relative(record["matrix_plan"])) == binding["sha256"], "producer did not pin its matrix")
    workloads = plan["workloads"]
    manifests = {}
    for key, name in (
        ("workload_manifest_sha256", "manifest.json"),
        ("profile_manifest_sha256", "short_video8/manifest.json"),
    ):
        path = child(workloads, name)
        manifests[key] = evidence.read(path)
        require(evidence.sha(path) == plan[key] == bound.get(path), "producer workload manifest binding differs")
    profile = manifests["profile_manifest_sha256"]
    top_manifest = manifests["workload_manifest_sha256"]
    require(
        top_manifest.get("protocol") == "cosmos.native-qlora-runtime-workloads.v1"
        and top_manifest.get("runtime_stress_only") is True
        and top_manifest.get("quality_evidence") is False
        and top_manifest.get("seed") == 42
        and top_manifest["variants"]["short_video8"]
        == {"manifest_sha256": plan["profile_manifest_sha256"], **profile["policy"]},
        "workload manifest semantics differ",
    )
    for split, count in (("train", 32), ("validation", 8)):
        path = child(workloads, f"short_video8/{split}.jsonl")
        rows = evidence.read(path, lines=True)
        measurements = profile["measurements"][split]["rows"]
        require(
            evidence.sha(path) == profile["files"][split] == bound.get(path)
            and len(rows) == len(measurements) == count
            and [digest_json(row) for row in rows] == [row["output_row_sha256"] for row in measurements],
            "runtime fixture rows or measured shapes differ",
        )
    require(
        profile["policy"]
        == {
            "max_length": 4096,
            "processor_kwargs": {
                "images_kwargs": {"size": {"shortest_edge": 65536, "longest_edge": 262144}},
                "videos_kwargs": {
                    "num_frames": 8,
                    "fps": None,
                    "size": {"shortest_edge": 32768, "longest_edge": 524288},
                },
            },
            "device_process_budget_bytes": BUDGET,
            "microbatches": [{"microbatch": b, "grad_accum": 8 // b} for b in (1, 2, 4)],
        },
        "short_video8 policy changed",
    )
    outputs = top["outputs"]
    require(len(outputs) == 8, "profile must retain six cells and each arm's boundary disposition")
    result, previous, accepted, specs, anchor = ([] if collected is None else collected), {}, {}, {}, None
    for entry, output in zip(entries, outputs[:6], strict=True):
        arm, micro = entry["arm"], entry["microbatch"]
        require(
            output.get("arm") == arm and output.get("microbatch") == micro, "runtime cell order differs from matrix"
        )
        spec_value = evidence.read(entry["path"])
        require(
            evidence.sha(entry["path"]) == entry["sha256"] == bound.get(entry["path"]), "runtime spec digest differs"
        )
        spec = runtime_spec(spec_value, arm, micro, ac)
        specs[arm, micro] = spec
        can_run = micro == 1 or previous.get(arm, False)
        if not can_run:
            require(
                output
                == {"kind": "runtime_skipped", "arm": arm, "microbatch": micro, "reason": "previous_cell_not_eligible"},
                "escalation skip differs from sealed rule",
            )
            result.append(
                {
                    "arm": arm,
                    "microbatch": micro,
                    "activation_checkpointing": ac,
                    "state": "skipped_after_ineligible_predecessor",
                    "eligible": False,
                }
            )
            continue
        require(output.get("kind") == "runtime", "feasible runtime cell is missing")
        row, where = cell(evidence, output, spec, entry["sha256"], plan, build, identity["allocation"], profile, tools)
        if anchor is not None:
            require(where == anchor, "runtime cells lack identical actual GPU/allocation/CPU placement")
        anchor = where
        previous[arm] = row["eligible"]
        if row["eligible"]:
            accepted[arm] = micro
        result.append(row)
    expected_outer = [specs[row["arm"], row["microbatch"]].name for row in outputs[:6] if row["kind"] == "runtime"]
    for arm, output in zip(ARMS, outputs[6:], strict=True):
        if arm not in accepted:
            require(
                output == {"kind": "no_accepted_runtime_cell", "arm": arm, "lifecycle_qualified": False},
                "unexpected lifecycle claim without an eligible runtime cell",
            )
            continue
        require(output.get("kind") == "lifecycle", "largest-accepted boundary lifecycle is missing")
        spec = specs[arm, accepted[arm]]
        sha = lifecycle(evidence, output, spec, build, profile)
        expected_outer.append(spec.name + "_largest_accepted_lifecycle")
        for row in result:
            if row["arm"] == arm and row["eligible"]:
                row["lifecycle_completion_sha256"] = sha
                row["boundary_lifecycle_microbatch"] = accepted[arm]
    healthy_outcomes(evidence, attempt, top["outcomes"], tuple(expected_outer))
    return result


def digest_plan(evidence: Evidence, identity: dict) -> str:
    candidates = []
    for path, sha in evidence.inputs["files"].items():
        if sha == identity.get("plan_sha256"):
            value = evidence.read(path)
            if value == identity["plan"]:
                candidates.append(sha)
    require(len(candidates) == 1, "producer's submitted plan bytes are missing or ambiguous")
    return candidates[0]


def validate_plans(evidence: Evidence) -> tuple[dict, dict, dict]:
    paths = evidence.inputs["plans"]
    require(
        set(paths) == {"confirmation_plan", "development_plan", "development_selection"},
        "selection plans are incomplete",
    )
    confirmation, development, decision = (
        evidence.read(paths[key]) for key in ("confirmation_plan", "development_plan", "development_selection")
    )
    require(
        confirmation.get("protocol") == "cosmos.qlora-confirmation-and-best-fit-plan.v2"
        and confirmation.get("development_plan_sha256") == evidence.sha(paths["development_plan"])
        and confirmation.get("paired_seeds") == list(SEEDS),
        "confirmation plan differs",
    )
    rule = confirmation["best_fit"]["execution_selection_contract"]
    require(
        rule.get("activation_checkpointing") == ["full", "none"]
        and rule.get("microbatches") == [1, 2, 4]
        and rule.get("profile") == "short_video8"
        and rule.get("timed_updates_inclusive") == [3, 20]
        and rule.get("primary_metric") == "completion.native.windows.after_warmup.seconds.mean"
        and rule.get("sort_keys")
        == [
            "unrounded after-warmup mean seconds ascending",
            "training_budget.observed_process_peak_lower_bound_bytes ascending",
            "microbatch ascending",
            "activation_checkpointing full before none",
        ]
        and rule.get("fixed_model_execution")
        == {"adapter_compute_dtype": "float32", "logits_mode": "full", "compile_language_model": False}
        and all(
            confirmation["best_fit"].get(key) == value
            for key, value in (("steps", 300), ("warmup_steps", 30), ("effective_batch", 8))
        ),
        "execution selection rule changed",
    )
    require(
        development.get("protocol") == "cosmos.qlora-development-plan.v1"
        and decision.get("protocol") == "cosmos.qlora-development-selection.v1"
        and decision.get("complete") is True
        and decision.get("development_plan_sha256") == evidence.sha(paths["development_plan"])
        and decision.get("development_manifest_sha256") == development.get("development_manifest_sha256")
        and decision.get("confirmation_seeds_required") == list(SEEDS)
        and decision.get("fixed_release_gates") == development.get("fixed_gates")
        and decision.get("development_rows") == 584
        and decision.get("development_groups") == 12,
        "development decision is incomplete or differs from sealed development",
    )
    candidates = {"lr_5e5", "baseline", "lr_2e4", "rank32"}
    require(
        len(decision.get("candidates", [])) == 8
        and {(row["arm"], row["candidate"]) for row in decision["candidates"]}
        == {(arm, candidate) for arm in ARMS for candidate in candidates}
        and all(row.get("state") in ("completed", "failed") for row in decision["candidates"]),
        "development decision must retain all eight terminal candidates",
    )
    gates = development["fixed_gates"]
    require(
        gates.get("relative_supervised_token_nll_one_sided_95_upper_max") == 0.02
        and gates.get("accuracy_loss_one_sided_95_upper_max") == 0.01
        and gates.get("paired_training_seeds") == list(SEEDS),
        "release quality gates changed",
    )
    for arm in ARMS:
        require(
            next(row for row in decision["candidates"] if row["arm"] == arm and row["candidate"] == "baseline").get(
                "state"
            )
            == "completed",
            "failed development baseline prevents this arm's selection",
        )
        winner = decision["selected"][arm]
        records = [
            row for row in decision["candidates"] if row["arm"] == arm and row["candidate"] == winner["candidate"]
        ]
        require(
            decision["arm_status"].get(arm) == "development_candidate_selected"
            and len(records) == 1
            and records[0].get("eligible") is True
            and records[0].get("state") == "completed"
            and winner.get("evidence_sha256") == digest_json(records[0])
            and winner.get("contract_sha256") == records[0].get("contract_sha256")
            and winner.get("hyperparameters") == records[0].get("hyperparameters")
            and set(winner["hyperparameters"]) == {"learning_rate", "rank", "alpha"},
            "development winner lacks eligible candidate provenance",
        )
    return confirmation, development, decision


def analyze(evidence: Evidence) -> tuple[dict, dict | None]:
    report = {
        "protocol": PROTOCOL,
        "complete": False,
        "cells": [],
        "errors": [],
        "release_qualified": False,
        "untouched_scoring_authorized": False,
        "scope": "Configuration screening on32source runtime fixtures; full/none allocations may differ. Rank16 calibration does not qualify selected rank32 or300-update quality. Sampled memory peaks are lower bounds.",
    }
    try:
        confirmation, _, decision = validate_plans(evidence)
        profiles = evidence.inputs["profiles"]
        require(
            len(profiles) == 2 and {row["activation_checkpointing"] for row in profiles} == {"full", "none"},
            "both full and none runtime calibrations are required",
        )
        for record in profiles:
            collected = []
            try:
                audit_profile(evidence, record, confirmation, collected=collected)
            except (InvalidExecutionSelection, OSError, KeyError, TypeError, ValueError, StopIteration):
                report["errors"].append("invalid_or_incomplete_" + record["activation_checkpointing"] + "_calibration")
                known = {(row["arm"], row["microbatch"]) for row in collected}
                for arm in ARMS:
                    for microbatch in (1, 2, 4):
                        if (arm, microbatch) not in known:
                            collected.append(
                                {
                                    "arm": arm,
                                    "microbatch": microbatch,
                                    "activation_checkpointing": record["activation_checkpointing"],
                                    "state": "failed_missing_or_unvalidated",
                                    "eligible": False,
                                }
                            )
            report["cells"].extend(collected)
        require(not report["errors"], "all feasible calibration cells and boundary lifecycles must complete")
        completed = [row for row in report["cells"] if row["state"] == "completed"]
        require(
            all(
                len({row[key] for row in completed}) == 1
                for key in ("execution_environment_sha256", "workload_identity_sha256")
            ),
            "full/none calibration changed software, mathematics or measured workload",
        )
        winners = {arm: choose([row for row in report["cells"] if row["arm"] == arm]) for arm in ARMS}
        report["selected"] = winners
        report["complete"] = True
        return report, decision
    except (InvalidExecutionSelection, OSError, KeyError, TypeError, ValueError, StopIteration) as error:
        report["errors"].append(
            str(error)
            if isinstance(error, InvalidExecutionSelection)
            else "malformed_or_missing_sealed_selection_evidence"
        )
        return report, None


def publish(inputs_path: Path, inputs_sha: str, root: Path, output: Path) -> dict:
    raw = inputs_path.read_bytes()
    require(is_sha(inputs_sha) and hashlib.sha256(raw).hexdigest() == inputs_sha, "input manifest digest differs")
    evidence = Evidence(root, json.loads(raw))
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            "execution_selection.py",
            "performance.py",
            "protocol.py",
            "lifecycle.py",
            "processes.py",
            "common.py",
        )
    }
    report, development_decision = analyze(evidence)
    evidence.stable()
    require(
        hashlib.sha256(inputs_path.read_bytes()).hexdigest() == inputs_sha, "input manifest changed during analysis"
    )
    report["inputs_sha256"] = inputs_sha
    report["analysis_source_sha256"] = source_hashes
    report["evidence_sha256"] = sorted(set(evidence.used.values()))
    if report["complete"]:
        paths = evidence.inputs["plans"]
        bindings = {
            key: {"path": str(evidence.path(path)), "sha256": evidence.sha(path)} for key, path in paths.items()
        }
        execution = {
            "protocol": PROTOCOL,
            "complete": True,
            "confirmation_plan_sha256": bindings["confirmation_plan"]["sha256"],
            "development_selection_sha256": bindings["development_selection"]["sha256"],
            "runtime_matrix_plan_sha256s": [
                row["sha256"] for row in evidence.read(paths["confirmation_plan"])["runtime_matrix_plans"]
            ],
            "evidence_sha256": report["evidence_sha256"],
            "confirmation_specs": [],
            "selected": {},
            "inputs_sha256": inputs_sha,
            "analysis_source_sha256": report["analysis_source_sha256"],
            "release_qualified": False,
            "untouched_scoring_authorized": False,
        }
        for arm in ARMS:
            winner = report["selected"][arm]
            execution["selected"][arm] = {
                key: winner[key]
                for key in (
                    "activation_checkpointing",
                    "microbatch",
                    "runtime_completion_sha256",
                    "lifecycle_completion_sha256",
                )
            }
            for seed in SEEDS:
                spec = Experiment(
                    f"best_fit_{arm}_seed{seed}",
                    arm,
                    seed=seed,
                    **development_decision["selected"][arm]["hyperparameters"],
                    activation_checkpointing=winner["activation_checkpointing"],
                    microbatch=winner["microbatch"],
                )
                path = output / (spec.name + ".json")
                write_json(path, spec.as_dict())
                execution["confirmation_specs"].append(
                    {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                )
        # Validate the exact unchanged consumer before publishing a usable receipt.
        # The temporary validation file is removed on both success and failure.
        with tempfile.TemporaryDirectory(prefix="qlora-execution-validation-") as directory:
            pending = Path(directory) / "execution-selection.json"
            write_json(pending, execution)
            bindings["execution_selection"] = {
                "path": str(pending),
                "sha256": hashlib.sha256(pending.read_bytes()).hexdigest(),
            }
            selection = load_best_fit_selection(bindings)
            selection.assert_stable()
        evidence.stable()
        require(
            hashlib.sha256(inputs_path.read_bytes()).hexdigest() == inputs_sha,
            "input manifest changed before publication",
        )
        require(
            all(
                hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == sha
                for name, sha in source_hashes.items()
            ),
            "analyzer source changed before publication",
        )
        write_json(output / "execution-selection.json", execution)
    write_json(output / "decision.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = publish(args.inputs, args.inputs_sha256, args.evidence_root, args.output)
    if not result["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
