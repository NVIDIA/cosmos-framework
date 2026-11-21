# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Qualify native adapter interruption, exact continuation and portable reload.

Run inside an allocated, sealed single-GPU runtime. This tool submits no jobs and
contains no training loop: every update runs through the native SFT entrypoint.
Failure attempts and logs remain in the new output directory. Training horizon,
schedule and validation controls are identical within the four-update comparison.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tools.native_qlora_qualification.processes import OwnedProcessTree
except ModuleNotFoundError as error:
    if error.name != "tools":
        raise
    # Direct script invocation must not put the checkout root ahead of the sealed
    # installed cosmos_framework package. Load only this dependency-light sibling.
    import importlib.util

    _process_spec = importlib.util.spec_from_file_location(
        "native_qlora_processes", Path(__file__).with_name("processes.py")
    )
    _process_module = importlib.util.module_from_spec(_process_spec)
    _process_spec.loader.exec_module(_process_module)
    OwnedProcessTree = _process_module.OwnedProcessTree

INTEGRATION = "cosmos_framework.posttrain.tests.qlora_integration"
NATIVE = "cosmos_framework.posttrain.entrypoints.sft"
FIXED = {
    "rl.max_steps": 4,
    "rl.lr_schedule.cycle_steps": 4,
    "rl.optimizer.warmup_steps": 1,
    "rl.save_every": 1,
    "rl.keep_optim_last": 4,
    "rl.keep_model_last": 4,
    "rl.validation.enabled": True,
    "rl.validation.before_training": True,
    "rl.validation.every_steps": 2,
    "rl.validation.max_samples": 4,
    "rl.validation.save_checkpoint": True,
    "rl.deterministic": True,
    "rl.save_async": False,
    "rl.checkpoint_format": "dcp",
    "rl.checkpoint_store.backend": "local",
}
RESERVED = {"job.name", "job.group", "job.project", "rl.run_ts", "rl.resume", "rl.init_from"}
NUMERICAL_METRICS = (
    "train/loss_avg",
    "train/grad_norm",
    "train/learning_rate",
    "train/valid_token_count",
    "train/valid_sequence_count",
    "optim/optimizer_step",
    "optim/attempt_id",
    "optim/nonfinite_gradient",
    "optim/nonfinite_loss",
    "optim/nonfinite_skips_total",
    "optim/nonfinite_skips_consecutive",
    "optim/step_skipped",
    "optim/finite_gradient_check_enabled",
)


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def controlled_overrides(requested: list[str]) -> list[str]:
    """Refuse hidden identity/cadence changes instead of shadowing conflicting dotlists."""
    result, seen = [], set()
    for item in requested:
        key, separator, raw = item.partition("=")
        if not separator or key.startswith("+") or key in seen:
            raise ValueError(f"overrides require unique, explicit native keys: {key!r}")
        seen.add(key)
        if key in RESERVED:
            raise ValueError(f"qualification owns output and resume override {key!r}")
        if key in FIXED:
            try:
                value = json.loads(raw)
            except ValueError:
                value = raw
            if type(value) is not type(FIXED[key]) or value != FIXED[key]:
                raise ValueError(f"qualification fixes {key}={FIXED[key]!r}; conflicting override refused")
        else:
            result.append(item)
    return result + [f"{key}={json.dumps(value)}" for key, value in FIXED.items()]


def compare_next_metrics(reference: Path, candidate: Path, step: int) -> dict[str, Any]:
    def one(path: Path) -> dict[str, Any]:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        selected = [row for row in rows if row.get("step") == step and "train/loss_avg" in row]
        if len(selected) != 1:
            raise ValueError(f"expected exactly one native update metric at step {step}")
        row = selected[0]
        if row.get("optim/update_outcome") != "applied":
            raise ValueError("qualification requires an applied native optimizer update")
        values = {key: row[key] for key in NUMERICAL_METRICS}
        if not all(type(v) in (int, float) and math.isfinite(v) for v in values.values()):
            raise ValueError("missing or nonfinite native numerical update metric")
        if any(values[key] != step for key in ("optim/optimizer_step", "optim/attempt_id")):
            raise ValueError("native optimizer and attempt clocks differ")
        if any(
            values[key] != 0
            for key in NUMERICAL_METRICS
            if key.startswith("optim/")
            and key not in {"optim/optimizer_step", "optim/attempt_id", "optim/finite_gradient_check_enabled"}
        ):
            raise ValueError("native continuation reported a skipped/nonfinite update")
        if values["optim/finite_gradient_check_enabled"] != 1:
            raise ValueError("finite-gradient checks must remain enabled")
        return values

    left, right = one(reference), one(candidate)
    if left != right:
        raise AssertionError("native next-update numerical metrics differ")
    return {
        "passed": True,
        "optimizer_step": step,
        "numerical_metrics": left,
        "excluded_observational_metrics": ["timer/*", "train/preprocessing_*", "train/grad_clip_fraction"],
        "scope": (
            "Exact native numerical update metrics only; cold-cache counters, timings and reporter-history clip "
            "fractions are observational. Model, optimizer, control, RNG and data payload equality requires a "
            "separately executed checkpoint comparison and is not established by this metrics result."
        ),
    }


def activation_checkpointing_metadata_delta(reference: dict, candidate: dict) -> dict[str, Any]:
    """Accept only the recorded full-to-none execution change for an observation.

    This never edits an artifact or relaxes native resume/reload identity checks.
    All base, quantization, adapter, optimizer, build and training identities must
    remain exact. Both artifacts must first pass the native metadata reader.
    """
    expected = {
        ("identity", "strategy", "activation_checkpointing"): ("full_nonreentrant", "none"),
        ("identity", "execution_policy", "activation_checkpointing", "mode"): ("full", "none"),
        ("identity", "execution_policy", "activation_checkpointing", "implementation"): ("nonreentrant", "none"),
    }
    adjusted = copy.deepcopy(candidate)
    changes = {}
    try:
        for metadata in (reference, candidate):
            policy = metadata["identity"]["execution_policy"]
            if (
                policy["protocol"] != "cosmos.native-adapter-execution.v1"
                or policy["adapter_compute_dtype"] != "float32"
                or policy["sft_logits_mode"] != "full"
                or policy["language_compile"]["enabled"] is not False
            ):
                raise ValueError("cross-AC observation requires ordinary eager FP32-adapter execution with full logits")
        for path, (before, after) in expected.items():
            left, right = reference, adjusted
            for key in path[:-1]:
                left, right = left[key], right[key]
            key = path[-1]
            if left[key] != before or right[key] != after:
                raise ValueError("cross-AC observation requires an explicit full-to-none pair")
            changes[".".join(path)] = {"reference": before, "candidate": after}
            right[key] = before
    except (KeyError, TypeError) as error:
        raise ValueError("cross-AC observation requires complete explicit execution identities") from error
    if digest(reference) != digest(adjusted):
        raise ValueError("cross-AC observation found an additional metadata difference")
    return {
        "reference_metadata_sha256": digest(reference),
        "candidate_metadata_sha256": digest(candidate),
        "acknowledged_execution_differences": changes,
    }


def compare_activation_checkpointing_checkpoints(reference: Path, candidate: Path, *, step: int) -> dict[str, Any]:
    """Compare trusted own-run full/no-AC payloads, without constructing a model.

    Run in the allocated sealed runtime. This is a numerical experiment between
    distinct execution identities, not permission to resume one as the other.
    Reuse native commit validation, DCP loading and value/control comparisons.
    """
    import torch

    from cosmos_framework.posttrain.checkpoint.adapter import read_adapter_metadata
    from cosmos_framework.posttrain.tests.qlora_integration import (
        _assert_close_tree,
        _load_flat_dcp,
        _portable_trainer_controls,
        inspect_checkpoint,
    )

    if type(step) is not int or step < 1:
        raise ValueError("cross-AC comparison requires a positive completed optimizer step")
    paths = (Path(reference), Path(candidate))
    for path in paths:
        inspect_checkpoint(str(path), expected_step=step)
    metadata = [read_adapter_metadata(str(path), require_resume=True) for path in paths]
    identity = activation_checkpointing_metadata_delta(*metadata)
    for component in ("model", "optim"):
        left, right = [_load_flat_dcp(path / component) for path in paths]
        _assert_close_tree(left, right, atol=0, rtol=0, path=component)
        del left, right
    for component in ("trainer", "data"):
        # These are the allocated experiment's trusted native artifacts, not
        # arbitrary uploaded pickle files. Publication IDs alone may differ.
        left, right = [torch.load(path / f"{component}.pt", map_location="cpu", weights_only=False) for path in paths]
        if component == "trainer":
            left, right = _portable_trainer_controls(left), _portable_trainer_controls(right)
        _assert_close_tree(left, right, atol=0, rtol=0, path=component)
    return {
        "passed": True,
        "optimizer_step": step,
        "atol": 0,
        "rtol": 0,
        "model_optimizer_values_exact": True,
        "controls_rng_and_data_exact": True,
        "observation_only": True,
        "resume_compatible": False,
        "literal_tensor_bytes_or_serialized_dcp_equality_claimed": False,
        "excluded_storage_identity_fields": [
            "validation.publication.destination",
            "validation.publication.id",
            "validation.publication.manifest",
        ],
        **identity,
    }


def compare_wts_scores(native: Path, fresh: Path, *, checkpoint: Path, expected_rows: int) -> dict[str, Any]:
    """Require exact selected rows, every numerical score and loaded adapter identity."""
    from tools.native_qlora_qualification.quality import validate_scores

    summaries, selections, score_rows, selected_rows, pins = [], [], [], [], {}
    for folder in (Path(native), Path(fresh)):
        for name in ("summary.json", "selection.json", "scores.jsonl", "selected_rows.jsonl"):
            pins[str(folder / name)] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        summary = json.loads((folder / "summary.json").read_text())
        selection = json.loads((folder / "selection.json").read_text())
        scores = [json.loads(line) for line in (folder / "scores.jsonl").read_text().splitlines() if line.strip()]
        selected = [
            json.loads(line) for line in (folder / "selected_rows.jsonl").read_text().splitlines() if line.strip()
        ]
        validate_scores(scores)
        if (
            summary.get("complete") is not True
            or summary.get("checkpoint") != str(checkpoint.resolve())
            or summary.get("scores_sha256") != pins[str(folder / "scores.jsonl")]
            or summary.get("selection_sha256") != digest(selection)
            or selection.get("selected_rows_file_sha256") != pins[str(folder / "selected_rows.jsonl")]
            or len(scores) != expected_rows
            or len(selected) != expected_rows
        ):
            raise ValueError("native/fresh WTS parity requires complete owned scoring artifacts")
        summaries.append(summary)
        selections.append(selection)
        score_rows.append(scores)
        selected_rows.append(selected)
    if score_rows[0] != score_rows[1] or selected_rows[0] != selected_rows[1]:
        raise AssertionError("native and fresh WTS per-example scores or selected rows differ")
    for key in (
        "source_sha256",
        "split_manifest_sha256",
        "split_role",
        "ordered_ids",
        "ordered_row_identities",
        "selected_content_sha256",
        "selected_rows_file_sha256",
    ):
        if selections[0][key] != selections[1][key]:
            raise AssertionError(f"native/fresh WTS selection differs: {key}")
    for key in (
        "metrics",
        "adapter_tensors_sha256",
        "processor_contract",
        "objective_identity",
        "model_identity_sha256",
    ):
        if summaries[0][key] != summaries[1][key]:
            raise AssertionError(f"native/fresh WTS numerical or loaded identity differs: {key}")
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != pin for path, pin in pins.items()):
        raise ValueError("WTS parity artifacts changed during comparison")
    return {
        "passed": True,
        "rows": expected_rows,
        "per_example_and_selection_exact": True,
        "adapter_and_processor_identity_exact": True,
        "source_sha256": pins,
        "quality_qualified": False,
    }


def wts_metadata_delta(reference: dict, candidate: dict, expected: dict) -> dict[str, Any]:
    """Acknowledge only enabling the declared evaluation contract; never permit cross-resume."""
    adjusted = copy.deepcopy(candidate)
    try:
        left = reference["identity"]["training_contract"]["validation"]["fields"]
        right = adjusted["identity"]["training_contract"]["validation"]["fields"]
        if (
            "wts" in left
            or right.pop("wts") != expected
            or expected["__dataclass_type__"] != "cosmos_framework.posttrain.config.sft.SFTWTSConfig"
            or expected["fields"]["enabled"] is not True
        ):
            raise ValueError("expected explicit disabled-to-declared-WTS evaluation identity")
    except (KeyError, TypeError) as error:
        raise ValueError("missing explicit native WTS identity") from error
    if digest(reference) != digest(adjusted):
        raise ValueError("an additional native metadata difference is not an evaluation journal")
    return {
        "reference_metadata_sha256": digest(reference),
        "candidate_metadata_sha256": digest(candidate),
        "acknowledged_metadata_field": "identity.training_contract.validation.fields.wts",
        "wts_config": expected,
    }


def wts_control_delta(reference: dict, candidate: dict, *, expected: dict, journal: dict) -> tuple[dict, dict]:
    """Normalize the explicit evaluation-only journal, retaining all ordinary controls."""
    from cosmos_framework.posttrain.tests.qlora_integration import _portable_trainer_controls

    left, right = _portable_trainer_controls(reference), _portable_trainer_controls(candidate)
    a, b = left["validation"], right["validation"]
    if a["version"] != 2 or b["version"] != 3 or "wts" in a or b.pop("wts") != journal:
        raise ValueError("unexpected evaluation journal/version delta")
    if "wts" in a["identity"]:
        raise ValueError("disabled reference unexpectedly contains WTS identity")
    identity = b["identity"].pop("wts")
    expected = expected["fields"]
    fixed = {
        "protocol": "cosmos.native-sft-wts.v1",
        "source_sha256": expected["source_sha256"],
        "manifest_sha256": expected["manifest_sha256"],
        "expected_rows": expected["expected_rows"],
        "steps": expected["steps"],
        "cache_bytes": expected["cache_bytes"],
        "min_correct": expected["min_correct"],
        "max_token_ce": expected["max_token_ce"],
        "stop_on_target": expected["stop_on_target"],
    }
    if set(identity) != {*fixed, "source_fingerprint", "processor", "objective", "selection_sha256"} or any(
        identity[key] != value for key, value in fixed.items()
    ):
        raise ValueError("evaluation journal identity differs from its fixed configuration")
    if any(not re.fullmatch("[a-f0-9]{64}", identity[key]) for key in ("source_fingerprint", "selection_sha256")):
        raise ValueError("evaluation source/selection identity is not pinned")
    b["version"] = 2
    return left, right


def hard_gold_evaluation_identity(training_objective: dict) -> dict:
    """Remove only recorded training context and the exact training-only WTS regularizer."""
    scoring = {
        key: value
        for key, value in training_objective.items()
        if key not in {"training_source", "micro_batch_size", "grad_accum", "dp_world", "precision", "preprocessing"}
    }
    if "wts_answer_smoothing" in scoring:
        smoothing = scoring["wts_answer_smoothing"]
        epsilon = smoothing.get("epsilon") if isinstance(smoothing, dict) else None
        if (
            type(epsilon) not in (int, float)
            or not 0 < epsilon < 1
            or type(smoothing.get("version")) is not int
            or scoring.get("reduction") != "global_supervised_token_mean"
            or "ce_exponent" in scoring
            or "moe_lbl" in scoring
            or smoothing
            != {
                "version": 1,
                "epsilon": epsilon,
                "distribution": "gold_plus_uniform_actual_2_to_4_choices",
                "denominator": "full_vocabulary_log_softmax",
                "alignment": "context_stable_first_answer_after_native_assistant_header_v1",
                "supervision": "single_answer_and_im_end",
                "scope": "training_only_first_answer",
                "remaining_tokens_and_evaluation": "hard_gold_cross_entropy",
            }
        ):
            raise ValueError("unrecognized training-only WTS smoothing contract")
        scoring.pop("wts_answer_smoothing")
    return scoring


def compare_wts_checkpoints(
    reference: Path, candidate: Path, *, step: int, expected: dict, journal: dict
) -> dict[str, Any]:
    """Allocated own-run tensor/control parity with only explicit evaluation bookkeeping differences."""
    import torch

    from cosmos_framework.posttrain.checkpoint.adapter import read_adapter_metadata
    from cosmos_framework.posttrain.tests.qlora_integration import (
        _assert_close_tree,
        _load_flat_dcp,
        inspect_checkpoint,
    )

    paths = (Path(reference), Path(candidate))
    before = [inspect_checkpoint(str(path), expected_step=step) for path in paths]
    metadata = [read_adapter_metadata(str(path), require_resume=True) for path in paths]
    delta = wts_metadata_delta(*metadata, expected)
    for component in ("model", "optim"):
        left, right = [_load_flat_dcp(path / component) for path in paths]
        _assert_close_tree(left, right, atol=0, rtol=0, path=component)
        del left, right
    for component in ("trainer", "data"):
        values = [torch.load(path / f"{component}.pt", map_location="cpu", weights_only=False) for path in paths]
        if component == "trainer":
            identity = values[1]["validation"]["identity"]["wts"]
            if identity["processor"] != metadata[1]["identity"]["processor_contract"]:
                raise ValueError("evaluation and training processor identities differ")
            training_objective = metadata[1]["identity"]["training_contract"]["objective"]
            scoring_objective = hard_gold_evaluation_identity(training_objective)
            if identity["objective"] != scoring_objective:
                raise ValueError("evaluation objective differs from the native training arithmetic")
            values = wts_control_delta(*values, expected=expected, journal=journal)
        _assert_close_tree(*values, atol=0, rtol=0, path=component)
    if [inspect_checkpoint(str(path), expected_step=step) for path in paths] != before:
        raise ValueError("WTS comparison checkpoint source changed during payload reads")
    return {
        "passed": True,
        "optimizer_step": step,
        "atol": 0,
        "rtol": 0,
        "model_optimizer_values_exact": True,
        "remaining_controls_rng_and_data_exact": True,
        "observation_only": True,
        "resume_compatible": False,
        "wts_journal_sha256": digest(journal),
        **delta,
    }


class Stages:
    def __init__(self, output: Path, environment: dict[str, str], timeout: int):
        self.process_capability = OwnedProcessTree.check_support()
        self.output, self.environment, self.timeout = output, environment, timeout
        (output / "stages").mkdir()
        write_json(output / "process-capability.json", self.process_capability)
        self.outcomes: list[dict[str, Any]] = []

    def run(
        self, name: str, command: list[str], *, interrupted: bool = False, interruption_receipt: Path | None = None
    ) -> Path:
        signal_dir = self.environment.get("SLURM_LOG_DIR")
        if signal_dir and (Path(signal_dir) / "SIGUSR1_RECEIVED").exists():
            raise RuntimeError("scheduler stop marker forbids beginning another qualification stage")
        folder = self.output / "stages" / name
        folder.mkdir()
        write_json(folder / "command.json", command)
        environment = {**self.environment, "COSMOS_QLORA_METRICS_FILE": str(folder / "memory.jsonl")}
        started = time.monotonic()
        child = None
        code = None
        tracker = None
        cleanup = None
        operation_error = None
        timeout = False
        try:
            with (folder / "output.log").open("x") as stream:
                child = subprocess.Popen(
                    command, stdout=stream, stderr=subprocess.STDOUT, env=environment, start_new_session=True
                )
                tracker = OwnedProcessTree(child.pid)
                try:
                    code = child.wait(timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    timeout = True
                    cleanup = tracker.cleanup()
                    tracker = None
                    code = child.wait()
                if interrupted and code != 0 and not timeout:
                    if interruption_receipt is None:
                        raise ValueError("interrupted stage requires its own rank receipt")
                    event = json.loads(interruption_receipt.read_text())
                    tracker.validate_interruption(event["process_snapshot"])
        except BaseException as error:
            operation_error = error
        finally:
            if tracker is not None:
                try:
                    cleanup = tracker.cleanup()
                except BaseException as error:
                    operation_error = error
                    cleanup = {"failed": True, "error_type": type(error).__name__}
        outcome = {
            "name": name,
            "exit_code": code,
            "elapsed_seconds": time.monotonic() - started,
            "timed_out": timeout,
            "injected_interruption_expected": interrupted,
            "owned_process_cleanup": cleanup,
        }
        write_json(folder / "outcome.json", outcome)
        self.outcomes.append(outcome)
        if operation_error is not None:
            raise RuntimeError(f"qualification stage {name!r} failed; see preserved log") from operation_error
        if timeout or (code == 0 if interrupted else code != 0):
            raise RuntimeError(f"qualification stage {name!r} returned an unexpected status; see preserved log")
        return folder


def run(args: argparse.Namespace) -> dict[str, Any]:
    from cosmos_framework.posttrain.config.build import select_policy_backend
    from cosmos_framework.posttrain.entrypoints.sft import validate_config_structure
    from cosmos_framework.posttrain.tests.qlora_integration import inspect_checkpoint
    from cosmos_framework.posttrain.utils.identity import canonical_config_value
    from cosmos_framework.utils.config import load_config
    from cosmos_framework.utils.lazy_config import instantiate

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    stages = None
    try:
        overrides = controlled_overrides(args.override)
        config = load_config(args.config, ["--", *overrides])
        rl = validate_config_structure(config)
        model = instantiate(select_policy_backend(config).model_config)
        qualify_wts = getattr(args, "qualify_wts", False)
        if qualify_wts and (
            not rl.validation.wts.enabled
            or not rl.validation.wts.stop_on_target
            or rl.validation.wts.steps != [2, 4]
            or rl.validation.wts.expected_rows != 4
            or rl.validation.wts.min_correct != 4
            or rl.validation.wts.max_token_ce != 0
        ):
            raise ValueError(
                "WTS qualification requires a sealed four-row development source, steps[2,4], correct4/CE0"
            )
        if getattr(model, "adapter", None) is None or rl.grad_accum < 2:
            raise ValueError("recovery qualification requires native adapters and grad_accum>=2")
        receipt = {
            "code_revision": os.environ.get("COSMOS_TRAINING_CODE_REVISION", ""),
            "wheel_sha256": os.environ.get("COSMOS_TRAINING_WHEEL_SHA256", ""),
        }
        if not re.fullmatch("[a-f0-9]{40}", receipt["code_revision"]) or not re.fullmatch(
            "[a-f0-9]{64}", receipt["wheel_sha256"]
        ):
            raise ValueError("qualification requires the sealed build receipt environment")
        spec = {
            "protocol": "cosmos.native-qlora-recovery-spec.v1",
            "config": args.config,
            "requested_overrides": args.override,
            "qualification_overrides": overrides,
            "build_receipt": receipt,
            "resolved_model": canonical_config_value(model),
            "resolved_sft": canonical_config_value(rl),
            "selected": {
                "arm": "nf4" if model.quantized_base is not None else "bf16",
                "seed": rl.seed,
                "adapter_initialization_seed": model.adapter.initialization_seed,
                "lr": rl.optimizer.lr,
                "rank": model.adapter.rank,
                "alpha": model.adapter.alpha,
                "microbatch": rl.micro_batch_size,
                "grad_accum": rl.grad_accum,
                "processor_kwargs": canonical_config_value(rl.processor_kwargs),
                "quantization": canonical_config_value(model.quantized_base),
            },
            "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
            "exact_atol": 0,
            "exact_rtol": 0,
            "scope": "Four-update qualification; full quality training uses its independently recorded horizon and schedule.",
        }
        if args.spec and json.loads(Path(args.spec).read_text()) != spec:
            raise ValueError("immutable recovery experiment spec differs from resolved native controls")
        write_json(output / "experiment.json", spec)
        if args.plan_only:
            return {"complete": False, "plan_only": True, "experiment_sha256": digest(spec)}
        env = {**os.environ, "IMAGINAIRE_OUTPUT_ROOT": str(output / "native")}
        stages = Stages(output, env, args.stage_timeout)
        launcher = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1", "-m"]
        roots = {}

        def native(
            name: str, *, mode: str | None = None, source: str = "off", disable_wts: bool = False
        ) -> tuple[Path, list[str]]:
            root = output / "native/qualification/recovery" / name / "canary"
            roots[name] = root
            owned_overrides = (
                [v for v in overrides if not v.startswith("rl.validation.wts.")] + ["rl.validation.wts.enabled=false"]
                if disable_wts
                else overrides
            )
            opts = owned_overrides + [
                "job.project=qualification",
                "job.group=recovery",
                f"job.name={name}",
                "rl.run_ts=canary",
                f"rl.resume={source}",
            ]
            if mode:
                marker = output / f"{name}-interruption.json"
                command = launcher + [
                    INTEGRATION,
                    "interrupt-native",
                    "--config",
                    args.config,
                    "--checkpoint-root",
                    str(root / "checkpoints"),
                    "--receipt",
                    str(marker),
                    "--mode",
                    mode,
                    "--after-step",
                    "2",
                ]
                command += [part for value in opts for part in ("--override", value)]
            else:
                command = launcher + [NATIVE, "--config", args.config, "--", *opts]
            stages.run(name, command, interrupted=mode is not None, interruption_receipt=marker if mode else None)
            if mode:
                event = json.loads(marker.read_text())
                expected = inspect_checkpoint(str(root / "checkpoints/iter_000002"), expected_step=2)
                if event.get("exit_code") != 86 or event.get("mode") != mode or event["checkpoint"] != expected:
                    raise ValueError("native interruption receipt/source differs from its committed checkpoint")
                if (root / "checkpoints/iter_000003").exists():
                    raise ValueError("interrupted run unexpectedly published its next update")
            return root, opts

        native("reference")
        proof = {}
        for mode in ("boundary", "accumulation"):
            root, _ = native(mode, mode=mode)
            source = str(root / "checkpoints/iter_000002")
            stages.run(
                mode + "_source_compare",
                [
                    sys.executable,
                    "-m",
                    INTEGRATION,
                    "compare-checkpoints",
                    "--reference",
                    str(roots["reference"] / "checkpoints/iter_000002"),
                    "--candidate",
                    source,
                    "--atol",
                    "0",
                    "--rtol",
                    "0",
                ],
            )
            resumed, _ = native(mode + "_resumed", source=source)
            comparisons = {}
            for step in (3, 4):
                record = inspect_checkpoint(str(resumed / f"checkpoints/iter_{step:06d}"), expected_step=step)
                folder = stages.run(
                    f"{mode}_compare_{step}",
                    [
                        sys.executable,
                        "-m",
                        INTEGRATION,
                        "compare-checkpoints",
                        "--reference",
                        str(roots["reference"] / f"checkpoints/iter_{step:06d}"),
                        "--candidate",
                        record["path"],
                        "--atol",
                        "0",
                        "--rtol",
                        "0",
                    ],
                )
                metrics = compare_next_metrics(roots["reference"] / "metrics.jsonl", resumed / "metrics.jsonl", step)
                write_json(folder / "numerical-update.json", metrics)
                comparisons[str(step)] = {"checkpoint": record, "metrics": metrics}
            proof[mode] = comparisons
        source = str(roots["accumulation_resumed"] / "checkpoints/iter_000004")
        portable_hint = output / "portable/checkpoints/iter_000004"
        stages.run(
            "publish_portable",
            [
                sys.executable,
                "-m",
                "cosmos_framework.posttrain.checkpoint.adapter",
                "--source",
                source,
                "--output-iter",
                str(portable_hint),
            ],
        )
        portable = output / "portable/checkpoint_artifacts/iter_000004"
        for kind, checkpoint in (("dcp", source), ("portable", str(portable))):
            command = launcher + [
                INTEGRATION,
                "probe",
                "--config",
                args.config,
                "--checkpoint",
                checkpoint,
                "--data",
                args.data,
                "--output",
                str(output / "probes" / kind),
                "--require-updates",
            ]
            image = (
                args.image
                if kind == "dcp"
                else str(Path(args.image).resolve())
                if args.image
                else str(output / "probes/dcp/video_first_frame.png")
            )
            if image:
                command += ["--image", image]
            command += [part for value in overrides for part in ("--override", value)]
            stages.run("probe_" + kind, command)
        stages.run(
            "compare_reload",
            [
                sys.executable,
                "-m",
                INTEGRATION,
                "compare-probes",
                "--reference",
                str(output / "probes/dcp"),
                "--candidate",
                str(output / "probes/portable"),
                "--atol",
                "0",
                "--rtol",
                "0",
            ],
        )
        wts_proof = None
        if qualify_wts:
            reference = roots["reference"]
            result_path = reference / "wts/target-result.json"
            native_result = json.loads(result_path.read_text())
            observations = native_result["observations"]
            if (
                native_result["protocol"] != "cosmos.native-sft-wts-target.v1"
                or native_result["complete"] is not True
                or native_result["final_step"] != 4
                or native_result["first_passing_step"] is not None
                or native_result["horizon_censored"] is not True
                or [v["step"] for v in observations] != [2, 4]
                or any(v["target_met"] is not False for v in observations)
            ):
                raise ValueError("WTS qualification must complete both nonpassing scheduled observations")
            checkpoint = reference / "checkpoints/iter_000004"
            selected = (reference / observations[-1]["directory"]).resolve()
            if not selected.is_relative_to((reference / "wts").resolve()):
                raise ValueError("native WTS qualification scores are not owned")
            fresh = output / "fresh-wts-scores"
            score = Path(__file__).with_name("score.py")
            stages.run(
                "fresh_wts_score",
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc-per-node=1",
                    str(score),
                    "--config",
                    args.config,
                    "--checkpoint",
                    str(checkpoint),
                    "--data",
                    rl.validation.wts.data.path,
                    "--output",
                    str(fresh),
                    "--all-rows",
                    "--split",
                    "development",
                    "--split-manifest",
                    rl.validation.wts.manifest_path,
                    "--split-manifest-sha256",
                    rl.validation.wts.manifest_sha256,
                    "--preprocessing-cache-bytes",
                    str(rl.validation.wts.cache_bytes),
                    *[part for value in overrides for part in ("--override", value)],
                ],
            )
            parity = compare_wts_scores(selected, fresh, checkpoint=checkpoint, expected_rows=4)
            write_json(output / "wts-score-parity.json", parity)
            disabled, _ = native("without_wts", disable_wts=True)
            comparisons = {}
            journal_keys = {
                "step",
                "rows",
                "correct",
                "token_weighted_ce",
                "target_met",
                "metrics",
                "scores_sha256",
                "selected_content_sha256",
                "processed_inputs_sha256",
            }
            for step in (2, 4):
                settings = {
                    "expected": canonical_config_value(rl.validation.wts),
                    "journal": {
                        "version": 1,
                        "observations": [{k: v[k] for k in journal_keys} for v in observations if v["step"] <= step],
                    },
                }
                settings_path = output / f"wts-comparison-{step}.json"
                write_json(settings_path, settings)
                comparison_path = output / f"wts-state-parity-{step}.json"
                stages.run(
                    f"wts_state_compare_{step}",
                    [
                        sys.executable,
                        "-c",
                        "import json,sys; from pathlib import Path; "
                        "from tools.native_qlora_qualification.lifecycle import compare_wts_checkpoints,write_json; "
                        "settings=json.loads(Path(sys.argv[4]).read_text()); "
                        "write_json(Path(sys.argv[5]),compare_wts_checkpoints(Path(sys.argv[1]),Path(sys.argv[2]),"
                        "step=int(sys.argv[3]),**settings))",
                        str(disabled / f"checkpoints/iter_{step:06d}"),
                        str(reference / f"checkpoints/iter_{step:06d}"),
                        str(step),
                        str(settings_path),
                        str(comparison_path),
                    ],
                )
                comparisons[str(step)] = json.loads(comparison_path.read_text())
            wts_proof = {
                "fresh_score_parity": parity,
                "evaluation_disabled_state_comparisons": comparisons,
                "native_result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                "scope": "Four-row CUDA parity and own recovery only; not full-development quality or continuous timing.",
            }
        result = {
            "protocol": "cosmos.native-qlora-recovery-result.v1",
            "complete": True,
            "experiment_sha256": digest(spec),
            "build_receipt": receipt,
            "recovery": proof,
            "outcomes": stages.outcomes,
            "reload_exact": True,
            "release_quality_qualified": False,
            **({"wts_qualification": wts_proof} if qualify_wts else {}),
        }
        write_json(output / "completion.json", result)
        return result
    except BaseException as error:
        write_json(
            output / "failure.json",
            {
                "complete": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "outcomes": stages.outcomes if stages else [],
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="New directory; failed attempts are never overwritten")
    parser.add_argument("--data", required=True)
    parser.add_argument("--image")
    parser.add_argument("--spec", help="Optional exact experiment.json from a reviewed identical plan")
    parser.add_argument(
        "--plan-only", action="store_true", help="Resolve and persist the immutable spec without launching"
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument(
        "--qualify-wts",
        action="store_true",
        help="After own recovery, qualify four-row native/fresh scoring and evaluation-only state differences.",
    )
    parser.add_argument("--stage-timeout", type=int, default=3600)
    args = parser.parse_args()
    if args.stage_timeout < 1:
        parser.error("stage-timeout must be positive")
    result = run(args)
    print(
        json.dumps({"complete": result["complete"], "experiment_sha256": result["experiment_sha256"]}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
