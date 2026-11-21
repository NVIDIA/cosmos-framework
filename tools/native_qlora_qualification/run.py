# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Measure native training through portable publication and sealed-split evaluation.

This is an experiment orchestrator, not a training implementation. Every optimizer
update runs in cosmos_framework.posttrain.entrypoints.sft under torchrun. Invoke
inside a single-GPU scheduler allocation with the sealed posttrain environment.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from processes import OwnedProcessTree
from protocol import Experiment, file_sha256, write_json
from storage import StorageProbeError, require_storage


def verify_build(build: Path) -> dict:
    receipt = json.loads((build / "receipt.json").read_text())
    wheel = build / receipt["wheel_file"]
    if file_sha256(wheel) != receipt["wheel_sha256"]:
        raise ValueError("sealed wheel digest differs")
    import cosmos_framework

    if Path(cosmos_framework.__file__).resolve().parent.parent != (build / "site").resolve():
        raise ValueError("framework must resolve to the sealed wheel overlay")
    count = 0
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith("cosmos_framework/") and not name.endswith("/"):
                if archive.read(name) != (build / "site" / name).read_bytes():
                    raise ValueError(f"installed package differs from sealed wheel: {name}")
                count += 1
    if count != receipt["package_files"]:
        raise ValueError("sealed package inventory differs")
    for name, expected in {"torch": "2.11.0+cu128", "transformers": "5.8.1", "bitsandbytes": "0.49.2"}.items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"unsupported runtime version for {name}")
    return receipt


def verify_kernel_gate(path: Path, receipt: dict) -> dict:
    gate = json.loads(path.read_text())
    if gate.get("completed") is not True or gate.get("build") != receipt:
        raise ValueError("training requires the completed real-kernel gate for this exact build")
    for label, path_key, digest_key, count in (
        ("real CUDA", "junit_xml", "junit_sha256", 14),
        ("higher-precision multimodal", "higher_precision_junit_xml", "higher_precision_junit_sha256", 1),
    ):
        xml_path, expected_digest = gate.get(path_key), gate.get(digest_key)
        if not isinstance(xml_path, str) or not xml_path or not isinstance(expected_digest, str):
            raise ValueError(f"kernel gate is missing its {label} XML identity")
        try:
            junit = Path(xml_path)
            if file_sha256(junit) != expected_digest:
                raise ValueError(f"kernel gate {label} XML differs from its receipt")
            cases = list(ET.parse(junit).getroot().iter("testcase"))
        except (OSError, ET.ParseError) as error:
            raise ValueError(f"kernel gate {label} XML is unavailable or invalid") from error
        if len(cases) != count or any(
            case.find(tag) is not None for case in cases for tag in ("skipped", "failure", "error")
        ):
            raise ValueError(f"kernel gate requires all {count} {label} tests without skips or failures")
    return gate


def preflight(output: Path) -> None:
    """Executed in a disposable process so the orchestrator retains no CUDA context."""
    import torch

    from cosmos_framework.posttrain.model.quantization.loader import runtime_precision_policy
    from cosmos_framework.posttrain.utils.determinism import enable_determinism

    enable_determinism(strict=True)
    props = torch.cuda.get_device_properties(0)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (8, 0):
        raise ValueError("qualification requires one visible SM80 accelerator")
    if "A100" not in props.name or props.total_memory < 79_000_000_000:
        raise ValueError("this qualification is bounded to A100 80GB")
    topology = []
    for cpu in sorted(os.sched_getaffinity(0)):
        directory = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        topology.append(
            {
                "cpu": cpu,
                "package": int((directory / "topology/physical_package_id").read_text()),
                "core": int((directory / "topology/core_id").read_text()),
                "numa_nodes": sorted(int(path.name.removeprefix("node")) for path in directory.glob("node[0-9]*")),
            }
        )
    physical_cores = len({(row["package"], row["core"]) for row in topology})
    if physical_cores != 8:
        raise ValueError(f"controlled qualification requires eight physical CPU cores; got {physical_cores}")
    write_json(
        output,
        {
            "name": props.name,
            "device_bytes": props.total_memory,
            "uuid": "GPU-" + str(uuid.UUID(str(props.uuid).removeprefix("GPU-"))),
            "runtime_precision": runtime_precision_policy(),
            "torch_git_version": torch.version.git_version,
            "torch_cuda": torch.version.cuda,
            "python": sys.version,
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "bitsandbytes",
                    "torchcodec",
                    "torchvision",
                    "safetensors",
                    "accelerate",
                )
            },
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "cpu_topology": topology,
            "physical_cpu_cores": physical_cores,
            "allocation_id": os.environ.get("SLURM_JOB_ID"),
            "host_id_sha256": hashlib.sha256(os.uname().nodename.encode()).hexdigest(),
            "thread_environment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TOKENIZERS_PARALLELISM")
            },
        },
    )


def kernel_gate(args: argparse.Namespace) -> None:
    output = Path(args.output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    receipt = verify_build(Path(args.build))
    stages = Stages(output, Path(args.time_binary))
    stages.run(
        "preflight",
        [sys.executable, str(Path(__file__).resolve()), "preflight", "--output", str(output / "runtime.json")],
    )
    junit = output / "gpu-tests.xml"
    tests = Path(args.build) / "site/cosmos_framework/posttrain/tests"
    stages.run(
        "real_cuda_kernels",
        [
            sys.executable,
            "-m",
            "pytest",
            "--noconftest",
            "-c",
            "/dev/null",
            "-q",
            f"--junitxml={junit}",
            "-o",
            f"cache_dir={output / 'pytest-cache'}",
            *[
                str(tests / name)
                for name in (
                    "test_model_qwen3_vl_qlora_gpu.py",
                    "test_native_adapter_execution_gpu.py",
                    "test_sft_selected_logits_gpu.py",
                )
            ],
        ],
    )
    gate = {
        "completed": True,
        "build": receipt,
        "junit_xml": str(junit),
        "junit_sha256": file_sha256(junit),
        "outcomes": stages.outcomes,
    }
    # Verify before publication: an XML containing skips cannot become a gate receipt.
    cases = list(ET.parse(junit).getroot().iter("testcase"))
    if len(cases) != 14 or any(case.find(tag) is not None for case in cases for tag in ("skipped", "failure", "error")):
        raise RuntimeError("expected all 14 real CUDA tests without skips")
    cpu_junit = output / "higher-precision-tests.xml"
    stages.run(
        "higher_precision_multimodal_regression",
        [
            sys.executable,
            "-m",
            "pytest",
            "--noconftest",
            "-c",
            "/dev/null",
            "-q",
            f"--junitxml={cpu_junit}",
            "-o",
            f"cache_dir={output / 'pytest-cache'}",
            str(tests / "test_entrypoints_sft.py") + "::test_real_qwen_multimodal_recipe_cpu",
        ],
        env_extra={"CUDA_VISIBLE_DEVICES": ""},
    )
    cpu_cases = list(ET.parse(cpu_junit).getroot().iter("testcase"))
    if len(cpu_cases) != 1 or any(
        case.find(tag) is not None for case in cpu_cases for tag in ("skipped", "failure", "error")
    ):
        raise RuntimeError("existing higher-precision multimodal regression must pass without skips")
    gate.update(higher_precision_junit_xml=str(cpu_junit), higher_precision_junit_sha256=file_sha256(cpu_junit))
    write_json(output / "completion.json", gate)


class Stages:
    def __init__(self, output: Path, time_binary: Path, *, timeout_seconds: int = 7200):
        self.process_capability = OwnedProcessTree.check_support()
        self.output, self.time_binary, self.timeout = output, time_binary, timeout_seconds
        self.outcomes: list[dict] = []
        self.work = output / "work"
        self.work.mkdir()
        (output / "stages").mkdir()
        write_json(output / "process-capability.json", self.process_capability)
        self.marker = output / "current-stage.txt"
        from tools.native_qlora_qualification.monitor_stage import publish_stage

        publish_stage(self.marker, "initialization", initialize=True)

    def run(self, name: str, command: list[str], *, env_extra: dict | None = None) -> None:
        stage = self.output / "stages" / name
        stage.mkdir()
        from tools.native_qlora_qualification.monitor_stage import publish_stage

        write_json(stage / "stage-publication.json", publish_stage(self.marker, name))
        write_json(stage / "command.json", command)
        env = dict(os.environ, COSMOS_QLORA_METRICS_FILE=str(stage / "memory.jsonl"), **(env_extra or {}))
        started = time.monotonic()
        with (stage / "output.log").open("x") as stream:
            child = subprocess.Popen(
                [str(self.time_binary), "-v", "-o", str(stage / "time.txt"), *command],
                cwd=self.work,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            tree = OwnedProcessTree(child.pid)
            try:
                status = child.wait(timeout=self.timeout)
            except BaseException:
                tree.cleanup()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                write_json(stage / "interrupted.json", {"elapsed_seconds": time.monotonic() - started})
                raise
            finally:
                tree.cleanup()
        result = {"name": name, "exit_code": status, "elapsed_seconds": time.monotonic() - started}
        write_json(stage / "outcome.json", result)
        self.outcomes.append(result)
        print(json.dumps(result), flush=True)
        if status:
            raise RuntimeError(f"stage {name} exited {status}; failed evidence retained")


def final_checkpoint(output: Path, experiment: Experiment, receipt: dict, *, step: int | None = None) -> Path:
    from cosmos_framework.posttrain.checkpoint.adapter import read_adapter_metadata
    from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager

    manager = CheckpointManager(checkpoint_format="dcp")
    expected_step = experiment.steps if step is None else step
    if type(expected_step) is not int or not 0 < expected_step <= experiment.steps:
        raise ValueError("checkpoint step must be within the unchanged training horizon")
    candidates = list((output / "native").rglob(f"checkpoints/iter_{expected_step:06d}"))
    if len(candidates) != 1:
        raise ValueError("expected exactly one own-run final checkpoint")
    checkpoint = candidates[0]
    # read_adapter_metadata verifies the atomic publication/manifest contract.
    metadata = read_adapter_metadata(str(checkpoint), require_resume=True, manager=manager)
    controls = manager.load_component(str(checkpoint), "trainer")
    if any(controls[key] != expected_step for key in ("optimizer_step", "iter", "attempt_id")):
        raise ValueError("training did not complete all scheduled updates")
    if controls["accum"] or controls["nonfinite_skips_total"] or controls.get("zero_grad_skips_total", 0):
        raise ValueError("final checkpoint has partial accumulation or skipped updates")
    validation = controls["validation"]
    if validation["progress"]["pending_step"] is not None or validation["publication"] is not None:
        raise ValueError("final validation/publication is unfinished")
    identity = metadata["identity"]
    for key in ("code_revision", "wheel_sha256"):
        if identity["training_contract"][key] != receipt[key]:
            raise ValueError("checkpoint build identity differs")
    if (identity["quantization"] is not None) != (experiment.arm == "nf4"):
        raise ValueError("checkpoint base format differs from experiment")
    write_json(
        output / ("final-checkpoint.json" if step is None else f"checkpoint-{step:06d}.json"),
        {
            "checkpoint": str(checkpoint),
            "optimizer_step": expected_step,
            "metadata": metadata,
            "provenance": manager.checkpoint_provenance(str(checkpoint)),
        },
    )
    return checkpoint


def read_validation_target(path: str | None, args: argparse.Namespace, experiment: Experiment) -> dict | None:
    """Freeze a development-only point target before any model or process starts.

    Crossing this target is not a repeated noninferiority test or a release gate.
    The external paired/grouped analysis and independent-data requirement remain.
    """
    if path is None:
        return None
    if args.action != "train-eval" or args.split != "development":
        raise ValueError("validation-target cadence requires fresh development training")
    value = json.loads(Path(path).read_text())
    keys = {
        "protocol",
        "evaluation_steps",
        "expected_rows",
        "minimum_correct",
        "maximum_token_ce",
        "score_data_sha256",
        "split_manifest_sha256",
    }
    if not isinstance(value, dict) or set(value) != keys or value["protocol"] != "cosmos.qlora-validation-target.v1":
        raise ValueError("invalid validation-target contract")
    steps = value["evaluation_steps"]
    if (
        not isinstance(steps, list)
        or not steps
        or any(type(step) is not int or not 0 < step <= experiment.steps for step in steps)
        or steps != sorted(set(steps))
        or steps[-1] != experiment.steps
    ):
        raise ValueError("evaluation_steps must increase uniquely through the unchanged training horizon")
    if (
        type(value["expected_rows"]) is not int
        or value["expected_rows"] <= 0
        or type(value["minimum_correct"]) is not int
        or not 0 <= value["minimum_correct"] <= value["expected_rows"]
        or type(value["maximum_token_ce"]) not in (int, float)
        or not math.isfinite(value["maximum_token_ce"])
        or value["maximum_token_ce"] < 0
    ):
        raise ValueError("invalid fixed task-accuracy or token-CE target")
    for key, source in (("score_data_sha256", args.score_data), ("split_manifest_sha256", args.split_manifest)):
        if value[key] != file_sha256(source):
            raise ValueError(f"validation target differs from sealed {key}")
    from tools.native_qlora_qualification.split import select_sealed_rows

    rows, selection = select_sealed_rows(
        Path(args.score_data), Path(args.split_manifest), value["split_manifest_sha256"]
    )
    if selection["split_role"] != "development" or len(rows) != value["expected_rows"]:
        raise ValueError("validation target requires the complete sealed development partition")
    return value


def score_command(args: argparse.Namespace, experiment: Experiment, output: Path, checkpoint: Path | None) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=1",
        str(Path(__file__).resolve().with_name("score.py")),
        "--config",
        experiment.recipe,
        "--data",
        args.score_data,
        "--output",
        str(output),
        "--split-manifest",
        args.split_manifest,
        "--split-manifest-sha256",
        file_sha256(args.split_manifest),
        "--split",
        args.split,
        "--all-rows",
        *[part for item in experiment.overrides() for part in ("--override", item)],
    ]
    if args.selection_receipt:
        command += [
            "--selection-receipt",
            args.selection_receipt,
            "--selection-receipt-sha256",
            file_sha256(args.selection_receipt),
        ]
    if checkpoint is not None:
        command += ["--checkpoint", str(checkpoint)]
    if getattr(args, "validation_target", None) is not None:
        command += ["--preprocessing-cache-bytes", str(experiment.cache_bytes)]
    return command


def target_observation(
    directory: Path, checkpoint: Path, target: dict, expected_inputs: str | None, *, manifest_path: Path
) -> dict:
    """Check complete owned scoring rows, not a rounded accuracy summary alone."""
    from tools.native_qlora_qualification.common import digest_json
    from tools.native_qlora_qualification.quality import INPUT_KEYS, validate_scores

    summary = json.loads((directory / "summary.json").read_text())
    rows_file = directory / "scores.jsonl"
    rows = [json.loads(line) for line in rows_file.read_text().splitlines() if line.strip()]
    validate_scores(rows)
    selection = json.loads((directory / "selection.json").read_text())
    selected_file = directory / "selected_rows.jsonl"
    selected = [json.loads(line) for line in selected_file.read_text().splitlines() if line.strip()]
    if file_sha256(manifest_path) != target["split_manifest_sha256"]:
        raise ValueError("validation target manifest changed before score acceptance")
    sealed_rows = json.loads(manifest_path.read_text())["rows"]
    if (
        selection.get("source_sha256") != target["score_data_sha256"]
        or selection.get("split_manifest_sha256") != target["split_manifest_sha256"]
        or selection.get("split_role") != "development"
        or selection.get("selected_rows_file_sha256") != file_sha256(selected_file)
        or summary.get("selection_sha256") != digest_json(selection)
        or not len(rows) == len(selected) == len(sealed_rows) == target["expected_rows"]
    ):
        raise ValueError("actual scoring selection differs from the frozen target")
    for index, (row, frozen, sealed) in enumerate(zip(rows, selected, sealed_rows, strict=True), 1):
        if (
            row["source_line"] != index
            or frozen["source_line"] != index
            or any(
                row[key] != frozen[key]
                for key in ("id", "row_identity", "raw_row_sha256", "media_sha256", "gold", "choices")
            )
            or any(row[key] != sealed[key] for key in ("id", "raw_row_sha256", "media_sha256"))
        ):
            raise ValueError("scored ordered rows differ from the sealed target partition")
    if (
        summary.get("complete") is not True
        or summary.get("checkpoint") != str(checkpoint.resolve())
        or summary.get("scores_sha256") != file_sha256(rows_file)
        or len(rows) != target["expected_rows"]
        or any(type(row["restricted_choice_correct"]) is not bool for row in rows)
    ):
        raise ValueError("incomplete, changed or foreign checkpoint scores cannot meet a target")
    inputs = hashlib.sha256(
        json.dumps(
            [{key: row[key] for key in INPUT_KEYS} for row in rows],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    if expected_inputs is not None and inputs != expected_inputs:
        raise ValueError("processed scoring inputs changed across validation checkpoints")
    correct = sum(row["restricted_choice_correct"] for row in rows)
    tokens = sum(row["native_supervised_tokens"] for row in rows)
    ce = math.fsum(row["native_sft_ce"] * row["native_supervised_tokens"] for row in rows) / tokens
    metrics = summary["metrics"]
    if (
        metrics["rows"] != len(rows)
        or metrics["restricted_choice_accuracy"] != correct / len(rows)
        or metrics["token_weighted_ce"] != ce
    ):
        raise ValueError("scoring summary disagrees with complete rows")
    return {
        "correct": correct,
        "rows": len(rows),
        "native_supervised_tokens": tokens,
        "token_weighted_ce": ce,
        "processed_inputs_sha256": inputs,
        "score_summary_sha256": file_sha256(directory / "summary.json"),
        "target_met": correct >= target["minimum_correct"] and ce <= target["maximum_token_ce"],
    }


def validate_action_checkpoint(action: str, checkpoint: str | None) -> None:
    """Keep frozen, fresh training and trained evaluation unambiguous before work."""
    if action not in {"train-eval", "evaluate", "frozen"}:
        raise ValueError("unsupported native qualification action")
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint.strip()):
        raise ValueError("checkpoint must be an explicit nonempty source path")
    if action == "evaluate" and checkpoint is None:
        raise ValueError("trained evaluation requires an explicit checkpoint")
    if action != "evaluate" and checkpoint is not None:
        raise ValueError(f"{action} requires a fresh base and rejects an inherited checkpoint")


def evaluation_mode(args: argparse.Namespace) -> str:
    """Preserve historical restart mode; native mode needs an explicit fixed target."""
    mode = getattr(args, "evaluation_mode", "restart")
    if mode not in {"restart", "native"}:
        raise ValueError("evaluation mode must be restart or native")
    if mode == "native" and (
        not getattr(args, "validation_target", None) or args.action != "train-eval" or args.split != "development"
    ):
        raise ValueError("native evaluation requires fresh development training and a validation target")
    return mode


def native_evaluation_overrides(args: argparse.Namespace, experiment: Experiment, target: dict) -> list[str]:
    """Bind the resident native evaluator to the same full partition and point target."""
    settings = {
        "enabled": True,
        "stop_on_target": True,
        "data.path": str(Path(args.score_data).resolve()),
        "source_sha256": target["score_data_sha256"],
        "manifest_path": str(Path(args.split_manifest).resolve()),
        "manifest_sha256": target["split_manifest_sha256"],
        "expected_rows": target["expected_rows"],
        "steps": target["evaluation_steps"],
        "cache_bytes": experiment.cache_bytes,
        "min_correct": target["minimum_correct"],
        "max_token_ce": target["maximum_token_ce"],
    }
    return [f"rl.validation.wts.{key}={json.dumps(value)}" for key, value in settings.items()]


def verify_segment_boundary(output: Path, checkpoint: Path, *, start: int, stop: int, horizon: int) -> dict:
    """A saved earlier checkpoint cannot conceal an ignored pause or extra work."""
    metrics = checkpoint.parent.parent / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    updates = [row for row in rows if "train/loss_avg" in row]
    steps = [row.get("step") for row in updates]
    if any(type(step) is not int for step in steps) or steps != list(range(1, stop + 1)):
        raise ValueError("native metrics must stop exactly at the requested boundary without duplicate/missing updates")
    observed = sorted(
        int(path.name.removeprefix("iter_"))
        for path in checkpoint.parent.iterdir()
        if path.is_dir() and re.fullmatch(r"iter_[0-9]+", path.name)
    )
    if not observed or observed[-1] != stop:
        raise ValueError("native checkpoint inventory extends beyond or lacks the requested boundary")
    log = output / "stages" / f"training_{stop:06d}" / "output.log"
    summaries = []
    for line in log.read_text().splitlines():
        if "SFT DONE " in line:
            summaries.append(ast.literal_eval(re.sub(r"\x1b\[[0-9;]*m", "", line.split("SFT DONE ", 1)[1]).strip()))
    if len(summaries) != 1 or not isinstance(summaries[0], dict):
        raise ValueError("one native completion summary is required for each segment")
    summary = summaries[0]
    if (
        type(summary.get("start")) is not int
        or type(summary.get("final_step")) is not int
        or summary.get("start") != start
        or summary.get("final_step") != stop
        or "preempted" in summary
        or summary.get("final_loss") != updates[-1]["train/loss_avg"]
        or (stop < horizon and (summary.get("paused") is not True or summary.get("checkpoint") != str(checkpoint)))
        or (stop == horizon and summary.get("paused", False) is not False)
    ):
        raise ValueError("native completion does not establish the requested committed pause/completion")
    result = {
        "start": start,
        "stop": stop,
        "new_updates": stop - start,
        "native_summary": summary,
        "checkpoint_steps": observed,
        "metrics_sha256": file_sha256(metrics),
        "log_sha256": file_sha256(log),
    }
    write_json(output / f"boundary-{stop:06d}.json", result)
    return result


def train_to_target(args, experiment, output, receipt, stages, target, started):
    """Native committed-boundary segments; every restart and scoring cost is timed.

    This deliberately measures a segmented workflow, whose CPU caches restart.
    It is not interchangeable with uninterrupted training or full cold preparation.
    """
    checkpoint, observations, inputs = None, [], None
    script = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=1",
        "-m",
        "cosmos_framework.posttrain.entrypoints.sft",
        "--config",
        experiment.recipe,
        "--",
    ]
    for step in target["evaluation_steps"]:
        # Refuse mutated data/targets before a new segment; identity guards in native
        # resume additionally verify the original schedule, optimizer and cursor.
        if read_validation_target(args.validation_target, args, experiment) != target:
            raise ValueError("validation target changed after experiment start")
        overrides = [*experiment.overrides(), "rl.resume=" + json.dumps(str(checkpoint) if checkpoint else "off")]
        if step < experiment.steps:
            overrides.append(f"rl.pause_after_step={step}")
        stages.run(f"training_{step:06d}", [*script, *overrides])
        checkpoint = final_checkpoint(output, experiment, receipt, step=step)
        boundary = verify_segment_boundary(
            output,
            checkpoint,
            start=observations[-1]["optimizer_step"] if observations else 0,
            stop=step,
            horizon=experiment.steps,
        )
        directory = output / f"scores_{step:06d}"
        if read_validation_target(args.validation_target, args, experiment) != target:
            raise ValueError("validation target changed during native training")
        stages.run(f"scoring_{step:06d}", score_command(args, experiment, directory, checkpoint))
        if read_validation_target(args.validation_target, args, experiment) != target:
            raise ValueError("validation target changed during scoring")
        observation = target_observation(directory, checkpoint, target, inputs, manifest_path=Path(args.split_manifest))
        inputs = observation["processed_inputs_sha256"]
        observation.update(optimizer_step=step, elapsed_from_runner_start_seconds=time.monotonic() - started)
        observation["native_boundary"] = boundary
        write_json(output / f"target-{step:06d}.json", observation)
        observations.append(observation)
        if observation["target_met"]:
            break
    result = {
        "protocol": "cosmos.qlora-validation-target-result.v1",
        "observations": observations,
        "target_met": observations[-1]["target_met"],
        "first_validated_checkpoint_seconds": observations[-1]["elapsed_from_runner_start_seconds"]
        if observations[-1]["target_met"]
        else None,
        "last_validated_step": observations[-1]["optimizer_step"],
        "censored_at_training_horizon": not observations[-1]["target_met"],
        "timing_scope": "Runner start through first passing scheduled full-data score: includes verification, preflight, native model/processor loading, preprocessing, every resumed segment, validation and checkpointing. Pre-existing base/data acquisition, conversion, environment/build preparation and scheduler wait are excluded and must be reported separately. Final portable export/reload follows separately.",
        "execution_scope": "Each validation checkpoint starts a fresh native process and preprocessing cache. Apply the same cadence to both arms; do not substitute uninterrupted-run timings.",
        "quality_qualified": False,
    }
    write_json(output / "target-result.json", result)
    return checkpoint, directory, result


def train_native_to_target(args, experiment, output, receipt, stages, target, started):
    """One native process owns training and scheduled full-data evaluation."""
    if read_validation_target(args.validation_target, args, experiment) != target:
        raise ValueError("validation target changed after experiment start")
    child_started = time.monotonic()
    stages.run(
        "training_wts",
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=1",
            "-m",
            "cosmos_framework.posttrain.entrypoints.sft",
            "--config",
            experiment.recipe,
            "--",
            *experiment.overrides(),
            "rl.resume=off",
            *native_evaluation_overrides(args, experiment, target),
        ],
    )
    child_finished = time.monotonic()
    if read_validation_target(args.validation_target, args, experiment) != target:
        raise ValueError("validation target changed during native training")
    candidates = list((output / "native").rglob("wts/target-result.json"))
    if len(candidates) != 1:
        raise ValueError("exactly one own native evaluation result is required")
    path = candidates[0]
    native_root = path.parent.parent.resolve()
    if not native_root.is_relative_to((output / "native").resolve()):
        raise ValueError("native evaluation result is outside the own run")
    source_sha = file_sha256(path)
    native = json.loads(path.read_text())
    records = native.get("observations")
    if (
        native.get("protocol") != "cosmos.native-sft-wts-target.v1"
        or native.get("complete") is not True
        or native.get("stop_on_target") is not True
        or native.get("resumed_from_checkpoint") is not False
        or type(native.get("horizon")) is not int
        or native["horizon"] != experiment.steps
        or not isinstance(records, list)
        or not records
        or [record.get("step") for record in records] != target["evaluation_steps"][: len(records)]
        or any(type(record.get("step")) is not int for record in records)
        or type(native.get("final_step")) is not int
        or native["final_step"] != records[-1]["step"]
    ):
        raise ValueError("native evaluation result lacks the complete scheduled prefix")
    final_step = native["final_step"]
    metrics = native_root / "metrics.jsonl"
    updates = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    updates = [row for row in updates if "train/loss_avg" in row]
    if any(type(row.get("step")) is not int for row in updates) or [row["step"] for row in updates] != list(
        range(1, final_step + 1)
    ):
        raise ValueError("native updates must end exactly at the final evaluated checkpoint")
    checkpoint_steps = sorted(
        int(p.name.removeprefix("iter_"))
        for p in (native_root / "checkpoints").iterdir()
        if p.is_dir() and re.fullmatch(r"iter_[0-9]+", p.name)
    )
    if not checkpoint_steps or checkpoint_steps[-1] != final_step:
        raise ValueError("native checkpoint inventory exceeds or lacks the selected endpoint")
    log = output / "stages/training_wts/output.log"
    summaries = [
        ast.literal_eval(re.sub(r"\x1b\[[0-9;]*m", "", line.split("SFT DONE ", 1)[1]).strip())
        for line in log.read_text().splitlines()
        if "SFT DONE " in line
    ]
    if (
        len(summaries) != 1
        or not isinstance(summaries[0], dict)
        or summaries[0].get("start") != 0
        or summaries[0].get("final_step") != final_step
        or "preempted" in summaries[0]
        or summaries[0].get("final_loss") != updates[-1]["train/loss_avg"]
    ):
        raise ValueError("native completion does not establish the exact fresh training boundary")
    observations, inputs, seen_directories = [], None, set()
    previous_available = child_started
    for index, record in enumerate(records):
        step = record["step"]
        relative = record.get("directory")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError("native evaluation directory must be an own relative path")
        directory = (native_root / relative).resolve()
        if not directory.is_relative_to(path.parent.resolve()) or directory in seen_directories:
            raise ValueError("native evaluation directories must be unique and owned")
        seen_directories.add(directory)
        checkpoint = final_checkpoint(output, experiment, receipt, step=step)
        if checkpoint.resolve() != native_root / "checkpoints" / f"iter_{step:06d}" or record.get("checkpoint") != str(
            checkpoint.resolve()
        ):
            raise ValueError("native evaluation names a foreign checkpoint")
        observation = target_observation(directory, checkpoint, target, inputs, manifest_path=Path(args.split_manifest))
        if record.get("target_met") is not observation["target_met"] or record.get("scores_sha256") != file_sha256(
            directory / "scores.jsonl"
        ):
            raise ValueError("native target publication disagrees with complete scoring artifacts")
        if index < len(records) - 1 and observation["target_met"]:
            raise ValueError("native training continued after its first passing evaluation")
        available = record.get("available_monotonic_seconds")
        if (
            type(available) not in (int, float)
            or not math.isfinite(available)
            or not previous_available < available <= child_finished
        ):
            raise ValueError("native checkpoint/score availability must belong to this training attempt")
        previous_available = available
        inputs = observation["processed_inputs_sha256"]
        observation.update(
            optimizer_step=step,
            elapsed_from_runner_start_seconds=available - started,
            native_publication={
                "checkpoint": str(checkpoint),
                "directory": str(directory),
                "available_monotonic_seconds": available,
            },
        )
        observations.append(observation)
    passed = observations[-1]["target_met"]
    if (
        native.get("first_passing_step") != (final_step if passed else None)
        or native.get("horizon_censored") is not (not passed)
        or native.get("checkpoint") != str(checkpoint.resolve())
        or summaries[0].get("target_reached") is not passed
        or not passed
        and final_step != experiment.steps
    ):
        raise ValueError("native target termination or horizon censoring is inconsistent")
    if file_sha256(path) != source_sha or read_validation_target(args.validation_target, args, experiment) != target:
        raise ValueError("native result or validation target changed during acceptance")
    result = {
        "protocol": "cosmos.qlora-validation-target-result.v1",
        "evaluation_mode": "native",
        "observations": observations,
        "target_met": passed,
        "first_validated_checkpoint_seconds": observations[-1]["elapsed_from_runner_start_seconds"] if passed else None,
        "last_validated_step": final_step,
        "censored_at_training_horizon": not passed,
        "native_result_sha256": source_sha,
        "native_metrics_sha256": file_sha256(metrics),
        "native_completion_log_sha256": file_sha256(log),
        "native_process_complete_from_runner_start_seconds": child_finished - started,
        "native_process_elapsed_seconds": child_finished - child_started,
        "publication_to_child_return_seconds": child_finished - previous_available,
        "timing_scope": "Runner start through native durable checkpoint and full-data score publication/readback on the same host monotonic clock. Includes initial runner verification/preflight, model loading, continuous training, evaluation and checkpointing. Final native target-index serialization and process cleanup are excluded from publication time but included in native process completion. Subsequent reader verification and final portable export/reload are separately included in complete runner time. Pre-existing acquisition/build preparation and scheduler waiting are excluded.",
        "execution_scope": "One native process retains the training model and preprocessing cache; scheduled full-data evaluation uses its separate persistent cache. No per-boundary model reload or training restart.",
        "quality_qualified": False,
    }
    for observation in observations:
        write_json(output / f"target-{observation['optimizer_step']:06d}.json", observation)
    write_json(output / "target-result.json", result)
    return checkpoint, directory, result


def execute(args: argparse.Namespace) -> None:
    # Cadence measurements include their initial sealed-data/target validation.
    # Keep the established default experiment's clock origin unchanged.
    cadence_started = time.monotonic() if getattr(args, "validation_target", None) is not None else None
    validate_action_checkpoint(args.action, args.checkpoint)
    mode = evaluation_mode(args)
    experiment = Experiment.read(args.spec)
    target = read_validation_target(getattr(args, "validation_target", None), args, experiment)
    output = Path(args.output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic() if cadence_started is None else cadence_started
    try:
        storage_probe = require_storage([output], context="compute" if os.environ.get("SLURM_JOB_ID") else "process")
    except StorageProbeError as error:
        # Storage failure can also prevent writing a receipt; preserve it in the
        # existing parent stage log before any build/model-loading work.
        print(json.dumps({"storage_probe": error.report}), file=sys.stderr, flush=True)
        raise
    write_json(output / "storage-probe.json", storage_probe)
    receipt = verify_build(Path(args.build))
    verify_kernel_gate(Path(args.kernel_gate), receipt)
    if target is not None:
        from cosmos_framework.posttrain.config.sft import SFTConfig

        if mode == "restart" and not hasattr(SFTConfig, "pause_after_step"):
            raise ValueError("validation-target cadence requires native committed-boundary pause support")
        if mode == "native":
            from cosmos_framework.posttrain.config.sft import SFTValidationConfig

            if "wts" not in SFTValidationConfig.__dataclass_fields__:
                raise ValueError("native evaluation requires the full-data WTS validation implementation")
    if os.environ.get("PYTHONHASHSEED") != str(experiment.seed):
        raise ValueError("PYTHONHASHSEED must match the experiment before process start")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("deterministic cuBLAS environment must precede process start")
    os.environ.update(
        COSMOS_TRAINING_CODE_REVISION=receipt["code_revision"],
        COSMOS_TRAINING_WHEEL_SHA256=receipt["wheel_sha256"],
        IMAGINAIRE_OUTPUT_ROOT=str(output / "native"),
        SFT_MODEL_PATH=args.model,
        SFT_DATA_ROOT=args.data_root,
    )
    os.environ.pop("COSMOS_SFT_PROFILE_DIR", None)
    common = experiment.overrides()
    script = Path(__file__).resolve()
    torchrun = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1"]
    provenance_files = [
        Path(args.spec),
        Path(args.build) / "receipt.json",
        Path(args.split_manifest),
        Path(args.score_data),
        Path(args.data_root) / "train.jsonl",
        Path(args.data_root) / "validation.jsonl",
        Path(args.kernel_gate),
    ]
    if target is not None:
        provenance_files.append(Path(args.validation_target))
    write_json(
        output / "contract.json",
        {
            "protocol": "cosmos.native-qlora-experiment.v1",
            "experiment": experiment.as_dict(),
            "action": args.action,
            "split": args.split,
            "build": receipt,
            "overrides": common,
            "files": [{"path": str(p), "sha256": file_sha256(p)} for p in provenance_files],
            "tools": {p.name: file_sha256(p) for p in script.parent.glob("*.py")},
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "checkpoint": args.checkpoint,
            "preflight_measurement": "separately timed before external sampler",
            **({"validation_target": target} if target is not None else {}),
            **({"evaluation_mode": mode} if mode == "native" else {}),
        },
    )
    stages = Stages(output, Path(args.time_binary))
    sampler = None
    passed = False
    failure = None
    target_result = None
    scores_directory = output / "scores"
    try:
        stages.run("preflight", [sys.executable, str(script), "preflight", "--output", str(output / "runtime.json")])
        runtime = json.loads((output / "runtime.json").read_text())
        os.environ["COSMOS_QLORA_GPU_UUID"] = runtime["uuid"]
        with (output / "sampler.log").open("x") as log:
            sampler = subprocess.Popen(
                [
                    sys.executable,
                    str(script.parent / "monitor.py"),
                    str(os.getpid()),
                    str(output / "nvml-host.jsonl"),
                    str(stages.marker),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        checkpoint = Path(args.checkpoint) if args.checkpoint else None
        if target is not None:
            train = train_native_to_target if mode == "native" else train_to_target
            checkpoint, scores_directory, target_result = train(
                args,
                experiment,
                output,
                receipt,
                stages,
                target,
                started,
            )
        elif args.action == "train-eval":
            stages.run(
                "training",
                [
                    *torchrun,
                    "-m",
                    "cosmos_framework.posttrain.entrypoints.sft",
                    "--config",
                    experiment.recipe,
                    "--",
                    *common,
                    "rl.resume=off",
                ],
            )
            checkpoint = final_checkpoint(output, experiment, receipt)
        integration = "cosmos_framework.posttrain.tests.qlora_integration"
        probe_overrides = [part for item in common for part in ("--override", item)]
        if checkpoint is not None:
            portable_iter = (
                output
                / "portable"
                / "checkpoints"
                / (checkpoint.name if target is not None else f"iter_{experiment.steps:06d}")
            )
            stages.run(
                "portable_publication",
                [
                    sys.executable,
                    "-m",
                    "cosmos_framework.posttrain.checkpoint.adapter",
                    "--source",
                    str(checkpoint),
                    "--output-iter",
                    str(portable_iter),
                ],
            )
            portable = output / "portable" / "checkpoint_artifacts" / portable_iter.name
            if not portable.is_dir():
                raise ValueError("portable adapter publication missing")
            for name, location in (("dcp", checkpoint), ("portable", portable)):
                command = [
                    *torchrun,
                    "-m",
                    integration,
                    "probe",
                    "--config",
                    experiment.recipe,
                    "--checkpoint",
                    str(location),
                    "--data",
                    str(Path(args.data_root) / "validation.jsonl"),
                    "--output",
                    str(output / "probes" / name),
                    "--require-updates",
                    *probe_overrides,
                ]
                if name == "portable":
                    command += ["--image", str(output / "probes/dcp/video_first_frame.png")]
                stages.run("probe_" + name, command)
            stages.run(
                "reload_exact",
                [
                    sys.executable,
                    "-m",
                    integration,
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
        if target is None:
            stages.run("scoring", score_command(args, experiment, scores_directory, checkpoint))
        if sampler.poll() is not None:
            raise RuntimeError("external sampler failed; complete resource evidence unavailable")
        passed = True
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        if sampler is not None:
            sampler.terminate()
            try:
                sampler.wait(timeout=15)
            except subprocess.TimeoutExpired:
                sampler.kill()
                sampler.wait()
        write_json(
            output / "completion.json",
            {
                "completed": passed,
                "failure": failure,
                "experiment": experiment.as_dict(),
                "outcomes": stages.outcomes,
                "complete_elapsed_seconds": time.monotonic() - started,
                "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "quality_qualified": False,
                "quality_gate_requires_external_paired_analysis": True,
                **(
                    {
                        "validation_target_result": target_result,
                        "scores_directory": str(scores_directory),
                        "post_target_export_reload_seconds": time.monotonic()
                        - started
                        - target_result["observations"][-1]["elapsed_from_runner_start_seconds"],
                    }
                    if target_result is not None
                    else {}
                ),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("preflight")
    command.add_argument("--output", required=True)
    command = commands.add_parser("kernel-gate")
    for name in ("output", "build", "time-binary"):
        command.add_argument("--" + name, required=True)
    command = commands.add_parser("experiment")
    for name in (
        "spec",
        "output",
        "build",
        "model",
        "data-root",
        "score-data",
        "split-manifest",
        "time-binary",
        "kernel-gate",
    ):
        command.add_argument("--" + name, required=True)
    command.add_argument("--action", choices=("train-eval", "evaluate", "frozen"), default="train-eval")
    command.add_argument(
        "--split", choices=("development", "final", "regression", "untouched_diagnostic"), default="development"
    )
    command.add_argument("--checkpoint")
    command.add_argument("--selection-receipt")
    command.add_argument(
        "--evaluation-mode",
        choices=("restart", "native"),
        default="restart",
        help="Fixed-target evaluation: historical pause/reload segments, or one resident native training process.",
    )
    command.add_argument(
        "--validation-target",
        help="Frozen development target/grid JSON. Restart mode pauses and reloads each segment; native mode evaluates in one resident process. Both preserve the declared training horizon and require a qualified build.",
    )
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(Path(args.output))
    elif args.command == "kernel-gate":
        kernel_gate(args)
    else:
        if (args.action == "evaluate") != bool(args.checkpoint):
            parser.error("evaluate requires a checkpoint; other actions forbid one")
        execute(args)


if __name__ == "__main__":
    main()
