# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Run one bounded native SFT sequence/media cell and audit its resource evidence.

Invoke inside the sealed single-device allocation. This orchestrator submits no
jobs and implements no training loop. Synthetic long targets are runtime stress
fixtures, never task-quality evidence. A completed over-budget cell is retained;
missing memory evidence never authorizes a larger microbatch.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from protocol import Experiment, file_sha256, write_json
from run import Stages, final_checkpoint, verify_build, verify_kernel_gate

from tools.native_qlora_qualification.performance import (
    artifact_size,
    distribution,
    memory_summary,
    metric_summary,
)

PROTOCOL = "cosmos.native-qlora-runtime-cell.v1"
PROFILES = {"short_video8", "long4k_video8", "short_video16_large", "long4k_video16_large"}
BUDGET = 24 * 1024**3
INTEGRATION = "cosmos_framework.posttrain.tests.qlora_integration"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_lines(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("required evidence JSONL is empty or contains blank records")
    rows = [json.loads(line) for line in lines]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("evidence JSONL records must be objects")
    return rows


def validate_experiment(experiment: Experiment) -> None:
    required = {
        "steps": 20,
        "warmup_steps": 2,
        "effective_batch": 8,
        "max_length": 4096,
        "save_every": 10,
        "validation_every": 10,
        "validation_samples": 8,
    }
    if any(getattr(experiment, key) != value for key, value in required.items()):
        raise ValueError(
            "runtime cells require 20 updates, warmup2, effective8, max4096, save/validation10 and validation8"
        )
    if experiment.microbatch not in (1, 2, 4):
        raise ValueError("runtime cells support microbatch1/2/4 only")


def verify_workloads(args: argparse.Namespace, experiment: Experiment) -> dict:
    """Bind recorded measurements to exact JSONL bytes before any model allocation."""
    validate_experiment(experiment)
    root = Path(args.workloads).absolute()
    if args.profile not in PROFILES:
        raise ValueError("unsupported measured workload profile")
    directory = root / args.profile
    if file_sha256(root / "manifest.json") != args.workload_manifest_sha256:
        raise ValueError("workload manifest digest differs")
    if file_sha256(directory / "manifest.json") != args.profile_manifest_sha256:
        raise ValueError("profile manifest digest differs")
    top = json.loads((root / "manifest.json").read_text())
    profile = json.loads((directory / "manifest.json").read_text())
    if (
        top.get("protocol") != "cosmos.native-qlora-runtime-workloads.v1"
        or top.get("runtime_stress_only") is not True
        or top.get("quality_evidence") is not False
        or top.get("seed") != experiment.seed
    ):
        raise ValueError("workload purpose, protocol or selection seed differs")
    policy = profile["policy"]
    variant = top["variants"][args.profile]
    if variant != {"manifest_sha256": args.profile_manifest_sha256, **policy}:
        raise ValueError("profile manifest and top-level policy differ")
    expected_processor = {
        "images_kwargs": {"size": {"shortest_edge": 65536, "longest_edge": 262144}},
        "videos_kwargs": {
            "num_frames": experiment.frames,
            "fps": None,
            "size": {"shortest_edge": experiment.video_min_pixels, "longest_edge": experiment.video_max_pixels},
        },
    }
    if (
        policy.get("max_length") != experiment.max_length
        or policy.get("processor_kwargs") != expected_processor
        or policy.get("device_process_budget_bytes") != BUDGET
        or policy.get("microbatches") != [{"microbatch": b, "grad_accum": 8 // b} for b in (1, 2, 4)]
    ):
        raise ValueError("experiment differs from measured processor, length, batching or memory policy")
    for split, count in (("train", 32), ("validation", 8)):
        path = directory / f"{split}.jsonl"
        if file_sha256(path) != profile["files"][split]:
            raise ValueError("prepared dataset digest differs")
        rows, measured = read_lines(path), profile["measurements"][split]["rows"]
        if len(rows) != count or len(measured) != count:
            raise ValueError("runtime fixtures require exactly32 training and8 validation rows")
        for row, record in zip(rows, measured, strict=True):
            if digest(row) != record.get("output_row_sha256"):
                raise ValueError("measured shape is not bound to its exact output row")
            length, tokens = record.get("sequence_length"), record.get("supervised_tokens")
            if type(length) is not int or not 2 <= length <= experiment.max_length:
                raise ValueError("invalid measured sequence length")
            if type(tokens) is not int or not 1 <= tokens < length:
                raise ValueError("invalid measured supervised-token count")
            if args.profile.startswith("long4k") and (length < 3840 or tokens < 1024):
                raise ValueError("long runtime fixture lacks meaningful actual sequence/target length")
            grid = record.get("video_grid_thw")
            if (
                not isinstance(grid, list)
                or len(grid) != 3
                or any(type(v) is not int or v <= 0 for v in grid)
                or grid[0] * 2 != experiment.frames
            ):
                raise ValueError("invalid measured video grid or temporal slots")
    return {"top": top, "profile": profile, "directory": str(directory)}


def expected_updates(data: Any, measurements: list[dict], experiment: Experiment) -> list[dict]:
    """Use the native cursor, including its epoch reshuffles; never invent a row order."""
    updates = []
    prefix = data.source_fingerprint + ":"
    for step in range(1, experiment.steps + 1):
        indices = []
        for _ in range(experiment.effective_batch // experiment.microbatch):
            for trajectory in data.get_samples(experiment.microbatch):
                if not trajectory.id.startswith(prefix):
                    raise ValueError("native trajectory source identity differs")
                index = int(trajectory.id[len(prefix) :])
                if not 0 <= index < len(measurements):
                    raise ValueError("native cursor selected an unavailable measured row")
                indices.append(index)
        if len(indices) != experiment.effective_batch:
            raise ValueError("native source returned an incomplete update")
        lengths = [measurements[index]["sequence_length"] for index in indices]
        padded = sum(
            max(lengths[start : start + experiment.microbatch]) * experiment.microbatch
            for start in range(0, len(lengths), experiment.microbatch)
        )
        updates.append(
            {
                "step": step,
                "indices": indices,
                "examples": len(indices),
                "supervised_tokens": sum(measurements[index]["supervised_tokens"] for index in indices),
                "unpadded_tokens": sum(lengths),
                "padded_tokens": padded,
            }
        )
    return updates


def native_workload_check(args: argparse.Namespace) -> None:
    """CPU-only native identities/cursors. Previously measured media are not decoded again."""
    from cosmos_framework.posttrain.data.packer.sft import build_sft_packer
    from cosmos_framework.posttrain.roles.env.sft import build_sft_env
    from cosmos_framework.utils.config import load_config
    from cosmos_framework.utils.lazy_config import instantiate

    experiment = Experiment.read(args.spec)
    verified = verify_workloads(args, experiment)
    profile, top = verified["profile"], verified["top"]
    os.environ.update(SFT_DATA_ROOT=verified["directory"], SFT_MODEL_PATH=args.model)
    config = load_config(experiment.recipe, ["--", *experiment.overrides()])
    # The training entrypoint performs full model-source verification. This CPU
    # check needs the exact typed data/processor policy, without hashing all model
    # shards again merely to recover the deterministic data traversal.
    rl = instantiate(config.rl)
    if rl.processor_kwargs != profile["policy"]["processor_kwargs"] or rl.max_length != profile["policy"]["max_length"]:
        raise ValueError("resolved native recipe differs from measured processor policy")
    train = build_sft_env(rl.data, rl.seed)
    validation = build_sft_env(rl.validation.data, rl.validation.seed, training=False)
    media = {**train.dataset.media_hashes, **validation.dataset.media_hashes}
    if media != top["media_sha256"]:
        raise ValueError("native dataset's freshly hashed media differ from the workload manifest")
    for split, env in (("train", train), ("validation", validation)):
        actual = [digest(row) for row in env.dataset.rows]
        recorded = [row["output_row_sha256"] for row in profile["measurements"][split]["rows"]]
        if actual != recorded:
            raise ValueError("native normalized rows differ from measured rows")
    packer = build_sft_packer(
        args.model,
        max_length=rl.max_length,
        processor_kwargs=rl.processor_kwargs,
        mask_think_opener=rl.mask_think_opener,
        trust_remote_code=False,
    )
    identity = digest(packer.validation_semantics_identity())
    if identity != profile["processor_identity_sha256"]:
        raise ValueError("native processor/tokenizer/template identity differs from measured fixtures")
    updates = expected_updates(train, profile["measurements"]["train"]["rows"], experiment)
    write_json(
        Path(args.output),
        {
            "protocol": PROTOCOL,
            "completed": True,
            "processor_identity_sha256": identity,
            "dataset_fingerprints": {"train": train.source_fingerprint, "validation": validation.source_fingerprint},
            "media_sha256": media,
            "expected_updates": updates,
            "expected_final_data_state_sha256": digest(train.state_dict()),
            "expected_validation_tokens": sum(
                r["supervised_tokens"] for r in profile["measurements"]["validation"]["rows"]
            ),
            "manifest_measured_shapes": profile["measurements"],
            "shape_scope": "All32/8 rows measured during workload preparation; CPU preflight verifies native processor, exact rows/media and cursor without redecode. Fresh native reload probes are recorded separately.",
        },
    )


def validate_metrics(records: list[dict], expected: dict, experiment: Experiment) -> dict:
    result = metric_summary(records, steps=experiment.steps, effective_batch=experiment.effective_batch)
    if len(result["updates"]) != len(expected["expected_updates"]):
        raise ValueError("native update count differs from measured data traversal")
    for actual, desired in zip(result["updates"], expected["expected_updates"], strict=True):
        for key in ("step", "examples", "supervised_tokens"):
            if actual[key] != desired[key]:
                raise ValueError("native update tokens/examples differ from measured masks and exact cursor")
    if [row["step"] for row in result["validation"]] != [0, 10, 20]:
        raise ValueError("native validation schedule is incomplete or differs")
    if any(
        row["examples"] != experiment.validation_samples
        or row["supervised_tokens"] != expected["expected_validation_tokens"]
        for row in result["validation"]
    ):
        raise ValueError("native validation tokens/examples differ from measured masks")
    for row in records:
        if "train/loss_avg" in row and row.get("optim/finite_gradient_check_enabled") != 1:
            raise ValueError("native finite-gradient checks must remain enabled")
    windows = {}
    for label, rows in (("all20", result["updates"]), ("after_warmup", result["updates"][experiment.warmup_steps :])):
        elapsed = math.fsum(row["seconds"] for row in rows)
        windows[label] = {
            "updates": len(rows),
            "seconds": distribution([row["seconds"] for row in rows]),
            "examples_per_second": sum(row["examples"] for row in rows) / elapsed,
            "supervised_tokens_per_second": sum(row["supervised_tokens"] for row in rows) / elapsed,
        }
    result["windows"] = windows
    return result


def budget_eligibility(training: dict) -> dict:
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
        evidence = training["phases"].get(phase)
        if evidence is None or evidence["allocator_peak_allocated_bytes"] is None:
            reasons.append("missing_phase_" + phase)
    maximum_gap = sampled["sampling_gap_seconds"]["maximum"]
    if maximum_gap is None or maximum_gap > 5:
        reasons.append("insufficient_sampling_cadence")
    valid = not reasons
    # Allocator peaks are event-recorded lower bounds on total process residency;
    # they can expose an over-budget spike between external samples.
    observed_lower_bound = max(
        (
            value
            for value in (
                peak,
                training.get("allocator_peak_allocated_bytes"),
                training.get("allocator_peak_reserved_bytes"),
            )
            if type(value) is int
        ),
        default=None,
    )
    within = observed_lower_bound <= BUDGET if valid else None
    return {
        "budget_bytes": BUDGET,
        "scope": "Observed owned-device-process peak during native training, including load, validation and checkpoint writes; not physical device capacity.",
        "observed_peak_within_budget": within,
        "observed_process_peak_lower_bound_bytes": observed_lower_bound,
        "eligible_for_higher_microbatch": within is True,
        "evidence_complete": valid,
        "reasons": reasons + (["observed_training_process_peak_exceeds_budget"] if within is False else []),
        "sampled_peaks_are_lower_bounds": True,
        "unseen_short_peaks_remain_possible": True,
    }


def probe_commands(experiment: Experiment, output: Path, checkpoint: Path, data: Path) -> list[tuple[str, list[str]]]:
    torchrun = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1"]
    overrides = [part for item in experiment.overrides() for part in ("--override", item)]
    portable_iter = output / "portable/checkpoints" / f"iter_{experiment.steps:06d}"
    portable = output / "portable/checkpoint_artifacts" / portable_iter.name
    commands = [
        (
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
    ]
    for name, location in (("dcp", checkpoint), ("portable", portable)):
        command = [
            *torchrun,
            "-m",
            INTEGRATION,
            "probe",
            "--config",
            experiment.recipe,
            "--checkpoint",
            str(location),
            "--data",
            str(data),
            "--output",
            str(output / "probes" / name),
            "--require-updates",
            *overrides,
        ]
        if name == "portable":
            command += ["--image", str(output / "probes/dcp/video_first_frame.png")]
        commands.append(("probe_" + name, command))
    commands.append(
        (
            "reload_exact",
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
    )
    return commands


def trace_environment(window: str | None, output: Path) -> dict[str, str]:
    """Explicit bounded diagnostics; inherited profiler environment is never admitted."""
    if window is None:
        return {}
    if window not in {"early", "steady"}:
        raise ValueError("trace window must be early or steady")
    return {
        "COSMOS_SFT_PROFILE_DIR": str(output / "trace"),
        "COSMOS_SFT_PROFILE_WAIT": "0" if window == "early" else "5",
        "COSMOS_SFT_PROFILE_WARMUP": "1",
        "COSMOS_SFT_PROFILE_ACTIVE": "2",
    }


def execute(args: argparse.Namespace) -> None:
    experiment = Experiment.read(args.spec)
    validate_experiment(experiment)
    output = Path(args.output).absolute()
    trace_window = getattr(args, "trace_window", None)
    trace_env = trace_environment(trace_window, output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    stages = Stages(output, Path(args.time_binary), timeout_seconds=args.stage_timeout)
    sampler, passed, failure, report = None, False, None, {}
    try:
        receipt = verify_build(Path(args.build))
        verify_kernel_gate(Path(args.kernel_gate), receipt)
        verified = verify_workloads(args, experiment)
        if (
            os.environ.get("PYTHONHASHSEED") != str(experiment.seed)
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            raise ValueError("deterministic Python/cuBLAS environment must precede process start")
        os.environ.update(
            COSMOS_TRAINING_CODE_REVISION=receipt["code_revision"],
            COSMOS_TRAINING_WHEEL_SHA256=receipt["wheel_sha256"],
            IMAGINAIRE_OUTPUT_ROOT=str(output / "native"),
            SFT_MODEL_PATH=args.model,
            SFT_DATA_ROOT=verified["directory"],
        )
        os.environ.pop("COSMOS_SFT_PROFILE_DIR", None)
        script = Path(__file__).resolve()
        write_json(
            output / "contract.json",
            {
                "protocol": PROTOCOL,
                "experiment": experiment.as_dict(),
                "build": receipt,
                "profile": args.profile,
                "workload_manifest_sha256": args.workload_manifest_sha256,
                "profile_manifest_sha256": args.profile_manifest_sha256,
                "overrides": experiment.overrides(),
                "spec_sha256": file_sha256(args.spec),
                "kernel_gate_sha256": file_sha256(args.kernel_gate),
                "tools": {p.name: file_sha256(p) for p in script.parent.glob("*.py")},
                **({"diagnostic_profile": trace_window, "execution_selection_eligible": False} if trace_window else {}),
                "runtime_stress_only": True,
                "quality_evidence": False,
                "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )
        check = [sys.executable, str(script), "--action", "check-workload"]
        for key in ("spec", "workloads", "profile", "workload_manifest_sha256", "profile_manifest_sha256", "model"):
            check += ["--" + key.replace("_", "-"), str(getattr(args, key))]
        check += ["--output", str(output / "workload-verification.json")]
        stages.run("workload_verification", check, env_extra={"CUDA_VISIBLE_DEVICES": ""})
        expected = json.loads((output / "workload-verification.json").read_text())
        stages.run(
            "preflight",
            [sys.executable, str(script.with_name("run.py")), "preflight", "--output", str(output / "runtime.json")],
        )
        runtime = json.loads((output / "runtime.json").read_text())
        os.environ["COSMOS_QLORA_GPU_UUID"] = runtime["uuid"]
        with (output / "sampler.log").open("x") as stream:
            sampler = subprocess.Popen(
                [
                    sys.executable,
                    str(script.with_name("monitor.py")),
                    str(os.getpid()),
                    str(output / "nvml-host.jsonl"),
                    str(stages.marker),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        torchrun = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1"]
        stages.run(
            "training",
            [
                *torchrun,
                "-m",
                "cosmos_framework.posttrain.entrypoints.sft",
                "--config",
                experiment.recipe,
                "--",
                *experiment.overrides(),
                "rl.resume=off",
            ],
            env_extra=trace_env,
        )
        checkpoint = final_checkpoint(output, experiment, receipt)
        from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager

        state = CheckpointManager(checkpoint_format="dcp").load_component(str(checkpoint), "data")
        if digest(state) != expected["expected_final_data_state_sha256"]:
            raise ValueError("checkpoint data cursor differs from the exact20-update native traversal")
        metric_paths = list((output / "native").rglob("metrics.jsonl"))
        if len(metric_paths) != 1:
            raise ValueError("native metrics source is missing or ambiguous")
        report["native"] = validate_metrics(read_lines(metric_paths[0]), expected, experiment)
        write_json(output / "native-metrics-audit.json", report["native"])
        for name, command in probe_commands(
            experiment, output, checkpoint, Path(verified["directory"]) / "validation.jsonl"
        ):
            stages.run(name, command)
        probes = {
            name: json.loads((output / "probes" / name / "probe.json").read_text()) for name in ("dcp", "portable")
        }
        measured_video = verified["profile"]["measurements"]["validation"]["rows"][0]
        for probe in probes.values():
            if probe["optimizer_parameters"] != probe["trainable_tensors"] or not all(
                probe["adapter_updates"].values()
            ):
                raise ValueError("fresh native probe lacks exact optimizer membership or both adapter updates")
            actual = probe["probes"]["video"]
            if any(actual[key] != measured_video[key] for key in ("sequence_length", "supervised_tokens")):
                raise ValueError("fresh native video probe differs from manifest-measured token/mask counts")
        report["fresh_native_probe_shapes"] = {name: probe["probes"] for name, probe in probes.items()}
        report["artifacts"] = {
            "dcp": artifact_size(checkpoint),
            "portable": artifact_size(output / "portable/checkpoint_artifacts" / checkpoint.name),
        }
        if sampler.poll() is not None:
            raise RuntimeError("external memory sampler exited before lifecycle completion")
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
        try:
            if (output / "nvml-host.jsonl").exists() and (output / "runtime.json").exists():
                samples = read_lines(output / "nvml-host.jsonl")
                runtime = json.loads((output / "runtime.json").read_text())
                phases = (
                    read_lines(output / "stages/training/memory.jsonl")
                    if (output / "stages/training/memory.jsonl").exists()
                    else []
                )
                report["training_memory"] = memory_summary(
                    phases, [r for r in samples if r.get("stage") == "training"], runtime
                )
                all_phases = [
                    r for p in (output / "stages").glob("*/memory.jsonl") if p.stat().st_size for r in read_lines(p)
                ]
                report["complete_lifecycle_memory"] = memory_summary(all_phases, samples, runtime)
                report["training_budget"] = budget_eligibility(report["training_memory"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            report["memory_evidence_failure"] = f"{type(error).__name__}: {error}"
        eligible = (
            not trace_window
            and passed
            and report.get("training_budget", {}).get("eligible_for_higher_microbatch") is True
        )
        write_json(
            output / "completion.json",
            {
                "protocol": PROTOCOL,
                "completed": passed,
                "failure": failure,
                "experiment": experiment.as_dict(),
                "outcomes": stages.outcomes,
                **report,
                "eligible_for_higher_microbatch": eligible,
                **({"diagnostic_profile": trace_window, "execution_selection_eligible": False} if trace_window else {}),
                "complete_elapsed_seconds": time.monotonic() - started,
                "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "quality_qualified": False,
                "resume_qualified": False,
                "scope": "Native20-update execution, completed checkpoint and exact fresh portable reload only. Independent lifecycle.py interruption/resume and task-quality gates remain required.",
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("run", "check-workload"), default="run")
    for key in (
        "spec",
        "workloads",
        "profile",
        "workload-manifest-sha256",
        "profile-manifest-sha256",
        "model",
        "output",
    ):
        parser.add_argument("--" + key, required=True)
    for key in ("build", "kernel-gate", "time-binary"):
        parser.add_argument("--" + key)
    parser.add_argument("--stage-timeout", type=int, default=3600)
    parser.add_argument(
        "--trace-window",
        choices=("early", "steady"),
        help="Diagnostic20-update trace; ineligible for execution selection",
    )
    args = parser.parse_args()
    if args.stage_timeout <= 0:
        parser.error("stage timeout must be positive")
    if args.action == "check-workload":
        native_workload_check(args)
    else:
        if any(getattr(args, key) is None for key in ("build", "kernel_gate", "time_binary")):
            parser.error("run requires --build, --kernel-gate and --time-binary")
        execute(args)


if __name__ == "__main__":
    main()
