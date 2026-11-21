# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

DIRECTORY = Path(__file__).parents[1]
sys.path.insert(0, str(DIRECTORY))
sys.path.insert(0, str(DIRECTORY.parents[1]))
SPEC = importlib.util.spec_from_file_location("runtime_cells", DIRECTORY / "runtime.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def experiment(**kwargs):
    return runtime.Experiment(
        **{
            "name": "runtime_fixture",
            "arm": "nf4",
            "steps": 20,
            "warmup_steps": 2,
            "save_every": 10,
            "validation_every": 10,
            "validation_samples": 8,
            **kwargs,
        }
    )


def fixture(tmp_path):
    directory = tmp_path / "short_video8"
    directory.mkdir()
    media = {split: tmp_path / f"{split}.mp4" for split in ("train", "validation")}
    for split, path in media.items():
        path.write_bytes((split + " no decoder in CPU contract tests").encode())
    policy = {
        "max_length": 4096,
        "processor_kwargs": {
            "images_kwargs": {"size": {"shortest_edge": 65536, "longest_edge": 262144}},
            "videos_kwargs": {"num_frames": 8, "fps": None, "size": {"shortest_edge": 32768, "longest_edge": 524288}},
        },
        "device_process_budget_bytes": runtime.BUDGET,
        "microbatches": [{"microbatch": b, "grad_accum": 8 // b} for b in (1, 2, 4)],
    }
    profile = {
        "policy": policy,
        "files": {},
        "measurements": {},
        "processor_identity_sha256": runtime.digest({"fixture": True}),
    }
    for split, count in (("train", 32), ("validation", 8)):
        rows, measured = [], []
        for i in range(count):
            row = {
                "id": f"{split}-{i}",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": str(media[split])},
                            {"type": "text", "text": f"question{i}"},
                        ],
                    },
                    {"role": "assistant", "content": "B"},
                ],
            }
            rows.append(row)
            measured.append(
                {
                    "output_row_sha256": runtime.digest(row),
                    "sequence_length": 300 + i,
                    "supervised_tokens": i + 2,
                    "video_grid_thw": [4, 12, 20],
                }
            )
        path = directory / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        profile["files"][split] = runtime.file_sha256(path)
        profile["measurements"][split] = {"rows": measured}
    (directory / "manifest.json").write_text(json.dumps(profile))
    top = {
        "protocol": "cosmos.native-qlora-runtime-workloads.v1",
        "runtime_stress_only": True,
        "quality_evidence": False,
        "seed": 42,
        "media_sha256": {str(path): runtime.file_sha256(path) for path in media.values()},
        "variants": {"short_video8": {"manifest_sha256": runtime.file_sha256(directory / "manifest.json"), **policy}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(top))
    return SimpleNamespace(
        workloads=str(tmp_path),
        profile="short_video8",
        workload_manifest_sha256=runtime.file_sha256(tmp_path / "manifest.json"),
        profile_manifest_sha256=runtime.file_sha256(directory / "manifest.json"),
    )


def test_manifest_requires_exact_data_and_resolved_policy(tmp_path):
    args = fixture(tmp_path)
    verified = runtime.verify_workloads(args, experiment())
    assert len(verified["profile"]["measurements"]["train"]["rows"]) == 32
    with pytest.raises(ValueError, match="processor"):
        runtime.verify_workloads(args, experiment(video_max_pixels=4194304))
    with pytest.raises(ValueError, match="manifest digest"):
        runtime.verify_workloads(SimpleNamespace(**{**vars(args), "profile_manifest_sha256": "0" * 64}), experiment())
    path = tmp_path / "short_video8/train.jsonl"
    path.write_text(path.read_text().replace("question0", "changed0"))
    with pytest.raises(ValueError, match="dataset digest"):
        runtime.verify_workloads(args, experiment())


@pytest.mark.parametrize(
    "change", [{"steps": 21}, {"warmup_steps": 1}, {"validation_samples": 4}, {"microbatch": 8}, {"save_every": 1}]
)
def test_bounded_experiment_contract(change):
    with pytest.raises(ValueError):
        runtime.validate_experiment(experiment(**change))


def test_cpu_native_preflight_uses_real_cursor_and_hashed_media(tmp_path, monkeypatch):
    # Only the processor is replaced; real config construction, datasets, media
    # hashing, deterministic epoch traversal and rank-state serialization run.
    from cosmos_framework.posttrain.data.packer import sft

    args = fixture(tmp_path)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(experiment().as_dict()))
    args.spec, args.model, args.output = str(spec), str(tmp_path / "model"), str(tmp_path / "check.json")
    monkeypatch.setenv("SFT_MODEL_PATH", args.model)
    monkeypatch.setenv("SFT_DATA_ROOT", str(tmp_path / "short_video8"))
    monkeypatch.setattr(
        sft,
        "build_sft_packer",
        lambda *a, **k: SimpleNamespace(validation_semantics_identity=lambda: {"fixture": True}),
    )
    runtime.native_workload_check(args)
    result = json.loads(Path(args.output).read_text())
    assert result["completed"] is True
    assert len(result["expected_updates"]) == 20
    assert sum(row["examples"] for row in result["expected_updates"]) == 160
    assert result["expected_validation_tokens"] == sum(range(2, 10))
    first_epoch = [i for row in result["expected_updates"][:4] for i in row["indices"]]
    second_epoch = [i for row in result["expected_updates"][4:8] for i in row["indices"]]
    assert sorted(first_epoch) == list(range(32)) == sorted(second_epoch)
    assert first_epoch != second_epoch
    assert len(result["expected_final_data_state_sha256"]) == 64
    (tmp_path / "train.mp4").write_bytes(b"changed actual source video")
    args.output = str(tmp_path / "rejected.json")
    with pytest.raises(ValueError, match="freshly hashed media"):
        runtime.native_workload_check(args)
    assert not Path(args.output).exists()


def metric_records():
    rows = []
    for step in range(21):
        if step:
            rows.append(
                {
                    "step": step,
                    "timer/e2e_step_s": 2.5,
                    "train/loss_avg": 0.3,
                    "train/grad_norm": 1.5,
                    "train/valid_sequence_count": 8,
                    "train/valid_token_count": step + 20,
                    "optim/update_outcome": "applied",
                    "optim/optimizer_step": step,
                    "optim/attempt_id": step,
                    "optim/step_skipped": 0,
                    "optim/nonfinite_skips_total": 0,
                    "optim/nonfinite_loss": 0,
                    "optim/nonfinite_gradient": 0,
                    "optim/finite_gradient_check_enabled": 1,
                }
            )
        if step in (0, 10, 20):
            rows.append(
                {
                    "step": step,
                    "timer/validation_s": 1.5,
                    "val/loss_avg": 0.4,
                    "val/valid_token_count": 44,
                    "val/valid_sequence_count": 8,
                }
            )
    expected = {
        "expected_updates": [{"step": step, "examples": 8, "supervised_tokens": step + 20} for step in range(1, 21)],
        "expected_validation_tokens": 44,
    }
    return rows, expected


def test_exact_masks_and_native_clocks_gate_metrics():
    rows, expected = metric_records()
    result = runtime.validate_metrics(rows, expected, experiment())
    assert result["windows"]["all20"]["examples_per_second"] == 3.2
    assert result["windows"]["after_warmup"]["updates"] == 18
    for key, value in (
        ("train/valid_token_count", 23),
        ("train/valid_sequence_count", 7),
        ("optim/update_outcome", "skipped"),
        ("optim/attempt_id", 3),
        ("train/grad_norm", float("nan")),
        ("optim/finite_gradient_check_enabled", 0),
    ):
        changed = copy.deepcopy(rows)
        changed[1][key] = value
        with pytest.raises(ValueError):
            runtime.validate_metrics(changed, expected, experiment())
    rows[-1]["val/valid_token_count"] = 43
    with pytest.raises(ValueError, match="validation tokens"):
        runtime.validate_metrics(rows, expected, experiment())


def memory_fixture():
    return {
        "allocator_evidence_complete": True,
        "allocator_peak_allocated_bytes": 8 * 1024**3,
        "allocator_peak_reserved_bytes": 9 * 1024**3,
        "sampled": {
            "gpu_memory_evidence_valid": True,
            "owned_device_process_peak_bytes": 10 * 1024**3,
            "host_pss_complete": True,
            "sampling_gap_seconds": {"maximum": 1.2},
        },
        "phases": {
            key: {"allocator_peak_allocated_bytes": 1}
            for key in ("loading", "first_update", "updates", "checkpoint_write")
        },
    }


def test_observed_budget_unknown_and_over_cap_are_not_passes():
    evidence = memory_fixture()
    assert runtime.budget_eligibility(evidence)["eligible_for_higher_microbatch"] is True
    for key, value in (
        ("gpu_memory_evidence_valid", False),
        ("owned_device_process_peak_bytes", None),
        ("host_pss_complete", False),
    ):
        changed = copy.deepcopy(evidence)
        changed["sampled"][key] = value
        result = runtime.budget_eligibility(changed)
        assert result["observed_peak_within_budget"] is None
        assert result["eligible_for_higher_microbatch"] is False
    evidence["sampled"]["owned_device_process_peak_bytes"] = runtime.BUDGET + 1
    assert runtime.budget_eligibility(evidence)["observed_peak_within_budget"] is False
    evidence = memory_fixture()
    evidence["allocator_peak_reserved_bytes"] = runtime.BUDGET + 1
    assert runtime.budget_eligibility(evidence)["observed_peak_within_budget"] is False
    evidence = memory_fixture()
    del evidence["phases"]["loading"]
    assert runtime.budget_eligibility(evidence)["eligible_for_higher_microbatch"] is False


def test_real_analyzer_rejects_wrong_uuid_and_nvml_sentinels():
    record = {
        "unix_seconds": 100.0,
        "device_uuid": "GPU-other",
        "device_memory_total_bytes": 80 * 1024**3,
        "device_memory_used_bytes": 15 * 1024**3,
        "device_compute_processes": [{"pid": 8, "belongs_to_run": True, "used_gpu_memory_bytes": 2**64 - 1}],
        "selected_host_rss_sum_bytes": 1,
        "selected_host_pss_sum_bytes": 1,
        "host_pss_complete": True,
    }
    report = runtime.memory_summary([], [record], {"uuid": "GPU-owned", "device_bytes": 80 * 1024**3})
    assert report["sampled"]["gpu_memory_evidence_valid"] is False
    assert runtime.budget_eligibility(report)["eligible_for_higher_microbatch"] is False


def test_commands_only_native_publication_and_fresh_exact_probes(tmp_path):
    commands = runtime.probe_commands(
        experiment(microbatch=4), tmp_path, tmp_path / "native/checkpoints/iter_000020", tmp_path / "validation.jsonl"
    )
    assert [name for name, _ in commands] == ["portable_publication", "probe_dcp", "probe_portable", "reload_exact"]
    for name, command in commands:
        assert "score.py" not in " ".join(command)
        if name.startswith("probe_"):
            assert "torch.distributed.run" in command and "--nproc-per-node=1" in command
            assert "--require-updates" in command
            assert "rl.micro_batch_size=4" in command and "rl.grad_accum=2" in command
    assert commands[-1][1][-4:] == ["--atol", "0", "--rtol", "0"]
    assert "--image" in commands[2][1] and "--image" not in commands[1][1]


def test_execution_failure_retains_receipt_and_never_authorizes_more_memory(tmp_path, monkeypatch):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(dataclasses.asdict(experiment())))
    args = SimpleNamespace(
        spec=str(spec), output=str(tmp_path / "failed"), time_binary="/usr/bin/time", stage_timeout=1, build="missing"
    )
    monkeypatch.setattr(runtime, "verify_build", lambda path: (_ for _ in ()).throw(ValueError("wrong build")))
    with pytest.raises(ValueError, match="wrong build"):
        runtime.execute(args)
    completion = json.loads((Path(args.output) / "completion.json").read_text())
    assert completion["completed"] is False and completion["eligible_for_higher_microbatch"] is False
    assert completion["failure"] == "ValueError: wrong build"
    assert completion["quality_qualified"] is False and completion["resume_qualified"] is False
