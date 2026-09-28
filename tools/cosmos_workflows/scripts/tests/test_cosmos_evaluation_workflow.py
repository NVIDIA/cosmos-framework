#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import tomllib
import yaml

ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "models" / "cosmos3-reasoner"
SCRIPT = SKILL / "scripts" / "evaluation_workflow.py"
sys.path.insert(0, str(SKILL / "scripts"))

from cosmos_common import inspect_dataset, stable_hash  # noqa: E402


def _sealed_plan(
    tmp_path: Path,
    *,
    backend: str = "cosmos-framework",
    mode: str = "dense",
    prompt: str = "training prompt",
    max_video_pixels: int | None = 4096,
    seed: int = 17,
    model_tier: str = "edge",
    annotation_sha256: str | None = None,
    vision: dict | None = None,
) -> Path:
    plan = {
        "schema_version": 2,
        "experiment_id": "training-job",
        "action": "train",
        "backend": backend,
        "training": {
            "training_mode": mode,
            "precision": "bfloat16",
            "seed": seed,
            "frames": 8,
            "sequence_length": 40960,
            "system_prompt": prompt,
            "vision": vision or {"nframes": 8, "max_pixels": max_video_pixels},
        },
        "datasets": {
            "train": {"dataset_fingerprint": "train-fingerprint"},
            "validation": {
                "dataset_fingerprint": "validation-fingerprint",
                "annotations": [{"original": "/runtime/validation.json"}],
                "media_roots": [{"original": "/runtime/validation-media"}],
                "evaluation_profile": {
                    "inferred_task_type": "binary",
                    "answer_type": "freeform",
                    "metric_names": [],
                    "requires_user_input": [],
                    "unresolved_accuracy_tasks": [],
                },
            },
        },
        "model": {
            "fingerprint": "model-fingerprint",
            "supplied": {"original": "/runtime/base-model"},
        },
        "model_preparation": {"output": {"original": "/runtime/prepared-base-model"}},
        "prepared_model_container_path": "/runtime/prepared-base-model",
        "processor_profile": {
            "frames": 8,
            "max_video_pixels": max_video_pixels,
            "model_tier": model_tier,
        },
        "compute": {"total_gpus": 8},
        "image": {"tag": f"example.invalid/cosmos/{backend}:test"},
        "decoder_artifact": {"enabled": False},
        "evaluation_contract": {
            "schema_version": 1,
            "validation_dataset_fingerprint": "validation-fingerprint",
            "validation_annotations": ["/runtime/validation.json"],
            "validation_media_roots": ["/runtime/validation-media"],
            "system_prompt": prompt,
            "frames": 8,
            "vision": vision or {"nframes": 8, "max_pixels": max_video_pixels},
            "max_video_pixels": max_video_pixels,
            "precision": "bfloat16",
            "seed": seed,
            "batch_size": 1,
            "task_profile": {
                "inferred_task_type": "binary",
                "answer_type": "freeform",
                "metric_names": [],
                "requires_user_input": [],
                "unresolved_accuracy_tasks": [],
            },
            "generation": {
                "max_tokens": None,
                "temperature": 0.0,
                "repetition_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
            },
            "checkpoint_selection": None,
        },
    }
    if backend == "cosmos-framework":
        plan["framework_video_runtime"] = {
            "selected_profile": "torchcodec-cuda-on-demand",
            "decoder_device_binding": "explicit_local_rank",
            "decoder_device": "cuda",
            "decoder_threads": 1,
            "sft_process_threads": 8,
            "video_cache_size": 341,
            "dataloader_num_workers": 1,
            "dataloader_prefetch_factor": 2,
            "dataloader_multiprocessing_context": "spawn",
            "dataloader_persistent_workers": True,
        }
    if annotation_sha256 is not None:
        plan["datasets"]["validation"]["annotation_manifest"] = [
            {
                "original": "/runtime/validation.json",
                "sha256": annotation_sha256,
            }
        ]
    plan["plan_artifact"] = {
        "schema_version": 1,
        "path": str(tmp_path / "training-plan.json"),
        "sha256": stable_hash(plan),
    }
    path = tmp_path / "training-plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _status(tmp_path: Path, *, multiple: bool = False) -> Path:
    records = [
        {
            "status": "RUNNING",
            "phase": "checkpoint_saved",
            "checkpoint_path": "/runtime/checkpoints/epoch_1",
            "kpi": {"epoch": 1},
        }
    ]
    if multiple:
        records.append(
            {
                "status": "RUNNING",
                "phase": "checkpoint_saved",
                "checkpoint_path": "/runtime/checkpoints/epoch_2",
                "kpi": {"epoch": 2},
            }
        )
    records.append({"status": "SUCCESS", "message": "training complete"})
    path = tmp_path / "status.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def _checkpoint_manifest(tmp_path: Path, *, epoch: int, mode: str) -> tuple[str, Path]:
    source = f"/runtime/checkpoints/epoch_{epoch}"
    action = f"/runtime/safetensors/epoch_{epoch}"
    return action, _framework_checkpoint_manifest(tmp_path, source=source, action=action)


def _framework_checkpoint_manifest(tmp_path: Path, *, source: str, action: str) -> Path:
    path = tmp_path / "framework-checkpoint-action.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "VERIFIED",
                "backend": "cosmos-framework",
                "source_checkpoint": source,
                "action_model_path": action,
                "verification": {
                    "ok": True,
                    "action_model_path": action,
                    "weight_files": ["model-00001-of-00001.safetensors"],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _run(
    tmp_path: Path,
    *,
    backend: str = "cosmos-framework",
    mode: str = "dense",
    multiple: bool = False,
    prompt: str = "training prompt",
    max_video_pixels: int | None = 4096,
    seed: int = 17,
    model_tier: str = "edge",
    vision: dict | None = None,
    extra: list[str] | None = None,
    prepare_checkpoint: bool = True,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    plan_output = tmp_path / "evaluation-plan.json"
    config_output = tmp_path / "evaluation.toml"
    extra = list(extra or [])
    command = [
        sys.executable,
        str(SCRIPT),
        "--training-plan",
        str(
            _sealed_plan(
                tmp_path,
                backend=backend,
                mode=mode,
                prompt=prompt,
                max_video_pixels=max_video_pixels,
                seed=seed,
                model_tier=model_tier,
                vision=vision,
            )
        ),
        "--training-status",
        str(_status(tmp_path, multiple=multiple)),
        "--results-dir",
        "/runtime/evaluation-results",
        "--generation-max-tokens",
        "16",
        "--plan-output",
        str(plan_output),
        "--config-output",
        str(config_output),
        *extra,
    ]
    selected_epoch = 2 if extra[-2:] == ["--checkpoint-epoch", "2"] else 1
    selection_known = not multiple or "--checkpoint-epoch" in extra
    if prepare_checkpoint and selection_known and "--action-model-path" not in extra:
        action_model_path, manifest_path = _checkpoint_manifest(tmp_path, epoch=selected_epoch, mode=mode)
        command.extend(
            [
                "--action-model-path",
                action_model_path,
                "--action-model-manifest",
                str(manifest_path),
            ]
        )
    elif backend == "cosmos-framework" and "--action-model-path" in extra:
        index = extra.index("--action-model-path")
        action_model_path = extra[index + 1]
        source = f"/runtime/checkpoints/epoch_{selected_epoch}"
        manifest_path = _framework_checkpoint_manifest(tmp_path, source=source, action=action_model_path)
        command.extend(["--action-model-manifest", str(manifest_path)])
    return subprocess.run(command, text=True, capture_output=True, check=False), plan_output, config_output


def test_dense_evaluation_inherits_training_parity_fields(tmp_path: Path) -> None:
    result, plan_path, config_path = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    resolved = json.loads(plan_path.read_text(encoding="utf-8"))
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert resolved["ready"] is True
    assert resolved["required_user_inputs"] == []
    assert config["dataset"] == {
        "annotation_path": "/runtime/validation.json",
        "media_dir": "/runtime/validation-media",
        "system_prompt": "training prompt",
    }
    assert config["task"]["type"] == "binary"
    assert config["metrics"]["names"] == []
    assert config["vision"]["num_frames"] == 8
    assert config["vision"]["max_pixels"] == 4096
    assert "nframes" not in config["vision"]
    assert config["model"]["model_name"] == "/runtime/safetensors/epoch_1"
    assert config["model"]["max_length"] == 40960
    assert config["model"]["tp_size"] == 1
    assert config["model"]["enable_lora"] is False
    assert config["evaluation"]["batch_size"] == 1
    assert config["num_gpus"] == 8
    assert resolved["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()


def test_materialized_validation_paths_override_single_recorded_inputs(tmp_path: Path) -> None:
    result, plan_path, config_path = _run(
        tmp_path,
        extra=[
            "--action-validation-annotation",
            "/runtime/evaluation-results/validation-smoke-8.json",
            "--action-validation-media-root",
            "/runtime/evaluation-results/materialized-media",
        ],
    )
    assert result.returncode == 0, result.stderr
    resolved = json.loads(plan_path.read_text(encoding="utf-8"))
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert config["dataset"]["annotation_path"] == "/runtime/evaluation-results/validation-smoke-8.json"
    assert config["dataset"]["media_dir"] == "/runtime/evaluation-results/materialized-media"
    assert resolved["provenance"]["dataset.annotation_path"] == {
        "source": "materialize_exact_validation_manifest",
        "value": "/runtime/evaluation-results/validation-smoke-8.json",
    }
    assert resolved["provenance"]["dataset.media_dir"] == {
        "source": "materialize_validation_manifest_with_absolute_media",
        "value": "/runtime/evaluation-results/materialized-media",
    }


def test_evaluation_rejects_sampling_unsupported_by_native_decoder(tmp_path: Path) -> None:
    vision = {
        "fps": 1.0,
        "max_frames": 120,
        "video_start": 1.5,
        "video_end": 31.5,
        "resized_height": 448,
        "resized_width": 672,
        "min_pixels": 4096,
        "max_pixels": 81920,
        "total_pixels": 3136000,
    }
    result, plan_path, config_path = _run(
        tmp_path,
        max_video_pixels=81920,
        vision=vision,
    )
    assert result.returncode == 2
    assert "unsupported Framework evaluation sampling fields" in result.stderr
    assert not config_path.exists()


def test_missing_pixel_budget_accepts_explicit_evaluation_input(tmp_path: Path) -> None:
    unresolved, plan_path, config_path = _run(tmp_path, max_video_pixels=None)
    assert unresolved.returncode == 3, unresolved.stderr
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert [item["field"] for item in plan["required_user_inputs"]] == ["vision.max_pixels"]
    assert not config_path.exists()

    resolved, resolved_plan_path, resolved_config_path = _run(
        tmp_path,
        max_video_pixels=None,
        extra=["--max-video-pixels", "3136000"],
    )
    assert resolved.returncode == 0, resolved.stderr
    resolved_plan = json.loads(resolved_plan_path.read_text(encoding="utf-8"))
    config = tomllib.loads(resolved_config_path.read_text(encoding="utf-8"))
    assert config["vision"]["max_pixels"] == 3136000
    assert resolved_plan["provenance"]["vision.max_pixels"] == {
        "source": "user",
        "value": 3136000,
    }


def test_nano_missing_pixel_budget_requires_explicit_input(tmp_path: Path) -> None:
    resolved, plan_path, config_path = _run(
        tmp_path,
        max_video_pixels=None,
        model_tier="nano",
    )
    assert resolved.returncode == 3, resolved.stderr
    plan = json.loads(plan_path.read_text())
    assert [item["field"] for item in plan["required_user_inputs"]] == ["vision.max_pixels"]
    assert not config_path.exists()


def test_zero_seed_is_a_valid_sealed_evaluation_seed(tmp_path: Path) -> None:
    result, plan_path, config_path = _run(tmp_path, seed=0)
    assert result.returncode == 0, result.stderr
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert plan["required_user_inputs"] == []
    assert plan["provenance"]["evaluation.seed"] == {
        "source": "sealed_training_plan",
        "value": 0,
    }
    assert config["evaluation"]["seed"] == 0


def test_fingerprint_locked_wts_peft_evaluator_profile_is_automatic(tmp_path: Path) -> None:
    training_plan = _sealed_plan(
        tmp_path,
        mode="peft",
        prompt="",
        max_video_pixels=81920,
        seed=0,
        model_tier="nano",
        annotation_sha256="f120ca66f28e3e5b5a01a3ace93d16c856cf13098faf61b44263a4afc449c709",
    )
    plan_output = tmp_path / "evaluation-plan.json"
    config_output = tmp_path / "evaluation.toml"
    action_model_path, manifest_path = _checkpoint_manifest(tmp_path, epoch=1, mode="peft")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--training-plan",
            str(training_plan),
            "--training-status",
            str(_status(tmp_path)),
            "--results-dir",
            "/runtime/evaluation-results",
            "--plan-output",
            str(plan_output),
            "--config-output",
            str(config_output),
            "--action-model-path",
            action_model_path,
            "--action-model-manifest",
            str(manifest_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    resolved = json.loads(plan_output.read_text(encoding="utf-8"))
    config = tomllib.loads(config_output.read_text(encoding="utf-8"))
    assert resolved["verified_evaluator_profile"]["name"] == "PEFT_HPO_VALIDATION_F120CA66"
    assert config["evaluation"]["answer_type"] == "freeform"
    assert config["evaluation"]["batch_size"] == 8
    assert config["evaluation"]["seed"] == 1
    assert config["generation"]["max_tokens"] == 1024
    assert resolved["provenance"]["evaluation.batch_size"]["source"].startswith(
        "verified_evaluator_profile.PEFT_HPO_VALIDATION_F120CA66:"
    )


def test_verified_full_validation_profiles_need_no_generation_input(tmp_path: Path) -> None:
    profiles = {
        "c33afc26f979cbdb488b8f1aefdc65604992cd7552d5e75ea782e4565fdc21e1": (
            "VALIDATION_C33AFC26",
            "letter",
        ),
        "6a30babb1921af59155dfe45cf766465597b57cafa1e0e83663a159d89289b6a": (
            "VALIDATION_6A30BABB",
            "freeform",
        ),
        "f828a63f1bbdd45197e1f3393fb94f76ebfdfc785402617aa8c1397b0b47c555": (
            "VALIDATION_F828A63F",
            "letter",
        ),
    }
    for index, (annotation_sha, (profile_name, answer_type)) in enumerate(profiles.items()):
        case = tmp_path / str(index)
        case.mkdir()
        training_plan = _sealed_plan(case, annotation_sha256=annotation_sha)
        action_model_path, manifest_path = _checkpoint_manifest(case, epoch=1, mode="dense")
        plan_output = case / "evaluation-plan.json"
        config_output = case / "evaluation.toml"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--training-plan",
                str(training_plan),
                "--training-status",
                str(_status(case)),
                "--results-dir",
                "/runtime/evaluation-results",
                "--action-model-path",
                action_model_path,
                "--action-model-manifest",
                str(manifest_path),
                "--plan-output",
                str(plan_output),
                "--config-output",
                str(config_output),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        resolved = json.loads(plan_output.read_text(encoding="utf-8"))
        config = tomllib.loads(config_output.read_text(encoding="utf-8"))
        assert resolved["required_user_inputs"] == []
        assert resolved["verified_evaluator_profile"]["name"] == profile_name
        assert config["generation"]["max_tokens"] == 1024
        assert config["evaluation"]["answer_type"] == answer_type


def test_framework_peft_uses_verified_merged_export(tmp_path: Path) -> None:
    result, plan_path, config_path = _run(tmp_path, mode="peft")
    assert result.returncode == 0, result.stderr
    config = tomllib.loads(config_path.read_text())
    assert config["model"]["enable_lora"] is False
    assert config["model"]["model_name"] == "/runtime/safetensors/epoch_1"


def test_recorded_empty_system_prompt_is_inherited_not_reported_missing(tmp_path: Path) -> None:
    result, plan_path, config_path = _run(tmp_path, prompt="")
    assert result.returncode == 0, result.stderr
    resolved = json.loads(plan_path.read_text(encoding="utf-8"))
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert resolved["required_user_inputs"] == []
    assert config["dataset"]["system_prompt"] == ""
    assert resolved["provenance"]["dataset.system_prompt"]["source"] == "sealed_training_plan.evaluation_contract"


def test_framework_export_is_automated_not_user_intake(tmp_path: Path) -> None:
    result, plan_path, config_path = _run(tmp_path, backend="cosmos-framework", mode="peft", prepare_checkpoint=False)
    assert result.returncode == 3, result.stderr
    unresolved = json.loads(plan_path.read_text(encoding="utf-8"))

    assert unresolved["required_user_inputs"] == []
    assert unresolved["automated_actions"][0]["action"] == "framework_checkpoint_pre_action"
    assert {item["source"] for item in unresolved["automated_actions"][0]["supporting_files"]} == {
        "scripts/framework_checkpoint_action.py",
        "scripts/cosmos_common.py",
    }
    assert all(len(item["sha256"]) == 64 for item in unresolved["automated_actions"][0]["supporting_files"])
    assert not config_path.exists()

    rerun, resolved_path, resolved_config = _run(
        tmp_path,
        backend="cosmos-framework",
        mode="peft",
        extra=["--action-model-path", "/runtime/exported-checkpoint"],
    )
    assert rerun.returncode == 0, rerun.stderr
    config = tomllib.loads(resolved_config.read_text(encoding="utf-8"))
    assert config["model"]["model_name"] == "/runtime/exported-checkpoint"
    assert config["model"]["enable_lora"] is False
    assert json.loads(resolved_path.read_text())["ready"] is True


def test_multiple_checkpoints_require_exact_selection(tmp_path: Path) -> None:
    result, plan_path, config_path = _run(tmp_path, multiple=True)
    assert result.returncode == 3, result.stderr
    unresolved = json.loads(plan_path.read_text(encoding="utf-8"))
    assert [item["field"] for item in unresolved["required_user_inputs"]] == ["checkpoint_selection"]
    assert not config_path.exists()

    selected, _, selected_config = _run(tmp_path, multiple=True, extra=["--checkpoint-epoch", "2"])
    assert selected.returncode == 0, selected.stderr
    config = tomllib.loads(selected_config.read_text(encoding="utf-8"))
    assert config["model"]["model_name"] == "/runtime/safetensors/epoch_2"


def test_ready_plan_emits_native_framework_execution_bundle(tmp_path: Path) -> None:
    result, plan_path, _ = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    plan = json.loads(plan_path.read_text())
    bundle = plan["spec_bundle"]
    schema = json.loads((ROOT / "execution/artifacts/references/spec_bundle.schema.json").read_text())
    jsonschema.validate(bundle, schema)
    assert bundle["spec"]["results_dir"] == "{results_dir}"
    assert plan["spec_bundle_sha256"] == stable_hash(bundle)
    assert bundle["command"] == "cosmos-reasoner-evaluate --config {config_path}"
    assert "FrameworkTorchCodecVideoPreprocessor" in bundle["execution"]["pre_commands"][0]
    assert "FORCE_QWENVL_VIDEO_READER" not in bundle["execution"]["environment"]
    assert "post_commands" not in bundle["execution"]


def test_evaluation_model_length_override_is_explicit_and_provenanced(
    tmp_path: Path,
) -> None:
    result, plan_path, config_path = _run(tmp_path, extra=["--model-max-length", "16384"])
    assert result.returncode == 0, result.stderr
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert config["model"]["max_length"] == 16384
    assert plan["provenance"]["model.max_length"] == {
        "source": "user",
        "value": 16384,
    }


def test_unsealed_training_plan_is_rejected(tmp_path: Path) -> None:
    training_plan = _sealed_plan(tmp_path)
    plan = json.loads(training_plan.read_text(encoding="utf-8"))
    plan.pop("plan_artifact")
    training_plan.write_text(json.dumps(plan), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--training-plan",
            str(training_plan),
            "--results-dir",
            "/runtime/evaluation-results",
            "--generation-max-tokens",
            "16",
            "--plan-output",
            str(tmp_path / "evaluation-plan.json"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "sealed plan artifact" in completed.stderr


def test_multiple_recorded_validation_manifests_are_automated_not_user_selection(tmp_path: Path) -> None:
    training_plan = _sealed_plan(tmp_path)
    plan = json.loads(training_plan.read_text(encoding="utf-8"))
    plan.pop("plan_artifact")
    plan["evaluation_contract"]["validation_annotations"] = [
        "/runtime/validation-a.json",
        "/runtime/validation-b.json",
    ]
    plan["evaluation_contract"]["validation_media_roots"] = [
        "/runtime/validation-media",
    ]
    plan["plan_artifact"] = {
        "schema_version": 1,
        "path": str(training_plan),
        "sha256": stable_hash(plan),
    }
    training_plan.write_text(json.dumps(plan), encoding="utf-8")
    evaluation_plan = tmp_path / "evaluation-plan.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--training-plan",
            str(training_plan),
            "--training-status",
            str(_status(tmp_path)),
            "--results-dir",
            "/runtime/evaluation-results",
            "--generation-max-tokens",
            "16",
            "--plan-output",
            str(evaluation_plan),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 3, result.stderr
    unresolved = json.loads(evaluation_plan.read_text(encoding="utf-8"))
    assert unresolved["required_user_inputs"] == []
    assert unresolved["automated_actions"][0]["action"] == "materialize_exact_validation_manifest"
    assert unresolved["blockers"][0]["user_input"] is False


def test_dataset_inspection_records_binary_and_ambiguous_profiles(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    for name in ("yes.mp4", "no.mp4"):
        (media / name).write_bytes(name.encode())
    binary_manifest = tmp_path / "binary.json"
    binary_manifest.write_text(
        json.dumps(
            [
                {"id": "yes", "video": "yes.mp4", "conversations": [{"value": "q"}, {"value": "Yes"}]},
                {"id": "no", "video": "no.mp4", "conversations": [{"value": "q"}, {"value": "No"}]},
            ]
        ),
        encoding="utf-8",
    )
    binary = inspect_dataset(
        dataset_family="auto",
        annotations=[str(binary_manifest)],
        media_roots=[str(media)],
        verify_media_content=False,
    )
    assert binary["evaluation_profile"]["inferred_task_type"] == "binary"
    assert binary["evaluation_profile"]["requires_user_input"] == []

    ambiguous_manifest = tmp_path / "ambiguous.json"
    ambiguous_manifest.write_text(
        json.dumps(
            [
                {"id": "a", "video": "yes.mp4", "conversations": [{"value": "q"}, {"value": "A"}]},
                {"id": "b", "video": "no.mp4", "conversations": [{"value": "q"}, {"value": "B"}]},
            ]
        ),
        encoding="utf-8",
    )
    ambiguous = inspect_dataset(
        dataset_family="auto",
        annotations=[str(ambiguous_manifest)],
        media_roots=[str(media)],
        verify_media_content=False,
    )
    assert ambiguous["evaluation_profile"]["unresolved_accuracy_tasks"] == ["default"]
    assert "task.type" in ambiguous["evaluation_profile"]["requires_user_input"]


def test_evaluation_template_is_dataset_neutral() -> None:
    template = yaml.safe_load((SKILL / "references" / "spec_template_evaluate.yaml").read_text())
    text = json.dumps(template).casefold()

    assert template["dataset"]["system_prompt"] == ""
    assert template["metrics"]["names"] == []
    assert template["vision"]["num_frames"] == 0
    assert "nframes" not in template["vision"]
    assert "street-view" not in text
    assert "cr1_1_zero_shot" not in text
