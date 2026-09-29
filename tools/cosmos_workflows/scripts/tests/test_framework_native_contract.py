# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Validate the port against Framework APIs, without importing GPU runtimes."""

import argparse
import ast
import json
import runpy
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomllib
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models/cosmos3-reasoner/scripts"))
import evaluation_workflow  # noqa: E402
from test_cosmos_evaluation_workflow import _checkpoint_manifest  # noqa: E402
from test_cosmos_skill_workflow import args_for, checkpoint_preparation, make_model, workflow  # noqa: E402

FRAMEWORK = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(FRAMEWORK))
from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig  # noqa: E402
from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides  # noqa: E402


@pytest.mark.parametrize("model", ["nvidia/Cosmos3-Nano", "nvidia/Cosmos3-Edge"])
@pytest.mark.parametrize("mode", ["dense", "peft"])
@pytest.mark.parametrize("dataset", ["video_conversation", "task_aware_video_reasoning"])
def test_generated_toml_matches_native_schema(tmp_path, model, mode, dataset):
    args = args_for(tmp_path, model_name=model, training_mode=mode, dataset_family=dataset)
    args.warmup = 0.25
    args.minimum_lr_factor = 0.1
    plan = workflow.build_plan(args)
    native = SFTExperimentConfig.model_validate(plan["spec"])
    assert native.job.task == "vlm"
    assert native.model.lora_enabled == (mode == "peft")
    assert native.trainer.callbacks.loss_spike_rollback.enabled == (mode == "peft")
    assert native.scheduler.f_min == [0.1]
    assert all(isinstance(step, int) for step in plan["spec"]["scheduler"]["warm_up_steps"])
    overrides = build_hydra_overrides(plan["spec"])
    assert f"trainer.callbacks.loss_spike_rollback.enabled={str(mode == 'peft').lower()}" in overrides
    assert "COSMOS_VIDEO_NUM_FRAMES" in plan["environment"]


@pytest.mark.parametrize(
    "schedule,native", [("linear", "lambdalinear"), ("cosine", "lambdacosine"), ("constant", "lambdalinear")]
)
@pytest.mark.parametrize("optimizer,native_optimizer", [("AdamW", "adamw"), ("FusedAdam", "fusedadamw")])
def test_optimizer_and_scheduler_reach_native_cli(tmp_path, schedule, native, optimizer, native_optimizer):
    args = args_for(tmp_path)
    args.optimizer = optimizer
    args.scheduler = schedule
    args.optimizer_epsilon = 1e-6
    plan = workflow.build_plan(args)
    command = shlex.split(plan["command"])
    assert f"optimizer={native_optimizer}" in command
    assert f"scheduler={native}" in command
    assert "++optimizer.eps=1e-06" in command
    assert plan["spec"]["scheduler"]["f_min"] == [1.0 if schedule == "constant" else 0.0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("warmup", -1),
        ("minimum_lr_factor", 2),
        ("loss_spike_rollback", -1),
        ("scheduler", "typo"),
        ("optimizer", "typo"),
    ],
)
def test_invalid_training_settings_do_not_silently_fall_back(tmp_path, field, value):
    args = args_for(tmp_path)
    setattr(args, field, value)
    with pytest.raises(workflow.WorkflowError):
        workflow.build_plan(args)


def test_rollback_can_be_disabled_for_peft(tmp_path):
    args = args_for(tmp_path, training_mode="peft")
    args.loss_spike_rollback = 0
    assert not workflow.build_plan(args)["spec"]["trainer"]["callbacks"]["loss_spike_rollback"]["enabled"]


@pytest.mark.parametrize(
    "change", [None, "revision", "image", "output", "source", "corrupt_provenance", "invalid_provenance"]
)
def test_conversion_reuse_checks_request_and_content(tmp_path, monkeypatch, change):
    base = make_model(tmp_path / "base", "cosmos3_omni")
    donor = make_model(tmp_path / "donor")
    output = tmp_path / "prepared" / "model"
    argv = [
        "--base-model-path-or-uri",
        str(base),
        "--vlm-architecture-model-path-or-uri",
        str(donor),
        "--output-path",
        str(output),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--runtime-image",
        "example/framework:test",
        "--runtime-image-digest",
        "sha256:original",
    ]
    calls = []

    def convert(command, **kwargs):
        calls.append(command)
        prepared = make_model(tmp_path / "prepared")
        (prepared / "tokenizer_config.json").write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(checkpoint_preparation.subprocess, "run", convert)
    assert checkpoint_preparation.main(argv) == 0
    if change == "revision":
        argv.extend(["--base-model-revision", "changed"])
    elif change == "image":
        argv[-1] = "sha256:changed"
    elif change == "output":
        (output / "tokenizer.json").write_text('{"changed": true}')
    elif change == "source":
        (base / "config.json").write_text('{"model_type": "cosmos3_omni", "changed": true}')
    elif change == "corrupt_provenance":
        (output / checkpoint_preparation.PROVENANCE_FILE).write_text("invalid json")
    elif change == "invalid_provenance":
        (output / checkpoint_preparation.PROVENANCE_FILE).write_text("[]")
    assert checkpoint_preparation.main(argv) == (0 if change is None else 2)
    assert len(calls) == 1  # mismatched outputs are preserved, never overwritten implicitly
    assert output.exists()


def test_request_frame_mapping_is_used_by_native_recipe():
    schema = json.loads((workflow.SKILL_DIR / "schemas/train_request.schema.json").read_text())
    mapping = schema["properties"]["training"]["properties"]["video_profile"]["x_cosmos_native_mapping"]
    recipe = (FRAMEWORK / "cosmos_framework/configs/base/reasoner/experiment/video_sft.py").read_text()
    for environment_name in mapping.values():
        assert environment_name in recipe


@pytest.mark.parametrize(
    "weight_map",
    [
        {},
        [],
        {"tensor": "/outside.safetensors"},
        {"tensor": "../outside.safetensors"},
        {"tensor": "missing.safetensors"},
    ],
)
def test_conversion_rejects_invalid_checkpoint_index(tmp_path, weight_map):
    model = make_model(tmp_path)
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    with pytest.raises(ValueError, match="weight_map|unsafe weight path|missing indexed shards"):
        checkpoint_preparation.validate(model)


@pytest.mark.parametrize("value", ["/missing/model", "./missing-model", "../missing-model", "s3://bucket/model"])
def test_checkpoint_conversion_does_not_treat_missing_paths_as_hub_ids(value):
    assert not checkpoint_preparation.is_uri(value)


@pytest.mark.parametrize("value", ["org/model", "hf_model://org/model", "hf://models/org/model"])
def test_checkpoint_conversion_accepts_hub_identifiers(value):
    assert checkpoint_preparation.is_uri(value)


@pytest.mark.parametrize("revision", ["", "main", "v1.0", "deadbeef"])
def test_checkpoint_conversion_requires_immutable_hub_commit(tmp_path, monkeypatch, revision):
    monkeypatch.setattr(checkpoint_preparation.subprocess, "run", lambda *a, **kw: pytest.fail("launched conversion"))
    code = checkpoint_preparation.main(
        [
            "--base-model-path-or-uri",
            "org/model",
            "--base-model-revision",
            revision,
            "--output-path",
            str(tmp_path / "output"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--runtime-image",
            "example/framework:test",
            "--runtime-image-digest",
            "sha256:test",
        ]
    )
    assert code == 2
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("target", ["base", "donor", "parent", "child", "cache", "cache-child", "symlink"])
def test_checkpoint_force_never_overwrites_inputs_or_cache(tmp_path, monkeypatch, target):
    base = make_model(tmp_path / "base", "cosmos3_omni")
    donor = make_model(tmp_path / "donor")
    cache = tmp_path / "cache"
    cache.mkdir()
    marker = cache / "existing"
    marker.write_text("preserved")
    link = tmp_path / "linked-output"
    link.symlink_to(base, target_is_directory=True)
    output = {
        "base": base,
        "donor": donor,
        "parent": tmp_path,
        "child": base / "converted",
        "cache": cache,
        "cache-child": cache / "converted",
        "symlink": link,
    }[target]
    monkeypatch.setattr(checkpoint_preparation.subprocess, "run", lambda *a, **kw: pytest.fail("launched conversion"))
    code = checkpoint_preparation.main(
        [
            "--base-model-path-or-uri",
            str(base),
            "--vlm-architecture-model-path-or-uri",
            str(donor),
            "--output-path",
            str(output),
            "--cache-dir",
            str(cache),
            "--force",
            "--runtime-image",
            "example/framework:test",
            "--runtime-image-digest",
            "sha256:test",
        ]
    )
    assert code == 2
    assert (base / "config.json").is_file()
    assert (donor / "config.json").is_file()
    assert marker.read_text() == "preserved"
    assert link.is_symlink()


def test_inference_contract_calls_native_cli_with_dcp_options(monkeypatch):
    metadata = yaml.safe_load((workflow.SKILL_DIR / "references/skill_info.yaml").read_text())
    action = metadata["actions"]["inference"]
    assert action["mode"] == "args"
    assert "config_format" not in action
    project = tomllib.loads((FRAMEWORK / "pyproject.toml").read_text())
    entrypoint = project["project"]["scripts"][action["command"]]
    module, _ = entrypoint.split(":")
    source = FRAMEWORK / (module.replace(".", "/") + ".py")
    # Execute the actual parser without importing the GPU inference engine.
    tree = ast.parse(source.read_text())
    tree.body = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"parse_args", "str_to_bool"}
    ]
    namespace = {"argparse": argparse}
    exec(compile(tree, str(source), "exec"), namespace)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            action["command"],
            "--model_path",
            "/checkpoints/dcp",
            "--media",
            "/data/clip.mp4",
            "--prompt",
            "Describe this video.",
            "--results_dir",
            "/results/inference",
            "--config_file",
            "/checkpoints/config.yaml",
            "--export_dir",
            "/checkpoints/exported",
            "--vit_checkpoint_path",
            "/checkpoints/base",
            "--num_frames",
            "8",
        ],
    )
    parsed = namespace["parse_args"]()
    assert parsed.config_file == "/checkpoints/config.yaml"
    assert parsed.export_dir == "/checkpoints/exported"
    assert parsed.vit_checkpoint_path == "/checkpoints/base"
    assert parsed.num_frames == 8
    assert not parsed.enable_lora


@pytest.mark.parametrize("dataset", ["video_conversation", "task_aware_video_reasoning"])
@pytest.mark.parametrize("workers", [0, 1])
@pytest.mark.parametrize("sharding", ["stride", "media_balanced"])
def test_training_plan_handoff_matches_native_evaluation_loader(tmp_path, monkeypatch, dataset, workers, sharding):
    args = args_for(tmp_path, dataset_family=dataset)
    plan = workflow.build_plan(args)
    assert plan["framework_video_runtime"]["dataloader_prefetch_factor"] == (
        4 if dataset == "video_conversation" else 2
    )
    sealed = workflow.save_plan_artifact(args, plan, str(tmp_path / "training-plan.json"))
    model, manifest = _checkpoint_manifest(tmp_path, epoch=1, mode="dense")
    evaluation_args = evaluation_workflow.parse_args(
        [
            "--training-plan",
            str(sealed),
            "--checkpoint",
            "/runtime/checkpoints/epoch_1",
            "--action-model-path",
            model,
            "--action-model-manifest",
            str(manifest),
            "--results-dir",
            "/results/evaluation",
            "--generation-max-tokens",
            "16",
            "--framework-dataloader-num-workers",
            str(workers),
            "--evaluation-shard-strategy",
            sharding,
            "--plan-output",
            str(tmp_path / "evaluation-plan.json"),
        ]
    )
    result = evaluation_workflow.resolve(evaluation_args)
    vision = result["config"]["vision"]
    assert vision["dataloader_prefetch_factor"] == (2 if workers else 0)
    # Only construct the native preprocessor; no decode, model load, or GPU call.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=SimpleNamespace()))
    native = runpy.run_path(str(FRAMEWORK / "cosmos_framework/inference/reasoner/framework_torchcodec_video.py"))
    preprocessor = native["FrameworkTorchCodecVideoPreprocessor"](
        num_frames=vision["num_frames"],
        cache_size=vision["video_cache_size"],
        dataloader_num_workers=vision["dataloader_num_workers"],
        dataloader_prefetch_factor=vision["dataloader_prefetch_factor"],
        dataloader_persistent_workers=vision["dataloader_persistent_workers"],
        dataloader_multiprocessing_context=vision["dataloader_multiprocessing_context"],
    )
    assert preprocessor.dataloader_num_workers == workers
