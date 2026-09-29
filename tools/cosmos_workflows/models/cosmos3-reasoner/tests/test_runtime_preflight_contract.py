# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "cosmos_workflow.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("cosmos_workflow", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _video_args() -> SimpleNamespace:
    return SimpleNamespace(
        dataset_family="video_conversation",
        minimum_lr_factor=None,
        container_checkpoint_dir="/checkpoints",
        learning_rate=1.1e-5,
        weight_decay=0.09,
        scheduler="linear",
        optimizer_epsilon=1e-8,
        warmup=0,
        gradient_clip=1.0,
        precision="bfloat16",
        async_checkpoint=False,
        validation_batch_size=1,
        seed=42,
        sequence_length=40960,
        nodes=1,
        gpus_per_node=8,
        training_mode="dense",
        experiment_id="video-smoke",
        frames=8,
        fps=None,
        min_frames=None,
        max_frames=None,
        video_start=None,
        video_end=None,
        video_resized_height=None,
        video_resized_width=None,
        video_min_pixels=None,
        video_max_pixels=81920,
        video_total_pixels=None,
        system_prompt="You are a helpful assistant.",
        container_cache_dir="/cache",
        run_mode="smoke",
        cosmos_job_id="video-smoke",
        container_results_dir="/results",
        nccl_debug="INFO",
        cuda_allocator="expandable_segments:True",
    )


def test_dataset_manifest_preserves_supplied_shared_filesystem_alias(
    tmp_path: Path,
) -> None:
    physical = tmp_path / "fs11" / "dataset"
    physical.mkdir(parents=True)
    (physical / "clip.mp4").write_bytes(b"video")
    annotation = physical / "train.json"
    annotation.write_text(
        json.dumps(
            [
                {
                    "id": "sample-1",
                    "video": "clip.mp4",
                    "conversations": [
                        {"from": "human", "value": "What happened?"},
                        {"from": "gpt", "value": "Nothing."},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    alias = tmp_path / "fsw"
    alias.symlink_to(tmp_path / "fs11", target_is_directory=True)

    inspected = MODULE.inspect_dataset(
        dataset_family="video_conversation",
        annotations=[str(alias / "dataset" / "train.json")],
        media_roots=[str(alias / "dataset")],
        verify_media_content=False,
    )

    assert inspected["media_manifest"][0]["path"] == str(alias / "dataset" / "clip.mp4")


def test_slurm_renderer_creates_writable_mount_roots_before_pyxis() -> None:
    args = SimpleNamespace(
        platform="slurm",
        partition="polar3",
        account="workflow_status",
        sqsh_path="/images/cosmos.sqsh",
        use_requeue=False,
        timeout="00:10:00",
        container_mount=[
            "/inputs:/data:ro",
            "/runs/results:/results",
            "/runs/checkpoints:/checkpoints",
            "/runs/cache:/cache",
        ],
        nodes=1,
        gpus_per_node=8,
        cpus_per_task=64,
        cosmos_job_id="cosmos-reason-train-test",
        experiment_id="test",
        time_limit="00:15:00",
        stdout_path="/runs/logs/main.out",
        stderr_path="/runs/logs/main.err",
        qos="",
        reservation="",
        exclusive=True,
        results_dir="/runs/results",
        checkpoint_dir="/runs/checkpoints",
        cache_dir="/runs/cache",
        master_port=29500,
    )
    script = MODULE.render_slurm(
        args,
        {
            "environment": {},
            "command": "true",
        },
    )

    pyxis_offset = script.index("srun ")
    for path in ("/runs/results", "/runs/checkpoints", "/runs/cache"):
        setup = f"mkdir -p -- {path}"
        assert setup in script
        assert script.index(setup) < pyxis_offset
    assert "mkdir -p -- /inputs" not in script
    assert script.count("--container-mounts=") == 1
    assert (
        "--container-mounts=/inputs:/data:ro,/runs/results:/results,/runs/checkpoints:/checkpoints,/runs/cache:/cache"
    ) in script


def _omni_preparation_args(platform: str) -> SimpleNamespace:
    return SimpleNamespace(
        base_model_format="cosmos3_omni",
        model="nvidia/Cosmos3-Nano",
        prepared_checkpoint_path="",
        base_model_path_or_uri="nvidia/Cosmos3-Nano",
        base_model_revision="b" * 40,
        checkpoint_dir="/checkpoints",
        cache_dir="/cache",
        container_cache_dir="/cache",
        vlm_architecture_model_path_or_uri="Qwen/Qwen3-VL-8B-Instruct",
        vlm_architecture_model_revision="a" * 40,
        image_tag="registry.example.invalid/cosmos-framework:test",
        sqsh_path="/images/cosmos-framework.sqsh" if platform == "slurm" else "",
        platform=platform,
        results_dir="/results",
        cosmos_job_id="cosmos-reason-prepare-test",
        experiment_id="prepare-test",
        container_mount=[
            "/results:/results",
            "/checkpoints:/checkpoints",
            "/cache:/cache",
        ],
    )


def _omni_model() -> dict[str, object]:
    return {
        "format": "cosmos3_omni",
        "source_type": "uri",
        "fingerprint": "c" * 64,
        "revision_resolution": {"repo_id": "nvidia/Cosmos3-Nano"},
        "vlm_architecture_revision_resolution": {"repo_id": "Qwen/Qwen3-VL-8B-Instruct"},
    }


def test_docker_model_preparation_resolves_digest_without_placeholder() -> None:
    args = _omni_preparation_args("docker")
    _, preparation = MODULE._model_preparation(args, _omni_model())

    command = preparation["command"]
    assert isinstance(command, str)
    assert "<" + "RESOLVE_AFTER_CLEAN_BUILD" + ">" not in command
    assert command.count("--runtime-image-digest") == 1
    assert "docker image inspect --format '{{index .RepoDigests 0}}'" in command
    assert "docker image inspect --format '{{.Id}}'" in command
    assert "unable to resolve runtime image digest" in command
    syntax = subprocess.run(["bash", "-n"], input=command, text=True, capture_output=True, check=False)
    assert syntax.returncode == 0, syntax.stderr


def test_slurm_model_preparation_resolves_authoritative_sqsh_digest() -> None:
    args = _omni_preparation_args("slurm")
    _, preparation = MODULE._model_preparation(args, _omni_model())
    action = preparation["platform_action"]
    assert isinstance(action, dict)

    container_command = action["container_command"]
    assert "<" + "RESOLVE_AFTER_CLEAN_BUILD" + ">" not in container_command
    assert container_command.count("--runtime-image-digest") == 1
    assert "COSMOS_PREPARATION_IMAGE_DIGEST" in container_command
    assert "/images/cosmos-framework.sqsh" in container_command

    args.partition = "polar3"
    args.account = "workflow_status"
    args.use_requeue = False
    args.timeout = "00:10:00"
    args.nodes = 1
    args.gpus_per_node = 8
    args.cpus_per_task = 64
    args.time_limit = "00:15:00"
    args.stdout_path = "/results/main.out"
    args.stderr_path = "/results/main.err"
    args.qos = ""
    args.reservation = ""
    args.exclusive = True
    args.master_port = 29500
    script = MODULE.render_slurm(
        args,
        {
            "environment": {},
            "command": "true",
            "model_preparation": preparation,
        },
    )

    assert "sha256sum -- /images/cosmos-framework.sqsh" in script
    assert "unable to resolve runtime image digest" in script
    assert ("--container-env=HF_TOKEN,HUGGING_FACE_HUB_TOKEN,COSMOS_PREPARATION_IMAGE_DIGEST") in script


def _framework_spec_args(
    validation_batch_size: int,
    validation_shard_strategy: str = "media_grouped",
) -> SimpleNamespace:
    return SimpleNamespace(
        nodes=1,
        gpus_per_node=8,
        effective_global_batch=8,
        framework_per_forward_batch=1,
        run_mode="diagnostic",
        smoke_train_samples=8,
        smoke_validation_samples=8,
        train_sample_limit=8,
        validation_sample_limit=0,
        validation_batch_size=validation_batch_size,
        framework_validation_shard_strategy=validation_shard_strategy,
        model="Cosmos3-Nano",
        dataset_family="video_conversation",
        experiment_id="framework-validation-test",
        attention_implementation="cosmos",
        precision="bfloat16",
        optimizer_epsilon=1e-8,
        learning_rate=1.1e-5,
        weight_decay=0.09,
        scheduler="linear",
        warmup=0.0,
        gradient_clip=1.0,
        async_checkpoint=False,
        sequence_length=40960,
        training_mode="dense",
    )


def test_framework_validation_batch_derives_exact_padded_rank_iterations() -> None:
    spec = MODULE._framework_spec(
        _framework_spec_args(validation_batch_size=5),
        train_count=8,
        val_count=2676,
        contract={
            "epochs": 1,
            "train_sample_multiplier": 1,
            "lora": None,
            "minimum_lr_factor": 0.0,
            "loss_spike_rollback": 0.0,
        },
    )

    # 2,676 records are padded to 2,680 across 8 ranks: 335/rank. Batch 5
    # therefore consumes the exact established multiset in 67 iterations.
    assert spec["trainer"]["max_val_iter"] == 67


def test_framework_prefetch_default_is_video_conversation_profile_specific() -> None:
    args = SimpleNamespace(
        dataset_family="video_conversation",
        run_mode="full",
        nodes=1,
        gpus_per_node=8,
        validation_batch_size=5,
        framework_validation_shard_strategy="media_grouped",
        framework_validation_video_feature_cache_size=0,
        framework_baked_overlay_pythonpath=("/cosmos-patches-framework-c312482-evalval-lab-v13/site-packages"),
        framework_video_cache_size=None,
        framework_sft_process_threads=0,
        framework_video_decoder_threads=0,
        framework_dataloader_num_workers=None,
        framework_dataloader_prefetch_factor=None,
    )
    runtime = MODULE._framework_video_runtime(
        args,
        train_data={"profile": {"unique_media_count": 341}},
        val_data={
            "record_count": 2676,
            "profile": {"unique_media_count": 171},
        },
    )

    assert runtime["dataloader_prefetch_factor"] == 4
    assert runtime["dataloader_pin_memory"] is True

    args.dataset_family = "task_aware_video_reasoning"
    runtime = MODULE._framework_video_runtime(
        args,
        train_data={"profile": {"unique_media_count": 341}},
        val_data={
            "record_count": 2676,
            "profile": {"unique_media_count": 171},
        },
    )
    assert runtime["dataloader_prefetch_factor"] == 2


def test_framework_repeated_media_auto_profile_emits_measured_validation_contract() -> None:
    args = SimpleNamespace(
        dataset_family="video_conversation",
        run_mode="full",
        nodes=1,
        gpus_per_node=8,
        validation_batch_size=1,
        framework_validation_shard_strategy="auto",
        framework_validation_video_feature_cache_size=None,
        framework_baked_overlay_pythonpath="",
        framework_video_cache_size=None,
        framework_sft_process_threads=0,
        framework_video_decoder_threads=0,
        framework_dataloader_num_workers=None,
        framework_dataloader_prefetch_factor=None,
    )
    runtime = MODULE._framework_video_runtime(
        args,
        train_data={"record_count": 5555, "profile": {"unique_media_count": 341}},
        val_data={"record_count": 2676, "profile": {"unique_media_count": 171}},
        runtime_model_type="qwen3_vl",
    )

    assert runtime["validation_repeated_media"] is True
    assert runtime["validation_shard_strategy"] == "media_grouped"
    assert runtime["validation_video_feature_cache_size"] == 341
    assert runtime["validation_processed_video_cache_size"] == 341
    assert runtime["validation_cache_frontload_unique_per_batch"] == 0
    assert runtime["dataloader_pin_memory"] is True


def test_framework_repeated_media_derives_frontload_from_user_batch() -> None:
    args = SimpleNamespace(
        dataset_family="video_conversation",
        run_mode="full",
        nodes=1,
        gpus_per_node=8,
        validation_batch_size=16,
        framework_validation_shard_strategy="auto",
        framework_validation_video_feature_cache_size=None,
        framework_validation_processed_video_cache_size=None,
        framework_validation_cache_frontload_unique_per_batch=None,
        framework_baked_overlay_pythonpath="",
        framework_video_cache_size=None,
        framework_sft_process_threads=0,
        framework_video_decoder_threads=0,
        framework_dataloader_num_workers=None,
        framework_dataloader_prefetch_factor=None,
    )
    runtime = MODULE._framework_video_runtime(
        args,
        train_data={"record_count": 5555, "profile": {"unique_media_count": 341}},
        val_data={"record_count": 2676, "profile": {"unique_media_count": 171}},
        runtime_model_type="qwen3_vl",
    )

    assert runtime["validation_batch_size"] == 16
    assert runtime["validation_processed_video_cache_size"] == 341
    assert runtime["validation_cache_frontload_unique_per_batch"] == 8
    assert runtime["validation_cache_frontload_batch_size"] == 16


def test_framework_media_grouped_validation_allows_equal_partial_final_batch() -> None:
    spec = MODULE._framework_spec(
        _framework_spec_args(validation_batch_size=8),
        train_count=8,
        val_count=2676,
        contract={
            "epochs": 1,
            "train_sample_multiplier": 1,
            "lora": None,
            "minimum_lr_factor": 0.0,
            "loss_spike_rollback": 0.0,
        },
    )

    # Every rank receives 335 records and emits 41 full batches plus one
    # seven-record final batch from the finite media-grouped stream.
    assert spec["trainer"]["max_val_iter"] == 42


def test_framework_infinite_validation_rejects_next_epoch_spill() -> None:
    try:
        MODULE._framework_spec(
            _framework_spec_args(
                validation_batch_size=16,
                validation_shard_strategy="stride",
            ),
            train_count=8,
            val_count=2676,
            contract={
                "epochs": 1,
                "train_sample_multiplier": 1,
                "lora": None,
                "minimum_lr_factor": 0.0,
                "loss_spike_rollback": 0.0,
            },
        )
    except MODULE.WorkflowError as exc:
        assert "335 records/rank is not divisible by batch 16" in str(exc)
    else:
        raise AssertionError("Framework accepted a validation batch that spills epochs")


def test_framework_v12_preflight_requires_finite_validation_stream() -> None:
    args = _video_args()
    args.gpus_per_node = 8
    args.framework_baked_overlay_pythonpath = "/cosmos-patches-framework-c312482-evalval-lab-v12/site-packages"
    args.framework_baked_overlay_module_prefix = "/cosmos-patches-framework-c312482-evalval-lab-v12/modules"
    args.results_dir = "/results"
    args.checkpoint_dir = "/checkpoints"
    args.cache_dir = "/cache"
    args.train_annotation = ["/data/train.json"]
    args.train_media_root = ["/data/train"]
    args.validation_annotation = ["/data/val.json"]
    args.validation_media_root = ["/data/val"]
    args.platform = "docker"
    args.sqsh_path = ""
    runtime = {
        "video_cache_size": 341,
        "sft_process_threads": 8,
        "dataloader_num_workers": 1,
        "decoder_threads": 1,
        "validation_batch_size": 8,
        "validation_shard_strategy": "media_grouped",
        "validation_video_feature_cache_size": 0,
        "validation_partial_final_batch": True,
    }

    preflight = MODULE._preflight_contract(
        args,
        {"tag": "example.invalid/cosmos-framework:test"},
        "/models/cosmos3",
        "/data/video.mp4",
        framework_video_runtime=runtime,
    )

    startup = preflight["container_startup"]
    assert "cosmos_framework.scripts" in startup
    assert "convert_model_to_vlm_safetensors" in startup
    assert "cosmos_rl.model_preparation" not in startup
    assert "COSMOS_PREFLIGHT_ASSERTION_FAILED:model_preparation_runtime" in startup
    assert "framework_finite_validation_stream" in startup


def test_framework_validation_cache_preflight_attests_inherited_and_new_modules() -> None:
    args = _video_args()
    args.gpus_per_node = 8
    args.framework_baked_overlay_pythonpath = "/cosmos-patches-framework-c312482-evalval-lab-v18/site-packages"
    args.framework_baked_overlay_module_prefix = "/cosmos-patches-framework-c312482-evalval-lab-v18/modules"
    args.results_dir = "/results"
    args.checkpoint_dir = "/checkpoints"
    args.cache_dir = "/cache"
    args.train_annotation = ["/data/train.json"]
    args.train_media_root = ["/data/train"]
    args.validation_annotation = ["/data/val.json"]
    args.validation_media_root = ["/data/val"]
    args.platform = "docker"
    args.sqsh_path = ""
    runtime = {
        "video_cache_size": 341,
        "sft_process_threads": 8,
        "dataloader_num_workers": 1,
        "decoder_threads": 1,
        "validation_batch_size": 1,
        "validation_shard_strategy": "media_grouped",
        "validation_video_feature_cache_size": 341,
        "validation_partial_final_batch": False,
    }

    preflight = MODULE._preflight_contract(
        args,
        {"tag": "example.invalid/cosmos-framework:test"},
        "/models/cosmos3",
        "/data/video.mp4",
        framework_video_runtime=runtime,
    )

    startup = preflight["container_startup"]
    assert "/cosmos-patches-framework-c312482-evalval-lab-v18/modules" in startup
    assert "framework_validation_visual_forward_cache" in startup
    assert "framework_validation_only_feature_cache" in startup
    assert "framework_validation_feature_cache_collective_safety" in startup
    assert "COSMOS_PREFLIGHT_ASSERTION_FAILED:nccl_min_max_scalars" in startup

    args.framework_baked_overlay_pythonpath = "/cosmos-patches-framework-c312482-evalval-lab-v19/site-packages"
    args.framework_baked_overlay_module_prefix = "/cosmos-patches-framework-c312482-evalval-lab-v19/modules"
    preflight = MODULE._preflight_contract(
        args,
        {"tag": "example.invalid/cosmos-framework:test"},
        "/models/cosmos3",
        "/data/video.mp4",
        framework_video_runtime=runtime,
    )
    startup = preflight["container_startup"]
    assert "/cosmos-patches-framework-c312482-evalval-lab-v19/modules" in startup

    args.framework_baked_overlay_pythonpath = "/cosmos-patches-framework-c312482-evalval-lab-v20/site-packages"
    args.framework_baked_overlay_module_prefix = "/cosmos-patches-framework-c312482-evalval-lab-v20/modules"
    preflight = MODULE._preflight_contract(
        args,
        {"tag": "example.invalid/cosmos-framework:test"},
        "/models/cosmos3",
        "/data/video.mp4",
        framework_video_runtime=runtime,
    )
    startup = preflight["container_startup"]
    assert "/cosmos-patches-framework-c312482-evalval-lab-v20/modules" in startup

    args.framework_baked_overlay_pythonpath = "/cosmos-patches-framework-c312482-evalval-lab-v21/site-packages"
    args.framework_baked_overlay_module_prefix = "/cosmos-patches-framework-c312482-evalval-lab-v21/modules"
    preflight = MODULE._preflight_contract(
        args,
        {"tag": "example.invalid/cosmos-framework:test"},
        "/models/cosmos3",
        "/data/video.mp4",
        framework_video_runtime=runtime,
    )
    startup = preflight["container_startup"]
    assert "/cosmos-patches-framework-c312482-evalval-lab-v21/modules" in startup


def test_framework_validation_feature_cache_is_gated_on_supported_model_type() -> None:
    """NVBUG 6669758: Cosmos Framework supports the cache only for qwen3_vl.

    Enabling it for cosmos3_edge aborts at model construction, so the planner
    must not seal a nonzero size for unsupported families.
    """
    args = SimpleNamespace(
        dataset_family="video_conversation",
        run_mode="full",
        nodes=1,
        gpus_per_node=4,
        validation_batch_size=1,
        framework_validation_shard_strategy="auto",
        framework_validation_video_feature_cache_size=None,
        framework_validation_processed_video_cache_size=None,
        framework_validation_cache_frontload_unique_per_batch=None,
        framework_baked_overlay_pythonpath="",
        framework_video_cache_size=None,
        framework_sft_process_threads=0,
        framework_video_decoder_threads=0,
        framework_dataloader_num_workers=None,
        framework_dataloader_prefetch_factor=None,
    )
    train_data = {"record_count": 5555, "profile": {"unique_media_count": 341}}
    val_data = {"record_count": 2676, "profile": {"unique_media_count": 171}}

    edge = MODULE._framework_video_runtime(args, train_data, val_data, runtime_model_type="cosmos3_edge")
    assert edge["validation_video_feature_cache_size"] == 0
    assert edge["validation_cache_frontload_unique_per_batch"] == 0
    assert edge["validation_video_feature_cache_supported"] is False
    assert edge["validation_video_feature_cache_model_type"] == "cosmos3_edge"
    # The gate disables only the GPU-embedding cache; grouped sharding and the
    # processed-video cache are unrelated and must survive.
    assert edge["validation_shard_strategy"] == "media_grouped"
    assert edge["validation_processed_video_cache_size"] == 341

    supported = MODULE._framework_video_runtime(args, train_data, val_data, runtime_model_type="qwen3_vl")
    assert supported["validation_video_feature_cache_size"] == 341
    assert supported["validation_video_feature_cache_supported"] is True

    # Unresolved family fails safe: cache off, never a model-init abort.
    unknown = MODULE._framework_video_runtime(args, train_data, val_data)
    assert unknown["validation_video_feature_cache_size"] == 0
    assert unknown["validation_video_feature_cache_supported"] is False


def test_framework_explicit_feature_cache_rejected_for_unsupported_model_type() -> None:
    """An explicit request on an unsupported family fails at plan time.

    Silently zeroing a user's explicit size would hide the incompatibility;
    raising here replaces a post-distributed-init ChildFailedError with an
    actionable planner error. NVBUG 6669758.
    """
    args = SimpleNamespace(
        dataset_family="video_conversation",
        run_mode="full",
        nodes=1,
        gpus_per_node=4,
        validation_batch_size=1,
        framework_validation_shard_strategy="auto",
        framework_validation_video_feature_cache_size=512,
        framework_validation_processed_video_cache_size=None,
        framework_validation_cache_frontload_unique_per_batch=None,
        framework_baked_overlay_pythonpath="",
        framework_video_cache_size=None,
        framework_sft_process_threads=0,
        framework_video_decoder_threads=0,
        framework_dataloader_num_workers=None,
        framework_dataloader_prefetch_factor=None,
    )
    with pytest.raises(MODULE.WorkflowError, match="qwen3_vl"):
        MODULE._framework_video_runtime(
            args,
            train_data={"record_count": 5555, "profile": {"unique_media_count": 341}},
            val_data={"record_count": 2676, "profile": {"unique_media_count": 171}},
            runtime_model_type="cosmos3_edge",
        )
