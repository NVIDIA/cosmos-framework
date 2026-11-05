# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Stage-3 self-forcing DMD recipes for causal Transfer2 training and inference."""

import copy
from typing import Any

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.configs.base.defaults.reasoner import Cosmos3NanoReasoner_VLM_GCP_Config_0517
from cosmos_framework.configs.simulation.common.resolution import RECTIFIED_FLOW_SHIFT
from cosmos_framework.configs.simulation.common.transfer_distillation import (
    _TF_DCM_TEACHER_NEGATIVE_PROMPT,
)
from cosmos_framework.data.generator.indexed_transfer_length_loader import (
    IndexedTransferLengthDataLoader,
)



_TEMPORAL_COMPRESSION_FACTOR: int = 4
_TRANSFER_MAX_PIXEL_FRAMES_BY_RESOLUTION: dict[str, int] = {"480": 201, "720": 81}
_TRANSFER_MAX_LATENT_FRAMES_BY_RESOLUTION: dict[str, int] = {"480": 51, "720": 21}
_TRANSFER_MIN_PIXEL_FRAMES: int = 9
_TRANSFER_TEACHER_FORCING_FRAMES_PER_CHUNK: int = 1
_TRANSFER_MODALITY_WEIGHTS: dict[str, float] = {"edge": 1.0, "blur": 1.0, "depth": 1.0, "seg": 1.0}
_TRANSFER_CONDITIONING_CONFIG: dict[int, float] = {0: 0.7, 1: 0.2, 2: 0.1}
_TRANSFER_STREAM_CONFIGS: tuple[tuple[str, int, int], ...] = (("480", 2, 4), ("720", 1, 2))
_TRANSFER_SHUFFLE_SEEDS: dict[str, int] = {"480": 340_480, "720": 340_720}
# One mixed-resolution model config serves both streams. Size the logical
# transfer window for the longer 480p stream; the 720p stream tops out at 21
# latent frames and therefore fits within the shared 51-frame capacity.
_STREAMING_KV_WINDOW_LATENT_FRAMES: int = _TRANSFER_MAX_LATENT_FRAMES_BY_RESOLUTION["480"]
# Transfer streaming interprets this as one logical temporal frame. The paired
# control/RGB cache implementation is responsible for retaining the whole pair.
_STREAMING_ATTENTION_SINK_LATENT_FRAMES: int = 1
_DATA_PARALLEL_SHARD_DEGREE: int = 16

_OPTIMIZER_KEYS: list[str] = ["moe_gen", "time_embedder", "vae2llm", "llm2vae"]
_FIXED_STEP_SAMPLER_CONFIG: dict[str, Any] = {
    "t_list": [1.0, 15 / 16, 5 / 6, 5 / 8],
    "sample_type": "sde",
}
_DISTILL_CALLBACKS: list[str] = [
    "basic_cosmos3_distill",
    "optimization_cosmos3_distill",
    "job_monitor_cosmos3_distill",
]
_DEFAULT_REPLACEMENTS: dict[str, Any] = {
    "override /model": "mot_fsdp_sf_dmd",
    "override /optimizer": "dmd2",
    "override /scheduler": "constant",
    "override /callbacks": _DISTILL_CALLBACKS,
}






def _make_indexed_transfer_dataloader_train(manifest_path: str) -> LazyDict:
    """Use one worker pool for all population-weighted latent horizons."""
    return L(IndexedTransferLengthDataLoader)(
        dataset={"_target_": "???"},
        tokenizer_spatial_compression_factor="${model.config.tokenizer.spatial_compression_factor}",
        tokenizer_temporal_compression_factor="${model.config.tokenizer.temporal_compression_factor}",
        patch_spatial="${model.config.diffusion_expert_config.patch_spatial}",
        num_workers=1,
        prefetch_factor=1,
        pin_memory=True,
    )


def _make_stage3_defaults(source_defaults: list[Any]) -> list[Any]:
    """Replace method defaults and add the multi-model distillation checkpointer."""
    defaults: list[Any] = []
    replaced: set[str] = set()
    inserted_ckpt_type = False
    for default in source_defaults:
        if default == "_self_":
            defaults.append({"override /ckpt_type": "dcp_distill_cosmos3"})
            defaults.append(copy.deepcopy(default))
            inserted_ckpt_type = True
            continue
        if isinstance(default, dict):
            matching_keys = set(default).intersection(_DEFAULT_REPLACEMENTS)
            if len(matching_keys) > 1:
                raise ValueError(f"Expected one Hydra default per entry, got replacements for {matching_keys}")
            if matching_keys:
                key = matching_keys.pop()
                defaults.append({key: copy.deepcopy(_DEFAULT_REPLACEMENTS[key])})
                replaced.add(key)
                continue
        defaults.append(copy.deepcopy(default))

    missing_replacements = set(_DEFAULT_REPLACEMENTS).difference(replaced)
    if missing_replacements:
        raise ValueError(f"Missing source Hydra defaults for {sorted(missing_replacements)}")
    if not inserted_ckpt_type:
        raise ValueError("Expected the source Hydra defaults to contain '_self_'")
    return defaults


def make_transfer_sf_dmd_experiment(
    base_experiment: dict[str, Any],
    *,
    name: str,
    student_checkpoint: dict[str, str],
    teacher_checkpoint: dict[str, str],
    stream_configs: tuple[tuple[str, int, int], ...] = _TRANSFER_STREAM_CONFIGS,
    job_group: str | None = None,
    length_index_manifest: str | None = None,
) -> dict[str, Any]:
    """Convert an aligned Transfer recipe into its Stage-3 SF-DMD recipe."""
    experiment = copy.deepcopy(base_experiment)
    experiment["model"]["config"]["teacher_forcing_target_only_no_text_pass2"] = False
    experiment["dataloader_train"] = (
        {"_target_": "???"}
        if length_index_manifest is None
        else _make_indexed_transfer_dataloader_train(length_index_manifest)
    )
    experiment["defaults"] = _make_stage3_defaults(experiment["defaults"])
    experiment["job"]["name"] = name
    if job_group is not None:
        experiment["job"]["group"] = job_group
    experiment["trainer"] = {
        "max_iter": 10_000,
        "logging_iter": 10,
        "callbacks": {
            "grad_clip": {"clip_norm": 1.0},
            "manual_gc": {"every_n": 200},
            # Reuse the Stage-1/2 VAE encode compiler. Distillation callback
            # defaults omit this callback, so supply its target explicitly.
            "compile_tokenizer": L("cosmos_framework.callbacks.compile_tokenizer.CompileTokenizer")(
                enabled=True,
                compile_after_iterations=3,
                warmup_resolutions=[resolution for resolution, _, _ in stream_configs],
            ),
        },
    }

    model_config = experiment["model"]["config"]
    model_config.update(
        {
            # Match the selected Stage-3 streams; mixed recipes retain their
            # historical maximum model resolution.
            "resolution": max((resolution for resolution, _, _ in stream_configs), key=int),
            "causal_training_strategy": "none",
            "simulation_mode": "self_forcing",
            "teacher_forcing_frames_per_chunk": _TRANSFER_TEACHER_FORCING_FRAMES_PER_CHUNK,
            # Replay one frame at a time for indexed 480p training: keeping all
            # frame graphs also retains their growing attention-history inputs.
            # Transfer conditioning prefixes are synchronized before rollout,
            # so every FSDP rank executes the same number of replay closures.
            "sf_replayed_training": length_index_manifest is not None,
            "sf_replay_block_size": 1,
            "sf_replay_include_conditioned_frames": False,
            "backward_grad_steps": 1,
            "sf_simulation_steps_per_frame": [4],
            "sf_backward_timestep_schedules": [[15 / 16, 5 / 6, 5 / 8]],
            "sf_rollout_step_mode": "max",
            "sf_require_equal_rollout_lengths": length_index_manifest is not None,
            # Repeated frame/denoising forwards reuse gathered student weights;
            # release them when rollout ends, before either scoring network runs.
            "sf_keep_student_fsdp_unsharded_during_rollout": length_index_manifest is not None,
            # As in Stage 2, keep frozen teacher parameters gathered across
            # the two CFG forwards, then release them before student backward.
            "sf_defer_teacher_fsdp_reshard": length_index_manifest is not None,
            "fixed_step_sampler_config": copy.deepcopy(_FIXED_STEP_SAMPLER_CONFIG),
            "load_teacher_weights": True,
            "loss_scale_fake_score": 1.0,
            "loss_scale_sid": 1.0,
            "student_update_freq": 6,
            "teacher_guidance": 6.0,
            # Match transfer inference and TF-dCM, retaining each sample's
            # duration, FPS, resolution, and transfer system prompt under CFG.
            "teacher_negative_prompt": _TF_DCM_TEACHER_NEGATIVE_PROMPT,
            "teacher_negative_prompt_keep_metadata": True,
            "teacher_load_from": copy.deepcopy(teacher_checkpoint),
            "student_load_from": copy.deepcopy(student_checkpoint),
            # The indexed 480p recipe reuses Stage 2's selective attention
            # checkpointing; mixed-resolution recipes retain full recompute.
            "activation_checkpointing": {"mode": "selective" if length_index_manifest is not None else "full"},
            "kv_cache_inference_size": _STREAMING_KV_WINDOW_LATENT_FRAMES,
            "attention_sink_size": _STREAMING_ATTENTION_SINK_LATENT_FRAMES,
            # Quantized KV storage is not supported for exact interleaved
            # Transfer control/RGB history.
            "kv_cache_dtype": None,
            "vlm_config_teacher": Cosmos3NanoReasoner_VLM_GCP_Config_0517,
            "vlm_config_fake_score": Cosmos3NanoReasoner_VLM_GCP_Config_0517,
        }
    )
    model_config["compile"]["enabled"] = False
    # Retain SF-DMD's larger shard group for its additional critic and rollout
    # graphs.
    model_config["parallelism"]["data_parallel_shard_degree"] = _DATA_PARALLEL_SHARD_DEGREE
    model_config["parallelism"]["context_parallel_shard_degree"] = 1
    model_config["parallelism"]["fsdp_reduce_dtype"] = None
    model_config["parallelism"]["vae_load_balance_group_size"] = 1
    model_config["rectified_flow_training_config"]["train_time_video_distribution"] = "uniform"
    model_config["rectified_flow_training_config"]["shift"] = (
        RECTIFIED_FLOW_SHIFT[stream_configs[0][0]] if len(stream_configs) == 1 else copy.deepcopy(RECTIFIED_FLOW_SHIFT)
    )

    experiment["optimizer"] = {
        "net": {
            "lr": 1e-6,
            "weight_decay": 0.01,
            "betas": [0.0, 0.999],
            "eps": 1e-8,
            "keys_to_select": copy.deepcopy(_OPTIMIZER_KEYS),
        },
        "fake_score": {
            "lr": 2e-7,
            "weight_decay": 0.01,
            "betas": [0.0, 0.999],
            "eps": 1e-8,
            "keys_to_select": copy.deepcopy(_OPTIMIZER_KEYS),
        },
    }
    experiment["scheduler"] = {}
    experiment["checkpoint"] = {
        "save_iter": 250,
        "load_from_object_store": {"enabled": False},
        # Bootstrap from the selected Stage-2 model weights above. A later
        # Stage-3 full-state resume is a distinct operation and
        # must not be silently substituted for this initialization.
        "load_training_state": False,
        "strict_resume": True,
    }
    return experiment
