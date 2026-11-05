# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Bimanual Stage 0--3 model and optimization contracts.

These builders expose the canonical single-view training workflow and require
caller-owned checkpoints and data instead of historical resumes.
The local-bundle entry point supplies data without changing these contracts.
"""

from __future__ import annotations

from typing import Any

from cosmos_framework.callbacks.initialize_ema import InitializeEmaFromRegular
from cosmos_framework.configs.base.defaults.reasoner import Cosmos3NanoReasoner_VLM_GCP_Config_0517
from cosmos_framework.utils.lazy_config import LazyCall as L

EXPERIMENT_NAMES: tuple[str, ...] = (
    "sim_bimanual_stage0_bidir",
    "sim_bimanual_stage1_tf",
    "sim_bimanual_stage2_tfdcm",
    "sim_bimanual_stage3_sfdmd",
)
LOCAL_EXPERIMENT_NAMES: tuple[str, ...] = tuple(
    name.replace("sim_bimanual_", "sim_local_bimanual_") for name in EXPERIMENT_NAMES
)
GA_CHECKPOINT = "${oc.env:COSMOS_SIM_BIMANUAL_GA_CHECKPOINT}"
STAGE0_CHECKPOINT = "${oc.env:COSMOS_SIM_BIMANUAL_STAGE0_CHECKPOINT}"
STAGE1_CHECKPOINT = "${oc.env:COSMOS_SIM_BIMANUAL_STAGE1_CHECKPOINT}"
STAGE2_CHECKPOINT = "${oc.env:COSMOS_SIM_BIMANUAL_STAGE2_CHECKPOINT}"
_CREDENTIAL = "${oc.env:CHECKPOINT_CREDENTIAL_FILE,null}"
_OPTIMIZER_KEYS: tuple[str, ...] = (
    "moe_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "action2llm",
    "action_modality_embed",
)
# Shift-5 schedule: preserve the source recipe's four-step SDE ladder.
# Samplers append the terminal zero; midpoint schedules must exclude it.
_T_LIST: tuple[float, ...] = (1.0, 0.9375, 0.8333333333333334, 0.625)
_MAX_ITERS: tuple[int, ...] = (35_000, 8_000, 4_000, 1_500)


def _checkpoint_source(root: str) -> dict[str, Any]:
    return {"load_path": f"{root}/model", "credentials": _CREDENTIAL}


def make_bimanual_experiment(stage: int) -> dict[str, Any]:
    """Build one stage; a concrete dataloader must be supplied before launch."""
    if stage not in range(4):
        raise ValueError("Bimanual stage must be 0, 1, 2, or 3.")
    distillation = stage == 3
    max_iter = _MAX_ITERS[stage]
    model_preset = ("mot_fsdp", "mot_fsdp_causal", "mot_fsdp_tf_dcm", "mot_fsdp_sf_dmd")[stage]
    model_config: dict[str, Any] = {
        "video_temporal_causal": stage != 0,
        "joint_attn_implementation": "two_way" if stage == 0 else "three_way",
        "causal_training_strategy": ("none", "teacher_forcing", "teacher_forcing_dcm", "none")[stage],
        "resolution": "480",
        "state_ch": 48,
        "latent_downsample_factor": 16,
        "state_t": 300,
        "max_num_tokens_after_packing": 400_000 if stage == 0 else 500_000,
        "action_gen": True,
        "action_schema": "unified_v1",
        "action_global_asinh": True,
        "action_metric_gripper": True,
        "sound_gen": False,
        "max_action_dim": 64,
        "num_embodiment_domains": 32,
        "rectified_flow_training_config": {
            "train_time_weight": "uniform",
            "train_time_video_distribution": "uniform" if distillation else "waver",
            "loss_scale": 100.0 if stage == 2 else 10.0,
            "use_discrete_rf": False,
            "shift": {"480": 5} if distillation else {"256": 3, "480": 5},
        },
        "diffusion_expert_config": {
            "patch_spatial": 2,
            "unified_3d_mrope_reset_spatial_ids": True,
            "unified_3d_mrope_temporal_modality_margin": 15_000,
            "enable_fps_modulation": True,
            "base_fps": 24,
        },
        "tokenizer": {
            "bucket_name": "",
            "object_store_credential_path_pretrained": "",
            "encode_chunk_frames": {"480": 24} if distillation else {"256": 68, "480": 24},
        },
        "parallelism": {"data_parallel_shard_degree": 8, "context_parallel_shard_degree": 4},
        "compile": {"enabled": False},
        "activation_checkpointing": {"mode": "full" if stage < 2 else "selective"},
        "vlm_config": {
            "pretrained_weights": {"enabled": False},
            "use_system_prompt": False,
            "tokenizer": {"config_variant": "hf"},
        },
    }
    if stage == 0:
        model_config["diffusion_expert_config"]["load_weights_from_pretrained"] = False
        model_config["rectified_flow_training_config"]["loss_reduction_mode"] = "local_item_mean"
    else:
        model_config.update(teacher_forcing_detach_clean_kv=False, teacher_forcing_frames_per_chunk=2)
    if stage == 2:
        model_config.update(
            teacher_forcing_dcm_total_steps=48,
            teacher_forcing_dcm_skipping_interval_steps=1,
            teacher_forcing_dcm_sample_timestep_shift=None,
            teacher_forcing_dcm_teacher_guidance=3.0,
            teacher_forcing_dcm_teacher_load_from=_checkpoint_source(STAGE1_CHECKPOINT),
            fixed_step_sampler_config={"t_list": list(_T_LIST), "sample_type": "sde"},
        )
    if distillation:
        model_config.update(
            vlm_config_teacher=Cosmos3NanoReasoner_VLM_GCP_Config_0517,
            vlm_config_fake_score=Cosmos3NanoReasoner_VLM_GCP_Config_0517,
            action_conditioned_teacher_guidance=1.0,
            simulation_mode="self_forcing",
            sf_keep_student_fsdp_unsharded_during_rollout=True,
            sf_replayed_training=True,
            sf_replay_block_size=1,
            sf_replay_include_conditioned_frames=False,
            backward_grad_steps=1,
            sf_simulation_steps_per_frame=[4],
            sf_backward_timestep_schedules=[list(_T_LIST[1:])],
            sf_rollout_step_mode="max",
            fixed_step_sampler_config={"t_list": list(_T_LIST), "sample_type": "sde"},
            load_teacher_weights=True,
            loss_scale_fake_score=1.0,
            loss_scale_sid=1.0,
            student_update_freq=6,
            warmup_student_steps=0,
            warmup_critic_steps=0,
            teacher_load_from=_checkpoint_source(STAGE0_CHECKPOINT),
            student_load_from=_checkpoint_source(STAGE2_CHECKPOINT),
        )
    callbacks: dict[str, Any] = {
        "grad_clip": {"clip_norm": 1.0},
        "manual_gc": {"every_n": 5 if distillation else 200, "warm_up": 5 if stage < 2 else 0},
    }
    if not distillation:
        callbacks["compile_tokenizer"] = {"enabled": stage < 2, "warmup_resolutions": ["256", "480"]}
    if stage == 0:
        callbacks["initialize_ema_from_regular"] = L(InitializeEmaFromRegular)(source_checkpoint=GA_CHECKPOINT)
    optimizer: dict[str, Any]
    if distillation:
        optimizer = {
            name: {
                "lr": lr,
                "weight_decay": 0.01,
                "betas": [0.0, 0.999],
                "eps": 1e-8,
                "keys_to_select": list(_OPTIMIZER_KEYS),
            }
            for name, lr in (("net", 1e-6), ("fake_score", 2e-7))
        }
    else:
        optimizer = {
            "lr": 2e-6 if stage == 2 else 1e-5,
            "betas": [0.9, 0.99] if stage == 0 else ([0.9, 0.999] if stage == 1 else [0.0, 0.999]),
            "weight_decay": 0.05 if stage == 0 else 0.01,
            "keys_to_select": list(_OPTIMIZER_KEYS),
        }
        if stage == 0:
            optimizer["lr_multipliers"] = {"action2llm": 5.0, "action_modality_embed": 5.0}
    checkpoint: dict[str, Any] = {
        "save_iter": 100,
        "load_training_state": False,
        "strict_resume": True,
    }
    if not distillation:
        checkpoint.update(
            load_path=(GA_CHECKPOINT, STAGE0_CHECKPOINT, STAGE1_CHECKPOINT)[stage],
            load_ema_to_reg=False,
            keys_not_to_resume=[],
            keys_to_skip_loading=(
                ["net_ema.", "action2llm", "llm2action", "action_modality_embed"] if stage == 0 else []
            ),
        )
    defaults: list[Any] = [
        {"override /data_train": None},
        {"override /data_val": None},
        {"override /model": model_preset},
        {"override /optimizer": "dmd2" if distillation else "fusedadamw"},
        {"override /scheduler": "constant" if distillation else "lambdalinear"},
        {"override /tokenizer": "wan2pt2_tokenizer"},
        {"override /sound_tokenizer": None},
        {"override /vlm_config": "cosmos3_nano_reasoner_vlm_gcp_0517"},
        {"override /checkpoint": "local"},
        {"override /ema": "power"},
        {"override /ckpt_type": "dcp_distill_cosmos3" if distillation else "dcp"},
        {
            "override /callbacks": (
                ["basic_cosmos3_distill", "optimization_cosmos3_distill", "job_monitor_cosmos3_distill"]
                if distillation
                else ["basic", "optimization", "job_monitor"]
            )
        },
        "_self_",
    ]
    return {
        "defaults": defaults,
        "job": {"project": "cosmos3_simulation", "group": "bimanual", "name": EXPERIMENT_NAMES[stage]},
        "trainer": {"max_iter": max_iter, "logging_iter": 10, "callbacks": callbacks},
        "model": {"config": model_config},
        "optimizer": optimizer,
        "scheduler": {}
        if distillation
        else {
            "f_max": [1.0],
            "f_min": [1.0],
            "warm_up_steps": [0],
            "cycle_lengths": [max_iter],
        },
        "checkpoint": checkpoint,
        "dataloader_train": {"_target_": "???"},
    }


def make_local_bimanual_experiment(stage: int) -> dict[str, Any]:
    """Supply the matching local raw-data pipeline without changing stage roles."""
    from cosmos_framework.data.generator.local_simulation.bimanual_config import make_bimanual_local_dataloader
    from cosmos_framework.data.generator.local_simulation.resume import LocalSimulationResumeCallback

    experiment = make_bimanual_experiment(stage)
    experiment["job"]["name"] = LOCAL_EXPERIMENT_NAMES[stage]
    # Model-side and dataloader-side caption IDs must use the captured vocabulary.
    experiment["model"]["config"]["vlm_config"]["tokenizer"]["pretrained_model_name"] = (
        "${oc.env:COSMOS_SIM_BIMANUAL_LOCAL_BUNDLE}/resources/tokenizer"
    )
    experiment["dataloader_train"] = L(make_bimanual_local_dataloader)(
        bundle_path="${oc.env:COSMOS_SIM_BIMANUAL_LOCAL_BUNDLE}",
        stage=stage,
        tokenizer_spatial_compression_factor="${model.config.tokenizer.spatial_compression_factor}",
        tokenizer_temporal_compression_factor="${model.config.tokenizer.temporal_compression_factor}",
        patch_spatial="${model.config.diffusion_expert_config.patch_spatial}",
        max_sequence_length="${model.config.max_num_tokens_after_packing}" if stage == 0 else None,
        max_samples_per_batch=None if stage == 0 else 1,
        seed=42,
        num_workers=0,
    )
    experiment["trainer"]["callbacks"]["local_simulation_resume"] = L(LocalSimulationResumeCallback)()
    return experiment
