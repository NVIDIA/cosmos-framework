# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Sim-Auto TF → TF-dCM → joint SF-DMD with explicit captured MADS inputs.

The local adapter never resolves a remote catalog. A complete MADS export can
use the same data contract; a 100-sample QA bundle is not the original corpus.
"""

from __future__ import annotations

import copy
import json
from importlib.resources import files

from cosmos_framework.callbacks.compile_tokenizer import CompileTokenizer
from cosmos_framework.callbacks.sim_auto_phase_metrics import PhaseMetrics
from cosmos_framework.callbacks.wall_clock_checkpoint import WallClockCheckpoint
from cosmos_framework.configs.base.defaults.reasoner import Cosmos3NanoReasoner_VLM_GCP_Config_0517
from cosmos_framework.configs.simulation.common.auto_rigs import RIG_CATALOGS
from cosmos_framework.data.generator.local_simulation.dataloader import (
    LocalSimulationDataLoader,
    LocalSimulationJointDataLoader,
)
from cosmos_framework.data.generator.local_simulation.dataset import LocalSimulationDataset
from cosmos_framework.data.generator.local_simulation.resume import LocalSimulationResumeCallback
from cosmos_framework.data.generator.multiview.multiview_data_source import (
    MADS_CAMERAS,
    MADS_V1_400_ALL_CAPTION_KEY_MAPPING,
    MADS_V1_CONTROL_KEY_MAPPING,
)
from cosmos_framework.model.generator.distillation.self_forcing_dmd_joint import JointSelfForcingDMDModel
from cosmos_framework.model.generator.sim_auto_models import SimAutoTeacherForcingDCMModel
from cosmos_framework.model.generator.tokenizers.lidar.lidar_tokenizer_v1 import LidarTokenizerV1Interface
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

EXPERIMENT_NAMES = {
    1: "sim_auto_joint_stage1_tf_all",
    2: "sim_auto_joint_stage2_tfdcm_all",
    3: "sim_auto_joint_stage3_sfdmd_all",
}
KEYS = ["moe_gen", "time_embedder", "vae2llm", "llm2vae", "lidar2llm", "llm2lidar", "rig_view_embed"]
SIGMAS = [1.0, 0.9375, 0.8333333333333334, 0.625]
NEGATIVE_PROMPT = json.dumps(
    json.loads(files("cosmos_framework").joinpath("inference/defaults/neg_prompts.json").read_text())
)
BASE = "${oc.env:SIM_AUTO_STAGE0_CHECKPOINT}"
STAGE1 = "${oc.env:SIM_AUTO_STAGE1_CHECKPOINT}"
STAGE2_MODEL = "${oc.env:SIM_AUTO_STAGE2_CHECKPOINT_MODEL}"
TEACHER_MODEL = "${oc.env:SIM_AUTO_STAGE3_TEACHER_MODEL}"
_NETWORK = dict(
    base_channels=128,
    bottleneck_3d=True,
    bottleneck_3d_causal_time=True,
    bottleneck_3d_max_t=32,
    bottleneck_3d_rope=True,
    depths=[3, 3, 3, 3],
    dilation=[1, 1, 1, 1],
    formulation="VAE",
    in_channels=3,
    mapping_depth=2,
    mask_as_input=False,
    mlp_ratio=3.0,
    num_heads=[4, 4, 8, 8],
    out_channels=3,
    patch_size=[2, 2],
    positional_embedding="learnable_embedding",
    resolution=[128, 1808],
    temporal_downsample=[False, False, False],
    temporal_upsample=[False, False, False],
    window_size=[5, 45],
    z_dim=128,
)
_PROJECTION = dict(semantic_width=1800, model_width=1808, model_width_transform="circular_pad")


def _dataset(stage: int, resolution: str) -> LazyDict:
    # All transform changes are explicit overlays of a captured descriptor.
    # No source URI or catalog is opened by LocalSimulationDataset.
    return L(LocalSimulationDataset)(
        bundle_path="${oc.env:COSMOS_SIM_AUTO_LOCAL_BUNDLE}",
        model="auto",
        stratum=resolution,
        seed=42,
        _recursive_=False,
        transform_overrides=dict(
            num_video_frames=96 if stage == 3 else 400,
            joint_aligned_num_video_frames=93 if stage == 3 else None,
            joint_cam_lidar_transfer=True,
            camera_keys=list(MADS_CAMERAS),
            camera_caption_key_mapping=dict(MADS_V1_400_ALL_CAPTION_KEY_MAPPING),
            camera_control_key_mapping=dict(MADS_V1_CONTROL_KEY_MAPPING),
            camera_sampler=dict(
                _target_="cosmos_framework.data.generator.multiview.rig_sampler.ExplicitRigSampler",
                rigs=copy.deepcopy(RIG_CATALOGS[resolution]["rigs"]),
                weights=list(RIG_CATALOGS[resolution]["weights"]),
                seed=None,
            ),
            cfg_dropout_rate=0.2,
            per_view_caption_dropout_rate=0.2,
            separate_view_text_tokenization=True,
            add_view_prefix_to_caption=True,
            single_caption_camera_name=None,
            single_caption_probability=1.0,
            append_duration_fps_timestamps=True,
            append_resolution_info=True,
            fps_downsample_factor=1,
            lidar_fps=10.0,
            enable_per_camera_vae_encoding=True,
            lidar_dropout_rate=0.1,
            lidar_prefix_dropout_rate=0.0,
            control_dropout_rate=0.0,
            conditioning_config={0: 0.7, 1: 0.2, 2: 0.1},
            rgb_view_completion_probability=0.0,
            lidar_conditioning_probability=0.0,
        ),
    )


def _loader(stage: int) -> LazyDict:
    streams = {}
    for resolution, ratio in (("480", 0.6), ("720", 0.4)):
        streams[f"cam_lidar_joint_{resolution}p_separate_view_text"] = dict(
            ratio=ratio,
            dataloader=L(LocalSimulationDataLoader)(
                dataset=_dataset(stage, resolution),
                batch_size=1,
                num_workers=0,
                prefetch_factor=1,
                pin_memory=False,
                shuffle=True,
                seed=42,
            ),
        )
    return L(LocalSimulationJointDataLoader)(
        _recursive_=False,
        dataloaders=streams,
        tokenizer_spatial_compression_factor="${model.config.tokenizer.spatial_compression_factor}",
        tokenizer_temporal_compression_factor="${model.config.tokenizer.temporal_compression_factor}",
        patch_spatial="${model.config.diffusion_expert_config.patch_spatial}",
        lidar_spatial_compression="${model.config.lidar_tokenizer.spatial_compression}",
        lidar_temporal_compression_factor="${model.config.lidar_tokenizer.temporal_compression_factor}",
        lidar_patch_spatial_hw=1,
        max_sequence_length=1_600_000,
        max_samples_per_batch=1 if stage == 3 else None,
        lazy_initialize_child_iterators=True,
        prewarm=False,
        prewarm_concurrency=1,
        enable_async_batch_building=True,
        async_batch_building_timeout_s=1800.0,
    )


def make_auto_experiment(stage: int) -> LazyDict:
    if stage not in EXPERIMENT_NAMES:
        raise ValueError(f"Unsupported Sim-Auto stage {stage}")
    method = {1: "mot_fsdp_causal", 2: "mot_fsdp_tf_dcm", 3: "mot_fsdp_sf_dmd"}[stage]
    callbacks = (
        ["basic", "optimization", "job_monitor"]
        if stage != 3
        else ["basic_cosmos3_distill", "optimization_cosmos3_distill", "job_monitor_cosmos3_distill"]
    )
    defaults = [
        {"override /data_train": None},
        {"override /data_val": None},
        {"override /model": method},
        {"override /optimizer": "dmd2" if stage == 3 else "fusedadamw"},
        {"override /scheduler": "constant" if stage == 3 else "lambdalinear"},
        {"override /tokenizer": "wan2pt2_tokenizer"},
        {"override /vlm_config": "cosmos3_nano_reasoner_vlm_gcp_0517"},
        {"override /checkpoint": "local"},
        {"override /callbacks": callbacks},
    ]
    if stage == 3:
        defaults.append({"override /ckpt_type": "dcp_distill_cosmos3"})
    defaults.append("_self_")
    model = dict(
        video_temporal_causal=True,
        joint_attn_implementation="multiview",
        resolution="720",
        state_ch=48,
        lidar_state_ch=128,
        lidar_fps=10.0,
        state_t=300,
        action_gen=False,
        sound_gen=False,
        correct_cp_gradients=True,
        exclude_reasoner_weights_from_checkpoint=False,
        grad_clip=stage == 3,
        max_num_tokens_after_packing=1_600_000,
        causal_training_strategy="teacher_forcing_dcm" if stage == 2 else "teacher_forcing",
        teacher_forcing_detach_clean_kv=False,
        teacher_forcing_frames_per_chunk=3,
        teacher_forcing_kv_implementation="multiview_maskless_kv",
        teacher_forcing_replay_policy=dict(
            control_visibility="causal",
            controls_read_strict_past_clean_rgb=True,
            clean_pass_causality="chunk",
            multiview_attention_scope="decomposed",
            decomposed_temporal_window_seconds=0.4,
            deduplicate_cross_view=True,
        ),
        multiview_attention=dict(
            backend="maskless",
            mask=dict(
                attention_scope="decomposed",
                cross_view_temporal_window=dict(bounds=[-0.4, 0.0], includes_first_frame=False),
                control_attends_sensor=True,
            ),
        ),
        diffusion_expert_config=dict(
            patch_spatial=2,
            lidar_patch_spatial_hw=1,
            num_view_embeddings=12,
            unified_3d_mrope_reset_spatial_ids=True,
            unified_3d_mrope_temporal_modality_margin=15000,
            enable_fps_modulation=True,
            base_fps=30,
            load_weights_from_pretrained=False,
        ),
        rectified_flow_training_config=dict(
            shift={"480": 5, "720": 10},
            train_time_weight="uniform",
            train_time_video_distribution="uniform" if stage == 3 else "waver",
            loss_scale=1.0 if stage == 1 else 100.0,
            lidar_loss_scale=2.0 if stage == 1 else 200.0,
            exclude_fully_conditioned_items=False,
            loss_reduction_mode="local_item_mean",
        ),
        parallelism=dict(
            context_parallel_shard_degree=32,
            data_parallel_shard_degree=64,
            fsdp_master_dtype="float32",
            fsdp_reduce_dtype="bfloat16",
        ),
        activation_checkpointing=dict(mode="full"),
        compile=dict(enabled=stage != 3, mark_unbacked=False),
        ema=dict(enabled=True, rate=0.1, iteration_shift=0),
        tokenizer=dict(
            bucket_name="",
            vae_path="${oc.env:RGB_VAE}",
            object_store_credential_path_pretrained=None,
            encode_chunk_frames={"256": 68, "480": 24, "720": 8},
        ),
        lidar_tokenizer=L(LidarTokenizerV1Interface)(
            vae_path="${oc.env:LIDAR_VAE}",
            bucket_name="",
            object_store_credential_path_pretrained=None,
            network_config=copy.deepcopy(_NETWORK),
            range_projection=copy.deepcopy(_PROJECTION),
            spatial_compression_factor=16,
            spatial_compression=[16, 16],
            temporal_compression_factor=1,
            streaming_context_frames=21,
            streaming_chunk_frames=20,
            dtype="float32",
        ),
        vlm_config=dict(
            use_system_prompt=False,
            pretrained_weights=dict(enabled=False),
            tokenizer=dict(config_variant="hf", pretrained_model_name="${oc.env:QWEN_TOKENIZER}"),
        ),
    )
    optimizer = dict(
        lr=1e-5 if stage == 1 else 2e-6,
        betas=[0.9, 0.999] if stage == 1 else [0.0, 0.999],
        weight_decay=0.01,
        eps=1e-8,
        keys_to_select=list(KEYS),
    )
    scheduler = dict(f_start=[1.0], f_max=[1.0], f_min=[1.0], warm_up_steps=[0], cycle_lengths=["${trainer.max_iter}"])
    checkpoint = dict(
        load_path=BASE if stage == 1 else STAGE1 if stage == 2 else "",
        load_training_state=False,
        load_ema_to_reg=False,
        strict_resume=True,
        keys_to_skip_loading=[],
        keys_not_to_resume=[],
        save_iter=100,
        dcp_async_mode_enabled=True,
    )
    if stage == 2:
        model.update(
            teacher_forcing_dcm_total_steps=48,
            teacher_forcing_dcm_skipping_interval_steps=1,
            teacher_forcing_dcm_sample_timestep_shift=None,
            teacher_forcing_dcm_detach_target_endpoint=True,
            teacher_forcing_dcm_teacher_guidance=1.5,
            teacher_forcing_dcm_teacher_checkpoint_prefix="net",
            teacher_forcing_dcm_teacher_load_from=dict(load_path=f"{STAGE1}/model", credentials=None),
            teacher_forcing_dcm_teacher_negative_prompt=NEGATIVE_PROMPT,
            teacher_forcing_dcm_teacher_negative_prompt_keep_metadata=True,
            fixed_step_sampler_config=dict(t_list=list(SIGMAS), sample_type="sde"),
        )
    if stage == 3:
        model.update(
            simulation_mode="self_forcing",
            joint_sensor_modalities=("vision", "lidar"),
            strict_regular_checkpoint_loading=True,
            teacher_negative_prompt_per_view_metadata=True,
            sf_replayed_training=True,
            sf_replay_block_size=1,
            sf_require_equal_rollout_lengths=True,
            backward_grad_steps=1,
            sf_simulation_steps_per_frame=[4],
            sf_backward_timestep_schedules=[SIGMAS[1:]],
            sf_rollout_step_mode="max",
            fixed_step_sampler_config=dict(t_list=list(SIGMAS), sample_type="sde"),
            load_teacher_weights=True,
            student_load_from=dict(load_path=STAGE2_MODEL, credentials=None),
            teacher_load_from=dict(load_path=TEACHER_MODEL, credentials=None),
            teacher_guidance=1.5,
            teacher_negative_prompt=NEGATIVE_PROMPT,
            teacher_negative_prompt_keep_metadata=True,
            loss_scale_sid=1.0,
            loss_scale_fake_score=1.0,
            student_update_freq=6,
            fake_score_compile=dict(enabled=True, mark_unbacked=False),
            vlm_config_teacher=copy.deepcopy(Cosmos3NanoReasoner_VLM_GCP_Config_0517),
            vlm_config_fake_score=copy.deepcopy(Cosmos3NanoReasoner_VLM_GCP_Config_0517),
        )
        for name in ("vlm_config_teacher", "vlm_config_fake_score"):
            model[name].pretrained_weights.enabled = False
            model[name].use_system_prompt = False
        optimizer = {
            name: dict(lr=lr, betas=[0.0, 0.999], weight_decay=0.01, eps=1e-8, keys_to_select=list(KEYS))
            for name, lr in (("net", 1e-6), ("fake_score", 2e-7))
        }
        scheduler = {}
    experiment = LazyDict(
        dict(
            defaults=defaults,
            job=dict(project="cosmos3_interactive", group="sim_auto", name=f"stage{stage}_seed0"),
            trainer=dict(
                max_iter=4000,
                grad_accum_iter=1,
                logging_iter=10,
                seed=0,
                callbacks=dict(
                    grad_clip=dict(clip_norm=1.0),
                    compile_tokenizer=L(CompileTokenizer)(enabled=False),
                    local_simulation_resume=L(LocalSimulationResumeCallback)(),
                    wall_clock_checkpoint=L(WallClockCheckpoint)(interval_minutes=0, keep_last=0),
                ),
            ),
            model_parallel=dict(context_parallel_size=32),
            model=dict(config=model),
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint=checkpoint,
            dataloader_train=_loader(stage),
        ),
        flags={"allow_objects": True},
    )
    if stage == 2:
        experiment["model"]["_target_"] = SimAutoTeacherForcingDCMModel
    elif stage == 3:
        experiment["model"]["_target_"] = JointSelfForcingDMDModel
        experiment["trainer"]["callbacks"]["dmd2_parity"] = L(PhaseMetrics)(
            output_path="${oc.env:IMAGINAIRE_OUTPUT_ROOT}/${job.project}/${job.group}/${job.name}/phase_metrics.jsonl"
        )
    return experiment
