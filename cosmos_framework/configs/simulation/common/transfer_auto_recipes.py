# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Transfer-Auto's four-phase curriculum on paired local RGB/WSM/LiDAR data.

Separate-sensor training is followed by short joint training, long joint training,
and a long-joint continuation. All joint phases use the released attention policy.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from cosmos_framework.callbacks.stage_handoff import ResetNewStageLearningRate
from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.generator.local_simulation.dataloader import (
    LocalSimulationDataLoader,
    LocalSimulationJointDataLoader,
)
from cosmos_framework.data.generator.local_simulation.dataset import LocalSimulationDataset, load_bundle_tokenizer
from cosmos_framework.data.generator.local_simulation.resume import LocalSimulationResumeCallback
from cosmos_framework.model.generator.tokenizers.lidar.lidar_tokenizer_v1 import (
    DEFAULT_NETWORK_CONFIG,
    LidarTokenizerV1Interface,
)
from cosmos_framework.model.generator.tokenizers.lidar.range_projection import V1P2_TRANSFER_RANGE_PROJECTION
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

AUTO_BUNDLE = "${oc.env:COSMOS_TRANSFER_AUTO_LOCAL_BUNDLE}"


def _auto_stream(resolution: str, ratio: float, *, phase: int, task: str) -> dict[str, Any]:
    overrides: dict[str, Any] = dict(
        conditioning_config={"__mapping__": [[0, 0.5], [1, 0.5]]},
        cfg_dropout_rate=0.1,
        first_caption_paragraph_only=phase == 2 or task == "lidar",
        joint_cam_lidar_transfer=phase >= 2,
        lidar_transfer=task == "lidar",
        load_lidar=phase >= 2 or task == "lidar",
        num_video_frames=65 if phase == 2 else 21 if task == "lidar" else 400,
        joint_cam_lidar_fixed_camera_frames=61 if phase == 2 else None,
        joint_aligned_num_video_frames=None,
        lidar_native_caption_window=False if task == "lidar" else True,
        lidar_dropout_rate=0.1 if phase >= 3 else 0.0,
        lidar_prefix_dropout_rate=0.0,
        control_dropout_rate=0.0,
        lidar_conditioning_probability=0.0,
        rgb_view_completion_probability=0.0,
        per_view_caption_dropout_rate=0.1 if phase >= 2 else 0.0,
        separate_view_text_tokenization=phase >= 2,
        add_view_prefix_to_caption=task != "lidar",
    )
    if phase == 1:
        overrides.update(camera_sampler=None, num_cameras_to_keep=1 if task != "lidar" else None)
    if task == "lidar":
        overrides.update(camera_keys=[], single_caption_camera_name=None, resolution_hw=[128, 1808])
    return dict(
        ratio=ratio,
        dataloader=L(LocalSimulationDataLoader)(
            dataset=L(LocalSimulationDataset)(
                bundle_path=AUTO_BUNDLE,
                model="auto",
                stratum=None if task == "lidar" else resolution,
                seed=42,
                transform_overrides=overrides,
            ),
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            seed=42,
        ),
    )


def make_transfer_auto_experiment(phase: int) -> LazyDict:
    """Build a fresh config; callers can shorten native stop/save iterations."""
    if phase not in (1, 2, 3, 4):
        raise ValueError("Transfer-Auto phase must be 1, 2, 3, or 4")
    name = f"transfer_auto_phase{phase}"
    model = deepcopy(NANO_MODEL_CONFIG)
    model.update(
        action_gen=False,
        sound_gen=False,
        video_temporal_causal=False,
        joint_attn_implementation="multiview" if phase >= 2 else "two_way",
        max_num_tokens_after_packing=240_000 if phase == 1 else 300_000 if phase == 2 else 1_500_000,
        lidar_state_ch=128,
        lidar_fps=10.0,
    )
    model["vlm_config"]["pretrained_weights"] = dict(
        enabled=False, backbone_path="", credentials_path="", enable_gcs_patch_in_boto3=False
    )
    model["vlm_config"]["tokenizer"] = L(load_bundle_tokenizer)(directory=f"{AUTO_BUNDLE}/resources/tokenizer")
    model["diffusion_expert_config"].update(
        num_view_embeddings=12, lidar_patch_spatial_hw=1, base_fps=30, load_weights_from_pretrained=False
    )
    model["rectified_flow_training_config"].update(
        loss_reduction_mode="local_item_mean",
        shift_lidar=5,
        lidar_loss_scale=1.0 if phase == 1 else 2.0,
        normalize_loss_by_active=False,
    )
    model["tokenizer"].update(vae_path="${oc.env:COSMOS_TRANSFER_AUTO_VIDEO_VAE}", encode_exact_durations=[17, 61, 73])
    model["ema"].update(iteration_shift=0 if phase == 1 else 70_000)
    model["parallelism"].update(data_parallel_shard_degree=32, context_parallel_shard_degree=32)
    model["activation_checkpointing"] = dict(mode="selective")
    if phase >= 2:
        model["activation_checkpointing"]["save_only_marked_ops"] = True
    network = deepcopy(DEFAULT_NETWORK_CONFIG)
    network.update(
        resolution=[128, 1808],
        patch_size=[2, 2],
        depths=[3, 3, 3, 3],
        num_heads=[4, 4, 8, 8],
        dilation=[1, 1, 1, 1],
        temporal_downsample=[False, False, False],
        temporal_upsample=[False, False, False],
    )
    model["lidar_tokenizer"] = L(LidarTokenizerV1Interface)(
        vae_path="${oc.env:COSMOS_TRANSFER_AUTO_LIDAR_VAE}",
        network_config=network,
        dtype="bfloat16",
        sample_posterior=False,
        apply_validity_mask=True,
        spatial_compression_factor=16,
        spatial_compression=[16, 16],
        temporal_compression_factor=1,
        range_projection=V1P2_TRANSFER_RANGE_PROJECTION.to_dict(),
        streaming_context_frames=21,
        streaming_chunk_frames=20 if phase >= 3 else 21,
    )
    if phase >= 2:
        # The maskless backend deduplicates cross-view keys. Use the same
        # preceding-0.4-second visibility window throughout joint training.
        model["multiview_attention"] = dict(
            backend="maskless",
            mask=dict(
                attention_scope="decomposed",
                control_attends_sensor=True,
                cross_view_temporal_window=dict(bounds=[-0.4, 0.0], includes_first_frame=False),
            ),
        )
    streams: dict[str, Any] = {}
    if phase == 1:
        for resolution in ("480", "720"):
            streams[f"rgb_{resolution}"] = _auto_stream(resolution, 0.15, phase=phase, task="rgb")
        streams["lidar"] = _auto_stream("480", 0.70, phase=phase, task="lidar")
    else:
        for resolution, ratio in (("480", 0.6), ("720", 0.4)):
            streams[f"joint_{resolution}"] = _auto_stream(resolution, ratio, phase=phase, task="joint")
    source = (
        "${oc.env:COSMOS_TRANSFER_AUTO_NANO_DCP}"
        if phase == 1
        else "${oc.env:COSMOS_TRANSFER_AUTO_PHASE" + str(phase - 1) + "_CHECKPOINT}"
    )
    schedule = (
        dict(f_start=[1e-6], f_max=[0.8], f_min=[0.1], warm_up_steps=[2000], cycle_lengths=[70_000])
        if phase == 1
        else dict(f_start=[0.1], f_max=[0.4], f_min=[0.1], warm_up_steps=[1000], cycle_lengths=[50_000])
    )
    handoff_callbacks = {"stage_initial_lr": L(ResetNewStageLearningRate)()} if phase == 2 else {}
    return LazyDict(
        dict(
            defaults=[
                {"override /model": "mot_fsdp"},
                {"override /data_train": None},
                {"override /data_val": None},
                {"override /optimizer": "adamw"},
                {"override /scheduler": "lambdalinear"},
                {"override /checkpoint": "local"},
                {"override /callbacks": ["basic", "optimization", "job_monitor"]},
                {"override /ema": "power"},
                {"override /tokenizer": "wan2pt2_tokenizer"},
                {"override /sound_tokenizer": None},
                {"override /vlm_config": None},
                {"override /ckpt_type": "dcp"},
                "_self_",
            ],
            job=dict(project="cosmos3", group="transfer_auto", name=name, wandb_mode="disabled"),
            model=dict(config=model),
            model_parallel=dict(context_parallel_size=32),
            optimizer=dict(
                optimizer_type="AdamW",
                lr=1e-4,
                betas=[0.9, 0.99],
                eps=1e-8,
                weight_decay=0.05,
                fused=True,
                keys_to_select=[
                    "moe_gen",
                    "time_embedder",
                    "vae2llm",
                    "llm2vae",
                    "lidar2llm",
                    "llm2lidar",
                    "rig_view_embed",
                ],
            ),
            scheduler=dict(lr_scheduler_type="LambdaLinear", **schedule),
            checkpoint=dict(
                load_path=source,
                load_training_state=phase != 1,
                load_ema_to_reg=False,
                strict_resume=True,
                keys_to_skip_loading=["lidar2llm", "llm2lidar", "rig_view_embed", "net_ema."] if phase == 1 else [],
                keys_not_to_resume=[]
                if phase == 1
                else ["dataloader", "scheduler", "trainer"]
                if phase == 2
                else ["dataloader"],
                save_iter=2000 if phase == 1 else 500,
            ),
            trainer=dict(
                max_iter={1: 70_000, 2: 10_000, 3: 22_000, 4: 50_000}[phase],
                grad_accum_iter=1,
                seed=0,
                logging_iter=1,
                run_validation=False,
                callbacks=dict(
                    grad_clip=dict(clip_norm=1.0, force_finite=True),
                    local_simulation_resume=L(LocalSimulationResumeCallback)(),
                    **handoff_callbacks,
                ),
            ),
            dataloader_train=L(LocalSimulationJointDataLoader)(
                restart_data_iterator_on_handoff=phase >= 3,
                dataloaders=streams,
                tokenizer_spatial_compression_factor=16,
                tokenizer_temporal_compression_factor=4,
                patch_spatial=2,
                lidar_spatial_compression=[16, 16],
                lidar_temporal_compression_factor=1,
                lidar_patch_spatial_hw=1,
                max_sequence_length=None,
                max_samples_per_batch=1,
                prewarm=False,
                default_lookahead_limit=1,
                seed=0,
            ),
            dataloader_val=None,
        ),
        flags={"allow_objects": True},
    )
