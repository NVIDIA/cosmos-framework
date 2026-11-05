# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Bidirectional Sim-Transfer teacher recipe; no internal checkpoint/data paths."""

import copy

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.generator.joint_dataloader import IterativeJointDataLoader
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

STAGE0_EXPERIMENT_NAME = "sim_transfer_stage0_bidir"
BASE_CHECKPOINT = "${oc.env:COSMOS_SIM_TRANSFER_BASE_CHECKPOINT}"
# Resolution, child fetch size, selection weight, RGB-frame ceiling.
STAGE0_STREAMS = (("480", 2, 4, 201), ("720", 1, 2, 149))


def make_transfer_bidirectional_experiment() -> LazyDict:
    """Train a new whole-window teacher from base regular weights, not teacher EMA."""
    model = copy.deepcopy(NANO_MODEL_CONFIG)
    model.update(
        resolution="720",
        max_num_tokens_after_packing=74_000,
        sound_gen=True,
        sound_dim=64,
        action_gen=True,
        joint_attn_implementation="two_way",
        video_temporal_causal=False,
        causal_training_strategy="none",
        activation_checkpointing={"mode": "selective"},
    )
    model["diffusion_expert_config"].update(
        load_weights_from_pretrained=False,
        max_vae_latent_side_after_patchify=52,
    )
    model["rectified_flow_training_config"].update(
        shift={"256": 3, "480": 5, "720": 10},
        loss_scale=10.0,
        image_loss_scale=None,
        sound_loss_scale=2.0,
        normalize_loss_by_active=False,
    )
    model["parallelism"].update(
        data_parallel_shard_degree=8,
        context_parallel_shard_degree=1,
        cfg_parallel_shard_degree=1,
        fsdp_master_dtype="float32",
        fsdp_reduce_dtype="bfloat16",
    )
    model["tokenizer"].update(
        encode_chunk_frames={"256": 68, "480": 24, "720": 8},
        encode_exact_durations=[17, 61, 73],
    )
    # Use the same reasoner architecture as the causal descendants; the full
    # base DCP, rather than a second pretrained-backbone load, supplies weights.
    model["vlm_config"] = {"pretrained_weights": {"enabled": False}, "use_system_prompt": False}
    return LazyDict(
        dict(
            defaults=[
                {"override /data_train": None},
                {"override /data_val": None},
                {"override /model": "mot_fsdp"},
                {"override /optimizer": "fusedadamw"},
                {"override /scheduler": "lambdalinear"},
                {"override /tokenizer": "wan2pt2_tokenizer"},
                {"override /sound_tokenizer": "avae_48k_25hz"},
                {"override /vlm_config": "cosmos3_nano_reasoner_vlm_gcp_0517"},
                {"override /checkpoint": "local"},
                {"override /ema": "power"},
                {"override /callbacks": ["basic", "optimization", "job_monitor"]},
                "_self_",
            ],
            job=dict(project="cosmos3_simulation", group="cosmos3_transfer_teacher", name=STAGE0_EXPERIMENT_NAME),
            model=dict(config=model),
            optimizer=dict(
                lr=1e-4,
                betas=[0.9, 0.99],
                eps=1e-8,
                weight_decay=0.05,
                keys_to_select=[
                    "moe_gen",
                    "time_embedder",
                    "vae2llm",
                    "llm2vae",
                    "sound2llm",
                    "llm2sound",
                    "sound_modality_embed",
                    "action2llm",
                    "llm2action",
                    "action_modality_embed",
                ],
                lr_multipliers={"action2llm": 5.0, "llm2action": 5.0, "action_modality_embed": 5.0},
            ),
            scheduler=dict(
                warm_up_steps=[0],
                cycle_lengths=[100_000],
                f_start=[1e-6],
                f_max=[0.4],
                f_min=[0.0],
            ),
            trainer=dict(
                max_iter=150_000,
                logging_iter=50,
                seed=0,
                grad_accum_iter=1,
                callbacks=dict(
                    grad_clip=dict(clip_norm=1.0),
                    manual_gc=dict(every_n=200),
                    compile_tokenizer=dict(
                        enabled=True, compile_after_iterations=3, warmup_resolutions=["256", "480", "720"]
                    ),
                    wall_clock_checkpoint=dict(interval_minutes=30.0, keep_last=3),
                ),
            ),
            checkpoint=dict(
                load_path=BASE_CHECKPOINT,
                load_training_state=False,
                load_ema_to_reg=False,
                strict_resume=True,
                save_iter=250,
                dcp_async_mode_enabled=True,
            ),
            # No private source catalog is shipped. Supply real source loaders or
            # explicitly opt into the local Transfer-bundle config.
            dataloader_train=L(IterativeJointDataLoader)(
                dataloaders={
                    f"video_transfer_4modality_data_{resolution}": dict(
                        dataloader={
                            "_target_": "???",
                            "batch_size": batch,
                            "dataset": {
                                "resolution": resolution,
                                "max_num_frames": frames,
                                "min_num_frames": 5,
                                "cfg_dropout_rate": 0.1,
                                "dense_caption_probability": 0.07,
                                "conditioning_config": {0: 0.7, 1: 0.2, 2: 0.1},
                                "resize_on_read": False,
                                "modality_weights": {"edge": 1.0, "blur": 1.0, "depth": 1.0, "seg": 1.0},
                            },
                        },
                        ratio=ratio,
                    )
                    for resolution, batch, ratio, frames in STAGE0_STREAMS
                },
                tokenizer_spatial_compression_factor="${model.config.tokenizer.spatial_compression_factor}",
                tokenizer_temporal_compression_factor="${model.config.tokenizer.temporal_compression_factor}",
                patch_spatial="${model.config.diffusion_expert_config.patch_spatial}",
                max_sequence_length="${model.config.max_num_tokens_after_packing}",
                sound_latent_fps="${model.config.sound_latent_fps}",
                audio_sample_rate=48_000,
                lookahead_limits={
                    f"video_transfer_4modality_data_{resolution}": 2 for resolution, *_ in STAGE0_STREAMS
                },
            ),
        ),
        flags={"allow_objects": True},
    )
