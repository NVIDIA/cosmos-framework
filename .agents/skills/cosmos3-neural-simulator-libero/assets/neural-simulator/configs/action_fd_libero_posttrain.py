# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_fd_libero_posttrain`` — LIBERO-10 forward-dynamics (neural simulator) post-training.

Adapted from ``action_fd_droid_posttrain``: same model config, optimizer, scheduler and
checkpoint settings. Deliberate deltas: the data source (below); single-GPU monitoring
cadence (device_monitor / iter_speed every 10 iterations); ``straggler_detection`` off (needs
an internal package); ``compile_tokenizer`` off (inductor AOT compile of the VAE needs a CUDA
toolkit); 24 dataloader workers with within-shard window shuffling (batch diversity on one GPU).
``LIBEROLeRobotDataset`` in ``mode="forward_dynamics"`` supplies concat_view
(third-person + wrist, 256x256 each -> 256x512, snapped to a 320x192 canvas) 17-frame
windows with 16 frame-wise-relative rot6d actions (10-D, ``quantile_rot`` normalized
with the bundled stats). The model conditions on frame 0 plus the 16 actions and
predicts frames 1..16.

Held-out episodes are pinned by an explicit id list (``LIBERO_HOLDOUT_PATH``); they never
enter training. See docs/action_fd_libero_posttrain.md.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_libero_sft_dataset
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


def _make_model_config() -> dict:
    """Identical to ``action_fd_droid_posttrain._make_model_config``."""
    cfg = copy.deepcopy(NANO_MODEL_CONFIG)

    cfg["sound_gen"] = False
    cfg["sound_dim"] = 64
    cfg["sound_latent_fps"] = 25
    cfg["max_num_tokens_after_packing"] = 74000
    cfg["resolution"] = "720"
    cfg["activation_checkpointing"]["mode"] = "selective"

    cfg["tokenizer"]["encode_exact_durations"] = [17]  # chunk_length 16 + 1 conditioning frame

    cfg["diffusion_expert_config"].update(
        base_fps=24,
        enable_fps_modulation=True,
        load_weights_from_pretrained=False,
        patch_spatial=2,
        unified_3d_mrope_temporal_modality_margin=15000,
        unified_3d_mrope_reset_spatial_ids=True,
    )
    cfg["rectified_flow_training_config"].update(
        image_loss_scale=None,
        loss_scale=10.0,
        shift={"256": 3, "480": 5, "720": 10},
        sound_loss_scale=2.0,
        train_time_video_distribution="waver",
        train_time_weight="uniform",
        use_discrete_rf=False,
    )
    return cfg


action_fd_libero_posttrain = LazyDict(
    dict(
        defaults=[
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /model": "mot_fsdp"},
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /checkpoint": "gcp"},
            {"override /callbacks": ["basic", "optimization", "job_monitor", "training_stats"]},
            {"override /ema": "power"},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3_action_fd",
            group="action_sft",
            name="${now:%Y-%m-%d_%H-%M-%S}_action_fd_libero_posttrain",
            wandb_mode="disabled",
        ),
        model=dict(
            config=_make_model_config(),
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            lr=1.0e-04,
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            cycle_lengths=[20000],
            f_max=[0.4],
            f_min=[0.0],
            f_start=[0.0],
            lr_scheduler_type="LambdaLinear",
            verbosity_interval=0,
            warm_up_steps=[0],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=50,
            max_iter=20000,
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=100, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            straggler_detection=dict(enabled=False, report_freq=50),  # needs the internal straggler package
            callbacks=dict(
                device_monitor=dict(every_n=10, log_memory_detail=True, save_s3=False, step_size=1),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=10, hit_thres=5, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=200, gc_level=1, warm_up=1),
                norm_monitor=dict(every_n=100),
                param_count=dict(save_s3=False),
                sigma_loss_analysis=dict(every_n=500, every_n_viz=500, save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
                # LIBERO concat_view (256x512) resolves to the "256" tier (320x192 canvas).
                compile_tokenizer=dict(
                    enabled=False, warmup_resolutions=["256"]
                ),  # AOT compile fails here (no CUDA toolkit)
            ),
        ),
        checkpoint=dict(
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # Skip net_ema. (EMA warm-starts from net, see dcp.py). The base Cosmos3-Nano
            # action heads (DROID ee_pose: [pos_delta, rot6d_delta, gripper], same 10-D
            # layout as LIBERO frame_wise_relative rot6d) are loaded and fine-tuned, as in
            # the DROID FD recipe.
            keys_to_skip_loading=[
                "net_ema.",
            ],
            load_ema_to_reg=False,
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
            load_path="???",  # Cosmos3-Nano DCP dir; supply via TOML/env
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=250,
            strict_resume=True,
            verbose=True,
        ),
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="action_libero_fd",
            max_samples_per_batch=None,
            max_sequence_length="${model.config.max_num_tokens_after_packing}",
            patch_spatial="${model.config.diffusion_expert_config.patch_spatial}",
            sound_latent_fps="${model.config.sound_latent_fps}",
            tokenizer_spatial_compression_factor="${model.config.tokenizer.spatial_compression_factor}",
            tokenizer_temporal_compression_factor="${model.config.tokenizer.temporal_compression_factor}",
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=False,
                num_workers=24,  # 24 episode streams = DROID FD's 8 ranks x 3 workers
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=2,
                sampler=None,
                datasets=dict(
                    libero=dict(
                        ratio=1,
                        dataset=L(get_action_libero_sft_dataset)(
                            # Local LeRobot dir of the libero_10 suite (20 FPS nvidia/LIBERO_LeRobot_v3):
                            #   hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset \
                            #     --include 'libero_10/**' --local-dir <dir>   # LIBERO_ROOT=<dir>/libero_10
                            root="${oc.env:LIBERO_ROOT}",
                            fps=20,  # metadata only (loader reads native fps from info.json)
                            chunk_length=16,
                            image_size=256,  # concat_view -> 256x512 -> "256" tier canvas 320x192
                            mode="forward_dynamics",
                            camera_mode="concat_view",
                            action_space="frame_wise_relative",
                            rotation_space="6d",
                            pose_coordinate_frame="native",
                            action_normalization="quantile_rot",
                            split="train",
                            # Explicit held-out episode ids (never used for training). Required.
                            holdout_episodes_path="${oc.env:LIBERO_HOLDOUT_PATH}",
                            iterable_shuffle=True,
                            episode_shuffle_seed=42,
                            # Single GPU feeds ~1,300 windows per step from 24 streams; shuffling window
                            # order inside each worker shard keeps consecutive samples from different episodes.
                            shuffle_within_shard=True,
                            resolution=None,
                            max_action_dim="${model.config.max_action_dim}",
                            cfg_dropout_rate=0.1,
                            append_idle_frames=True,
                            idle_frames_dropout=0.05,
                            format_prompt_as_json=True,
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                        ),
                    ),
                ),
            ),
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


cs.store(
    group="experiment",
    package="_global_",
    name="action_fd_libero_posttrain",
    node=action_fd_libero_posttrain,
)
