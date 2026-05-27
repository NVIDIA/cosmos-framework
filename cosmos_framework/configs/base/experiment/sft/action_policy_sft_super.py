# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Action policy SFT on the LIBERO suite — "super" variant.

LoRA fine-tune on top of the Qwen3-VL-32B-Instruct backbone (CP=2 / DP=4 with
activation checkpointing). Only the LoRA adapters on the gen MoE attention
projections train; the rest of the backbone stays frozen.

Dataset roots are taken from ``$DATASET_PATH``; export it before launching so
that the four LIBERO suites (``libero_10`` / ``libero_object`` /
``libero_spatial`` / ``libero_goal``) resolve under their corresponding
subdirectories.

``checkpoint.load_path`` is a required override (Hydra ``???`` placeholder);
supply it on the CLI or in a downstream experiment that inherits this one.

Usage::

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. torchrun --nproc_per_node=8 \\
        --master_port=12341 -m cosmos_framework.scripts.train \\
        --config=configs/base/config.py -- \\
        experiment=action_policy_sft_super checkpoint.load_path=<path>
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

from cosmos_framework.configs.base.experiment.sft.models.super_model_config import SUPER_MODEL_CONFIG
from cosmos_framework.data.vfm.action.dataloaders import InfiniteDataLoader
from cosmos_framework.data.vfm.action.libero_dataset import LIBERODataset
from cosmos_framework.data.vfm.action.unified_dataset import dataset_entry, wrap_dataset
from cosmos_framework.data.vfm.joint_dataloader import IterativeJointDataLoader

cs = ConfigStore.instance()


def _action_policy_super_model_config() -> dict:
    """SUPER_MODEL_CONFIG baseline + action-policy-specific overrides.

    Mirrors action_policy_sft_nano (action_gen on, encode_exact_durations=
    [17, 61, 73], log_enc_time_every_n=50, policy-style loss scales) on top
    of the 32B / LoRA / DP=4 / CP=2 super baseline.
    """
    cfg = copy.deepcopy(SUPER_MODEL_CONFIG)
    cfg["action_gen"] = True
    cfg["log_enc_time_every_n"] = 50
    cfg["max_action_dim"] = 64
    cfg["max_num_tokens_after_packing"] = 16384
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    cfg["parallelism"]["use_torch_compile"] = True
    cfg["rectified_flow_training_config"]["loss_scale"] = 10.0
    cfg["rectified_flow_training_config"]["image_loss_scale"] = None
    cfg["tokenizer"]["encode_exact_durations"] = [17, 61, 73]
    return cfg


_LIBERO_DATASETS = [
    L(dataset_entry)(
        name="libero",
        dataset=L(LIBERODataset)(
            action_normalization="quantile_rot",
            action_space="frame_wise_relative",
            action_stats_path=(
                "cosmos_framework/data/vfm/action/normalizers/"
                "libero_native_frame_wise_relative_rot6d.json"
            ),
            camera_mode="concat_view",
            chunk_length=16,
            download_videos=False,
            embodiment_type="libero",
            force_cache_sync=False,
            fps=20,
            image_size=256,
            mode="policy",
            pose_coordinate_frame="native",
            repo_id=[
                "libero_10",
                "libero_object",
                "libero_spatial",
                "libero_goal",
            ],
            root=[
                "${oc.env:DATASET_PATH}/libero_10",
                "${oc.env:DATASET_PATH}/libero_object",
                "${oc.env:DATASET_PATH}/libero_spatial",
                "${oc.env:DATASET_PATH}/libero_goal",
            ],
            rotation_space="6d",
            seed=0,
            skip_video_loading=False,
            split="train",
            tolerance_s=0.0001,
            val_ratio=0.01,
            video_backend="torchcodec",
        ),
        ratio=1.0,
        resolution=None,
    ),
]


action_policy_sft_super = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdacosine"},
            {"override /checkpoint": "s3"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                    "generation",
                ]
            },
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /cluster": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3_action_libero",
            group="action_libero",
            name="action_policy_sft_super",
            wandb_mode="offline",
        ),
        model=dict(
            config=_action_policy_super_model_config(),
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,
            keys_to_select=["lora_"],
            lr=5.0e-05,
            lr_multipliers={},
            weight_decay=0.05,
        ),
        scheduler=dict(
            cycle_lengths=[20000],
            f_max=[1.0],
            f_min=[0.0],
            f_start=[1.0e-06],
            verbosity_interval=0,
            warm_up_steps=[500],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=100,
            max_iter=16000,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=0,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=100, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                compile_tokenizer=dict(
                    compile_after_iterations=3,
                    enabled=True,
                    warmup_resolutions=["256", "480", "720"],
                ),
                dataloader_speed=dict(every_n=50, save_s3=False, step_size=1),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=50, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                manual_gc=dict(every_n=200, gc_level=1, warm_up=5),
                mfu=dict(
                    backwardpass_ratio=2.0,
                    every_n=50,
                    grad_accum_iter=2,
                    hit_thres=5,
                    include_padding=True,
                    include_vae_encoder=True,
                ),
                moe_specialization=dict(every_n=250),
                moe_stability=dict(every_n=250),
                norm_monitor=dict(
                    every_n=50,
                    layer_norm_only=False,
                    log_stat_wandb=True,
                    save_s3=False,
                    step_size=1,
                    track_activations=True,
                ),
                ofu=dict(every_n=50, hit_thres=5),
                param_count=dict(save_s3=False),
                sequence_packing_padding=dict(every_n=50),
                sigma_loss_analysis=dict(every_n=500, every_n_viz=500, save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                straggler_detection=dict(enabled=True, report_freq=50),
                training_stats=dict(log_freq=100),
                wandb_2x=dict(
                    logging_iter_multipler=2,
                    save_logging_iter_multipler=1,
                    save_s3=False,
                ),
                expert_heatmap=dict(every_n=500),
                device_monitor=dict(
                    every_n=200,
                    log_memory_detail=True,
                    save_s3=False,
                    step_size=1,
                    upload_every_n_mul=5,
                ),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=True,
            enable_gcs_patch_in_boto3=True,
            keys_to_skip_loading=[
                "net_ema.",
                # "action2llm",
                # "llm2action",
                # "action_modality_embed",
                # "action_pos_embed",
                "lora_",
            ],
            load_ema_to_reg=False,
            load_path="???",  # OmegaConf MISSING — must be set via override at launch
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            strict_resume=False,
            verbose=True,
            load_from_object_store=dict(
                bucket="",
                credentials="",
                enabled=False,
            ),
            save_to_object_store=dict(
                bucket="",
                credentials="",
                enabled=False,
            ),
        ),
        dataloader_train=L(IterativeJointDataLoader)(
            audio_sample_rate=48000,
            max_samples_per_batch=256,
            max_sequence_length=None,
            patch_spatial=2,
            seed=42,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloaders=dict(
                action_data=dict(
                    ratio=1,
                    dataloader=L(InfiniteDataLoader)(
                        batch_size=4,
                        in_order=False,
                        multiprocessing_context="spawn",
                        num_workers=4,
                        pin_memory=True,
                        seed=42,
                        use_deterministic_seed=True,
                        dataset=L(wrap_dataset)(
                            action_channel_masking=True,
                            append_duration_fps_timestamps=True,
                            append_idle_frames=False,
                            append_resolution_info=True,
                            caption_key="ai_caption",
                            cfg_dropout_rate=0.1,
                            format_prompt_as_json=False,
                            idle_frames_dropout=0.05,
                            keep_aspect_ratio=True,
                            list_of_datasets=_LIBERO_DATASETS,
                            max_action_dim=64,
                            pad_keys=None,
                            resolution=None,
                            shard_across_workers=True,
                            text_token_key="text_token_ids",
                            video_temporal_downsample=4,
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                        ),
                    ),
                ),
            ),
        ),
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


for _item in [action_policy_sft_super]:
    _name = [k for k, v in globals().items() if v is _item][0]
    _item["job"]["name"] = _name + "_${now:%Y-%m-%d}_${now:%H-%M-%S}"
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
