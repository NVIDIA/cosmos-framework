# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_droid_nano`` — Cosmos3-Nano DROID action policy SFT recipe.

Mirrors the vision SFT stack (PackingDataLoader + RankPartitionedDataLoader),
but feeds the DROID action dataset (``joint_pos`` 8D + ``use_state``, raw/
un-normalized) through ``ActionTransformPipeline``, and trains the generation +
action heads from the public ``nvidia/Cosmos3-Nano`` base.

Usage (1 node, 8 GPU)::

    DROID_ROOT=/path/to/droid_lerobot_640x360/success \\
    BASE_CHECKPOINT_PATH=<Cosmos3-Nano DCP dir> \\
    WAN_VAE_PATH=<Wan2.2_VAE.pth> \\
    torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \\
        --sft-toml examples/toml/sft_config/action_policy_droid_repro.toml
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_droid_sft_dataset
from cosmos_framework.data.generator.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


action_policy_droid_nano = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            # FusedAdam with fp32 master_weights + eps 1e-8 (bf16 params + eps 1e-6
            # diverged on the action loss).
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},  # linear LR decay
            {"override /checkpoint": "s3"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                ]
            },
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3",
            group="action_sft",
            name="action_policy_droid_nano",
            wandb_mode="disabled",
        ),
        model=dict(
            config=copy.deepcopy(NANO_MODEL_CONFIG),  # action_gen=True, max_action_dim=64
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,  # popped by build_optimizer for FusedAdam (fused by construction)
            # Train the generation + action heads.
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            lr=2.0e-04,  # for the 8192 global batch
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[100],  # smoke: 100 iters (real run sets via TOML)
            f_max=[0.4],
            f_min=[0.0],
            f_start=[0.0],
            verbosity_interval=0,
            warm_up_steps=[0],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=1,
            max_iter=100,  # smoke
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(
                    every_n=200, log_memory_detail=True, save_s3=False, step_size=1, upload_every_n_mul=5
                ),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                param_count=dict(save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # Skip net_ema. (EMA warm-starts from net, see dcp.py) and the action
            # heads, so they init fresh from the base (the base has no DROID-trained
            # action heads).
            keys_to_skip_loading=[
                "net_ema.",
                "action2llm",
                "llm2action",
                "action_modality_embed",
                "action_pos_embed",
            ],
            load_ema_to_reg=False,
            load_path="???",  # Cosmos3-Nano DCP dir; supply via TOML/env
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            strict_resume=False,  # base init: tolerate key set differences
            verbose=True,
            hf_export=dict(
                enabled=False,
                export_every_n=1,
                hf_repo_id=None,
                upload_to_object_store=dict(bucket="", credentials="", enabled=False),
            ),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="action_droid",
            max_samples_per_batch=128,  # per rank -> 8192 global batch at 64 ranks (16 nodes, shard 8 x replicate 8)
            max_sequence_length=None,  # None disables token packing (TOML can't express null)
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=False,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=4,
                sampler=None,
                # Shuffling is handled by the dataset (iterable_shuffle=True below):
                # ActionIterableShuffleDataset streams rank x worker-sharded, episode-order-
                # shuffled, sequential-within-episode. The map-style dataset has no internal
                # shuffle, so a SequentialSampler would feed every rank the SAME consecutive
                # overlapping windows -> global batch ~1 episode -> unstable grad-norm; a plain
                # RandomSampler decorrelates but does random-access I/O -> slow + OOM. The
                # iterable gives decorrelation with sequential reads.
                datasets=dict(
                    droid=dict(
                        ratio=1,
                        dataset=L(get_action_droid_sft_dataset)(
                            root="${oc.env:DROID_ROOT}",
                            fps=15.0,
                            chunk_length=32,
                            action_space="joint_pos",
                            # Policy-only task mode. "joint" would randomly pick
                            # forward_dynamics/inverse_dynamics/policy per sample (multi-task),
                            # which dilutes each per-task loss by ~1/3.
                            mode="policy",
                            use_state=True,
                            iterable_shuffle=True,  # rank x worker episode-shuffle stream
                            episode_shuffle_seed=42,
                            # SR boost: random crop+rescale + ColorJitter, applied CPU-side in the
                            # DROIDLeRobotDataset image augmentor (matches i4's pipeline stage).
                            use_image_augmentation=True,
                            # keep_ranges_1_0_1.json window filter (drops idle/non-task frames). Off by default;
                            # set use_filter_dict=True + filter_dict_path to enable.
                            use_filter_dict=False,
                            filter_dict_path=None,
                            action_normalization=None,
                            viewpoint="concat_view",  # wrist 480p (top) + L/R shoulder 320x180 (bottom)
                            resolution="480",  # 640x360 data @ 480p
                            max_action_dim="${model.config.max_action_dim}",
                            cfg_dropout_rate=0.1,
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                            # Match i4 GA (droid_lerobot_8b_ga / MR #9995): format the action
                            # prompt as JSON via ActionPromptJsonFormatter instead of plain text.
                            format_prompt_as_json=True,
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


# chunk_length=32 -> 33 observation frames; pin the VAE encode duration to match.
# Set post-construction so it lands on the deep-copied NANO_MODEL_CONFIG.tokenizer.
action_policy_droid_nano["model"]["config"]["tokenizer"]["encode_exact_durations"] = [33]


# Uncap the packed-sequence length. The NANO default (45056) caps the packed sequence,
# truncating long DROID windows to ~1/4 of their natural length; -1 (uncapped) processes
# the full vision sequence per step. Does not change the per-token loss; widens the
# effective vision context per step.
action_policy_droid_nano["model"]["config"]["max_num_tokens_after_packing"] = -1


# Weight the vision flow-matching loss 10x in the total loss (the NANO default is 1.0).
# loss_scale multiplies only the vision term, balancing it against the action loss
# (action_loss_weight=10) so both heads train at comparable gradient magnitude.
action_policy_droid_nano["model"]["config"]["rectified_flow_training_config"]["loss_scale"] = 10.0


for _item in [action_policy_droid_nano]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
