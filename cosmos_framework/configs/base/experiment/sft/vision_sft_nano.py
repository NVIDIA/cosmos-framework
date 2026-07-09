# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``vision_sft_nano`` — Cosmos3-Nano vision SFT recipe.

Qwen3-VL-8B / Cosmos3-Nano backbone, T2V/I2V/V2V SFT, PackingDataLoader +
RankPartitionedDataLoader stack. EMA enabled.

Notes:
    * ``_target_`` references for ``model``, ``optimizer``, ``scheduler``,
      ``checkpoint``, ``callbacks``, ``ema``, ``tokenizer``, ``cluster``,
      ``vlm_config``, and ``ckpt_type`` flow from the ``defaults:`` group
      choices, matching the YAML's ``defaults:`` list (``_self_`` is placed
      LAST per prerelease convention so the experiment overrides the
      defaults, but no setting changes semantically).
    * The YAML's giant ``model_parallel`` block, ``trainer.profiling``,
      ``trainer.straggler_detection`` and ``trainer.type`` are populated by
      the base ``Config`` (``cosmos_framework/configs/base/config.py``) defaults and
      are therefore omitted here.

``checkpoint.load_path`` is left as ``???`` (a Hydra MISSING marker); supply
via CLI / a downstream experiment that inherits from this one.

Usage::

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. torchrun --nproc_per_node=1 \\
        --master_port=12341 -m cosmos_framework.scripts.train \\
        --config=cosmos_framework/configs/base/config.py -- \\
        experiment=vision_sft_nano \\
        checkpoint.load_path=<path>
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.generator.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.data.generator.dataflow import (
    CosmosDataLoader,
    IdentityProcessor,
    RankPartitionedDistributor,
    SequentialPackingBatcher,
    VFMListCollator,
)
from cosmos_framework.data.generator.local_datasets.sft_dataset import get_sft_dataset
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


vision_sft_nano = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "adamw"},
            # YAML used `scheduler: warmup_cosine_lr` but that group is only
            # registered in cosmos_framework/configs/base/reasoner/defaults/optimizer.py
            # (reachable from the vlm config tree). The base vfm config path
            # only knows `lambdacosine`, which also sets
            # lr_scheduler_type="LambdaCosine" — behaviorally identical.
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
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3",
            group="sft",
            name="vision_sft_nano",
            wandb_mode="disabled",
        ),
        model=dict(
            config=copy.deepcopy(NANO_MODEL_CONFIG),
        ),
        optimizer=dict(
            betas=[0.9, 0.95],
            eps=1.0e-06,
            fused=True,
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
            ],
            lr=5.0e-04,
            lr_multipliers={},
            optimizer_type="AdamW",
            weight_decay=0,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaCosine",
            cycle_lengths=[1000],
            f_max=[1.0],
            f_min=[0.0],
            f_start=[0.0],
            verbosity_interval=0,
            warm_up_steps=[50],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=2,
            logging_iter=1,
            max_iter=500,
            max_val_iter=None,
            # YAML had `memory_format: preserve_format` as a string, but the
            # prerelease trainer passes this verbatim to model.to(memory_format=…)
            # which requires a torch.memory_format enum (not a string).
            # Omit and let the framework default apply, matching how
            # mixed_modality_sft_nano.py handles it.
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
                compile_tokenizer=dict(
                    compile_after_iterations=3,
                    enabled=False,
                    warmup_resolutions=None,
                ),
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(
                    every_n=200,
                    log_memory_detail=True,
                    save_s3=False,
                    step_size=1,
                    upload_every_n_mul=5,
                ),
                expert_heatmap=dict(every_n=1000),
                grad_clip=dict(clip_norm=0.1, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                norm_monitor=dict(
                    every_n=100,
                    layer_norm_only=False,
                    log_stat_wandb=True,
                    model_key=None,
                    save_s3=False,
                    step_size=1,
                    track_activations=True,
                ),
                param_count=dict(save_s3=False),
                sequence_packing_padding=dict(every_n=50),
                sigma_loss_analysis=dict(every_n=500, every_n_viz=500, save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
                wandb_2x=dict(
                    logging_iter_multipler=2,
                    save_logging_iter_multipler=1,
                    save_s3=False,
                ),
                wandb_val=dict(save_s3=False),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            keys_to_skip_loading=["net_ema."],
            load_ema_to_reg=False,
            load_path="???",  # supply via CLI / downstream experiment
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            strict_resume=True,
            verbose=True,
            hf_export=dict(
                enabled=False,
                export_every_n=1,
                hf_repo_id=None,
                upload_to_object_store=dict(
                    bucket="",
                    credentials="",
                    enabled=False,
                ),
            ),
            jit=dict(
                device="cuda",
                dtype="bfloat16",
                enabled=False,
                input_shape=None,
                strict=True,
            ),
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
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="default",
            max_samples_per_batch=None,
            max_sequence_length=45056,
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=True,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=4,
                sampler=None,
                datasets=dict(
                    video=dict(
                        ratio=1,
                        dataset=L(get_sft_dataset)(
                            append_duration_fps_timestamps=True,
                            append_resolution_info=True,
                            # Per-caption token cap. Structured-JSON captions are long, so
                            # default to 2048 (measured max ~1790); tune via the TOML knob
                            # [dataloader_train].max_caption_tokens. See sft_dataset.py
                            # _MAX_CAPTION_TOKENS.
                            max_caption_tokens=2048,
                            caption_suffix="",
                            cfg_dropout_keep_metadata=False,
                            cfg_dropout_rate=0.1,
                            # 70% T2V, 20% I2V (first frame), 10% V2V (first 5 frames / 2 latent frames)
                            conditioning_config={0: 0.7, 1: 0.2, 2: 0.1},
                            conditioning_fps=-1,
                            conditioning_fps_noise_std=0.0,
                            frame_selection_mode="first",
                            jsonl_paths=["${oc.env:DATASET_PATH}/train/video_dataset_file.jsonl"],
                            min_short_edge=0,
                            num_video_frames=-1,
                            resolution="256",
                            sample_by_window=False,
                            temporal_compression_factor=4,
                            temporal_interval_mode="max_30fps",
                            use_system_prompt=False,
                            # YAML spells this out as
                            #   _target_: create_qwen2_tokenizer_with_download
                            #   config_variant: gcp
                            # but that pins the dataset's tokenizer to the GCP
                            # variant, requiring credentials/gcp_checkpoint.secret.
                            # Use a Hydra interpolation instead so launchers
                            # (e.g. launch_vision_sft_nano_toml.sh) can flip
                            # model.config.vlm_config.tokenizer.config_variant=hf
                            # and have the dataset inherit the same setting.
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


# ``vision_sft_nano_mapstyle_dataloader`` — identical to ``vision_sft_nano`` except the training
# dataloader uses the four-role ``CosmosDataLoader`` stack
# (``RankPartitionedDistributor`` → ``IdentityProcessor`` →
# ``SequentialPackingBatcher`` → ``VFMListCollator``) instead of the legacy
# ``PackingDataLoader`` + ``RankPartitionedDataLoader``. Every other block is reused
# verbatim by deep-copying the base recipe and overriding only ``job.name`` and
# ``dataloader_train``.
vision_sft_nano_mapstyle_dataloader = copy.deepcopy(vision_sft_nano)
vision_sft_nano_mapstyle_dataloader.job.name = "vision_sft_nano_mapstyle_dataloader"
vision_sft_nano_mapstyle_dataloader.dataloader_train = L(CosmosDataLoader)(
    distributor=L(RankPartitionedDistributor)(
        datasets=dict(
            video=dict(
                ratio=1,
                dataset=L(get_sft_dataset)(
                    append_duration_fps_timestamps=True,
                    append_resolution_info=True,
                    caption_suffix="",
                    cfg_dropout_keep_metadata=False,
                    cfg_dropout_rate=0.1,
                    # 70% T2V, 20% I2V (first frame), 10% V2V (first 5 frames / 2 latent frames)
                    conditioning_config={0: 0.7, 1: 0.2, 2: 0.1},
                    conditioning_fps=-1,
                    conditioning_fps_noise_std=0.0,
                    frame_selection_mode="first",
                    jsonl_paths=["${oc.env:DATASET_PATH}/train/video_dataset_file.jsonl"],
                    min_short_edge=0,
                    num_video_frames=-1,
                    resolution="256",
                    sample_by_window=False,
                    temporal_compression_factor=4,
                    temporal_interval_mode="max_30fps",
                    use_system_prompt=False,
                    tokenizer_config="${model.config.vlm_config.tokenizer}",
                ),
            ),
        ),
    ),
    processor=L(IdentityProcessor)(),
    batcher=L(SequentialPackingBatcher)(
        max_sequence_length=45056,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        max_samples_per_batch=None,
        sound_latent_fps=0,
        audio_sample_rate=48000,
    ),
    collator=L(VFMListCollator)(),
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=4,
)


for _item in [vision_sft_nano, vision_sft_nano_mapstyle_dataloader]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
