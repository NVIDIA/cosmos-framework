# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Mirror of videophy2_sft_nano on the four-role dataflow loader.
Differs ONLY in dataloader wiring (CosmosDataLoader + roles) and job.name suffix _v2.
Loss-curve regression mirror."""

from __future__ import annotations

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.data.vfm.dataflow import CosmosDataLoader, IterableDistributor, PoolPackingBatcher
from cosmos_framework.data.vfm.processors import build_processor
from cosmos_framework.utils.vlm.constant import IGNORE_INDEX
from cosmos_framework.configs.base.vlm.experiment.dataflow_roles import VLMCollator
from cosmos_framework.configs.base.vlm.experiment.videophy2_dataflow_roles import VideoPhy2Processor
from cosmos_framework.configs.base.vlm.experiment.videophy2_sft_nano import build_videophy2_local_dataset

cs = ConfigStore.instance()


def _dl(dataset_key, split, num_workers):
    return L(CosmosDataLoader)(
        distributor=L(IterableDistributor)(
            iterable=L(build_videophy2_local_dataset)(dataset_key=dataset_key, split=split),
        ),
        processor=L(VideoPhy2Processor)(
            processor=L(build_processor)(
                tokenizer_type="${model.config.policy.backbone.model_name}",
                config_variant="hf",
            ),
            ignore_index=IGNORE_INDEX,
        ),
        batcher=L(PoolPackingBatcher)(
            max_tokens="${data_setting.max_tokens}",
            pool_size=16,
            max_batch_size=1,
            long_threshold=6400,
        ),
        collator=L(VLMCollator)(),
        num_workers=num_workers,
    )


videophy2_sft_nano_v2 = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "local"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "qwen3_vl_8b_instruct"},
            {"override /callbacks": ["basic_vlm", "basic_log", "hf_export"]},
            "_self_",
        ],
        job=dict(
            project="cosmos3_reasoner",
            group="sft",
            wandb_mode="disabled",
        ),
        trainer=dict(
            callbacks=dict(
                log_tensor_shape=dict(num_log=2),
            ),
            max_iter=50,
            logging_iter=1,
            run_validation=True,
            validation_iter=10,
            max_val_iter=50,
            grad_accum_iter=8,
        ),
        optimizer=dict(
            lr=1e-6,
            fused=True,
            weight_decay=0.05,
            betas=[0.9, 0.999],
            lr_multipliers={"mm_projector": 20.0, "merger": 20.0},
        ),
        scheduler=dict(
            warm_up_steps=[5],
            cycle_lengths=[50],
            f_min=[0.1],
        ),
        data_setting=dict(
            qwen_max_video_token_length=8192,
            qwen_max_image_token_length=2048,
            max_tokens=16000,
            max_batch_size=1,
            distributor_seed=1993,
        ),
        model=dict(
            config=dict(
                freeze=dict(
                    freeze_vision_encoder=True,
                    freeze_mm_projector=False,
                ),
                parallelism=dict(
                    data_parallel_shard_degree=8,
                    data_parallel_replicate_degree=-1,
                ),
                policy=dict(
                    monkey_patch_for_text_only_data=True,
                ),
            ),
        ),
        # hf_export so eval_videophy2 can read each save as HF safetensors.
        checkpoint=dict(
            save_iter=100,
            hf_export=dict(enabled=True),
        ),
        upload_reproducible_setup=False,
        dataloader_train=_dl("videophy2_train", "train", 2),
        dataloader_val=_dl("videophy2_val", "val", 0),
    ),
    flags={"allow_objects": True},
)

# Set job.name with _v2 suffix (mirrors original naming convention)
videophy2_sft_nano_v2["job"]["name"] = "videophy2_sft_nano_v2_${now:%Y-%m-%d}_${now:%H-%M-%S}"

cs.store(
    group="experiment",
    package="_global_",
    name="videophy2_sft_nano_v2",
    node=videophy2_sft_nano_v2,
)
