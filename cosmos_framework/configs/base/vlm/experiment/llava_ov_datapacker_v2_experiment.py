# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Mirror of pre_exp012_llava_ov_datapacker on the four-role dataflow loader.
Differs ONLY in dataloader wiring (CosmosDataLoader + roles). Loss-curve
regression mirror — see docs/superpowers/specs/2026-06-04-modular-dataflow-refactor-design.md."""

from __future__ import annotations

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.data.vfm.dataflow import CosmosDataLoader, IterableDistributor, PoolPackingBatcher
from cosmos_framework.data.vfm.processors import build_processor
from cosmos_framework.utils.vlm.constant import IGNORE_INDEX
from cosmos_framework.configs.base.vlm.experiment.dataflow_roles import VLMCollator, VLMProcessor
from cosmos_framework.configs.base.vlm.experiment.llava_ov_datapacker_experiment import (
    get_llava_ov_streaming,
)

cs = ConfigStore.instance()

pre_exp012_llava_ov_datapacker_v2 = LazyDict(
    dict(
        # Hydra defaults — inlined from the former pre_exp012_000_phase2_vlm_smoke_4gpu_8b
        # smoke recipe. data_train/data_val intentionally omitted because the
        # dataloader_train below is a self-contained CosmosDataLoader; pulling in
        # the smoke's s3 webdataset defaults would let storage_type schema bleed into
        # our CosmosDataLoader config.
        defaults=[
            {"override /checkpoint": "s3"},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "qwen3_vl_8b_instruct"},
            {"override /callbacks": ["basic_vlm", "basic_log"]},
            "_self_",
        ],
        job=dict(
            name="pre_exp012_llava_ov_datapacker_v2_${now:%Y-%m-%d}_${now:%H-%M-%S}",
            group="vlm_llava_ov_demo",
            wandb_mode="disabled",
        ),
        trainer=dict(
            max_iter=10,
            logging_iter=1,
            run_validation=False,
        ),
        optimizer=dict(
            lr=1e-5,
            fused=True,
        ),
        model=dict(
            config=dict(
                # Phase 2 requires a trainable_params regex; ".*" = full fine-tune.
                freeze=dict(
                    trainable_params=[".*"],
                ),
                parallelism=dict(
                    data_parallel_shard_degree=4,
                    data_parallel_replicate_degree=-1,
                ),
            ),
        ),
        # Local-only mode: disable the parent's object-store IO and clear the
        # S3 credentials/bucket so maybe_download_hf_model_from_s3 falls back
        # to HuggingFace Hub (avoids opening credentials/s3_training.secret in
        # OSS smoke runs). Pattern mirrors vision_sft_nano.py.
        checkpoint=dict(
            # Don't save checkpoints during smoke runs.
            save_iter=100000,
            load_from_object_store=dict(enabled=False, credentials="", bucket=""),
            save_to_object_store=dict(enabled=False, credentials="", bucket=""),
        ),
        # Replace the S3 WebDataset-based dataloader with CosmosDataLoader
        # pointing at lmms-lab/LLaVA-OneVision-Data streamed from HuggingFace Hub,
        # wired through the four-role dataflow (IterableDistributor, VLMProcessor,
        # PoolPackingBatcher, VLMCollator).
        dataloader_train=L(CosmosDataLoader)(
            distributor=L(IterableDistributor)(
                iterable=L(get_llava_ov_streaming)(subset="ai2d(gpt4v)", split="train"),
            ),
            processor=L(VLMProcessor)(
                processor=L(build_processor)(
                    tokenizer_type="${model.config.policy.backbone.model_name}",
                    config_variant="hf",
                ),
                ignore_index=IGNORE_INDEX,
            ),
            batcher=L(PoolPackingBatcher)(
                max_tokens=16000, pool_size=16, max_batch_size=1, long_threshold=6400,
            ),
            collator=L(VLMCollator)(),
            num_workers=2,
        ),
        dataloader_val=None,
        # Suppress S3 uploads in callbacks (iter_speed.save_s3, param_count.save_s3,
        # wandb_*.save_s3 all interpolate from ${upload_reproducible_setup}). Mirrors
        # the VFM SFT experiments under cosmos/configs/base/experiment/sft/.
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)

cs.store(
    group="experiment",
    package="_global_",
    name="pre_exp012_llava_ov_datapacker_v2",
    node=pre_exp012_llava_ov_datapacker_v2,
)
