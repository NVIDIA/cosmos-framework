# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""VideoPhy-2 SFT recipe: LocalSFTDataset + CosmosDataLoader on Qwen3-VL.

Launch via examples/launch_sft_videophy2_nano.sh after running
prepare_videophy2_from_hf to populate $VIDEOPHYSICS_ROOT.
"""

from __future__ import annotations

import io
import os
from typing import Any, Iterator

import torch
import torch.utils.data
from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.data.generator.dataflow import CosmosDataLoader, IterableDistributor, PoolPackingBatcher
from cosmos_framework.data.generator.processors import build_processor
from cosmos_framework.data.reasoner.local_sft_dataset import LocalSFTDataset
from cosmos_framework.data.reasoner.data_sources_videophy2.videophy2 import DATAINFO
from cosmos_framework.utils import log
from cosmos_framework.utils.reasoner.constant import IGNORE_INDEX, PROCESSOR_KEYS_TO_ADD
from cosmos_framework.configs.base.reasoner.experiment.dataflow_roles import VLMCollator
from cosmos_framework.configs.base.reasoner.experiment.videophy2_dataflow_roles import VideoPhy2Processor

cs = ConfigStore.instance()


class _UnshardedLocalSFTDataset(LocalSFTDataset):
    """Yield the full shuffled manifest per iteration.

    Why: ``CosmosDataLoader``'s IterableDistributor already shards by
    ``dp_rank * num_workers + worker_id``; stock ``LocalSFTDataset`` shards
    again inside ``__iter__``, double-sharding to ``1 / (world*workers)^2``.
    """

    def _per_partition_indices(self, epoch: int) -> list[int]:
        import random

        manifest = self._load_manifest()
        indices = list(range(len(manifest)))
        if self.shuffle:
            rng = random.Random(self.distributor_seed + epoch)
            rng.shuffle(indices)
        return indices


def build_videophy2_local_dataset(
    dataset_key: str,
    split: str,
) -> _UnshardedLocalSFTDataset:
    # augmentor_config=None: the Processor decodes+tokenizes inline; the
    # BytesToMedia/TokenizeData augmentors aren't shipped in OSS.
    source = DATAINFO[dataset_key]
    if split not in source.manifest_path:
        raise KeyError(
            f"split={split!r} not present in DATAINFO[{dataset_key!r}].manifest_path "
            f"(have: {list(source.manifest_path)})"
        )
    return _UnshardedLocalSFTDataset(
        manifest_path=source.manifest_path[split],
        data_root=source.data_root,
        media_field_name=source.media_field_name,
        augmentor_config=None,
        text_only=source.text_only,
        shuffle=True,
        distributor_seed=1993,
        is_infinite_loader=True,
        split=split,
        dataset_name=dataset_key,
    )


_MAX_VIDEO_FRAMES = 32
_TARGET_VIDEO_FPS = 2.0


def _decode_video_to_pil_frames(video_bytes: bytes) -> tuple[list, float]:
    from torchcodec.decoders import VideoDecoder
    from PIL import Image
    import numpy as np

    decoder = VideoDecoder(video_bytes)
    total_frames = decoder.metadata.num_frames or 0
    source_fps = float(decoder.metadata.average_fps or 0.0) or 30.0

    if total_frames <= 0:
        raise ValueError("video has zero frames")

    stride = max(1, int(round(source_fps / _TARGET_VIDEO_FPS)))
    indices = list(range(0, total_frames, stride))
    if len(indices) > _MAX_VIDEO_FRAMES:
        step = len(indices) / _MAX_VIDEO_FRAMES
        indices = [indices[int(i * step)] for i in range(_MAX_VIDEO_FRAMES)]

    frames_tensor = decoder.get_frames_at(indices=indices).data
    frames_np = frames_tensor.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.uint8)
    frames = [Image.fromarray(f) for f in frames_np]

    effective_fps = source_fps / stride if stride > 0 else source_fps
    return frames, float(effective_fps)


def _dl(dataset_key, split, num_workers, persistent_workers=False, pin_memory=False, prefetch_factor=None):
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
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )


videophy2_sft_nano = LazyDict(
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
        dataloader_train=_dl("videophy2_train", "train", 2, persistent_workers=True, pin_memory=True, prefetch_factor=2),
        dataloader_val=_dl("videophy2_val", "val", 0, persistent_workers=False, pin_memory=True, prefetch_factor=None),
    ),
    flags={"allow_objects": True},
)


for _item in [videophy2_sft_nano]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    if "job" not in _item:
        _item["job"] = dict(name=experiment_name + "_${now:%Y-%m-%d}_${now:%H-%M-%S}")
    else:
        _item["job"]["name"] = experiment_name + "_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    cs.store(group="experiment", package="_global_", name=experiment_name, node=_item)
