# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""VideoPhy-2 SFT recipe: LocalSFTDataset + DataPackerDataLoader on Qwen3-VL.

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
from cosmos_framework.utils.lazy_config import LazyDict, instantiate
from cosmos_framework.data.vfm.data_packer import DataPacker
from cosmos_framework.data.vfm.data_packer_dataloader import DataPackerDataLoader
from cosmos_framework.data.vfm.processors import build_processor
from cosmos_framework.data.vlm.local_sft_dataset import LocalSFTDataset
from cosmos_framework.data.vlm.data_sources_videophy2.videophy2 import DATAINFO
from cosmos_framework.utils import log
from cosmos_framework.utils.vlm.constant import IGNORE_INDEX, PROCESSOR_KEYS_TO_ADD

cs = ConfigStore.instance()


class _UnshardedLocalSFTDataset(LocalSFTDataset):
    """Yield the full shuffled manifest per iteration.

    Why: ``DataPackerDataLoader._IterableWrapper`` already shards by
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
    # augmentor_config=None: the DataPacker decodes+tokenizes inline; the
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


def build_videophy2_datapacker_dataloader(**kwargs) -> DataPackerDataLoader:
    for _spurious in ("storage_type",):
        kwargs.pop(_spurious, None)
    return DataPackerDataLoader(**kwargs)


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


class VideoPhy2DataPacker(DataPacker):
    """LocalSFTDataset + Qwen3-VL processor adapter; max_batch_size=1."""

    def __init__(
        self,
        tokenizer_config: Any,
        max_seq_len: int = 16000,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        self._max_seq_len = max_seq_len
        self._ignore_index = ignore_index
        self._processor = (
            tokenizer_config if hasattr(tokenizer_config, "apply_chat_template") else instantiate(tokenizer_config)
        )

    def _materialize_media_in_conversation(
        self,
        conversation: list,
        media_bytes_by_key: dict,
    ) -> list:
        # Resolve "video": "<key>" / "image": "<key>" references against
        # data_dict["media"] (bytes); decode each unique key once.
        decoded_cache: dict[str, tuple[list, float]] = {}
        new_messages: list[dict] = []
        for message in conversation:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                new_messages.append({"role": message.get("role", "user"), "content": content})
                continue
            if not isinstance(content, list):
                continue
            new_content: list[dict] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                kind = item.get("type")
                if kind == "video":
                    key = item.get("video")
                    if not isinstance(key, str):
                        new_content.append(item)
                        continue
                    if key not in media_bytes_by_key:
                        raise KeyError(
                            f"conversation references video key {key!r} not present in "
                            f"sample['media'] (keys: {list(media_bytes_by_key)})"
                        )
                    if key not in decoded_cache:
                        decoded_cache[key] = _decode_video_to_pil_frames(media_bytes_by_key[key])
                    frames, fps = decoded_cache[key]
                    new_content.append({"type": "video", "video": frames, "fps": fps})
                elif kind == "image":
                    key = item.get("image")
                    if not isinstance(key, str):
                        new_content.append(item)
                        continue
                    if key not in media_bytes_by_key:
                        raise KeyError(
                            f"conversation references image key {key!r} not present in "
                            f"sample['media'] (keys: {list(media_bytes_by_key)})"
                        )
                    from PIL import Image

                    img = Image.open(io.BytesIO(media_bytes_by_key[key])).convert("RGB")
                    new_content.append({"type": "image", "image": img})
                else:
                    new_content.append(item)
            new_messages.append({"role": message.get("role", "user"), "content": new_content})
        return new_messages

    def sft_process_sample(self, item: dict) -> dict:
        conversation = item.get("texts")
        if not isinstance(conversation, list):
            raise TypeError(
                f"LocalSFTDataset sample expected 'texts' to be a list, got {type(conversation).__name__}"
            )
        media_bytes_by_key = item.get("media") or {}
        messages = self._materialize_media_in_conversation(conversation, media_bytes_by_key)

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        input_ids = inputs["input_ids"]  # [N]

        token_mask = self._processor.add_assistant_tokens_mask(input_ids)  # [N] bool
        labels = input_ids.clone()
        labels[~token_mask] = self._ignore_index

        result: dict = {
            "input_ids": input_ids,
            "labels": labels,
        }
        for key in PROCESSOR_KEYS_TO_ADD:
            if key in inputs and inputs[key] is not None:
                result[key] = inputs[key]

        return result

    def compute_num_tokens(self, sample: dict) -> int:
        return int(sample["input_ids"].shape[0])

    def sft_collate_fn(
        self,
        samples: list[dict],
        max_len: int,
        ignore_label_id: int = IGNORE_INDEX,
    ) -> dict:
        assert len(samples) == 1, f"VideoPhy2DataPacker expects max_batch_size=1, got {len(samples)}"
        s = samples[0]

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        batch: dict = {
            "input_ids": s["input_ids"].unsqueeze(0),
            "labels": s["labels"].unsqueeze(0),
            "sample_worker_id": torch.tensor([worker_id]),
            "sample_epoch": torch.tensor([0]),
            "sample_index": torch.tensor([0]),
        }

        if "attention_mask" in s and s["attention_mask"] is not None:
            batch["attention_mask"] = s["attention_mask"].unsqueeze(0)

        for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "second_per_grid_ts"):
            if key in s and s[key] is not None:
                batch[key] = s[key]

        return batch


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
        dataloader_train=L(build_videophy2_datapacker_dataloader)(
            data_source=L(build_videophy2_local_dataset)(
                dataset_key="videophy2_train",
                split="train",
            ),
            data_packer=L(VideoPhy2DataPacker)(
                tokenizer_config=L(build_processor)(
                    tokenizer_type="${model.config.policy.backbone.model_name}",
                    config_variant="hf",
                ),
                max_seq_len="${data_setting.max_tokens}",
                ignore_index=IGNORE_INDEX,
            ),
            max_tokens="${data_setting.max_tokens}",
            max_batch_size=1,
            pool_size=16,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
        ),
        dataloader_val=L(build_videophy2_datapacker_dataloader)(
            data_source=L(build_videophy2_local_dataset)(
                dataset_key="videophy2_val",
                split="val",
            ),
            data_packer=L(VideoPhy2DataPacker)(
                tokenizer_config=L(build_processor)(
                    tokenizer_type="${model.config.policy.backbone.model_name}",
                    config_variant="hf",
                ),
                max_seq_len="${data_setting.max_tokens}",
                ignore_index=IGNORE_INDEX,
            ),
            max_tokens="${data_setting.max_tokens}",
            max_batch_size=1,
            pool_size=16,
            num_workers=0,
            prefetch_factor=None,
            persistent_workers=False,
            pin_memory=True,
        ),
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
