# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Action policy SFT on LIBERO — DataPackerDataLoader variant.

Equivalent to ``action_policy_sft_nano`` but replaces the
IterativeJointDataLoader + InfiniteDataLoader stack with DataPackerDataLoader
+ ActionDataPacker.  Sample processing is identical (same
ActionTransformPipeline params); batching changes from fixed batch_size=128 to
token-budget packing capped at max_batch_size=256.

Usage::

    DATASET_PATH=/path/to/libero_datasets torchrun \\
        --nproc_per_node=4 --master_port=12341 -m cosmos_framework.scripts.train \\
        --config=configs/base/config.py -- \\
        experiment=action_policy_sft_nano_datapacker \\
        checkpoint.load_path=<path>
"""

from __future__ import annotations

import copy
import math

import torch
from hydra.core.config_store import ConfigStore

from cosmos_framework.data.vfm.action.libero_dataset import LIBERODataset
from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline
from cosmos_framework.data.vfm.action.unified_dataset import MapToIterableAdapter
from cosmos_framework.data.vfm.data_packer import DataPacker
from cosmos_framework.data.vfm.data_packer_dataloader import DataPackerDataLoader
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.configs.base.experiment.sft.action_policy_sft_nano import action_policy_sft_nano

cs = ConfigStore.instance()


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------


def _get_libero_iterable_dataset(**kwargs) -> MapToIterableAdapter:
    """Wrap LIBERODataset in MapToIterableAdapter for DataPackerDataLoader.

    LIBERODataset is map-style; MapToIterableAdapter yields uniformly random
    items indefinitely so _IterableWrapper inside DataPackerDataLoader can
    apply DP sharding on top.
    """
    return MapToIterableAdapter(LIBERODataset(**kwargs))


# ---------------------------------------------------------------------------
# DataPacker
# ---------------------------------------------------------------------------


class ActionDataPacker(DataPacker):
    """DataPacker adapter for LIBERODataset + ActionTransformPipeline (policy mode).

    Applies the same ActionTransformPipeline as the wrap_dataset call in
    action_policy_sft_nano.  The sft_collate_fn output format matches what
    OmniMoTModel.training_step expects.
    """

    def __init__(
        self,
        tokenizer_spatial_compression_factor: int = 16,
        tokenizer_temporal_compression_factor: int = 4,
        patch_spatial: int = 2,
        tokenizer_config=None,
        cfg_dropout_rate: float = 0.1,
        max_action_dim: int = 64,
        action_channel_masking: bool = True,
    ) -> None:
        self._spatial = tokenizer_spatial_compression_factor
        self._temporal = tokenizer_temporal_compression_factor
        self._patch = patch_spatial
        self._transform = ActionTransformPipeline(
            pad_keys=["video"],
            tokenizer_config=tokenizer_config,
            cfg_dropout_rate=cfg_dropout_rate,
            max_action_dim=max_action_dim,
            action_channel_masking=action_channel_masking,
            append_duration_fps_timestamps=True,
            append_resolution_info=True,
        )

    def sft_process_sample(self, item: dict) -> dict:
        return self._transform(item, resolution=None)

    def compute_num_tokens(self, sample: dict) -> int:
        tokens = 1 + len(sample.get("text_token_ids", []))
        v = sample.get("video")
        if v is not None:
            _, T, H, W = v.shape
            latent_h = math.ceil(H / (self._spatial * self._patch))
            latent_w = math.ceil(W / (self._spatial * self._patch))
            latent_t = 1 + (T - 1) // self._temporal
            tokens += latent_h * latent_w * latent_t + 2
        return tokens

    def sft_collate_fn(self, samples: list, max_len: int, ignore_label_id: int = -100) -> dict:
        return {
            "text_token_ids": [[s["text_token_ids"]] for s in samples],
            "video": [s.get("video") for s in samples],
            "action": [s.get("action") for s in samples],
            "padding_mask": [s.get("padding_mask") for s in samples],
            "image_size": [s.get("image_size") for s in samples],
            "fps": torch.tensor([float(s.get("fps", 0.0)) for s in samples]),
            "domain_id": [s.get("domain_id") for s in samples],
            "sequence_plan": [s.get("sequence_plan") for s in samples],
            "raw_action_dim": [s.get("raw_action_dim") for s in samples],
            "ai_caption": [s.get("ai_caption", "") for s in samples],
        }


# ---------------------------------------------------------------------------
# Experiment registration
# ---------------------------------------------------------------------------

_exp = copy.deepcopy(action_policy_sft_nano)
_exp["dataloader_train"] = L(DataPackerDataLoader)(
    data_source=L(_get_libero_iterable_dataset)(
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
    data_packer=L(ActionDataPacker)(
        tokenizer_spatial_compression_factor="${model.config.tokenizer.spatial_compression_factor}",
        tokenizer_temporal_compression_factor="${model.config.tokenizer.temporal_compression_factor}",
        patch_spatial="${model.config.diffusion_expert_config.patch_spatial}",
        tokenizer_config="${model.config.vlm_config.tokenizer}",
        cfg_dropout_rate=0.1,
        max_action_dim="${model.config.max_action_dim}",
        action_channel_masking=True,
    ),
    max_tokens=999_999,
    max_batch_size=256,
    pool_size=16,
    num_workers=4,
    prefetch_factor=4,
    persistent_workers=True,
    pin_memory=True,
)
_exp["job"]["name"] = "action_policy_sft_nano_datapacker_${now:%Y-%m-%d}_${now:%H-%M-%S}"

# Smoke-test overrides: skip the S3 VLM backbone download and use the HF
# tokenizer variant so the run is self-contained (no object-store credentials).
_exp["model"]["config"]["tokenizer"]["bucket_name"] = ""
_exp["model"]["config"]["vlm_config"]["pretrained_weights"]["enabled"] = False
_exp["model"]["config"]["vlm_config"]["tokenizer"]["config_variant"] = "hf"

cs.store(group="experiment", package="_global_", name="action_policy_sft_nano_datapacker", node=_exp)
