# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from torch.utils.data import DataLoader

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.config_helper import ConfigStore
from cosmos_framework.data.vfm.vlm.collate_fn import custom_collate
from cosmos_framework.data.vfm.vlm.debug_data_qwen import DebugQwenDataset
from cosmos_framework.data.vfm.vlm.dummy_data_qwen import DummyQwenDataset
from cosmos_framework.data.vfm.processors import build_processor_lazy


# Debug dataset
def create_debug_dataloader_config_qwen(
    num_images, loss_on_completion_only: bool = True, use_dummy_image: bool = False
):
    return L(DataLoader)(
        dataset=L(DebugQwenDataset)(
            tokenizer=L(build_processor_lazy)(
                tokenizer_type="${model.config.policy.backbone.model_name}",
                credentials="${checkpoint.load_from_object_store.credentials}",
                bucket="${checkpoint.load_from_object_store.bucket}",
            ),
            num_images=num_images,
            seq_len="${model.config.policy.model_max_length}",
            image_token_len="${model.config.policy.qwen_max_video_token_length}",
            # use_dummy_image=use_dummy_image,
        ),
        num_workers=8,
        prefetch_factor=4,
        batch_size=1,
        sampler=None,
        persistent_workers=False,
        pin_memory=True,
        collate_fn=custom_collate,
    )


def create_dummy_dataloader_config_qwen():
    return L(DataLoader)(
        dataset=L(DummyQwenDataset)(
            tokenizer=L(build_processor_lazy)(
                tokenizer_type="${model.config.policy.backbone.model_name}",
                credentials="${checkpoint.load_from_object_store.credentials}",
                bucket="${checkpoint.load_from_object_store.bucket}",
            ),
            num_visual_tokens="${model.config.policy.qwen_max_video_token_length}",
            total_tokens="${model.config.policy.model_max_length}",
            batch_size="${dataloader_train.batch_size}",
        ),
        num_workers=8,
        prefetch_factor=4,
        batch_size=1,
        sampler=None,
        persistent_workers=False,
        pin_memory=True,
        collate_fn=custom_collate,
    )


def register_data_debug():
    cs = ConfigStore.instance()
    for split in ["train", "val"]:
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="debug_image_data_qwen",  # This data is from pixtral model output, expected to have low loss ~1.4
            node=create_debug_dataloader_config_qwen(1),
        )
        cs.store(
            group=f"data_{split}",
            package=f"dataloader_{split}",
            name="dummy_image_data_qwen",
            node=create_dummy_dataloader_config_qwen(),
        )


def register_data():
    register_data_debug()
