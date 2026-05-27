# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import functools
from uuid import uuid4

import pydantic

from cosmos_framework.inference.common.config import CONFIG_DIR
from cosmos_framework.utils.flags import TRAINING
from cosmos_framework.utils.checkpoint_db import (
    CheckpointConfig,
    CheckpointDirHf,
    CheckpointDirS3,
    CheckpointFileHf,
    CheckpointFileS3,
    RepositoryType,
    register_checkpoint,
)


@functools.cache
def register_checkpoints():
    """Register checkpoints used in hydra configs (tokenizers, VLM)."""
    for repository, revision in [
        ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
        ("Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203"),
        ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"),
        ("Qwen/Qwen3-VL-32B-Instruct", "0cfaf48183f594c314753d30a4c4974bc75f3ccb"),
    ]:
        for s3_prefix in [
            # 'cosmos_framework.configs.base.defaults.vlm.download_tokenizer_files'
            "cosmos3/pretrained/huggingface",
            # 'cosmos_framework.utils.vfm.vlm.pretrained_models_downloader.maybe_download_hf_model_from_s3'
            "cosmos_reason2/hf_models",
        ]:
            register_checkpoint(
                CheckpointConfig(
                    uuid=uuid4().hex,
                    name=repository,
                    s3=CheckpointDirS3(
                        uri=f"s3://bucket/{s3_prefix}/{repository}",
                    ),
                    hf=CheckpointDirHf(
                        repository=repository,
                        revision=revision,
                        include=() if TRAINING else ("*.json", "*.txt"),
                    ),
                ),
            )

    register_checkpoint(
        CheckpointConfig(
            uuid=uuid4().hex,
            name="Cosmos3-Reasoner-8B-Private",
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos3-Reasoner-8B-Private",
            ),
            hf=CheckpointDirHf(
                repository="nvidia/Cosmos3-Nano-Reasoner",
                revision="6406357cdc32fbf8db5f51ff7992343803b06961",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid=uuid4().hex,
            name="Cosmos3-Reasoner-32B-Private",
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos3-Reasoner-32B-Private",
            ),
            hf=CheckpointDirHf(
                repository="nvidia/Cosmos3-Super-Reasoner",
                revision="b9b716f3508dfa442e0c8ba32fb5d0c9adf2a32c",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="c5236e3a-e846-49e3-a40c-67dfceefff5d",
            name="Cosmos3-Nano-Reasoner-bb9c6f5",
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos3-Nano-Reasoner-bb9c6f5",
            ),
            hf=CheckpointDirHf(
                repository="nvidia/Cosmos3-Experimental",
                subdirectory="c5236e3a-e846-49e3-a40c-67dfceefff5d",
                revision="6ca42c5d0b96cb133e811c1bcced048d4acfaa12",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid="4cb0c125-49a8-4e66-aebb-06e100affdb0",
            name="Cosmos3-Super-Reasoner-b6df0d1",
            s3=CheckpointDirS3(
                uri="s3://bucket/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos3-Super-Reasoner-b6df0d1",
            ),
            hf=CheckpointDirHf(
                repository="nvidia/Cosmos3-Experimental",
                subdirectory="4cb0c125-49a8-4e66-aebb-06e100affdb0",
                revision="6ca42c5d0b96cb133e811c1bcced048d4acfaa12",
            ),
        )
    )

    register_checkpoint(
        CheckpointConfig(
            uuid=uuid4().hex,
            name="Wan2.1/vae",
            s3=CheckpointFileS3(
                uri="s3://bucket/pretrained/tokenizers/video/wan2pt1/Wan2.1_VAE.pth",
            ),
            hf=CheckpointFileHf(
                repository="Wan-AI/Wan2.1-T2V-14B",
                revision="a064a6c71f5be440641209c07bf2a5ce7a2ff5e4",
                filename="Wan2.1_VAE.pth",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid=uuid4().hex,
            name="Wan2.2/vae",
            s3=CheckpointFileS3(
                uri="s3://bucket/pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth",
            ),
            hf=CheckpointFileHf(
                repository="Wan-AI/Wan2.2-TI2V-5B",
                revision="921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
                filename="Wan2.2_VAE.pth",
            ),
        ),
    )

    register_checkpoint(
        CheckpointConfig(
            uuid=uuid4().hex,
            name="AVAE",
            s3=CheckpointDirS3(
                uri="s3://bucket/pretrained/tokenizers/audio/avae",
            ),
            hf=CheckpointDirHf(
                repository="nvidia/Cosmos3-Experimental",
                revision="c243efd72b3c9138196ba903deb4a0ad26f2bf20",
                subdirectory="avae",
            ),
        ),
    )


CHECKPOINTS: dict[str, CheckpointConfig] = {
    # Created using 'convert_model_to_dcp'
    "Cosmos3-Nano-Train": CheckpointConfig(
        name="Cosmos3-Nano-Train",
        uuid=uuid4().hex,
        config_file=str(CONFIG_DIR / "model/Cosmos3-Nano.yaml"),
        experiment="cosmos3_ga_16bm8b_v1_midtrain",
        s3=CheckpointDirS3(
            uri="s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/cosmos3_ga_midtraining/cosmos3_ga_16bm8b_v1_midtrain/checkpoints/iter_000012000/",
        ),
        hf=CheckpointDirHf(
            repository="nvidia/Cosmos3-Experimental",
            revision="a3743aa1092fbefc9c6f6ae8c8c17e56a78aea4b",
            subdirectory="e77a607f-af13-4321-bbf5-92f3e90f05e1-train",
        ),
    ),
    "Cosmos3-Super-Train": CheckpointConfig(
        name="Cosmos3-Super-Train",
        uuid=uuid4().hex,
        config_file=str(CONFIG_DIR / "model/Cosmos3-Super.yaml"),
        experiment="cosmos3_ga_64bm32b_v1_midtrain",
        s3=CheckpointDirS3(
            uri="s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/cosmos3_ga_midtraining/cosmos3_ga_64bm32b_v1_midtrain/checkpoints/iter_000005000/",
        ),
        hf=CheckpointDirHf(
            repository="nvidia/Cosmos3-Experimental",
            revision="a3743aa1092fbefc9c6f6ae8c8c17e56a78aea4b",
            subdirectory="d92be19a-42ab-4a96-bdf2-98d1c9724cd9-train",
        ),
    ),
}
"""Checkpoints used by tests."""


class DatasetConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    hf: CheckpointDirHf
    """Config for dataset on Hugging Face."""


DATASETS = {
    "nvidia/bridge-v2-subset-synthetic-captions": DatasetConfig(
        hf=CheckpointDirHf(
            repository_type=RepositoryType.DATASET,
            repository="nvidia/bridge-v2-subset-synthetic-captions",
            revision="46468e12ac0dd36901e9e3240d4fc7620942b5d7",
            subdirectory="sft_dataset_bridge",
        ),
    ),
    "nvidia/LIBERO_LeRobot_v3": DatasetConfig(
        hf=CheckpointDirHf(
            repository_type=RepositoryType.DATASET,
            repository="nvidia/LIBERO_LeRobot_v3",
            revision="ddc1edeb6e51e2b7d4d2ba7a1433daaecd37aa64",
        ),
    ),
    "nvidia/bridge_lerobot_v3": DatasetConfig(
        hf=CheckpointDirHf(
            repository_type=RepositoryType.DATASET,
            repository="nvidia/bridge_lerobot_v3",
            revision="b887e193b141f2fe5b6e3d567577aa51c475693b",
        ),
    ),
}
"""Datasets used by tests."""
