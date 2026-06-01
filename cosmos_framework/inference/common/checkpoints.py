# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import functools
import re
import shutil
from pathlib import Path
from uuid import uuid4

import pydantic

from cosmos_framework.inference.common.config import CONFIG_DIR
from cosmos_framework.utils.checkpoint_db import (
    CheckpointConfig,
    CheckpointDirHf,
    CheckpointDirS3,
    CheckpointFileHf,
    CheckpointFileS3,
    RepositoryType,
    register_checkpoint,
)
from cosmos_framework.utils.flags import TRAINING

_AVAE_LEGACY_CKPT_NAME = "avae_48k_noncausal_25hz_64ch.ckpt"
_AVAE_LEGACY_JSON_NAME = "avae_48k_noncausal_25hz_64ch.json"

# Inside a residual unit the legacy nn.Sequential layout is [snake1, conv1,
# snake2, conv2]; map the named diffusers attribute back to its sub-index.
_AVAE_RES_UNIT_INNER_INDEX = {"snake1": 0, "conv1": 1, "snake2": 2, "conv2": 3}


def _avae_block_key_to_legacy(key: str, num_blocks: int) -> str:
    """Map a diffusers OobleckDecoder key (`decoder.block.*`) back to the legacy
    nn.Sequential layout (`decoder.layers.*`) the native AVAE loader expects.

    Exact inverse of ``_sound_tokenizer_remap_flat_layout`` in
    ``cosmos_framework/scripts/_convert_model_to_diffusers.py``. The legacy decoder
    is ``Sequential([conv1, block_0..block_{N-1}, snake1, conv2])``; each block is
    ``Sequential([snake1, conv_t1, res_unit1, res_unit2, res_unit3])`` and each
    residual unit is ``Sequential([snake1, conv1, snake2, conv2])``.
    """
    snake1_idx = num_blocks + 1
    conv2_idx = num_blocks + 2

    m = re.fullmatch(r"decoder\.block\.(\d+)\.res_unit(\d+)\.(snake1|conv1|snake2|conv2)\.(.+)", key)
    if m:
        block_idx, res_idx, inner, rest = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
        return f"decoder.layers.{block_idx + 1}.layers.{res_idx + 1}.layers.{_AVAE_RES_UNIT_INNER_INDEX[inner]}.{rest}"
    m = re.fullmatch(r"decoder\.block\.(\d+)\.snake1\.(.+)", key)
    if m:
        return f"decoder.layers.{int(m.group(1)) + 1}.layers.0.{m.group(2)}"
    m = re.fullmatch(r"decoder\.block\.(\d+)\.conv_t1\.(.+)", key)
    if m:
        return f"decoder.layers.{int(m.group(1)) + 1}.layers.1.{m.group(2)}"
    m = re.fullmatch(r"decoder\.conv1\.(.+)", key)
    if m:
        return f"decoder.layers.0.{m.group(1)}"
    m = re.fullmatch(r"decoder\.snake1\.(.+)", key)
    if m:
        return f"decoder.layers.{snake1_idx}.{m.group(1)}"
    m = re.fullmatch(r"decoder\.conv2\.(.+)", key)
    if m:
        return f"decoder.layers.{conv2_idx}.{m.group(1)}"
    return key


def _materialize_avae_ckpt(local_dir: str) -> None:
    """Synthesize the legacy ``.ckpt`` + ``.json`` the native AVAE loader expects
    from the decoder-only ``sound_tokenizer/`` safetensors.

    The new HF layout ships ``sound_tokenizer/{config.json,
    diffusion_pytorch_model.safetensors}`` in the diffusers OobleckDecoder layout
    (``decoder.block.*`` keys, Snake1d ``alpha``/``beta`` shaped ``[1, C, 1]``). The
    native loader in ``cosmos_framework/model/vfm/tokenizers/audio/avae.py`` builds an
    ``nn.Sequential`` decoder keyed ``decoder.layers.*`` with Snake params shaped
    ``[C]`` and loads via ``load_state_dict(strict=False)`` — so without remapping
    the keys, none match and every decoder weight is silently left at init (noise).
    We invert the forward conversion (key remap + snake reshape) and wrap the result
    under ``state_dict``. Decoder-only is sufficient: generation only decodes sound
    latents to a waveform. Idempotent.
    """
    import torch
    from safetensors.torch import load_file

    local = Path(local_dir)
    ckpt_path = local / _AVAE_LEGACY_CKPT_NAME
    json_path = local / _AVAE_LEGACY_JSON_NAME
    if ckpt_path.exists() and json_path.exists():
        return

    safetensors_path = local / "diffusion_pytorch_model.safetensors"
    if not safetensors_path.exists():
        safetensors_path = local / "model.safetensors"
    config_path = local / "config.json"
    if not safetensors_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            f"AVAE shim: expected diffusion_pytorch_model.safetensors (or model.safetensors) "
            f"and {config_path.name} in {local}"
        )

    src = load_file(str(safetensors_path))
    block_ids = {int(m.group(1)) for k in src if (m := re.fullmatch(r"decoder\.block\.(\d+)\..+", k))}
    if not block_ids:
        raise RuntimeError(f"No `decoder.block.*` keys in {safetensors_path}; cannot remap AVAE decoder.")
    num_blocks = max(block_ids) + 1

    state_dict: dict = {}
    for key, value in src.items():
        legacy_key = _avae_block_key_to_legacy(key, num_blocks)
        if (legacy_key.endswith(".alpha") or legacy_key.endswith(".beta")) and value.ndim == 3:
            value = value.reshape(-1).contiguous()  # Snake1d [1, C, 1] -> [C]
        state_dict[legacy_key] = value
    if any(k.startswith("decoder.block.") for k in state_dict):
        raise RuntimeError("`decoder.block.*` keys remain after AVAE remap; conversion is incomplete.")

    if not ckpt_path.exists():
        torch.save({"state_dict": state_dict}, str(ckpt_path))
    if not json_path.exists():
        shutil.copyfile(str(config_path), str(json_path))


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
                repository="nvidia/Cosmos3-Nano",
                revision="main",
                subdirectory="sound_tokenizer",
            ),
            # The sound_tokenizer/ safetensors are decoder-only and use the diffusers
            # OobleckDecoder key layout; _materialize_avae_ckpt remaps them back to the
            # legacy decoder.layers.* layout the native AVAE loader expects.
            post_download=_materialize_avae_ckpt,
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
            uri="s3://bucket1/cosmos3_vfm/cosmos3_ga_midtraining/cosmos3_ga_16bm8b_v1_midtrain/checkpoints/iter_000012000/",
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
            uri="s3://bucket1/cosmos3_vfm/cosmos3_ga_midtraining/cosmos3_ga_64bm32b_v1_midtrain/checkpoints/iter_000005000/",
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
