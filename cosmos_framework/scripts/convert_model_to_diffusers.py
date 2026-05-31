# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Convert transformers checkpoint to diffusers checkpoint."""

from cosmos_framework.inference.common.init import init_script

init_script(
    env={
        "COSMOS_DEVICE": "cpu",
    }
)

import json
import shutil
import struct
from pathlib import Path
from typing import Annotated

import pydantic
import tyro

from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.args import CheckpointOverrides, ResolvedPath
from cosmos_framework.inference.common.checkpoints import register_checkpoints
from cosmos_framework.inference.common.config import deserialize_config_dict, serialize_config_dict
from cosmos_framework.inference.common.public_model_config import (
    build_public_model_config,
    load_model_config_from_hf_config,
)
from cosmos_framework.utils.checkpoint_db import CheckpointConfig, CheckpointDirHf


class Args(pydantic.BaseModel):
    checkpoint: CheckpointOverrides
    """Transformers checkpoint."""
    output_path: Annotated[ResolvedPath, tyro.conf.arg(aliases=("-o",))]
    """Output diffusers checkpoint directory."""

    config_only: bool = False
    """If True, only save config."""

    remap_sound_tokenizer_keys: bool = True
    """If True, remap the legacy AVAE sound tokenizer state dict into the
    diffusers OobleckDecoder layout (and save the result as
    `diffusion_pytorch_model.safetensors`). Off by default — the sound
    tokenizer is written verbatim under `model.safetensors`."""

    remap_time_embedder_keys: bool = True
    """If True, remap the transformer's `time_embedder` state dict from the
    legacy `nn.Sequential` layout (`mlp.0.*` / `mlp.2.*`) to the diffusers
    `TimestepEmbedding` layout (`linear_1.*` / `linear_2.*`). Off by default —
    keys are forwarded verbatim."""


class SafetensorsIndexMetadata(pydantic.BaseModel):
    total_size: int = 0


class SafetensorsIndex(pydantic.BaseModel):
    metadata: SafetensorsIndexMetadata = pydantic.Field(default_factory=SafetensorsIndexMetadata)
    weight_map: dict[str, str] = pydantic.Field(default_factory=dict)

    def update(self, safetensors_path: Path, rel_path: str):
        with safetensors_path.open("rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size).decode("utf-8"))

        for name, info in header.items():
            if name == "__metadata__":
                continue
            self.metadata.total_size += info["data_offsets"][1] - info["data_offsets"][0]
            if name in self.weight_map:
                raise ValueError(f"Key {name} already in weight map")
            self.weight_map[name] = rel_path

    def update_dir(self, safetensors_dir: Path, rel_path: str):
        for safetensors_path in safetensors_dir.glob("*.safetensors"):
            self.update(safetensors_path, f"{rel_path}/{safetensors_path.name}")


def convert_model_to_diffusers(args: Args):
    args.output_path.mkdir(parents=True, exist_ok=True)

    register_checkpoints()
    checkpoint_config = args.checkpoint.build_checkpoint(checkpoints=OmniSetupOverrides.CHECKPOINTS)
    checkpoint_path = checkpoint_config.download_checkpoint()
    model_dict = load_model_config_from_hf_config(deserialize_config_dict(checkpoint_path / "config.json"))

    assert model_dict["config"]["action_gen"]
    assert model_dict["config"]["sound_gen"]

    sound_tokenizer_dir, sound_tokenizer_name = model_dict["config"]["sound_tokenizer"]["avae_path"].rsplit("/", 1)
    sound_tokenizer_checkpoint = CheckpointConfig.maybe_from_uri(f"s3://bucket/{sound_tokenizer_dir}")
    assert sound_tokenizer_checkpoint is not None
    sound_tokenizer_path = Path(sound_tokenizer_checkpoint.hf.download()) / sound_tokenizer_name
    assert sound_tokenizer_path.is_file(), f"Sound tokenizer checkpoint not found: {sound_tokenizer_path}"
    sound_tokenizer_config_path = sound_tokenizer_path.with_suffix(".json")
    assert sound_tokenizer_config_path.is_file(), f"Sound tokenizer config not found: {sound_tokenizer_config_path}"

    vision_encoder_model = model_dict["config"]["vlm_config"]["tokenizer"]["pretrained_model_name"]

    if not args.config_only:
        from cosmos_framework.scripts._convert_model_to_diffusers import (
            Args as _Args,
        )
        from cosmos_framework.scripts._convert_model_to_diffusers import (
            convert_model_to_diffusers as _convert_model_to_diffusers,
        )

        _args = _Args(
            checkpoint_path=str(checkpoint_path),
            output=str(args.output_path),
            save_pipeline=True,
            dtype="bf16",
            sound_tokenizer_path=str(sound_tokenizer_path),
            sound_tokenizer_config_path=str(sound_tokenizer_config_path),
            include_sound_tokenizer=True,
            remap_sound_tokenizer_keys=args.remap_sound_tokenizer_keys,
            remap_time_embedder_keys=args.remap_time_embedder_keys,
            vision_encoder_model=vision_encoder_model,
            skip_vision_encoder=False,
        )
        _convert_model_to_diffusers(_args)

    # Add vlm files
    vlm_repository = model_dict["config"]["vlm_config"]["model_name"]
    vlm_checkpoint = CheckpointDirHf(
        repository=vlm_repository,
        revision="main",
        include=("*.jinja", "*.json", "*.txt"),
    )
    vlm_checkpoint_path = vlm_checkpoint.download()
    for pattern in vlm_checkpoint.include:
        for p in Path(vlm_checkpoint_path).glob(pattern):
            shutil.copy(p, args.output_path / p.name)

    # Add top-level config
    config_dict = deserialize_config_dict(args.output_path / "config.json")
    config_dict["architectures"] = ["Cosmos3ForConditionalGeneration"]
    # vLLM's `_prepare_weights` breaks after the first pattern with any match, so
    # collapse to a single glob spanning both component subdirs. The unified
    # `model.safetensors.index.json` written below dedupes the consolidated shard.
    config_dict["allow_patterns_overrides"] = ["*/*.safetensors"]
    config_dict["model"] = build_public_model_config(model_dict)
    serialize_config_dict(config_dict, args.output_path / "config.json")

    # Add top-level index
    index = SafetensorsIndex()
    index.update_dir(args.output_path / "transformer", "transformer")
    vision_encoder_rel = "vision_encoder/model.safetensors"
    index.update(args.output_path / vision_encoder_rel, vision_encoder_rel)
    (args.output_path / "model.safetensors.index.json").write_text(index.model_dump_json(indent=2))

    shutil.copy(checkpoint_path / "checkpoint.json", args.output_path / "checkpoint.json")

    print(f"Saved diffusers checkpoint to {args.output_path}")


def main():
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.OmitArgPrefixes,))
    convert_model_to_diffusers(args)


if __name__ == "__main__":
    main()
