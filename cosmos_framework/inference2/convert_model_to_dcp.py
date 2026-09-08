# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Convert a Cosmos3 checkpoint (diffusers repo) to a DCP checkpoint — inference2 port.

Migrated off the old Overrides/Args stack: the checkpoint is resolved by
``inference2.checkpoints.resolve_checkpoint`` instead of ``CheckpointOverrides``. The model
config is read from the checkpoint's own ``config.json["model"]`` (the complete,
deployment-correct config), matching the original ``load_model_config_dict`` JSON branch —
this is what lets the frozen Wan VAE / AVAE resolve. The DCP-save path
(``from_pretrained_dcp`` -> ``get_model_state_dict`` -> ``dcp.save``) is unchanged.

inference2 is self-contained: the diffusers loader (``_model_io``), config serialization
(``_config``), checkpoint registration (``_checkpoints_registry``) and public-alias
transforms (``_public_model_config``) are local copies of the former
``cosmos_framework.inference`` modules, so nothing is imported from that package.
"""

import os

# Self-contained env setup (replaces inference.common.init.init_script): force a CPU model
# build and quiet the tokenizers fork warning BEFORE any framework/torch import.
os.environ["COSMOS_DEVICE"] = "cpu"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import math
import shutil
from pathlib import Path
from typing import Annotated

import pydantic
import torch
import torch.distributed.checkpoint as dcp
import tyro
from torch.distributed.checkpoint.filesystem import FileSystemWriter
from torch.distributed.checkpoint.state_dict import get_model_state_dict

torch.set_grad_enabled(False)  # conversion: no autograd

from cosmos_framework.checkpoint.dcp import CustomSavePlanner
from cosmos_framework.inference2._checkpoints_registry import register_checkpoints
from cosmos_framework.inference2._config import deserialize_config_dict
from cosmos_framework.inference2._model_io import Cosmos3OmniConfig, Cosmos3OmniModel
from cosmos_framework.inference2._public_model_config import (
    build_public_model_config,
    load_model_config_from_hf_config,
)
from cosmos_framework.inference2.checkpoints import resolve_checkpoint
from cosmos_framework.utils.checkpoint_db import _CHECKPOINTS

_AVAE_REGISTRY_URI = "s3://bucket/pretrained/tokenizers/audio/avae"


def _redirect_avae_to_local(hf_path: Path) -> None:
    """Point the AVAE registry entry at ``hf_path/sound_tokenizer/`` so hydra instantiation
    resolves the sound tokenizer from the local sibling dir instead of the HF Hub."""
    sound_tokenizer_dir = hf_path / "sound_tokenizer"
    if not sound_tokenizer_dir.is_dir():
        return
    register_checkpoints()
    avae = _CHECKPOINTS.get(_AVAE_REGISTRY_URI)
    if avae is not None:
        avae.hf._path = str(sound_tokenizer_dir)


class Args(pydantic.BaseModel):
    checkpoint: str
    """Checkpoint: a local diffusers dir, a friendly name (e.g. ``Cosmos3-Nano``), or an HF repo id."""
    output_path: Annotated[Path, tyro.conf.arg(aliases=("-o",))]
    """Output DCP checkpoint directory."""


def convert_model_to_dcp(args: Args) -> None:
    print("Loading model...")
    hf_path = resolve_checkpoint(args.checkpoint)
    _redirect_avae_to_local(hf_path)

    # Source the model config from the checkpoint's own config.json["model"] — the complete,
    # deployment-correct config the publisher wrote (tokenizer _target_ + bucket_name="bucket",
    # credentials, resolution, sampler variant, ...). This is exactly what the original
    # convert_model_to_dcp does (load_model_config_dict -> JSON branch); it's why the frozen
    # Wan VAE / AVAE resolve via register_checkpoints. Reconstructing the config from the
    # diffusers repo instead cannot reproduce these deployment-only fields.
    model_dict = load_model_config_from_hf_config(deserialize_config_dict(hf_path / "config.json"))
    hf_config = Cosmos3OmniConfig(model=build_public_model_config(model_dict))

    hf_model = Cosmos3OmniModel.from_pretrained_dcp(hf_path, config=hf_config)
    state_dict = get_model_state_dict(hf_model.model)

    # Match transformers default max shard size = 5GB.
    max_shard_size = 5 * 1024**3
    model_size = sum(p.numel() * p.element_size() for p in state_dict.values() if isinstance(p, torch.Tensor))
    thread_count = math.ceil(model_size / max_shard_size)

    print("Saving model...")
    output_path = args.output_path.expanduser().absolute()
    storage_writer = FileSystemWriter(output_path / "model", thread_count=thread_count)
    dcp.save(state_dict=state_dict, storage_writer=storage_writer, planner=CustomSavePlanner())
    checkpoint_json = hf_path / "checkpoint.json"
    if checkpoint_json.is_file():
        shutil.copy(checkpoint_json, output_path / "checkpoint.json")
    hf_config.save_pretrained(output_path / "model")
    print(f"Saved checkpoint to {output_path}")


def main() -> None:
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.OmitArgPrefixes,))
    convert_model_to_dcp(args)


if __name__ == "__main__":
    main()
