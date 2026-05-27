# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Convert a Hugging Face model to a DCP checkpoint."""

from cosmos_framework.inference.common.init import init_script

init_script(
    env={
        "COSMOS_DEVICE": "cpu",
    }
)

import json
import math
import shutil
from typing import Annotated

import pydantic
import torch
import torch.distributed.checkpoint as dcp
import tyro
from torch.distributed.checkpoint.filesystem import FileSystemWriter
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.args import CheckpointOverrides, ResolvedPath
from cosmos_framework.inference.common.config import fix_config_dict
from cosmos_framework.inference.model import Cosmos3OmniModel
from cosmos_framework.checkpoint.dcp import CustomSavePlanner


class Args(pydantic.BaseModel):
    checkpoint: CheckpointOverrides
    """Hugging Face checkpoint."""
    output_path: Annotated[ResolvedPath, tyro.conf.arg(aliases=("-o",))]
    """Output DCP checkpoint directory."""


def convert_model_to_dcp(args: Args):
    print("Loading model...")
    checkpoint_config = args.checkpoint.build_checkpoint(checkpoints=OmniSetupOverrides.CHECKPOINTS)
    hf_path = checkpoint_config.download_checkpoint()
    model_dict = json.loads((hf_path / "config.json").read_text())["model"]
    model_dict = fix_config_dict(model_dict)
    hf_model = Cosmos3OmniModel.from_pretrained_dcp(hf_path)
    state_dict = get_model_state_dict(hf_model.model)

    # Match transformers default max shard size = 5GB.
    max_shard_size = 5 * 1024**3
    model_size = sum(p.numel() * p.element_size() for p in state_dict.values() if isinstance(p, torch.Tensor))
    thread_count = math.ceil(model_size / max_shard_size)

    print("Saving model...")
    storage_writer = FileSystemWriter(args.output_path / "model", thread_count=thread_count)
    dcp.save(state_dict=state_dict, storage_writer=storage_writer, planner=CustomSavePlanner())
    shutil.copy(hf_path / "checkpoint.json", args.output_path / "checkpoint.json")
    shutil.copy(hf_path / "config.json", args.output_path / "model/config.json")

    print(f"Saved checkpoint to {args.output_path}")


def main():
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.OmitArgPrefixes,))
    convert_model_to_dcp(args)


if __name__ == "__main__":
    main()
