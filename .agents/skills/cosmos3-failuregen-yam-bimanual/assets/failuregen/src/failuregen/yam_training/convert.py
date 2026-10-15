# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Initialize joint-domain expansion or existing-domain EEF fine-tuning."""

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

from cosmos_framework.checkpoint.dcp import CustomSavePlanner
from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.args import CheckpointOverrides, ResolvedPath
from cosmos_framework.inference.common.checkpoints import register_checkpoints
from cosmos_framework.inference.common.public_model_config import build_public_model_config
from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel
from cosmos_framework.utils.checkpoint_db import _CHECKPOINTS

_AVAE_REGISTRY_URI = "s3://bucket/pretrained/tokenizers/audio/avae"


def _redirect_avae_to_local(hf_path):
    """Point the AVAE registry entry at hf_path/sound_tokenizer/.

    Pre-seeds the CheckpointDirHf._path cache so the registered AVAE checkpoint
    resolves to the local sibling directory instead of fetching the pinned
    revision of nvidia/Cosmos3-Nano from the HF Hub during hydra instantiation.
    """
    sound_tokenizer_dir = hf_path / "sound_tokenizer"
    if not sound_tokenizer_dir.is_dir():
        return
    register_checkpoints()
    avae = _CHECKPOINTS.get(_AVAE_REGISTRY_URI)
    if avae is not None:
        avae.hf._path = str(sound_tokenizer_dir)


class Args(pydantic.BaseModel):
    checkpoint: CheckpointOverrides
    """Hugging Face checkpoint."""
    output_path: Annotated[ResolvedPath, tyro.conf.arg(aliases=("-o",))]
    """Output DCP checkpoint directory."""
    append_yam_domain: bool = False
    """Append YAM domain 32 to the unmodified 32-domain Nano checkpoint (CPU)."""
    reuse_yam_eef_domain: bool = False
    """Keep all 32 pretrained domains and fine-tune existing xdof_yam/domain 16."""


def convert_model_to_dcp(args: Args):
    if args.append_yam_domain == args.reuse_yam_eef_domain:
        raise ValueError("Choose exactly one of --append-yam-domain or --reuse-yam-eef-domain")
    if args.output_path.exists() and any(args.output_path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output {args.output_path}")
    print("Loading model...")
    checkpoint_config = args.checkpoint.build_checkpoint(checkpoints=OmniSetupOverrides.CHECKPOINTS)
    hf_path = checkpoint_config.download_checkpoint()
    _redirect_avae_to_local(hf_path)
    model_dict = checkpoint_config.load_model_config_dict()
    hf_config = Cosmos3OmniConfig(model=build_public_model_config(model_dict))
    if hf_config.model["config"].get("num_embodiment_domains") != 32:
        raise ValueError("YAM initialization expects the original 32-domain Cosmos3-Nano; inspect this checkpoint")
    hf_model = Cosmos3OmniModel.from_pretrained_dcp(hf_path, config=hf_config)
    migration = None
    if args.append_yam_domain:
        from failuregen.yam_training.contract import CONTRACT
        from failuregen.yam_training.expand import expand_action_domains

        migration = expand_action_domains(hf_model.model.net, 33)
        hf_config.model["config"]["num_embodiment_domains"] = 33
        hf_model.model.config.num_embodiment_domains = 33
        migration.update(source=str(hf_path), contract=CONTRACT)
    else:
        from failuregen.yam_training import register_domain
        from failuregen.yam_training.contract import EEF_CONTRACT

        register_domain("eef")
        if any(getattr(hf_model.model.net, name).num_domains != 32 for name in ("action2llm", "llm2action")):
            raise ValueError("Expected 32-domain pretrained action projectors")
        # No projector weights or biases are initialized or expanded in this path.
        migration = {
            "old_domain_count": 32,
            "new_domain_count": 32,
            "existing_rows_preserved": True,
            "initialization": "reuse_existing_domain",
            "domain_id": 16,
            "source": str(hf_path),
            "contract": EEF_CONTRACT,
        }
    state_dict = get_model_state_dict(hf_model.model)

    # Match transformers default max shard size = 5GB.
    max_shard_size = 5 * 1024**3
    model_size = sum(p.numel() * p.element_size() for p in state_dict.values() if isinstance(p, torch.Tensor))
    thread_count = math.ceil(model_size / max_shard_size)

    print("Saving model...")
    storage_writer = FileSystemWriter(args.output_path / "model", thread_count=thread_count)
    dcp.save(state_dict=state_dict, storage_writer=storage_writer, planner=CustomSavePlanner())
    # ``checkpoint.json`` only exists for DCP-format source repos (e.g. Cosmos3-Nano);
    # safetensors/diffusers-layout repos (e.g. Cosmos3-Edge) don't ship it. Copy when present.
    source_checkpoint_json = hf_path / "checkpoint.json"
    if source_checkpoint_json.exists():
        shutil.copy(source_checkpoint_json, args.output_path / "checkpoint.json")
    hf_config.save_pretrained(args.output_path / "model")
    if migration is not None:
        (args.output_path / "yam_checkpoint_migration.json").write_text(json.dumps(migration, indent=2) + "\n")
    print(f"Saved checkpoint to {args.output_path}")


def main():
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.OmitArgPrefixes,))
    convert_model_to_dcp(args)


if __name__ == "__main__":
    main()
