# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Convert DCP checkpoint to Hugging Face model."""

from cosmos_framework.inference.common.init import init_script

init_script(
    env={
        "COSMOS_DEVICE": "cpu",
        "COSMOS_TRAINING": "1",
    }
)

import json
from pathlib import Path
from typing import Annotated, Any, Callable

import attrs
import safetensors.torch
import torch.distributed.checkpoint as dcp
import tyro
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

from cosmos_framework.checkpoint.dcp import CustomLoadPlanner
from cosmos_framework.checkpoint.s3_filesystem import S3StorageReader
from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig
from cosmos_framework.inference.common.args import (
    CheckpointOverrides,
    ParallelismOverrides,
    ResolvedPath,
    tyro_cli,
)
from cosmos_framework.inference.common.checkpoints import register_checkpoints
from cosmos_framework.inference.common.config import serialize_config_dict
from cosmos_framework.inference.common.distillation_export import (
    build_student_checkpoint_metadata,
    resolve_vision_checkpoint_path,
    sanitize_student_model_config,
    sanitize_student_public_model_config,
)
from cosmos_framework.inference.common.init import is_rank0
from cosmos_framework.inference.common.public_model_config import build_public_model_config
from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.utils import log
from cosmos_framework.utils.checkpoint_db import CheckpointConfig, sanitize_uri
from cosmos_framework.utils.lazy_config.registry import convert_target_to_string

_INTERNAL_VISUAL_PREFIX = "model.net.language_model.visual."
_EXPORTED_VISUAL_PREFIX = "model.visual."


def _coerce_to_base_model(model_dict: dict[str, Any]) -> None:
    """For distillation training configs, rewrite the target to the base
    OmniMoTModel so the exported checkpoint only contains the student network."""
    target = model_dict.get("_target_", "")
    base_model_target = convert_target_to_string(OmniMoTModel)
    if target != base_model_target:
        log.info(f"Overriding model target from {target} to OmniMoTModel for export")

    sanitize_student_model_config(
        model_dict,
        base_model_target=base_model_target,
        base_config_type=convert_target_to_string(OmniMoTModelConfig),
        base_config_field_names={field.name for field in attrs.fields(OmniMoTModelConfig)},
    )


class Args(ParallelismOverrides):
    checkpoint: CheckpointOverrides = CheckpointOverrides.model_construct()
    output_dir: Annotated[ResolvedPath, tyro.conf.arg(aliases=("-o",))]
    """Output model directory."""
    config_only: bool = False
    """If True, only export config."""
    student_only_checkpoint_metadata: bool = False
    """If True, omit source checkpoint and credential paths from checkpoint metadata."""
    vit: bool = True
    """If True, export ViT weights."""
    vit_checkpoint_path: ResolvedPath | None = None
    """Optional local Hugging Face checkpoint directory containing ViT weights."""


def _load_safetensor_weights(model_dir: Path, predicate: Callable[[str], bool]) -> dict[str, Any]:
    """Load weights from a safetensors file."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards = {v for k, v in weight_map.items() if predicate(k)}
        vision_weights = {}
        for shard in shards:
            tensors = safetensors.torch.load_file(model_dir / shard)
            vision_weights.update({k: v for k, v in tensors.items() if predicate(k)})
    else:
        tensors = safetensors.torch.load_file(model_dir / "model.safetensors")
        vision_weights = {k: v for k, v in tensors.items() if predicate(k)}
    return vision_weights


def _rewrite_visual_fqns_for_vfm(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Map HF visual tower FQNs to OmniMoTModel's internal visual tower FQNs."""
    remapped_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(_EXPORTED_VISUAL_PREFIX):
            key = _INTERNAL_VISUAL_PREFIX + key[len(_EXPORTED_VISUAL_PREFIX) :]
        remapped_state_dict[key] = value
    return remapped_state_dict


def export_model(args: Args) -> None:
    register_checkpoints()
    checkpoint_args = args.checkpoint.build_checkpoint(checkpoints={})
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    log.info("Loading config...")
    model_dict = checkpoint_args.load_model_config_dict()
    if not model_dict["config"]["ema"]["enabled"]:
        checkpoint_args.use_ema_weights = False
    model_dict["config"]["ema"]["enabled"] = False

    # Download VLM checkpoint
    if args.vit:
        configured_vlm_checkpoint = model_dict["config"]["vlm_config"]["pretrained_weights"]["backbone_path"]

        def download_vlm_checkpoint(configured_uri: str) -> str:
            sanitized_uri = sanitize_uri(configured_uri)
            checkpoint: CheckpointConfig | None = CheckpointConfig.maybe_from_uri(sanitized_uri)
            if checkpoint is None:
                raise ValueError(f"Invalid checkpoint path: {configured_uri}")
            return checkpoint.hf.download()

        vlm_checkpoint_path = resolve_vision_checkpoint_path(
            local_path=str(args.vit_checkpoint_path) if args.vit_checkpoint_path is not None else None,
            configured_uri=configured_vlm_checkpoint,
            download_checkpoint=download_vlm_checkpoint,
        )
    else:
        vlm_checkpoint_path = None

    # Load model
    log.info("Loading model...")
    _coerce_to_base_model(model_dict)
    hf_config = Cosmos3OmniConfig(model=build_public_model_config(model_dict))
    hf_config.save_pretrained(args.output_dir)
    hf_model = Cosmos3OmniModel(hf_config)

    # Save model
    log.info("Saving model...")
    if not args.config_only:
        # Load checkpoint
        if checkpoint_args.checkpoint_path.startswith("s3://"):
            storage_reader = S3StorageReader(
                credential_path=checkpoint_args.credential_path,
                path=checkpoint_args.checkpoint_path,
            )
        else:
            storage_reader = FileSystemReader(checkpoint_args.checkpoint_path)
        state_dict = get_model_state_dict(hf_model.model)
        dcp.load(
            state_dict=state_dict,
            storage_reader=storage_reader,
            planner=CustomLoadPlanner(
                load_ema_to_reg=checkpoint_args.use_ema_weights,
            ),
        )
        state_dict = get_model_state_dict(
            hf_model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )
        if not is_rank0():
            return

        # Load ViT from VLM checkpoint
        if args.vit:
            assert vlm_checkpoint_path is not None
            vit_state_dict = _load_safetensor_weights(
                Path(vlm_checkpoint_path), lambda x: x.startswith("model.visual.")
            )
            assert vit_state_dict, "No vision weights found"
            state_dict.update(_rewrite_visual_fqns_for_vfm(vit_state_dict))

        # Save checkpoint
        hf_model.save_pretrained(
            args.output_dir,
            state_dict=state_dict,
        )

    # Re-write 'config.json' to apply replacements.
    hf_config_file = args.output_dir / "config.json"
    hf_config_json = json.loads(hf_config_file.read_text())
    if args.student_only_checkpoint_metadata:
        sanitize_student_public_model_config(hf_config_json["model"])
    hf_config_json["model_type"] = "cosmos3_omni"
    serialize_config_dict(hf_config_json, hf_config_file)

    # Write 'checkpoint.json' last to indicate that the model is complete.
    checkpoint_metadata = (
        build_student_checkpoint_metadata(use_ema_weights=checkpoint_args.use_ema_weights)
        if args.student_only_checkpoint_metadata
        else checkpoint_args.model_dump(mode="json")
    )
    serialize_config_dict(checkpoint_metadata, args.output_dir / "checkpoint.json")

    print(f"Saved model to {args.output_dir}")


def main() -> None:
    args = tyro_cli(Args, description=__doc__, config=(tyro.conf.OmitArgPrefixes,))
    export_model(args)


if __name__ == "__main__":
    main()
