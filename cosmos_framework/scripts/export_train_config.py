# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Convert internal training config to public config."""

from cosmos_framework.inference.common.init import init_script

init_script()

from pathlib import Path
from typing import Annotated

import omegaconf
import pydantic
import tyro

from cosmos_framework.inference.common.args import CheckpointOverrides, ResolvedFilePath, ResolvedPath, tyro_cli
from cosmos_framework.inference.common.config import (
    CONFIG_DIR,
    deserialize_config_dict,
    serialize_config,
    structure_config,
)
from cosmos_framework.inference.common.init import init_output_dir


def _validate_config_file(v: Path) -> Path:
    if v.suffix != ".yaml":
        raise ValueError(f"Config file must be a YAML file: {v}")
    return v


ConfigFilePath = Annotated[ResolvedFilePath, pydantic.AfterValidator(_validate_config_file)]


class Args(pydantic.BaseModel):
    output_dir: Annotated[ResolvedPath, tyro.conf.arg(aliases=("-o",))]

    config_file: ConfigFilePath
    """Hydra config yaml file."""
    config_overrides: list[str] = pydantic.Field(default_factory=list)
    """Hydra config overrides."""

    checkpoint: CheckpointOverrides = CheckpointOverrides.model_construct()
    """Checkpoint arguments."""


def export_train_config(args: Args) -> None:
    checkpoint_args = args.checkpoint.build_checkpoint(checkpoints={})
    init_output_dir(args.output_dir)

    # Load config
    config_dict = deserialize_config_dict(args.config_file)
    overrides_omegaconf = omegaconf.OmegaConf.from_dotlist(args.config_overrides)
    config_omegaconf = omegaconf.OmegaConf.merge(config_dict, overrides_omegaconf)

    config_omegaconf.job.wandb_mode = "disabled"

    # Set checkpoint
    checkpoint_dict = deserialize_config_dict(CONFIG_DIR / "checkpoint/local.yaml")
    checkpoint_dict["type"] = config_omegaconf.checkpoint.type
    checkpoint_dict["load_path"] = "???"
    checkpoint_dict["keys_to_skip_loading"] = ["net_ema."]
    config_omegaconf.checkpoint = checkpoint_dict

    # Set model
    model_dict = checkpoint_args.load_model_config_dict()
    model_dict["config"]["ema"] = config_omegaconf.model.config.ema
    config_omegaconf.model = model_dict

    # Filter callbacks
    def process_callback(name: str, callback: omegaconf.DictConfig) -> omegaconf.DictConfig | None:
        if hasattr(callback, "save_s3"):
            setattr(callback, "save_s3", False)
        return callback

    callbacks_dict = {}
    for k, v in config_omegaconf.trainer.callbacks.items():
        v = process_callback(k, v)
        if v is not None:
            callbacks_dict[k] = v
    config_omegaconf.trainer.callbacks = callbacks_dict

    omegaconf.OmegaConf.save(config_omegaconf, args.output_dir / "config_raw.yaml")
    omegaconf.OmegaConf.resolve(config_omegaconf)
    config = structure_config(config_omegaconf)
    config.validate()
    serialize_config(config, args.output_dir / "config.yaml")


def main() -> None:
    args = tyro_cli(Args, description=__doc__, config=(tyro.conf.OmitArgPrefixes,))
    export_train_config(args)


if __name__ == "__main__":
    main()
