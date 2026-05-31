# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Export config to yaml file."""

from cosmos_framework.inference.common.init import init_script

init_script(
    env={
        "COSMOS_TRAINING": "1",
        "COSMOS_DEVICE": "cpu",
    }
)

from typing import Annotated

import pydantic
import tyro

from cosmos_framework.inference.common.args import ConfigOverrides, ResolvedPath, tyro_cli
from cosmos_framework.inference.common.config import InvalidMode, serialize_config_dict, unstructure_config


class Args(pydantic.BaseModel):
    output_file: Annotated[ResolvedPath, tyro.conf.arg(aliases=("-o",))]
    config: tyro.conf.OmitArgPrefixes[ConfigOverrides] = ConfigOverrides.model_construct()

    invalid: InvalidMode = "error"
    """How to handle unknown field types."""
    config_key: str | None = None
    """Config key to export."""


def export_config(args: Args):
    config_args = args.config.build_config()
    if args.output_file.suffix not in [".yaml", ".yml"]:
        raise ValueError("Output file must have a .yaml or .yml extension")

    config = config_args.load_config()

    # Extract key
    if args.config_key:
        for k in args.config_key.split("."):
            config = getattr(config, k)

    config_dict: dict = unstructure_config(config, invalid=args.invalid)

    # Re-create key structure
    if args.config_key:
        c: dict = config_dict
        for k in reversed(args.config_key.split(".")):
            config_dict = {k: c}
            c = config_dict

    # Add metadata
    assert "_metadata" not in config_dict
    config_dict["_metadata"] = {
        "args": config_args.model_dump(mode="json"),
    }

    serialize_config_dict(config_dict, args.output_file)
    print(f"Saved config to {args.output_file}")


def main():
    args = tyro_cli(Args, description=__doc__)
    export_config(args)


if __name__ == "__main__":
    main()
