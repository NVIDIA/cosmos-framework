# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Export all nodes from the config store to yaml files."""

from cosmos_framework.inference.common.init import init_script

init_script(env={"COSMOS_TRAINING": "1"})

import importlib
from typing import Annotated, Any, Generator

import pydantic
import tyro
import yaml
from hydra.core.config_store import ConfigNode, ConfigStore

from cosmos_framework.inference.common.args import ResolvedPath, tyro_cli
from cosmos_framework.inference.common.config import _InvalidMode, apply_config_replacements, unstructure_config
from cosmos_framework.utils import config_helper, log


class Args(pydantic.BaseModel):
    output_dir: Annotated[ResolvedPath, tyro.conf.arg(aliases=("-o",))]

    config_file: str = "configs/base/config.py"
    """Config module path."""

    invalid: _InvalidMode = "error"
    """How to handle unknown field types."""


def _iter_config_nodes(repo: dict) -> Generator[ConfigNode]:
    """Iteratively yield all config store nodes."""
    # Depth-First Search.
    stack = [repo]
    while stack:
        current_dict = stack.pop()
        for value in current_dict.values():
            if isinstance(value, dict):
                stack.append(value)
            else:
                yield value


def _save_config(
    config: Any, name: str, group: str | None, package: str | None, output_dir: ResolvedPath, invalid: _InvalidMode
):
    config_dict = unstructure_config(config, invalid=invalid)
    config_str = yaml.safe_dump(config_dict)
    config_str = apply_config_replacements(config_str)

    node_file = output_dir.joinpath(*filter(None, [group, name])).with_suffix(".yaml")
    node_file.parent.mkdir(parents=True, exist_ok=True)
    with node_file.open("w") as f:
        if package:
            f.write(f"# @package {package}\n")
        f.write(config_str)


def export_config_store(args: Args):
    config_module = importlib.import_module(config_helper.get_config_module(args.config_file))
    config = config_module.make_config()

    cs = ConfigStore.instance()
    _save_config(
        config, "base_config.yaml", group=None, package="_global_", output_dir=args.output_dir, invalid=args.invalid
    )
    for node in _iter_config_nodes(cs.repo):
        if node.name.startswith("_"):
            continue
        if node.group in ["hydra"]:
            continue
        try:
            _save_config(
                node.node,
                name=node.name,
                group=node.group,
                package=node.package,
                output_dir=args.output_dir,
                invalid=args.invalid,
            )
        except Exception as e:
            log.error(f"Error saving config '{node.name}': {e}")
            continue


def main():
    args = tyro_cli(Args, description=__doc__)
    export_config_store(args)


if __name__ == "__main__":
    main()
