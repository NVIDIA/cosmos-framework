# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from cosmos_framework.inference.common.init import init_script

init_script(
    training=True,
    env={"COSMOS_TRAINING": "1"},
    default_env={"COSMOS_VERBOSE": "1"},
)

import contextlib
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import hydra
import omegaconf
import pydantic
import torch
import tyro

from cosmos_framework.inference.common.args import ResolvedFilePath, ResolvedPath, tyro_cli
from cosmos_framework.inference.common.checkpoints import register_checkpoints
from cosmos_framework.inference.common.config import (
    ROOT_DIR,
    deserialize_config_dict,
    serialize_config,
    structure_config,
)
from cosmos_framework.inference.common.init import init_output_dir, is_rank0
from cosmos_framework.utils.flags import SMOKE
from cosmos_framework.trainer import ImaginaireTrainer
from cosmos_framework.utils import log

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from cosmos_framework.utils.config import Config
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


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

    dry_run: bool = False
    """Dry run (no training)."""
    resume: bool = True
    """Resume training from the latest checkpoint."""


def _get_config_overrides(args: Args, config_dict: dict) -> list[str]:
    model_name = config_dict["model"]["config"]["vlm_config"]["model_name"]
    overrides = [
        *args.config_overrides,
    ]
    if SMOKE:
        overrides.extend(
            [
                "trainer.max_iter=2",
                "trainer.logging_iter=1",
            ]
        )
        if model_name.startswith("Qwen/Qwen3-VL-"):
            overrides.extend(
                [
                    "model.config.vlm_config.model_instance.config.text_config_overrides.num_hidden_layers=2",
                    "model.config.vlm_config.model_instance.config.text_config_overrides.num_window_layers=2",
                    "model.config.vlm_config.pretrained_weights.enabled=false",
                ]
            )
    return overrides


def _get_job_dir(project: str, group: str, name: str) -> Path:
    output_root = Path(os.environ.get("IMAGINAIRE_OUTPUT_ROOT", "/tmp/imaginaire4-output"))
    return output_root / project / group / name


def train(args: Args) -> None:
    # Build merged config (unresolved) from YAML + CLI overrides.
    config_dict = deserialize_config_dict(args.config_file)
    overrides = _get_config_overrides(args, config_dict)
    log.debug(f"Config overrides: {overrides}")
    overrides_omegaconf = omegaconf.OmegaConf.from_dotlist(overrides)
    config_omegaconf = omegaconf.OmegaConf.merge(config_dict, overrides_omegaconf)

    # Read job identity (literal in YAML, no interpolation) before resolution
    # so we can place per-invocation artifacts under a job.name-scoped subdir.
    job_project = str(config_omegaconf.job.project)
    job_group = str(config_omegaconf.job.group)
    job_name = str(config_omegaconf.job.name)
    job_dir = _get_job_dir(job_project, job_group, job_name)
    effective_output_dir = args.output_dir / job_name

    # Rank-0 directory mgmt. --resume=false wipes both the canonical job dir and
    # the local per-invocation output dir; --resume=true (default) preserves both.
    if is_rank0():
        if not args.resume:
            if job_dir.exists():
                shutil.rmtree(job_dir)
            if effective_output_dir.exists():
                shutil.rmtree(effective_output_dir)
        job_dir.mkdir(parents=True, exist_ok=True)
        effective_output_dir.mkdir(parents=True, exist_ok=True)
        symlink_path = effective_output_dir / "job"
        if symlink_path.is_symlink() or symlink_path.exists():
            os.remove(symlink_path)
        os.symlink(job_dir, symlink_path)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # File logging targets the per-job-name dir; pass job_name so loguru tags
    # every line with [job=<name>].
    init_output_dir(effective_output_dir, resume=args.resume, job_name=job_name)
    log.info(f"Job directory (canonical): {job_dir}")
    log.info(f"Output directory (this invocation): {effective_output_dir}")

    # Persist config snapshots in the per-job dir.
    omegaconf.OmegaConf.save(config_omegaconf, effective_output_dir / "config_raw.yaml")
    omegaconf.OmegaConf.resolve(config_omegaconf)
    config: "Config" = structure_config(config_omegaconf)
    config.validate()
    config.freeze()  # type: ignore
    serialize_config(config, effective_output_dir / "config.yaml")

    # Instantiate
    register_checkpoints()
    with contextlib.chdir(ROOT_DIR):
        # Trainer init sets the rank-local CUDA device before tokenizers allocate weights.
        trainer: "ImaginaireTrainer" = config.trainer.type(config)
        model: "OmniMoTModel" = hydra.utils.instantiate(config.model)
        dataloader_train: "DataLoader" = hydra.utils.instantiate(config.dataloader_train)
        dataloader_val: "DataLoader" = hydra.utils.instantiate(config.dataloader_val)

    if args.dry_run:
        return

    # Start training
    trainer.train(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
    )


def main() -> None:
    args = tyro_cli(Args, description=__doc__, config=(tyro.conf.OmitArgPrefixes,))
    train(args)


if __name__ == "__main__":
    main()
