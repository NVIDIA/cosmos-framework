# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Lazy config builders for explicitly selected local simulation experiments."""

from __future__ import annotations

import copy
from typing import Any

from omegaconf import DictConfig

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.data.generator.local_simulation.dataloader import (
    LocalSimulationDataLoader,
    LocalSimulationJointDataLoader,
)
from cosmos_framework.data.generator.local_simulation.dataset import LocalSimulationDataset
from cosmos_framework.data.generator.local_simulation.resume import LocalSimulationResumeCallback


def make_local_dataloader_config(
    model: str,
    *,
    stratum: str | None = None,
    transform_overrides: dict[str, Any] | None = None,
    max_num_frames: int | None = None,
) -> DictConfig:
    """Build a new loader without instantiating a dataset or reading any files."""
    return L(LocalSimulationDataLoader)(
        dataset=L(LocalSimulationDataset)(
            bundle_path=f"${{oc.env:COSMOS_SIM_{model.upper()}_LOCAL_BUNDLE}}",
            model=model,
            stratum=stratum,
            seed=42,
            transform_overrides=dict(transform_overrides or {}),
            max_num_frames=max_num_frames,
        ),
        batch_size=1,
        num_workers=0,
        prefetch_factor=1,
        pin_memory=False,
        shuffle=True,
        group_by_stratum=model == "transfer",
        seed=42,
    )


def make_local_experiment_config(
    source: dict[str, Any] | DictConfig,
    model: str,
    *,
    name: str | None = None,
    transform_overrides: dict[str, Any] | None = None,
    preserve_transfer_streams: bool = False,
) -> dict[str, Any] | DictConfig:
    """Copy a recipe and substitute only its explicitly opted-in data source.

    ``name`` defaults to ``sim_local_<model>``. ``transform_overrides`` replaces bundle data-policy
    fields for this recipe, e.g. the caption dropout a training stage applies.

    Retain the original packing and modality settings. Multi-stream recipes keep
    their per-resolution streams so synchronized stream selection still works.
    No production recipe or registered experiment is mutated.
    """
    job_name = name or f"sim_local_{model}"
    experiment = copy.deepcopy(source)
    loader = experiment["dataloader_train"]
    # Every Transfer stage reads one local stream through the single-sample packer below; its
    # source loaders differ (indexed Stage 3, joint WebDataset Stage 1/2, or none at all).
    if "dataloaders" in loader and (model != "transfer" or preserve_transfer_streams):
        loader["_target_"] = LocalSimulationJointDataLoader
        for name, stream in loader["dataloaders"].items():
            stratum = ("720" if "720" in name else "480") if model == "auto" else None
            overrides = dict(transform_overrides or {})
            max_frames = None
            batch_size = 1
            if model == "transfer":
                # Transfer strata are latent lengths, not resolution labels.
                # Render the original source at each recipe's model canvas.
                overrides["resolution"] = stream["dataloader"]["dataset"]["resolution"]
                max_frames = stream["dataloader"]["dataset"]["max_num_frames"]
                batch_size = stream["dataloader"]["batch_size"]
            stream["dataloader"] = make_local_dataloader_config(
                model, stratum=stratum, transform_overrides=overrides, max_num_frames=max_frames
            )
            stream["dataloader"]["batch_size"] = batch_size
        loader["lazy_initialize_child_iterators"] = True
    else:
        # Indexed Transfer normally constructs this one-sample packer inside
        # its loader. Keep that training batch contract with a local child.
        # Preserve model interpolations until the full Hydra config is composed.
        settings = dict(loader.items_ex(resolve=False)) if isinstance(loader, DictConfig) else dict(loader)
        dataloader = make_local_dataloader_config(model, transform_overrides=transform_overrides)
        experiment["dataloader_train"] = L(LocalSimulationJointDataLoader)(
            dataloaders={"local_data": {"dataloader": dataloader, "ratio": 1.0}},
            tokenizer_spatial_compression_factor=settings.get(
                "tokenizer_spatial_compression_factor", "${model.config.tokenizer.spatial_compression_factor}"
            ),
            tokenizer_temporal_compression_factor=settings.get(
                "tokenizer_temporal_compression_factor", "${model.config.tokenizer.temporal_compression_factor}"
            ),
            patch_spatial=settings.get("patch_spatial", "${model.config.diffusion_expert_config.patch_spatial}"),
            max_sequence_length=None,
            max_samples_per_batch=1,
            prewarm=False,
            lazy_initialize_child_iterators=True,
        )
    experiment["job"]["name"] = job_name
    experiment["job"]["group"] = "simulation_local_data"
    # DictConfig copies inserted mappings, so setdefault may return a detached dict.
    if "trainer" not in experiment:
        experiment["trainer"] = {}
    trainer = experiment["trainer"]
    if "callbacks" not in trainer:
        trainer["callbacks"] = {}
    callbacks = trainer["callbacks"]
    callbacks["local_simulation_resume"] = L(LocalSimulationResumeCallback)()
    return experiment
