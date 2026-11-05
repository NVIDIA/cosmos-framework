# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Standalone opt-in config entry point for the local simulation bundles.

Use ``--config=cosmos_framework/configs/simulation/local.py`` and select
``experiment=sim_local_transfer``.
The regular simulation config and its production experiment names are unchanged.
"""

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from cosmos_framework.configs.base.config import Config
from cosmos_framework.configs.simulation.common.transfer_profiles import (
    make_transfer_teacher_forcing_dcm_experiment,
    make_transfer_teacher_forcing_experiment,
)
from cosmos_framework.configs.simulation.common.transfer_stage0_profiles import make_transfer_bidirectional_experiment
from cosmos_framework.configs.simulation.common.transfer_stage3_profiles import (
    make_transfer_self_forcing_dmd_experiment,
)
from cosmos_framework.data.generator.local_simulation.config import make_local_experiment_config


def _local_transfer_experiment(
    source: dict | DictConfig,
    *,
    name: str = "sim_local_transfer",
    transform_overrides: dict | None = None,
    preserve_transfer_streams: bool = False,
) -> dict | DictConfig:
    """Use local data and load backbone weights from the selected stage DCPs."""
    experiment = make_local_experiment_config(
        source,
        "transfer",
        name=name,
        transform_overrides=transform_overrides,
        preserve_transfer_streams=preserve_transfer_streams,
    )
    model = experiment["model"]["config"]
    for key in ("vlm_config", "vlm_config_teacher", "vlm_config_fake_score"):
        if key in model:
            model[key].setdefault("pretrained_weights", {}).update(
                enabled=False,
                backbone_path="",
                credentials_path="",
                enable_gcs_patch_in_boto3=False,
            )
    return experiment


def register_local_experiments() -> None:
    """Register new local experiments without replacing any production entry."""
    ConfigStore.instance().store(
        group="experiment",
        package="_global_",
        name="sim_local_transfer_stage0_bidir",
        node=_local_transfer_experiment(
            make_transfer_bidirectional_experiment(),
            name="sim_local_transfer_stage0_bidir",
            transform_overrides={"cfg_dropout_rate": 0.1, "resize_on_read": False, "min_num_frames": 5},
            preserve_transfer_streams=True,
        ),
    )
    ConfigStore.instance().store(
        group="experiment",
        package="_global_",
        name="sim_local_transfer",
        node=_local_transfer_experiment(make_transfer_self_forcing_dmd_experiment()),
    )
    # Sim-Transfer Stage 1 (teacher forcing) and Stage 2 (TF-dCM) on the same Transfer bundle as
    # Stage 3 (``sim_local_transfer``). Both stages train with 10% caption dropout.
    for name, source in (
        ("sim_local_transfer_stage1_tf", make_transfer_teacher_forcing_experiment()),
        ("sim_local_transfer_stage2_tfdcm", make_transfer_teacher_forcing_dcm_experiment()),
    ):
        ConfigStore.instance().store(
            group="experiment",
            package="_global_",
            name=name,
            node=_local_transfer_experiment(source, name=name, transform_overrides={"cfg_dropout_rate": 0.1}),
        )


def make_config() -> Config:
    """Compose local recipes with the standard simulation model defaults."""
    # Loader config construction does not require the training-only model
    # imports performed by the full simulation entry point.
    from cosmos_framework.configs.simulation.config import make_config as make_simulation_config

    config = make_simulation_config(experiment_registries=())
    register_local_experiments()
    return config
