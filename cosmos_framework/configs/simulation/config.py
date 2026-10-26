# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unified Hydra config entry point for interactive Cosmos3 experiments.

This is the simulation counterpart of the base VFM config root.
It registers the shared VFM defaults and the interactive distillation / causal
model presets, then imports the experiment registries listed in
``_DEFAULT_EXPERIMENT_REGISTRIES`` unless the caller names its own.
``cosmos_framework.scripts.train_experiment`` discovers ``make_config`` by convention.
"""

import importlib.util

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.trainer import ImaginaireTrainer
from cosmos_framework.utils.config_helper import import_all_modules_from_package, import_module
from cosmos_framework.configs.base.config import Config as VFMConfig
from cosmos_framework.configs.base.defaults.callbacks import register_callbacks
from cosmos_framework.configs.base.defaults.checkpointer import (
    register_checkpoint,
    register_ckpt_type,
)
from cosmos_framework.configs.base.defaults.ema import register_ema
from cosmos_framework.configs.base.defaults.model import register_model
from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
from cosmos_framework.configs.base.defaults.reasoner import (
    register_vlm,  # noqa: F401 - release-only binding of the reasoner defaults
)
from cosmos_framework.configs.base.defaults.tokenizer import (
    register_sound_tokenizer,
    register_tokenizer,
)
from cosmos_framework.configs.base.defaults.distill_callbacks import (
    register_distill_callbacks,
)
from cosmos_framework.configs.base.defaults.distill_ckpt_type import (
    register_distill_ckpt_type,
)
from cosmos_framework.configs.base.defaults.distill_optimizer import (
    register_optimizer,
    register_scheduler,
)
from cosmos_framework.model.generator.consistency_model.tf_dcm import (
    TeacherForcingDCMModel,
    TeacherForcingDCMModelConfig,
)
from cosmos_framework.model.generator.distillation.self_forcing_dmd import (
    SelfForcingDMDConfig,
    SelfForcingDMDModel,
)
from cosmos_framework.trainer.distillation import (
    DistillationTrainer,  # noqa: F401 - release-only binding of the distillation trainer
)
from cosmos_framework.model.generator.omni_mot_causal_model import (
    OmniMoTCausalModel,
    OmniMoTCausalModelConfig,
)

# Experiment registries imported when make_config() is called without an explicit
# registry. The released copy imports only the families that ship with it.
_DEFAULT_EXPERIMENT_REGISTRIES: tuple[str, ...] = (
    "cosmos_framework.configs.simulation.experiment.transfer",
    "cosmos_framework.configs.simulation.experiment.auto",
    "cosmos_framework.configs.simulation.experiment.bimanual",
)


def _import_experiment_registry(registry: str) -> None:
    spec = importlib.util.find_spec(registry)
    if spec is None:
        raise ModuleNotFoundError(f"Could not find interactive experiment registry {registry}")
    if spec.submodule_search_locations is None:
        import_module(registry, reload=True)
    else:
        import_all_modules_from_package(registry, reload=True)


def _register_causal_model() -> None:
    """Register the ``mot_fsdp_causal`` model preset in the Hydra ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(
        group="model",
        package="_global_",
        name="mot_fsdp_causal",
        node=dict(
            trainer=dict(
                distributed_parallelism="fsdp",
            ),
            model=L(OmniMoTCausalModel)(
                config=OmniMoTCausalModelConfig(
                    parallelism=ParallelismConfig(
                        data_parallel_shard_degree=8,
                    ),
                ),
                _recursive_=False,
            ),
        ),
    )


def _register_causal_distill_model() -> None:
    """Register the ``mot_fsdp_tf_dcm`` model preset in the Hydra ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(
        group="model",
        package="_global_",
        name="mot_fsdp_tf_dcm",
        node=dict(
            trainer=dict(
                distributed_parallelism="fsdp",
            ),
            model=L(TeacherForcingDCMModel)(
                config=TeacherForcingDCMModelConfig(
                    parallelism=ParallelismConfig(
                        data_parallel_shard_degree=8,
                    ),
                ),
                _recursive_=False,
            ),
        ),
    )
    cs.store(
        group="model",
        package="_global_",
        name="mot_fsdp_sf_dmd",
        node=LazyDict(
            dict(
                trainer=dict(
                    type=DistillationTrainer,
                    distributed_parallelism="fsdp",
                ),
                model=L(SelfForcingDMDModel)(
                    config=SelfForcingDMDConfig(
                        parallelism=ParallelismConfig(
                            data_parallel_shard_degree=8,
                        ),
                    ),
                    _recursive_=False,
                ),
            ),
            flags={"allow_objects": True},
        ),
    )


def _register_defaults() -> None:
    """Register VFM defaults plus interactive-only config groups."""
    register_model()
    register_checkpoint()
    register_ckpt_type()
    register_distill_ckpt_type()
    register_optimizer()
    register_scheduler()
    register_callbacks()
    register_distill_callbacks()
    register_tokenizer()
    register_sound_tokenizer()
    register_ema()
    register_vlm()


def make_config(
    experiment_registries: tuple[str, ...] | None = None,
    experiment_module: str | None = None,
) -> VFMConfig:
    """Build a Hydra config for all interactive training experiments.

    Args:
        experiment_registries: When ``None`` (default) the registries in
            ``_DEFAULT_EXPERIMENT_REGISTRIES`` are imported. Pass an explicit tuple to restrict loading
            to only the listed module/package registries (used by the L0 test
            fixture to avoid re-importing every family).
        experiment_module: Single experiment registry requested by the training
            launcher. This is mutually exclusive with ``experiment_registries``.
    """
    if experiment_module is not None:
        if experiment_registries is not None:
            raise ValueError("experiment_module and experiment_registries are mutually exclusive")
        experiment_registries = (experiment_module,)

    c = VFMConfig(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    c.job.project = "cosmos3_simulation"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = ImaginaireTrainer
    c.trainer.straggler_detection.enabled = False
    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 20
    c.trainer.validation_iter = 100
    c.trainer.run_validation = False
    c.trainer.callbacks = None

    c.upload_reproducible_setup = False

    _register_defaults()
    _register_causal_model()
    _register_causal_distill_model()

    if experiment_registries is None:
        experiment_registries = _DEFAULT_EXPERIMENT_REGISTRIES
    for experiment_registry in experiment_registries:
        _import_experiment_registry(experiment_registry)

    return c
