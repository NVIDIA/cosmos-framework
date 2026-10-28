# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared root-config machinery for the posttrain entrypoints (DESIGN: config system).

The registration + base that the per-algorithm roots build on. `PosttrainConfig` subclasses the
cosmos3 root `Config` and adds the typed posttrain fields (`rl`/`grid` and model backends);
`build_root` reuses cosmos3's config-group registration, registers posttrain experiments, and stamps
the posttrain job/trainer defaults. The runnable roots are the thin per-algorithm modules
`config_dpo.py` (`rl: DPOConfig`), `config_grpo.py` (`rl: GRPOConfig`), and its OPD specialization
`config_opd.py` (`rl: OPDConfig`) — the `--config` targets the entries load.

The posttrain schema remains additive via subclassing; `build_root` uses the shared `job.project`;
W&B maps the generator storage namespace at initialization time.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import sys
from typing import TypeVar

import attrs

from cosmos_framework.trainer import ImaginaireTrainer as Trainer
from cosmos_framework.utils.config_helper import import_all_modules_from_package
from cosmos_framework.configs.base import config as c3
from cosmos_framework.posttrain.config.model import HF, I4, QWEN3_VL, HFModelConfig, Qwen3VLModelConfig
from cosmos_framework.posttrain.config.rl import GridConfig, RunConfig
from cosmos_framework.posttrain.utils.log import logger


@attrs.define(slots=False)
class PosttrainConfig(c3.Config):
    # Which algorithm this experiment is for ("dpo" | "grpo" | ...) — metadata / logging.
    # The runnable entry is per-algorithm (entrypoints/dpo.py etc.), so dispatch is the entry you
    # launch, not a field lookup. The per-algorithm roots override `rl` with their typed schema.
    algorithm: str = "dpo"
    rl: RunConfig = attrs.field(factory=RunConfig)
    grid: GridConfig = attrs.field(factory=GridConfig)
    # Model sourcing is top-level, not an RL concern: "i4" uses the inherited `model` tree, while
    # the typed load specs below are shared by every algorithm root.
    model_backend: str = "i4"
    model_hf: HFModelConfig = attrs.field(factory=HFModelConfig)
    model_qwen3_vl: Qwen3VLModelConfig = attrs.field(factory=Qwen3VLModelConfig)

    def __attrs_post_init__(self) -> None:
        """Reject unknown backend selectors when the typed root is constructed."""
        if self.model_backend not in (I4, HF, QWEN3_VL):
            raise ValueError(
                f"unknown model_backend {self.model_backend!r} "
                f"(expected {I4!r} or {HF!r}; {QWEN3_VL!r} is a compatibility alias)"
            )


_C = TypeVar("_C", bound=PosttrainConfig)


# Experiment catalogs are imported only so `experiment=` can select from them. They reach into
# another project that carries its own optional dependencies (robotics datasets, a database driver,
# video and array codecs), and posttrain installs none of those. A module-scope import failure in
# one experiment must not decide whether an unrelated recipe can load its config. Both catalogs get
# the same treatment: posttrain's own experiments inherit upstream experiments through `defaults`, so they
# import the same optional dependencies transitively.
_EXPERIMENT_CATALOGS = (
    "cosmos_framework.configs.base.experiment",
    "cosmos_framework.posttrain.config.experiment",
)
_FIRST_PARTY_ROOTS = ("cosmos_framework",)


def _missing_optional_dependency(error: ModuleNotFoundError) -> str | None:
    """Return the absent third-party distribution, or None when the failure is ours to fix.

    Args:
        error: The import failure raised while loading a catalog module.

    Returns:
        The top-level name of an absent third-party dependency, or None when the missing module is
        first-party — a broken in-repo import is a real bug and must keep propagating.
    """
    missing = (error.name or "").split(".")[0]
    if not missing or missing in _FIRST_PARTY_ROOTS:
        return None
    return missing


def _import_experiment_catalog(package_path: str) -> dict[str, str]:
    """Import an inherited experiment catalog module by module, tolerating absent optional deps.

    Mirrors `import_all_modules_from_package`'s recursive walk, but isolates each module so one
    unimportable experiment cannot abort the rest of the catalog. A package that cannot be imported
    is not descended into.

    Args:
        package_path: Dotted path of the catalog package to import.

    Returns:
        `{module path: absent dependency}` for every module skipped. Empty when all imported.

    Raises:
        ModuleNotFoundError: The missing module is first-party, so the failure is a real bug.
    """
    package = importlib.import_module(package_path)
    skipped: dict[str, str] = {}

    def walk(directory: str, prefix: str) -> None:
        for _, module_name, is_package in pkgutil.iter_modules([directory]):
            if module_name.startswith("_"):
                continue
            full_name = f"{prefix}.{module_name}"
            try:
                loaded = sys.modules.get(full_name)
                importlib.reload(loaded) if loaded is not None else importlib.import_module(full_name)
            except ModuleNotFoundError as error:
                dependency = _missing_optional_dependency(error)
                if dependency is None:
                    raise
                skipped[full_name] = dependency
                continue  # an unimportable package registers nothing below it either
            if is_package:
                walk(os.path.join(directory, module_name), full_name)

    for directory in package.__path__:
        walk(directory, package_path)
    return skipped


def _register_experiment_catalogs() -> None:
    """Register every experiment catalog, degrading rather than failing on absent optional deps.

    The fast path is the shared helper, so a complete environment behaves exactly as before and
    pays nothing. Only after it hits an absent third-party dependency does this fall back to the
    per-module walk. A first-party import failure still propagates from either path — that is a
    real bug, not a missing wheel.

    An experiment whose module was skipped cannot be selected; `experiment=<name>` then fails in
    config resolution. The warning names the absent dependencies so that error is diagnosable.
    """
    for package_path in _EXPERIMENT_CATALOGS:
        try:
            import_all_modules_from_package(package_path, reload=True)
            continue
        except ModuleNotFoundError as error:
            if _missing_optional_dependency(error) is None:
                raise
        skipped = _import_experiment_catalog(package_path)
        if skipped:
            logger.warning(
                f"{package_path}: skipped {len(skipped)} experiment module(s) needing absent "
                f"optional dependencies {sorted(set(skipped.values()))}. Those experiments cannot "
                f"be selected with experiment=<name>."
            )




def build_root(
    cls: type[_C], group: str, *, register_experiment_configs: bool = True, register_base_configs: bool = True
) -> _C:
    """Build a per-algorithm root of type `cls`: reuse cosmos3's config-group registration, register
    posttrain experiments when requested (so `experiment=` selects them; inherited upstream experiments
    stay available so a posttrain experiment can `defaults: [/experiment/<i4>]`), and stamp the
    posttrain job/trainer defaults. Concrete recipes that fully populate the root disable experiment
    registration so they do not import unrelated projects and their optional dependencies. Idempotent
    — the ConfigStore registration is safe to repeat. ``register_base_configs=False`` selects a
    self-contained native root without inherited model groups and requires experiment registration
    to be disabled too."""
    c = cls(model=None, optimizer=None, scheduler=None, dataloader_train=None, dataloader_val=None)

    c.job.project = "cosmos3_vfm"
    c.job.group = group
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = Trainer
    c.trainer.straggler_detection.enabled = False
    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 20
    c.trainer.validation_iter = 100
    c.trainer.run_validation = False
    c.trainer.callbacks = None

    c.upload_reproducible_setup = False

    if not register_base_configs:
        if register_experiment_configs:
            raise ValueError("standalone roots cannot register inherited experiment catalogs")
        c.defaults = ["_self_"]
        return c

    from cosmos_framework.configs.base.defaults.callbacks import register_callbacks
    from cosmos_framework.configs.base.defaults.checkpointer import register_checkpoint, register_ckpt_type
    from cosmos_framework.configs.base.defaults.ema import register_ema
    from cosmos_framework.configs.base.defaults.model import register_model
    from cosmos_framework.configs.base.defaults.optimizer import register_optimizer, register_scheduler
    from cosmos_framework.configs.base.defaults.reasoner import register_vlm
    from cosmos_framework.configs.base.defaults.tokenizer import register_sound_tokenizer, register_tokenizer

    # Reuse cosmos3's config-group registration (model/optimizer/tokenizer/cluster/...).
    register_model()
    register_checkpoint()
    register_ckpt_type()
    register_optimizer()
    register_scheduler()
    register_callbacks()
    register_tokenizer()
    register_sound_tokenizer()
    register_ema()
    register_vlm()

    # Register BOTH cosmos3 experiments (so posttrain can inherit them via defaults) AND posttrain
    # experiments (selectable via experiment=). Callers that fully populate the root opt out, as
    # does role-native PPO, which has no experiment-tree dependency: eager discovery pulls unrelated
    # optional action-training packages (for example lerobot) into even ``ppo --validate`` and makes
    # a dense-HF-only entrypoint unusable in its intentionally smaller runtime environment. Existing
    # DPO/GRPO callers retain the default.
    if register_experiment_configs:
        # Register BOTH cosmos3 experiments (so posttrain can inherit them via defaults) AND
        # posttrain experiments (selectable via experiment=).
        _register_experiment_catalogs()
    return c
