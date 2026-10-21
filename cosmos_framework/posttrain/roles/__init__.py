# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""roles — concrete role bodies as variant subpackages (env / rollout / reward /
trainer) + the role factory (`core.factory`, used as `from ..roles import roles`
-> `roles.env(...)`). Each role is declared via the factory and backed by a
WorkerGroup (workers/). New engine / algorithm variants drop into the matching
subpackage without touching callers.

Lazy submodule access (PEP 562, mirrors `posttrain/__init__.py`): importing `roles` —
or any submodule under it, e.g. `roles.core.dispatch` (Python must run this __init__
first to reach a submodule) — must NOT eager-import the env/rollout/reward/trainer
subpackages. `roles.rollout` in particular pulls a torch-heavy body at package-init
time (`rollout.hf` -> `import torch`), which would make `roles.core.dispatch`
un-importable in a torch-free driver process (DESIGN §2.1: "the driver stays
torch-free" — dispatch.py is consulted driver-side by `GridBackend.submit`). The
concrete classes below (`Env`, `HFRollout`, ...) still work as `roles.HFRollout` etc.
— resolved lazily on first access.
"""

from typing import TYPE_CHECKING

from cosmos_framework.posttrain.roles.core import factory as roles

__all__ = [
    "roles",
    "Env",
    "DatasetEnv",
    "ArithmeticEnv",
    "DatasetPreferenceEnv",
    "LocalRollout",
    "HFRollout",
    "RewardWorker",
    "LocalTrainer",
    "LLMTrainer",
    "DiffusionTrainer",
    "PPOPolicyTrainer",
    "PPOReferenceRole",
    "PPOCriticRole",
    "DeterministicArithmeticDataset",
]

if TYPE_CHECKING:
    from cosmos_framework.posttrain.roles.critic import PPOCriticRole
    from cosmos_framework.posttrain.roles.env import (
        ArithmeticEnv,
        DatasetEnv,
        DatasetPreferenceEnv,
        DeterministicArithmeticDataset,
        Env,
    )
    from cosmos_framework.posttrain.roles.reference import PPOReferenceRole
    from cosmos_framework.posttrain.roles.reward import RewardWorker
    from cosmos_framework.posttrain.roles.rollout import HFRollout, LocalRollout
    from cosmos_framework.posttrain.roles.trainer import DiffusionTrainer, LLMTrainer, LocalTrainer, PPOPolicyTrainer

_SUBMODULE_FOR = {
    "Env": "env",
    "DatasetEnv": "env",
    "ArithmeticEnv": "env",
    "DatasetPreferenceEnv": "env",
    "RewardWorker": "reward",
    "HFRollout": "rollout",
    "LocalRollout": "rollout",
    "LocalTrainer": "trainer",
    "LLMTrainer": "trainer",
    "DiffusionTrainer": "trainer",
    "PPOPolicyTrainer": "trainer",
    "PPOReferenceRole": "reference",
    "PPOCriticRole": "critic",
    "DeterministicArithmeticDataset": "env",
}


def __getattr__(name: str):
    """Lazily resolve a concrete role class (PEP 562), importing its subpackage only on first access.

    Args:
        name: The requested attribute; must be a known role class in `_SUBMODULE_FOR`, else `AttributeError`.

    Returns:
        The class `name` imported from its `env`/`rollout`/`reward`/`trainer` submodule.
    """
    submodule = _SUBMODULE_FOR.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"{__name__}.{submodule}")
    return getattr(module, name)
