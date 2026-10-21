# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""roles.env — Env / Source variants (DESIGN §1.3). Base in `base.py`; concrete
sources (dataset, sim, world-model) in sibling modules.

**Every re-export below is LAZY, and that is load-bearing rather than tidy.** This barrel
used to import `wfm` eagerly, which imports torch at module scope, so importing ANYTHING
under `roles.env` pulled the whole training stack in. That is the mechanism `embodied/`
cites to justify living at top level -- and `sim_server.py`, the process the SIMULATOR
container starts, is in this package. Measured before this change: importing
`roles.env.sim_server` added torch, dill, pynvml, torchgen and tqdm to a bare interpreter.

`SimBody` was already deferred for the same reason, one backend at a time. The rule is now
the package's, not one symbol's exception.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # type checkers only; never executed at runtime.
    # `X as X` is the PEP 484 re-export marker. Needed because `__all__` below is
    # COMPUTED from `_EXPORTS`, which a linter cannot follow.
    from cosmos_framework.posttrain.roles.env.arithmetic import (
        DeterministicArithmeticDataset as DeterministicArithmeticDataset,
    )
    from cosmos_framework.posttrain.roles.env.base import Env as Env
    from cosmos_framework.posttrain.roles.env.dataset import (
        ArithmeticEnv as ArithmeticEnv,
    )
    from cosmos_framework.posttrain.roles.env.dataset import (
        DatasetEnv as DatasetEnv,
    )
    from cosmos_framework.posttrain.roles.env.dataset import (
        JsonlRows as JsonlRows,
    )
    from cosmos_framework.posttrain.roles.env.dataset import (
        TrainValidationEnv as TrainValidationEnv,
    )
    from cosmos_framework.posttrain.roles.env.dataset import (
        hf_items as hf_items,
    )
    from cosmos_framework.posttrain.roles.env.dataset import (
        is_conversation_prompt as is_conversation_prompt,
    )
    from cosmos_framework.posttrain.roles.env.dataset import (
        jsonl_prompts as jsonl_prompts,
    )
    from cosmos_framework.posttrain.roles.env.dataset import (
        project_prompt_items as project_prompt_items,
    )
    from cosmos_framework.posttrain.roles.env.preference import (
        DatasetPreferenceEnv as DatasetPreferenceEnv,
    )
    from cosmos_framework.posttrain.roles.env.preference import (
        DPOPreferenceEnv as DPOPreferenceEnv,
    )
    from cosmos_framework.posttrain.roles.env.preference import (
        PreferenceSourceIdentity as PreferenceSourceIdentity,
    )
    from cosmos_framework.posttrain.roles.env.preference import (
        build_preference_env as build_preference_env,
    )
    from cosmos_framework.posttrain.roles.env.prompt import (
        DatasetSourceIdentity as DatasetSourceIdentity,
    )
    from cosmos_framework.posttrain.roles.env.prompt import (
        build_prompt_env as build_prompt_env,
    )
    from cosmos_framework.posttrain.roles.env.wfm import (
        WFMDataEnv as WFMDataEnv,
    )
    from cosmos_framework.posttrain.roles.env.wfm import (
        WFMSeedContract as WFMSeedContract,
    )
    from cosmos_framework.posttrain.roles.env.wfm import (
        wfm_first_seed as wfm_first_seed,
    )
    from cosmos_framework.posttrain.roles.env.wfm import (
        wfm_seed_set as wfm_seed_set,
    )

#: name -> the sibling module that defines it. A DICT rather than a chain of ifs so that
#: adding a re-export is a one-line data change and cannot forget a branch.
_EXPORTS = {
    "Env": "base",
    # Synthetic image/PIL generation stays lazy so its optional stack stays out of the sim venv.
    "DeterministicArithmeticDataset": "arithmetic",
    "ArithmeticEnv": "dataset",
    "DatasetEnv": "dataset",
    "TrainValidationEnv": "dataset",
    "JsonlRows": "dataset",
    "hf_items": "dataset",
    "is_conversation_prompt": "dataset",
    "jsonl_prompts": "dataset",
    "project_prompt_items": "dataset",
    "DatasetPreferenceEnv": "preference",
    "DPOPreferenceEnv": "preference",
    "PreferenceSourceIdentity": "preference",
    "build_preference_env": "preference",
    "DatasetSourceIdentity": "prompt",
    "build_prompt_env": "prompt",
    "WFMDataEnv": "wfm",
    "WFMSeedContract": "wfm",
    "wfm_first_seed": "wfm",
    "wfm_seed_set": "wfm",
    # robolab pulls isaaclab/robolab -- heavy, sim-venv only.
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> object:
    """Resolve a re-export on first access, importing only the module that defines it."""
    try:
        mod_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    mod = importlib.import_module(f"{__name__}.{mod_name}")
    return getattr(mod, name)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
