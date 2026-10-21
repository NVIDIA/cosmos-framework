# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos RL — async-native (also sync-capable) post-training framework.

See docs/design/DESIGN.md (incl. the version history) and the README. Roles = env(dataset) /
rollout / reward / trainer(policy) (+buffer), decoupled as model ↔ algorithm(objective) ↔
trainer(engine); typed config over inherited i4 experiments; per-algorithm entrypoints
(DPO end-to-end on the 8B); orchestration spans local(torchrun/FSDP) and Ray, single-node
to multinode. Weight transfer is still the no-op seam (real backends in review); the
*interfaces* are the general (final-form) ones, with TODO hooks for deferred features.
"""

# Lazy submodule access (PEP 562): importing `posttrain` (or any submodule, which runs
# this __init__ first) must NOT eager-import `roles` — that pulls torch-heavy role bodies
# (roles.rollout.hf -> import torch) and makes the whole package un-importable in a
# torch-free venv. The Ray runtime machinery (workers.ray._Host, comm, control.placement,
# workers.worker_group) is torch-free and must load in the robolab/sim venv for the
# sim-eval WorkerGroup driver path. `from posttrain import roles`/`posttrain.roles` still
# work — resolved lazily on first access.
from typing import TYPE_CHECKING

__all__ = ["roles", "workers"]

if TYPE_CHECKING:
    from cosmos_framework.posttrain import roles, workers


def __getattr__(name: str):
    if name in __all__:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
