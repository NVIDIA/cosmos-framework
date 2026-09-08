# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""W&B launch defaults.

``wandb_util`` resolves the W&B entity and project through this module. It has an
internal counterpart that is deliberately not released: that one also carries
cluster paths (container images, shared caches) and NVIDIA's shared W&B entity,
none of which mean anything outside NVIDIA. This file supplies the public
implementations at the same module path, so the released ``wandb_util`` resolves
against it.

Both values come from the environment, so runs land wherever the caller is
already configured to write.
"""

from __future__ import annotations

import os


def get_wandb_entity() -> str | None:
    """Return the W&B entity to log under, or ``None`` to let wandb decide.

    ``None`` is the important default: wandb then resolves the entity itself,
    using ``WANDB_ENTITY`` when set and otherwise the account the caller is
    logged in as. A run can only be written to an entity the caller belongs to,
    so this module must not name a specific team.
    """
    return os.environ.get("WANDB_ENTITY") or None


def get_wandb_project(config_project: str) -> str:
    """Return the W&B project, preferring ``WANDB_PROJECT`` over the job config.

    Args:
        config_project: The project from the job config, used when the
            environment does not override it.
    """
    return os.environ.get("WANDB_PROJECT") or config_project
