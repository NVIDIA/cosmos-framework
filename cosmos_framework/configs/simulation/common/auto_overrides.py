# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Bind preserved Sim-Auto recipe values to the native local-bundle config."""

from __future__ import annotations

import re

_STREAM = re.compile(r"^(dataloader_train\.dataloaders\.[^.]+\.dataloader\.dataset\.)(.+)$")


def framework_auto_overrides(overrides: list[str]) -> list[str]:
    """Translate configuration locations only; never change scientific values.

    A local captured descriptor is immutable. Its stage-specific changes live
    under transform_overrides. Storage credentials bind to their real readers,
    not an i4 cluster object, and local bundle iteration replaces named IO caches.
    """
    result = []
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator:
            result.append(override)
            continue
        if key.startswith("job.cluster.object_store_credential_"):
            continue
        if key.startswith("dataloader_train.dataloaders.") and key.endswith(".dataloader.cache_replay_name"):
            continue
        match = _STREAM.match(key)
        if match and not match[2].startswith("transform_overrides."):
            key = f"{match[1]}transform_overrides.{match[2]}"
        if key == "trainer.callbacks.dmd2_parity._target_" and value == "phase_metrics.PhaseMetrics":
            value = "cosmos_framework.callbacks.sim_auto_phase_metrics.PhaseMetrics"
        result.append(f"{key}={value}")
    return result
