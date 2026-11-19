# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""A rollout that always returns zero actions: the reference :class:`~embodied.rollout.Rollout`.

Used to bring up a simulator or a collection run without a checkpoint or a GPU.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from cosmos_framework.posttrain.embodied.registry import register_rollout
from cosmos_framework.posttrain.embodied.rollout import RolloutInfo


class ZeroRollout:
    """Returns one ``(horizon, action_dim)`` zero chunk per observation.

    ``action_dim`` defaults to the registered embodiment's. Pass it explicitly when the embodiment
    is registered only in the simulator's process (the gateway transport).
    """

    def __init__(self, action_dim: int | None = None, horizon: int = 1, *, embodiment: str = "example") -> None:
        if action_dim is None:
            from cosmos_framework.posttrain.embodied.embodiment import get_embodiment

            action_dim = get_embodiment(embodiment).action_dim
        self._dim, self._horizon, self._embodiment = int(action_dim), int(horizon), embodiment

    def act(
        self, obs_batch: Sequence[Mapping[str, Any]], *, context: Mapping[str, Any] | None = None
    ) -> list[np.ndarray]:
        del context
        return [np.zeros((self._horizon, self._dim), dtype=np.float32) for _ in obs_batch]

    def describe(self) -> RolloutInfo:
        return RolloutInfo(name="zero", action_dim=self._dim, horizon=self._horizon, embodiment=self._embodiment)


register_rollout("zero", ZeroRollout)
