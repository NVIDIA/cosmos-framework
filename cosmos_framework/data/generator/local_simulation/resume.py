# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Checkpoint the consumer's local stream counts through the existing DCP API."""

from __future__ import annotations

import copy
from typing import Any

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils.callback import Callback

LOCAL_SIMULATION_RESUME_KEY: str = "_local_simulation_resume"
_LOADED_STATE: dict[str, Any] | None = None


def consume_local_resume_state() -> dict[str, Any] | None:
    """Hand checkpoint state to the loader before child iterators are created."""
    global _LOADED_STATE
    state = _LOADED_STATE
    _LOADED_STATE = None
    return state


class LocalSimulationResumeCallback(Callback):
    """Save counts from batches handed to training, excluding async prefetch."""

    checkpoint_component: str = "dataloader"
    _state: dict[str, Any]

    def __init__(self) -> None:
        super().__init__()
        consume_local_resume_state()
        self._state = {}

    def has_checkpoint_state(self) -> bool:
        return True

    def on_training_step_batch_start(self, model: ImaginaireModel, data: dict[str, Any], iteration: int = 0) -> None:
        # CP slots reuse the same raw batch. Its absolute counts are idempotent,
        # and already include a partial window the trainer discards on resume.
        state = data.get(LOCAL_SIMULATION_RESUME_KEY)
        if state is not None:
            self._state = copy.deepcopy(state)

    def state_dict(self) -> dict[str, Any]:
        return {LOCAL_SIMULATION_RESUME_KEY: copy.deepcopy(self._state)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        global _LOADED_STATE
        self._state = copy.deepcopy(state_dict.get(LOCAL_SIMULATION_RESUME_KEY, {}))
        _LOADED_STATE = self._state or None
