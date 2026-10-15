# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""The boundary between this package and a forward-dynamics model.

Everything above this line works in dataset space: absolute measured states, seconds, episode
indices. Everything below is the model's business, including how absolute states become whatever
action representation it was trained on and how those get normalized. That split is deliberate:
the action encoding is a property of the trained model, not of the robot or the failure.

To support a different model, implement this protocol and point ``model.adapter`` in the profile
at it. Three methods, only the first two required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ForwardDynamicsModel(Protocol):
    def encode_actions(self, states: np.ndarray) -> list[list[float]]:
        """Turn ``(chunk_length + 1, state_dim)`` absolute states into one action chunk.

        Whatever the model expects: framewise deltas, absolute values, normalized or not. The
        reference implementation emits joint deltas with absolute gripper columns, quantile
        normalized, because that is how the reference model was trained.
        """
        ...

    def rollout_chunk(self, obs_image: Path, actions: list[list[float]], prompt: str, out_dir: Path, seed: int) -> Path:
        """Generate one chunk of video from a starting frame plus an action chunk.

        Returns the path to the written clip. The caller feeds the clip's last frame back in as
        the next chunk's ``obs_image``, which is what makes the rollout autoregressive.
        """
        ...

    def infer_actions(self, video: Path, prompt: str, out_dir: Path) -> np.ndarray | None:
        """Optional inverse pass: read actions back out of a generated clip.

        Used to check whether a channel the pixels barely show was nonetheless followed. Return
        ``None`` if the model has no inverse mode.
        """
        ...


def load_adapter(spec: str, profile, **kwargs) -> ForwardDynamicsModel:
    """Instantiate an adapter from a ``module:ClassName`` string in the profile."""
    import importlib

    if ":" not in spec:
        raise ValueError(f"adapter spec {spec!r} must look like 'package.module:ClassName'")
    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"{module_name} has no attribute {class_name}")
    return cls(profile, **kwargs)
