# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Reviewed runtime profiles for numerical post-training goldens.

A golden trajectory is evidence only under the runtime that produced it. Keeping
the runtime and tolerance beside the values makes that qualification executable
instead of relying on prose in each test module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class QualificationMismatch(RuntimeError):
    """The current runtime cannot interpret a qualified golden as a regression oracle."""


@dataclass(frozen=True)
class GoldenProfile:
    """Complete qualification envelope for one numerical loss trajectory."""

    name: str
    algorithm: str
    accelerator: str
    torch_version: str
    world_size: int
    steps: int
    seed: int
    rtol: float
    atol: float
    reference: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("golden profile name must be non-empty")
        if not self.algorithm.strip():
            raise ValueError("golden profile algorithm must be non-empty")
        if not self.accelerator.strip():
            raise ValueError("golden profile accelerator must be non-empty")
        if not self.torch_version.strip():
            raise ValueError("golden profile torch_version must be non-empty")
        if type(self.world_size) is not int or self.world_size < 1:
            raise ValueError(f"golden profile world_size must be positive, got {self.world_size!r}")
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError(f"golden profile steps must be positive, got {self.steps!r}")
        if type(self.seed) is not int:
            raise ValueError(f"golden profile seed must be an integer, got {self.seed!r}")
        if len(self.reference) != self.steps:
            raise ValueError(
                f"golden profile reference has {len(self.reference)} values for {self.steps} configured steps"
            )
        if any(not math.isfinite(value) for value in self.reference):
            raise ValueError("golden profile reference values must all be finite")
        if not math.isfinite(self.rtol) or self.rtol < 0:
            raise ValueError(f"golden profile rtol must be finite and non-negative, got {self.rtol!r}")
        if not math.isfinite(self.atol) or self.atol < 0:
            raise ValueError(f"golden profile atol must be finite and non-negative, got {self.atol!r}")

    def require_runtime(self, *, torch_version: str, world_size: int, accelerator: str) -> None:
        """Fail closed when a caller tries to use the golden outside its capture envelope."""

        observed = {
            "torch_version": torch_version,
            "world_size": world_size,
            "accelerator": accelerator,
        }
        expected = {
            "torch_version": self.torch_version,
            "world_size": self.world_size,
            "accelerator": self.accelerator,
        }
        if observed != expected:
            raise QualificationMismatch(
                f"golden profile {self.name!r} is qualified for {expected!r}, but this run is {observed!r}; "
                "re-capture the trajectory before changing the checked-in profile"
            )
