# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Typed, dependency-light adapter configuration for posttrain SFT."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass
class AdapterComponentConfig:
    """Full-path glob selectors, restricted to one native Qwen3-VL component.

    ``layers`` restricts language/vision block indices; projector selectors name the
    final merger or individual deepstack mergers directly. Every selector must match.
    Settings are applied per module, including method, initialization and dropout.
    """

    component: str = "language"
    targets: list[str] = field(default_factory=list)
    layers: list[int] | None = None
    method: str = "lora"
    initialization: str = "default"
    rank: int = 8
    alpha: float = 16.0
    use_rslora: bool = False
    dropout: float = 0.0
    dtype: str = "float32"
    pissa_niter: int = 4
    lr_multiplier: float = 1.0
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.targets, str):
            raise TypeError("adapter targets must be a list, not a string")
        self.targets = list(self.targets)
        if self.layers is not None:
            self.layers = list(self.layers)
        if self.component not in ("language", "projector", "vision"):
            raise ValueError(f"unknown adapter component {self.component!r}")
        if not self.targets or any(not isinstance(t, str) or not t.startswith("model.") for t in self.targets):
            raise ValueError("adapter targets must be non-empty full module paths/globs starting with 'model.'")
        if self.layers is not None:
            if (
                self.component == "projector"
                or not self.layers
                or any(type(i) is not int or i < 0 for i in self.layers)
            ):
                raise ValueError("layers must be non-empty non-negative language/vision block indices")
            if len(set(self.layers)) != len(self.layers):
                raise ValueError("duplicate adapter layer indices")
        if self.method not in ("lora", "dora", "pissa"):
            raise ValueError(f"unsupported adapter method {self.method!r}")
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("adapter rank must be a positive integer")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("adapter alpha must be finite and positive")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("adapter dropout must be in [0, 1)")
        if self.dtype != "float32":
            raise ValueError("posttrain adapters currently require float32 storage and accumulation")
        if self.method == "pissa":
            if self.initialization not in ("default", "svd", "svd_lowrank") or self.dropout:
                raise ValueError("PiSSA requires svd/svd_lowrank initialization and dropout=0")
        elif self.initialization != "default":
            raise ValueError("LoRA/DoRA currently support default (Kaiming A, zero B) initialization only")
        if type(self.pissa_niter) is not int or self.pissa_niter < 1:
            raise ValueError("pissa_niter must be a positive integer")
        if not math.isfinite(self.lr_multiplier) or self.lr_multiplier <= 0:
            raise ValueError("adapter lr_multiplier must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("adapter weight_decay must be finite and non-negative")
        if type(self.use_rslora) is not bool:
            raise TypeError("use_rslora must be bool")


@dataclass
class AdapterExtraModuleConfig:
    """Separately optimized native normalization modules (full-path glob selectors)."""

    component: str = "language"
    targets: list[str] = field(default_factory=list)
    lr_multiplier: float = 1.0
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.targets, str):
            raise TypeError("extra module targets must be a list, not a string")
        self.targets = list(self.targets)
        AdapterComponentConfig(
            component=self.component,
            targets=self.targets,
            lr_multiplier=self.lr_multiplier,
            weight_decay=self.weight_decay,
        )


@dataclass
class AdapterConfig:
    """One adapter per target; component settings may differ. No adapter stacking.

    ``warm_start`` is a portable adapter directory, never a training checkpoint.
    Exact resume comes exclusively from the existing ``rl.resume`` checkpoint path.
    """

    components: list[AdapterComponentConfig] = field(default_factory=list)
    extra_modules: list[AdapterExtraModuleConfig] = field(default_factory=list)
    warm_start: str = ""

    def __post_init__(self) -> None:
        self.components = [AdapterComponentConfig(**x) if isinstance(x, Mapping) else x for x in self.components]
        self.extra_modules = [
            AdapterExtraModuleConfig(**x) if isinstance(x, Mapping) else x for x in self.extra_modules
        ]
        if not self.components:
            raise ValueError("adapter configuration must select at least one component")
        for group in [*self.components, *self.extra_modules]:
            group.__post_init__()
