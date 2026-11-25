# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Adapter configuration and per-architecture component bindings (no torch dependency).

See :mod:`cosmos_framework.utils.generator.adapters` for the adapter methods themselves.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict, get_args

AdapterMethod = Literal["lora", "dora", "pissa"]
AdapterComponent = Literal["language", "projector", "vision"]
PissaInitialization = Literal["svd", "svd_lowrank"]
ADAPTER_METHODS: tuple[str, ...] = get_args(AdapterMethod)
ADAPTER_COMPONENTS: tuple[str, ...] = get_args(AdapterComponent)
PISSA_INITIALIZATIONS: tuple[str, ...] = get_args(PissaInitialization)


class AdapterTrainingSettings(TypedDict):
    """Per-target settings that only affect training."""

    dropout: float
    initialization: PissaInitialization | None
    pissa_niter: int
    lr_multiplier: float
    weight_decay: float | None


class AdapterTargetIdentity(TypedDict):
    """One adapted ``nn.Linear``: what defines its adapter tensors."""

    name: str
    component: AdapterComponent
    shape: list[int]
    method: AdapterMethod
    rank: int
    alpha: float
    use_rslora: bool


class AdapterTarget(AdapterTargetIdentity):
    """A resolved target: its identity plus its training settings."""

    training: AdapterTrainingSettings


class AdapterManifest(TypedDict):
    """The resolved targets of one model (``adapters.resolve_targets``)."""

    architecture: str
    targets: list[AdapterTarget]
    trainable_parameters: int


class AdapterTensorIdentity(TypedDict):
    """What defines a model's adapter tensors (``adapters.tensor_identity``)."""

    architecture: str
    targets: list[AdapterTargetIdentity]


class AdapterIdentity(AdapterTensorIdentity):
    """The tensor identity plus its format and base (``adapter_artifacts.adapter_identity``)."""

    format: str
    base_dtype: str
    base_fingerprint: str


class AdapterResumeSettings(TypedDict):
    """Per-target training settings an exact resume must keep (``adapters.training_identity``)."""

    dropout: float
    lr_multiplier: float
    weight_decay: float | None


_SELECTOR = re.compile(r"[A-Za-z_][A-Za-z0-9_.*?\[\]-]*")


@dataclass
class AdapterComponentConfig:
    """One group of targets inside one model component, with its own adapter settings.

    ``targets`` are full module-path globs (``fnmatch`` syntax) and every selector must match at
    least one module. ``layers`` optionally restricts language/vision targets to block indices;
    negative indices count from the last block of the component.

    ``method``, ``rank``, ``alpha`` and ``use_rslora`` define the adapter tensors and their meaning
    (the adapter identity). The other fields are training settings: ``dropout`` on the adapter
    input (LoRA/DoRA), PiSSA ``initialization`` (``None`` or ``svd``: exact SVD; ``svd_lowrank``:
    randomized with ``pissa_niter`` power iterations), ``lr_multiplier`` relative to the optimizer
    learning rate, and ``weight_decay`` (``None`` inherits the optimizer's).
    """

    component: AdapterComponent = "language"
    targets: list[str] = field(default_factory=list)
    layers: list[int] | None = None
    method: AdapterMethod = "lora"
    rank: int = 8
    alpha: float = 16.0
    use_rslora: bool = False
    dropout: float = 0.0
    initialization: PissaInitialization | None = None
    pissa_niter: int = 4
    lr_multiplier: float = 1.0
    weight_decay: float | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Normalize and check the settings; call again after changing a field."""
        if isinstance(self.targets, str):
            raise TypeError("adapter targets must be a list of module paths, not a string")
        self.targets = list(self.targets)
        if self.layers is not None:
            self.layers = list(self.layers)
        if self.component not in ADAPTER_COMPONENTS:
            raise ValueError(f"unknown adapter component {self.component!r}; expected one of {ADAPTER_COMPONENTS}")
        if not self.targets or any(not isinstance(t, str) or not _SELECTOR.fullmatch(t) for t in self.targets):
            raise ValueError("adapter targets must be non-empty full module paths or globs")
        if self.layers is not None:
            if self.component == "projector" or not self.layers:
                raise ValueError("adapter layers must be a non-empty list and apply to language/vision blocks only")
            if any(type(i) is not int for i in self.layers) or len(set(self.layers)) != len(self.layers):
                raise ValueError("adapter layers must be unique integers")
        if self.method not in ADAPTER_METHODS:
            raise ValueError(f"unsupported adapter method {self.method!r}; expected one of {ADAPTER_METHODS}")
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("adapter rank must be a positive integer")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)):
            raise TypeError("adapter alpha must be a number")
        self.alpha = float(self.alpha)  # 16 and 16.0 define the same adapter
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("adapter alpha must be finite and positive")
        if type(self.use_rslora) is not bool:
            raise TypeError("adapter use_rslora must be a bool")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("adapter dropout must be in [0, 1)")
        if self.method == "pissa":
            if self.initialization not in (None, *PISSA_INITIALIZATIONS):
                raise ValueError(f"PiSSA initialization must be one of {PISSA_INITIALIZATIONS}")
            if self.dropout:
                raise ValueError("PiSSA requires dropout=0")
            if type(self.pissa_niter) is not int or self.pissa_niter < 0:
                raise ValueError("pissa_niter must be a non-negative integer")
        elif self.initialization is not None:
            raise ValueError("initialization applies to PiSSA only; LoRA and DoRA use Kaiming A and zero B")
        if not math.isfinite(self.lr_multiplier) or self.lr_multiplier <= 0:
            raise ValueError("adapter lr_multiplier must be finite and positive")
        if self.weight_decay is not None and (not math.isfinite(self.weight_decay) or self.weight_decay < 0):
            raise ValueError("adapter weight_decay must be None or finite and non-negative")

    @property
    def scaling(self) -> float:
        return self.alpha / (math.sqrt(self.rank) if self.use_rslora else self.rank)

    @property
    def pissa_initialization(self) -> str:
        """The PiSSA initialization in effect (``svd`` when unset)."""
        return self.initialization or "svd"

    def tensor_settings(self) -> dict[str, Any]:
        """The settings that define the adapter tensors and their meaning (part of the identity)."""
        return {"method": self.method, "rank": self.rank, "alpha": self.alpha, "use_rslora": self.use_rslora}

    def training_settings(self) -> AdapterTrainingSettings:
        """The settings that only affect training; they may change between a run and its warm start."""
        return {
            "dropout": self.dropout,
            "initialization": self.initialization,
            "pissa_niter": self.pissa_niter,
            "lr_multiplier": self.lr_multiplier,
            "weight_decay": self.weight_decay,
        }


def adapter_method_settings(name: str) -> dict[str, Any]:
    """``method``/``use_rslora`` for one of lora, rslora, dora, rsdora, pissa, rspissa."""
    rank_stabilized = name.startswith("rs")
    method = name.removeprefix("rs")
    if method not in ADAPTER_METHODS:
        raise ValueError(f"unknown adapter method {name!r}; expected lora|rslora|dora|rsdora|pissa|rspissa")
    return {"method": method, "use_rslora": rank_stabilized}


@dataclass
class AdapterConfig:
    """Adapter groups for one model; each target module receives exactly one adapter."""

    components: list[AdapterComponentConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Normalize and check every component group; call again after changing the configuration."""
        self.components = [AdapterComponentConfig(**c) if isinstance(c, Mapping) else c for c in self.components]
        if not self.components:
            raise ValueError("adapter configuration must contain at least one component group")
        for component in self.components:
            if not isinstance(component, AdapterComponentConfig):
                raise TypeError(f"adapter components must be AdapterComponentConfig, got {type(component).__name__}")
            component.validate()


@dataclass(frozen=True)
class AdapterSetup:
    """What a model builder needs to install adapters: the configuration and the initial adapter state.

    The adapters start freshly initialized from the base weights, or as placeholders that a training
    checkpoint restores (``from_checkpoint``), or from a portable adapter directory (``warm_start``).
    """

    config: AdapterConfig
    from_checkpoint: bool = False
    warm_start: str | None = None

    def __post_init__(self) -> None:
        if self.from_checkpoint and self.warm_start:
            raise ValueError("adapters start from a training checkpoint or a warm start, not both")


@dataclass(frozen=True)
class AdapterArchitecture:
    """Explicit module-path boundaries of one architecture's components (keyed by HF ``model_type``).

    ``boundaries`` are ``(component, anchored regex)`` pairs; a named ``layer`` group extracts the
    repeated block index used by ``AdapterComponentConfig.layers``.
    """

    name: str
    boundaries: tuple[tuple[str, str], ...]

    def locate(self, name: str) -> tuple[str, int | None] | None:
        """Return ``(component, layer index or None)`` for a module path, or ``None`` if unbound."""
        found = []
        for component, pattern in self.boundaries:
            match = re.match(pattern, name)
            if match:
                layer = match.groupdict().get("layer")
                found.append((component, int(layer) if layer is not None else None))
        if len(found) > 1:
            raise ValueError(f"ambiguous {self.name} adapter component for {name!r}")
        return found[0] if found else None


# Module paths of the Hugging Face (and identical native) Qwen layouts SFT trains. Dense models only:
# MoE routers and fused experts are not plain linear layers called through their forward.
_VLM_BOUNDARIES = (
    ("language", r"^model\.language_model\.layers\.(?P<layer>\d+)\."),
    ("projector", r"^model\.visual\.(?:merger|deepstack_merger_list\.\d+)(?:\.|$)"),
    ("vision", r"^model\.visual\.blocks\.(?P<layer>\d+)\."),
)
_TEXT_BOUNDARIES = (("language", r"^model\.layers\.(?P<layer>\d+)\."),)

ADAPTER_ARCHITECTURES: dict[str, AdapterArchitecture] = {
    **{name: AdapterArchitecture(name, _VLM_BOUNDARIES) for name in ("qwen3_vl", "qwen3_5")},
    **{name: AdapterArchitecture(name, _TEXT_BOUNDARIES) for name in ("qwen2", "qwen3")},
}


def adapter_architecture(name: str | None) -> AdapterArchitecture:
    try:
        return ADAPTER_ARCHITECTURES[name or ""]
    except KeyError:
        raise ValueError(
            f"no adapter binding for architecture {name!r}; supported: {sorted(ADAPTER_ARCHITECTURES)}"
        ) from None
