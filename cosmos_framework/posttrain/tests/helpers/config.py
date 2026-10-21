# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Lightweight helpers for tests that inspect posttrain recipe fields."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TypeVar

import pytest

from cosmos_framework.posttrain.config.config import PosttrainConfig

_ConfigT = TypeVar("_ConfigT", bound=PosttrainConfig)


def skip_recipe_registration(
    monkeypatch: pytest.MonkeyPatch,
    root_factory: Callable[..., _ConfigT],
    config_type: type[_ConfigT],
) -> None:
    """Build a recipe root without registering the unrelated Hydra catalog.

    ``build_root`` registers the full Hydra experiment catalog for production
    entrypoints. Tests that only inspect fields set directly by a recipe do not
    consume that catalog or its generic trainer defaults, and loading it under
    every xdist worker dominates their runtime. Keep the recipe's real root
    factory in place and replace only the ``build_root`` global it calls, so a
    recipe wired to the wrong algorithm root still fails this test.
    """

    root_factory_module = sys.modules[root_factory.__module__]

    def build_unregistered_root(
        root_type: type[_ConfigT],
        group: str,
        *,
        register_experiment_configs: bool = True,
    ) -> _ConfigT:
        del group, register_experiment_configs
        assert root_type is config_type, "recipe uses an unexpected algorithm root"
        return root_type(model=None, optimizer=None, scheduler=None, dataloader_train=None, dataloader_val=None)

    monkeypatch.setattr(root_factory_module, "build_root", build_unregistered_root)
