# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Importable custom dataset factory resolution."""

from __future__ import annotations

from typing import Any


def validate_dataset_factory(factory_path: str) -> None:
    """Validate that a configured custom dataset factory is importable and callable."""
    if not isinstance(factory_path, str) or not factory_path.strip():
        raise ValueError("custom dataset factory must be a non-empty dotted callable path")
    from cosmos_framework.utils.serialization import load_callable

    try:
        factory = load_callable(factory_path)
    except Exception as error:  # noqa: BLE001 -- report arbitrary import failures with config context
        raise ValueError(
            f"custom dataset factory {factory_path!r} is not importable: {error}. "
            "It must be a dotted callable path available on every worker's PYTHONPATH."
        ) from error
    if not callable(factory):
        raise TypeError(f"custom dataset factory {factory_path!r} resolved to a non-callable object")


def build_custom_dataset(factory_path: str, factory_kwargs: dict[str, Any]) -> Any:
    """Build and validate one non-empty map-style custom dataset."""
    validate_dataset_factory(factory_path)
    from cosmos_framework.utils.serialization import load_callable

    factory = load_callable(factory_path)
    try:
        dataset = factory(**dict(factory_kwargs))
    except Exception as error:  # noqa: BLE001 -- preserve the original failure behind factory context
        raise RuntimeError(f"custom dataset factory {factory_path!r} failed: {error}") from error
    if not (hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")):
        raise TypeError(
            f"custom dataset factory {factory_path!r} must return a map-style dataset "
            f"(__len__ + __getitem__), got {type(dataset).__name__}"
        )
    if len(dataset) == 0:
        raise ValueError(f"custom dataset factory {factory_path!r} returned an empty dataset")
    return dataset
