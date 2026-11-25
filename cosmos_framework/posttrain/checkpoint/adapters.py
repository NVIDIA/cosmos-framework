# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Adapter state in SFT training checkpoints.

A training checkpoint is the only resumable state; its trainer component records the adapter
identity in the objective identity. Portable artifacts and merged exports are in
:mod:`cosmos_framework.utils.generator.adapter_artifacts`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch

from cosmos_framework.posttrain.utils.log import info0
from cosmos_framework.utils.generator.adapter_artifacts import validate_adapter_state, validate_identity
from cosmos_framework.utils.generator.adapters import adapter_state_spec, identity_differences


def checkpoint_adapter_identity(checkpoint_dir: str) -> dict[str, Any] | None:
    """The adapter identity a committed checkpoint was trained with, or ``None`` without adapters."""
    from cosmos_framework.posttrain.checkpoint.io import load_component

    trainer = load_component(checkpoint_dir, "trainer") or {}
    return trainer.get("objective_identity", {}).get("adapters")


def require_resumable_adapters(
    checkpoint_dir: str, identity: Mapping[str, Any], training: Mapping[str, Any] | None = None
) -> None:
    """Refuse an exact resume whose checkpoint adapters differ from the configured ones, naming each field.

    Compares the adapter identity and, separately, the settings a resumed run must keep
    (``adapters.training_identity``: dropout, learning-rate multipliers, weight decay), which the
    optimizer state depends on. Runs before the trainer's own objective-identity comparison so the error is
    about the adapters only.
    """
    from cosmos_framework.posttrain.checkpoint.io import load_component

    if not os.path.isfile(os.path.join(checkpoint_dir, "trainer.pt")):
        info0(f"adapter resume check skipped for {checkpoint_dir!r}: no local trainer component")
        return
    saved = (load_component(checkpoint_dir, "trainer") or {}).get("objective_identity", {})
    if saved.get("adapters") is None:
        raise ValueError(f"checkpoint {checkpoint_dir!r} was trained without adapters; it cannot resume an adapter run")
    differences = identity_differences(saved["adapters"], identity)
    if differences:
        raise ValueError(
            f"checkpoint {checkpoint_dir!r} was trained with different adapters: {'; '.join(differences[:10])}"
        )
    if training is not None:
        changed = [
            f"{name}.{key}: {saved_settings.get(key)!r} (saved) != {training.get(name, {}).get(key)!r} (configured)"
            for name, saved_settings in sorted(saved.get("adapter_training", {}).items())
            for key in sorted(saved_settings)
            if saved_settings.get(key) != training.get(name, {}).get(key)
        ]
        if changed:
            raise ValueError(
                f"checkpoint {checkpoint_dir!r} was trained with different adapter training settings; exact resume needs "
                "the same ones. To change them, export the checkpoint (python -m "
                "cosmos_framework.posttrain.checkpoint.tools.export_adapters --checkpoint <iter_dir> --out <adapter>) "
                f"and start a new run with rl.adapter_warm_start=<adapter>: {'; '.join(changed[:10])}"
            )


def checkpoint_adapter_state(checkpoint_dir: str) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Read only the adapter tensors of a committed SFT checkpoint (consolidated or DCP).

    DCP checkpoints load the requested tensors only; frozen base weights and optimizer state are
    never read.
    """
    from cosmos_framework.posttrain.checkpoint.io import (
        component_kind,
        is_complete_iter_dir,
        load_component,
        load_component_dcp,
    )

    if not is_complete_iter_dir(checkpoint_dir):
        raise ValueError(f"{checkpoint_dir!r} is not a complete training checkpoint (trainer and model required)")
    identity = checkpoint_adapter_identity(checkpoint_dir)
    if identity is None:
        raise ValueError(f"checkpoint {checkpoint_dir!r} was not trained with adapters (no adapter identity)")
    validate_identity(identity)
    spec = adapter_state_spec(identity)
    if component_kind(checkpoint_dir, "model") == "pt":
        model = (load_component(checkpoint_dir, "model") or {}).get("net", {})
        missing = sorted(name for name in spec if not isinstance(model.get(name), torch.Tensor))
        if missing:
            raise ValueError(f"checkpoint model state lacks adapter tensors: {missing[:5]}")
        state = {name: model[name].detach().clone() for name in spec}
    else:
        state = {name: torch.empty(shape, dtype=torch.float32) for name, shape in spec.items()}
        # Only the requested keys are planned; a missing adapter tensor fails the load.
        load_component_dcp(checkpoint_dir, "model", state)
    validate_adapter_state(state, identity)
    return identity, state
